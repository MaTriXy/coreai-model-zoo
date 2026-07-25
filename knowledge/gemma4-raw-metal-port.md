# Gemma-4-E2B raw-Metal port — hand-written mixed-bit decode loop (ship doc)

**What shipped**: Gemma-4-E2B running on a fully hand-written Metal decode loop — no
Core AI engine, no `.aimodel`, no MPSGraph. A 2.18 GB mmap'd pack of Google's official
QAT mixed-bit weights (int2/int4/int8 + PLE tables) is driven by 5 hand-tuned kernel
files and a ~250-dispatch-per-token host sequence with on-GPU argmax. **Lossless**
(token-exact vs the fp16 oracle, S1 gate 3/3 at every optimization step) and at
**LiteRT-LM speed parity on iPhone 17 Pro** (same-afternoon interleaved A/B vs
LiteRT-LM's own benchmark entry point, 2026-07-15: raw median 53.7 vs LiteRT 50.5
tok/s; session best 56.2). Mac M4 Max: S=1 124.1 tok/s (engine int4lin path: 82.4).

- User-facing engines: `apps/CoreAIChat` (`Gemma4MetalBackend`, picker "Gemma 4
  ⚡raw-Metal", headless `GEMMA_ENGINE=rawmetal`) and coreai-kit
  (`ChatSession(catalog: "gemma-4-e2b-metal")`, `Gemma4MetalRuntime`).
- Generalized lessons (kernel-vs-glue decomposition, when a raw loop pays, prefill
  physics vs losslessness, expansion candidates): `knowledge/raw-metal-loop-playbook.md`.
- Conversion pipeline: `conversion/gemma4_raw_metal/` (P0→P2b chain, gates at every
  stage). Weight provenance + transplant analysis:
  `knowledge/gemma4-mixedbit-qat-transplant.md`. A19 tuning levers that did/didn't
  work: `knowledge/gemma4-raw-metal-a19-levers.md`.

## Why this exists (the byte floor, and what sits above it)

Decode is bandwidth-bound. The shipped int4lin engine bundle reads ~2.0 GB/token; the
QAT mixed-bit weights read **783 MB/token** — but the stock engine graph cannot express
the int2/int4/int8 mix + PLE gather at full efficiency (36.5 tok/s on iPhone = 28.6
GB/s effective). The raw loop exists to harvest the missing bandwidth with exact
control over kernels and dispatch: 36.5 → 55–56 tok/s (43.8 GB/s effective vs
LiteRT-LM's 44.6 = 98%) with zero quality cost.

## Architecture

- **Pack** (`gemma4_pack.bin` + `gemma4_pack.json`): every tensor already in KERNEL
  layout (quantized words `qp`, scales `sc`, biases `bi`, per-layer norms, PLE tables,
  packed embeddings, rope inv-freq tables), 64 B aligned, mmap'd into ONE
  `bytesNoCopy` MTLBuffer — load = mmap + JSON parse, weights never copied. The
  shipped pack is the **interleave-4** variant (`interleave4: true`): QP words of 4
  consecutive rows sit in one uint4 for single-16B-load fetches; per-row word values
  and dot order are unchanged, so bit-exactness proofs carry.
- **Kernels** (6 files, compiled at load with `mathMode .safe`): mixed-bit matvecs
  (int2-symmetric with a constant-memory byte-LUT decode, int4-affine, int8), fused
  gate+up FFN kernels with residual/rmsnorm prologue folds, fused q/k/v projection,
  flash-SDPA with rope+v-norm merged in (`flash_sdpa_rope_occ`, G-way seq split),
  two-stage on-GPU argmax, the S=4 verify lane (+ drafter, bench only), and the
  wide-prefill M=8/16 widenings (`gemma4_prefill.metal`).
- **Host loop**: ~253 dispatches/token, all token-dependence resolved ON GPU (the
  argmax writes the next token id into the token buffer the next step's embed-gather
  reads — the CPU never touches the token chain). Steps are encoded 8 per command
  buffer, 3 CBs in flight.
- **Chat engine** (`Gemma4MetalEngine.swift`, identical file in CoreAIChat and
  coreai-kit): greedy generate over the verbatim dispatch sequence, streaming per-CB,
  stop ids `<eos>`=1 / `<turn|>`=106, **cross-turn KV prefix reuse** (longest common
  prefix with the previous call is not re-prefilled — the kit ChatSession
  `trimKVCache` contract maps straight onto it). Per-platform SDPA split defaults:
  iOS G_sliding=8/G_full=8 (A19-tuned), macOS 16/8 (M4-tuned).
- **Batched prefill (wide chunks, default M=8)**: prefill runs the MTP verify-lane
  dispatch sequence over M-position chunks (16/8/4 widest-fit + S=1 remainder;
  `prefillM`, env `G4_PREFILL_M`). M=4 uses the gated verify kernels verbatim;
  M=8/16 use `gemma4_prefill.metal` widenings that keep every output scalar's
  EXACT S=1 accumulation order (loop staging is the only difference), so chunked
  KV is **byte-identical to S=1 KV** — proven directly by a KV cache byte-compare
  (flat+il packs, window-crossing and unaligned-resume prompts) plus the S1 token
  gate on device. Measured: Mac M4 Max m8 **553-560 @p128 / 508-510 @p512 /
  464-465 @p1024** (+24% over the m4 chunks, ≈4.5× over S=1). iPhone 17 Pro: m8 ≈
  m4 parity — the A19 prefill is ALU/clock-bound well above its byte floor, so
  width alone doesn't pay there (m8 halves the byte floor for headroom).
  Variants measured and REJECTED (kept in the file, off by default): m16
  (register spill on both GPUs), staged x-stage bodies `_m8s/_m16s` (dead on A19,
  poor on M4), byte-LUT int2 `_m4l/_m8l` (neutral on A19 prefill, slightly worse
  on Mac — decode's LUT win does not transfer to the wide lane).
- **A19 prefill absolute numbers are DVFS-ramp dependent** (session-6 settled-day
  verdict, 17 single-variable runs): a p347 prefill launched from device-idle
  finishes before the GPU clock ramps (**66-68 tok/s**); a p~1000 prefill ramps
  mid-run (**87 tok/s**, m4 ≈ m8); runs launched right after sustained UI
  interaction hit **~95-102** (that is what session-4's single 94.7 reading was).
  Thermal, Low Power Mode, cable vs battery, screen state and brightness were all
  eliminated as causes. Quote the pair "≈87 tok/s @p1k / 66-68 @p347 cold-start"
  — never a pre-ramped burst number alone. Byte-bound decode barely moves between
  regimes (51-52 @ctx≈380), which is also the thermal tell: decode sliding below
  ~51 means the device is genuinely warm (charging + screen-on are the heaters).

## Discipline (do not relax)

1. **mathMode .safe + literal op sequences.** Fast math contracts/reassociates fp16
   chains differently per kernel shape; near-ties then fork. This bit twice (see
   GEMMA4_METAL_LOOP_STATE.md traps). Any kernel or dispatch-order change requires the
   S1 token gate on device (`G4CHAT_GATE=1` in CoreAIChat, oracle refs bundled) plus
   the python P0/P1 gates.
2. **Kernels version with the HOST, not the weights** — they ship inside the app/kit
   bundle (`Resources/g4msl`, `Gemma4Metal/g4msl`), never next to the pack on HF.
3. **The gates are the only proof.** tok/s claims come from settled fresh trial1 runs
   (see the A/B protocol in GEMMA4_METAL_LOOP_STATE.md SESSION 3); losslessness claims
   come from the S1 token gate, never from eyeballing text.

## Known limits (v1)

- **Greedy only** (argmax on GPU; no logits surface, no sampling, no guided gen).
- **Prefill is chunked, not GEMM-tiled** (bit-exact lane: Mac m8 ≈4.5× over S=1;
  iPhone ~+31% over S=1) — still well below LiteRT-LM's wide-batch prefill
  (452–3,250 tok/s). The A19 side is ALU/clock-bound well above its byte floor:
  wider chunks, staged bodies, and byte-LUT int2 were all measured session-5 and
  don't move it, and the fused wide lane (rmsnorm folds in the matvec prologues,
  `gemma4_prefill_fused.metal`) was built, gated bit-exact, measured session-6 and
  **killed: -34% Mac / -40% A19** — the fold recomputes at least once per
  threadgroup vs once globally for the glue, and in an ALU-bound lane that always
  exceeds the ~1-3% dispatch savings (the S=1 fused lane wins only because decode
  is byte-bound and the fold ALU hides under the weight stream). **The bit-exact
  prefill chapter is closed**; what remains on A19 is DVFS clock behavior and the
  parked M=64+ simdgroup_matrix GEMM step, which changes reduction order ⇒
  same-quality-class instead of bit-exact — a user decision.
- **MTP (multi-token prediction) stays OFF on iPhone** — with all levers it reaches
  48.0 < S=1's 55.9 on A19 (it wins on Mac: 181.9 vs 124.1, bench lane only, not in
  the chat engines yet).
- E2B only; ctx 4096 (pack export bound).

## Ship layout

HF `mlboydaisuke/gemma-4-E2B-CoreAI`, subtree `raw-metal/gemma4_e2b_raw_metal/`:
`gemma4_pack.bin` (2.18 GB) + `gemma4_pack.json` + `tokenizer/` (stock gemma
tokenizer files) + `metadata.json`. Both platforms share the one pack; kit catalog id
`gemma-4-e2b-metal`, CoreAIChat downloads the same subtree. Weights are Google's
Gemma-4-E2B QAT parameters from the official `google/gemma-4-E2B-it-qat-mobile-transformers`
release — Gemma Terms of Use apply, and the model card credits the source.
(Migrated from the earlier `.litertlm` extraction; bit-exact — see
`knowledge/gemma4-litertlm-to-official-migration.md`.)
