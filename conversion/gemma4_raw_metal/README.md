# Gemma-4-E2B raw-Metal pack — conversion pipeline

Builds `gemma4_pack.(bin|json)`, the mmap-able mixed-bit weight pack driven by the
hand-written Metal decode loop (`apps/CoreAIChat` `Gemma4MetalEngine.swift` /
`Gemma4MetalBackend.swift`). Everything is validated by exact-token gates at every
stage — see `knowledge/gemma4-raw-metal-port.md` for the full story and
`knowledge/gemma4-mixedbit-qat-transplant.md` for the weight provenance.

## Inputs (external to this repo)

1. **Mixed-bit QAT weights** — Google's official Gemma-4-E2B mobile mixed-bit QAT
   release `google/gemma-4-E2B-it-qat-mobile-transformers` (Apache-2.0; int2/int4/int8
   + PLE tables), read by `official_extract.py` → `gemma4e2b_mixedbit_weights.safetensors`
   + manifest + fp32 norms + `final_norm.f32.npy`. Point `GEMMA4_EXTRACT` (or `EXTRACT`
   in `p0_sdpa_parity.py`) at that output directory. (Bit-exact equivalent of the earlier
   `.litertlm` extraction — see `knowledge/gemma4-litertlm-to-official-migration.md`.)
2. **Oracle refs** — `oracle_refs.json` (3 prompts, fp16 + bf16 greedy ids from the
   delegate-exact oracle) produced by the mixed-bit transplant pipeline. Set
   `ORACLE_REFS` in `p0_sdpa_parity.py`.

## Chain (run in order, each gate must PASS)

| step | script | what / gate |
|---|---|---|
| P0 | `p0_sdpa_parity.py` | flash-SDPA kernel vs fp32 reference (cos + max-abs) |
| P1 | `p1_chain.py --gate` | python RawLoop (dispatches the EXACT Swift kernel sequence via `torch.mps.compile_shader`) greedy ids == oracle fp16/bf16, 3/3 |
| P2 | `p2_export_pack.py` | dump kernel-layout tensors → `pack/gemma4_pack.(bin|json)` |
| P2b | `p2b_interleave_pack.py` | `pack/` → `pack_il/`: 4-row-interleave the QP words for the A19 uint4-load kernels (values/dot order unchanged ⇒ bit-exactness carries) |
| P3 | `p3_verify.py` / `p3_mtp.py` | S=4 verify lane + MTP drafter parity (bench app only — the chat engine is S=1) |

`msl/` holds the canonical kernel sources. The app bundles copies as
`Resources/g4msl/*.metal.txt` (Xcode must NOT compile them — the engine compiles at
load with `mathMode .safe`). If you touch a kernel or the dispatch sequence, rerun
P0/P1 AND the on-device S1 token gate (`G4CHAT_GATE=1` app launch, or the g4bench
harness) before trusting any output.

## Ship layout (HF `mlboydaisuke/gemma-4-E2B-metal` — standalone engine repo)

The engine + this pipeline ship as the standalone `gemma4-metal` repo
(github.com/john-rocky/gemma4-metal); this zoo copy is the integration mirror.

```
gemma4_e2b_raw_metal/
  gemma4_pack.bin    # 2.18 GB, interleave-4 pack (pack_il output)
  gemma4_pack.json   # manifest + meta (interleave4: true)
  tokenizer/         # stock gemma tokenizer files
  metadata.json
```

Kernels ride the app/kit bundle (they version with the host dispatch code, not the
weights). Tokenizer = the stock gemma tokenizer already bundled by the apps.

## Numbers (iPhone 17 Pro, fresh settled trial1, p128 g256)

Engine 36.5 → raw loop 55–56.2 tok/s, lossless (S1 token gate 3/3 at every step);
same-afternoon interleaved A/B vs LiteRT-LM's own benchmark: raw median 53.7 vs
LiteRT 50.5 (2026-07-15). Mac M4 Max: S=1 124.1; MTP 181.9 (bench lane only).
