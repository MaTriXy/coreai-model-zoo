# Spec-decode C1 handoff (2026-07-03) — integrate the loop into the pipelined engine

> One page for the next session. Full detail: `spec-decode-hybrid-verify-design.md`
> (all measurements) + memory `project_spec_decode_port`.

## THE TASK

Make Qwen3.6-27B ⚡Spec **wall-clock faster than the shipped 15.9 tok/s decode** by
running the (fully validated) lossless spec-decode loop on **engine-grade forwards**
— then the user records the app and posts to X. Everything else is done.

Two integration shapes to choose between (open design decision):
- (a) **Engine-internal spec mode**: extend `CoreAIPipelinedEngine.swift` (zoo-patched
  fork under `coreai-models/swift/`) with a draft+verify path — PB_SPEC precedent on
  iOS (`ondevice/PipelinedBench/Sources/SpeculativeDecoder.swift`). C1 analysis
  (2026-07-01, in memory): chunked-prefill S=K path exists, KV rollback = just
  `processedTokenCount`, needs a `[K,vocab]→[K]` batch-argmax graph (~20 lines).
  ⚠️ That rollback is NOT sufficient for the 27B GDN hybrid — conv/rec states need
  the snapshot/tail/exact-S re-anchor discipline (below).
- (b) **Keep the two-model outer loop** (today's `SpecDecodeEngine.swift`) but route
  forwards through the engine's kernels instead of raw `AIModel` JIT.

## WHY (today's measurements, M4 Max)

- Loop itself is perfect: E2E in CoreAIChatMac, correct code output, α 2.71–3.19/round,
  zero MTL4/ANECompile noise (`expectFrequentReshapes=true`).
- Wall tiers of the RAW-driven window graph: cold 2.0 → disk-cache-warm 3.6 →
  **in-process warm (turn 2) 10.9 tok/s**. Bottlenecks: ① per-anchor in-process
  specialization load (~1 s/shape, from the 29 GB runtime cache; shape set is CLOSED —
  anchors advance in steps of S, so positions are multiples of 9, ≤227 shapes) and
  ② raw MPSGraph JIT forward = **135 ms vs the pipelined engine's 63 ms** (custom
  Metal kernels; ~half bandwidth). Spec cuts forwards 2.6× but the kernel gap eats it:
  10.9 < 15.9. Engine-grade forwards project **25–30 tok/s ≈ 1.6–1.9× shipped**.
- ⛔ AOT dead end on THIS OS build: beta `coreai-build` 3600.67 output → runtime
  26A5353q rejects with `invalidCompiledModel` (format skew; iOS works because the
  phone runs a matching beta). Engine tries AOT slice, falls back to JIT (0c1b0ab).
  Revisit after an OS/toolchain update. This Mac's GPU arch = h16s.

## FROZEN CONFIG (do not re-litigate)

- Target `exports/qwen3_6_27b_verify_s9_int8hu_block32_sym` (metadata has
  `spec_draft`) × draft `exports/qwen3_5_0_8b_verify_s9_int4lin`, **K=6, S=9**.
  S=13 K=8 pair on disk as a code-mode option. S=17 disproven (c_v cliff).
  Draft A/B: int4 holds α at half bytes; 2B dropped. No training needed
  (EAGLE-3 = C3, separate; pretrained head exists — Ex0bit/Qwen3.6-27B-PRISM-EAGLE3,
  τ2.2 ⇒ no gain without tree + hidden-state-tap export).

## REUSABLE PARTS (commits f574c75, 27f4fc4, 0c1b0ab in this repo)

- `apps/CoreAIChatMac/Sources/SpecDecodeEngine.swift` — WindowedModel discipline
  (snapshot conv/rec only; KV rows self-overwrite), fused greedy verify indices
  (python-proven), anchor bootstrap via held-back last prompt token (fixes the
  prompt%S==0 pad-row bug — also fixed in `ondevice/_spec_mac_two_model.py`,
  NOT under git), draft auto-pairing, ⚡Spec toggle + α stats in ChatView.
- Swift gotcha: `MutableViews` is ~Escapable — insert from DISTINCT local vars
  (4 states), never `&array[i]`.
- Build: `cd apps/CoreAIChatMac && xcodegen`, then Xcode-beta xcodebuild Release
  (`-derivedDataPath build/dd-release CODE_SIGNING_ALLOWED=NO`), ad-hoc codesign,
  launch binary directly with `CHATMAC_MODEL` / `CHATMAC_PROMPT` / `CHATMAC_PROMPT2`
  env hooks for hands-free runs. Bundles are symlinked into
  `~/Library/Application Support/CoreAIChatMac/models/`.

## MINOR OPEN ITEMS

- Stats `tok/fwd` is diluted by prefill re-anchor forwards (display only).
- Stop button stops display, loop runs to maxNewTokens (LLaDA-style).
- HF upload of the verify pair + catalog entries already point at
  `gpu-pipelined/<bundle>` layout — **user-gated** (SHIP-GATE), as are push/X.
- 27B/35B raw HF weights kept on disk (C3 hidden-state-tap export needs 27B raw).

Conventions: GPU-SOLO `_GPU_LOCK`; no "claude" in commits; UI/comments in English;
disk guard ≥120 G before 27B runs (mpsgraph temp is transient, df lags APFS reclaim).
