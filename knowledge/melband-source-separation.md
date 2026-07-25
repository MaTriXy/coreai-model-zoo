# Music source separation on Core AI (Mel-Band RoFormer)

Porting notes from the zoo's first source-separation model
([`KimberleyJSN/melbandroformer`](https://huggingface.co/KimberleyJSN/melbandroformer), MIT, ~228 M).
The shipped card is [`models/melband-roformer/README.md`](../models/melband-roformer/README.md); the recipe lives in
[`conversion/melband_roformer`](../conversion/melband_roformer).

## The shape of the problem

A separator is `waveform → STFT → (neural mask estimation) → iSTFT → waveform`. Two things in that
chain do not lower to Core AI:

1. **complex tensors** — the band-split, the mask, and the mask application are complex-valued;
2. **`scatter_add`** — the band-average step (overlapping mel bands write back into shared bins).

Everything else (band-split projections, 6 axial rotary transformer blocks alternating time/frequency,
the mask estimator MLPs) lowers as-is. So the port is not a rewrite — it is **two local
re-formulations**:

- **Real arithmetic.** Carry real/imag as an extra size-2 axis. The complex mask multiply becomes
  `(ar·br − ai·bi, ar·bi + ai·br)` on real tensors; the band-average scatter becomes a **constant
  matmul** with a precomputed `A` matrix (bands × bins) — the scatter pattern is static, so it is data,
  not control flow. Gate: cos **1.0000000** vs the reference model.
- **STFT/iSTFT as constant DFT matmuls.** With a fixed frame count, the DFT is just a `[2048 → 2·1025]`
  matrix; the analysis window folds into it, and the synthesis side is its transpose with the
  window-squared normalizer left to the host. Gate: cos **0.9999984** vs `torch.stft`.

The second move is what makes the mobile host trivial. The shipped graph is
`frames[1,2,801,2048] → recon[1,2,801,2048]`: the Swift host does reflect-pad, framing (stride = hop),
overlap-add, divide by `Σ window²`, trim. **No FFT, no vDSP packing, no complex buffers.** On Apple
platforms that is the difference between "port the model" and "port the model *and* re-derive
`vDSP_fft_zrip` packing conventions against a Python reference".

## Fixed shapes beat dynamic ones here

The model is chunked anyway (8 s = 352 800 samples @ 44.1 kHz, 801 frames, overlapping chunks
crossfaded by the host), so nothing needs a dynamic axis. That buys:

- no `--expect-frequent-reshapes`, no on-device recompiles (see
  [`aot-and-specialization.md`](aot-and-specialization.md) for why the hint is actively harmful when
  shapes are static);
- one AOT `.aimodelc` per architecture that covers every song length.

## Numbers (fp16)

| gate | result |
|---|---|
| re-authored real-arithmetic core vs reference | cos 1.0000000 |
| in-graph STFT/iSTFT vs `torch.stft` | cos 0.9999984 |
| Core AI fp16 Mac GPU, full framing + overlap-add | cos 0.9999453 |
| iPhone 17 Pro (h18p AOT, GPU) vs Mac golden | cos 1.000000, rms ratio 1.0000 |

iPhone 17 Pro: 8 s chunk in **1.23 s (6.5× real-time)** warm, 3.82 s (2.1×) on the cold first run —
the usual DVFS ramp; quote both, not the better one.

## Reusable bits

- The **real-arithmetic + constant-matmul-DFT** pattern applies to any STFT-domain model (other
  RoFormer/BS-RoFormer stems, denoisers, dereverb). Only the mask semantics change.
- `attend.Attend.flash_attn` in the lucidrains stack must be swapped for
  `F.scaled_dot_product_attention` before export, and `scaled_dot_product_attention` / `rope` must be
  dropped from `_EXTERNALIZE_SPECS` so they lower into the graph instead of becoming external ops.
- `model.load_state_dict(..., strict=False)` is correct here: the checkpoint carries training-only
  keys the inference model does not declare.
- Goldens (`golden_raw.f32` / `golden_vocals.f32`, stereo channel-major float32) double as the demo
  clip **and** the on-device self-test target — one artifact, two jobs.
