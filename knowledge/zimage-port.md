# Z-Image-Turbo port — a 6B diffusion DiT on Core AI, and why it is Mac-only

[`Tongyi-MAI/Z-Image-Turbo`](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) (6B,
Apache-2.0) → a Single-Stream DiT (S3-DiT) text-to-image model. Not the zoo's fastest —
FLUX.2 klein does 1024² in ~17 s to Z-Image's ~70 s — but the port is near-lossless (PSNR 42.6 dB)
and one graph serves every resolution and prompt length:
Qwen3-4B text encoder → 34-block DiT (8-step FlowMatchEuler + CFG) → 16ch AutoencoderKL.
Shipped as [mlboydaisuke/Z-Image-Turbo-CoreAI](https://huggingface.co/mlboydaisuke/Z-Image-Turbo-CoreAI).

The architecture was reverse-engineered from a **device-proven LiteRT/Android port** of the
same model (Pixel 8a, 256px, ~27 min). That port's write-up is the single most useful
document for anyone touching this model — it named the pad-token substitution, the negated
CFG, the penultimate-hidden conditioning and the fp16 NaN long before we hit them.

## Architecture → export mapping

| Piece | What it is | How it exported |
| --- | --- | --- |
| Text encoder | Qwen3-4B, causal, **penultimate hidden** `hidden_states[-2]` | fixed-L (64) graph over `input_ids` + an additive 4D mask (causal ∧ non-padding); `embed_tokens` is *inside* the graph. Host: chat-template tokenize → right-pad → build mask. Padding keys are masked, so valid-token outputs are pad-length independent. bf16 weights, fp32 boundary. |
| DiT | 3840 dim / 30 heads / hd 128; x-embed + cap-embed, 2 noise-refiner (adaLN) + 2 context-refiner (no adaLN) + 30 main + final; 3-axis RoPE | thin wrapper over the **stock diffusers blocks** (`NativeZDiT`). The only change: the attention processor's `view_as_complex` RoPE swapped for a bit-exact real interleaved form fed precomputed cos/sin. Host does patchify / RoPE / pad-mask / unpatchify. **bf16.** |
| VAE | 16ch AutoencoderKL decoder | fp32, scalar unscale `z/0.3611 + 0.1159`, `_patch_nearest_upsample`. Per-size (dynamic latent H/W trips `Constraints violated` — the decoder derives W from H). |
| Sampler | FlowMatchEuler, 8 steps, `guidance=1.0` | host loop |

Drivers: `conversion/zimage/{capture_oracle,export_dit,export_encoder,export_vae,pipeline_engine}.py`.
Working notes + every dead end: `conversion/zimage/ZIMAGE_STATE.md`.

## The three things that each cost a wrong image

1. **The DiT conditions on the encoder's PENULTIMATE hidden state** (`hidden_states[-2]`),
   not the last.
2. **Z-Image's CFG is negated**: `noise_pred = -(pos + g·(pos − neg))`, not the textbook
   `neg + g·(pos − neg)`.
3. **Caption padding is part of the computation.** `_pad_with_ids` pads the caption up to a
   multiple of `SEQ_MULTI_OF` (=32) and substitutes a *learned* pad token; those pad tokens
   are **real attention context** (the pipeline does not mask them). So `n_cap` must equal
   `round_up(L, 32)` exactly — truncating to the valid length drops per-step velocity
   correlation from 1.000000 to 0.994.
   Corollary: cond and uncond generally have **different** `n_cap` (a 45-token prompt → 64,
   the empty negative → 32), so each branch needs its own caption RoPE and pad mask.

## One graph for every resolution and every prompt length

`export_dit.py --dyn-cap --dyn-img` makes both the image-token and caption axes dynamic.
Cost is **+4.5–9 %** (6-layer probe 0.222 → 0.232 s/fwd; full 0.89 → 0.97), so there is no
reason to ship static-shape buckets — each would duplicate 11 GB of weights. `--dyn-cap` is
not optional in practice: a static `cap=32` graph cannot run a cond/uncond pair whose
`n_cap` differs.

## fp16 does not work. bf16 does. That decides everything.

Z-Image's residual stream reaches ~6.7e3 and its `feed_forward.w2` / `attention.to_out`
outputs reach 3.1e5. In an fp16 Core AI graph the DiT goes **all-NaN at sampler step 2**
(depth-driven — it happens at 256px too, with a perfectly healthy latent). The LiteRT port
hit the identical failure ("fp16 NaN in the adaLN path… the context refiner is clean") and
escaped with `GpuOptions(precision = FP32)`.

bf16 has the range and is exact. So the shipped Mac DiT is bf16.

**This is why the port is Mac-only.** `xcrun coreai-build compile` has **no flag to set
input element types**, and it feeds `f16` to the module:

```
Incompatible element type for parameter at index 0,
mlir module expected element type bf16 but received f16
```

so a bf16 module cannot be AOT-compiled, and iOS needs AOT for graphs this size.

### The activation-dtype ⇄ weight-storage coupling (general, affects every iOS port)

Measured on 6-layer probes (h18p):

| graph | source `.aimodel` | AOT `.aimodelc` |
| --- | --- | --- |
| int8 weights + **fp16** activations | 1.8 GB | **1.8 GB** (int8 kept, dequantized at run time) |
| int8 weights + fp32 activations | 1.9 GB | **8.7 GB** (weights constant-folded to fp32) |

MPSGraph folds the weight dequantization into fp32 constants when activations are fp32.
That single mechanism explains three separate observations: the fp32 graph was the *fastest*
int8 config (0.37 vs 0.68 s/fwd — the dequant is gone), it was 4× larger, and the shipped
iOS bundle `glm_ocr_decode_int8hu_s1` stays 764 MB through AOT (it uses fp16 activations;
`int8hu` is `int8lin` + an untied head — the quant scheme is identical). Placement is
irrelevant: `--preferred-compute gpu|none|neural-engine` all land on the MPSGraph delegate.

## int8 is a size trade, not a speed win — the DiT is compute-bound

| @512px, full depth | s/fwd (M4 Max) |
| --- | --- |
| bf16 | **0.89** |
| int8 weight-only + bf16 activations | 2.35 |
| int8 weight-only + fp16 activations | 2.35 |
| int8 per_channel / per_tensor granularity | 2.35 / 2.34 |

Activation dtype and granularity are irrelevant: a weight-only int8 graph dequantizes back
to 16-bit and runs the *same* matmul, so on a compute-bound shape (1056 tokens × 6B) it is
strictly more work than bf16. Weight-only int8 wins only on **bandwidth-bound** shapes
(LLM decode, S=1) and on footprint. The penalty is a constant ≈2.4–2.7× multiplier — it does
**not** get relatively worse at low resolution (0.678 vs 0.287 s/fwd at 256px).

⇒ **bf16 is the Mac default: faster *and* higher quality than int8.**

## Dead ends (do not repeat)

- **Hand-rolled ops lose to Core AI's composites, twice.** The LiteRT port's GPU-clean DiT
  (manual multi-head view/transpose/matmul attention) fails the versioned-IR pass
  (`cannot unwrap empty odiec_module_t`); Core AI wants attention as its SDPA composite,
  which the native diffusers attention emits. Likewise a hand-rolled max-norm RMSNorm —
  *exact in fp32 torch* — NaNs from step 0 inside the graph. Fix weights, not graphs.
- **fp16 NaN survives every weight transform.** `rescale_residual(K)` (residual 6.7e3 → 104),
  `rescale_fp16_safe(C)` (w2/to_out 3.1e5 → 9.8e3), and both together are output-exact
  (corr 1.000000) and leave **zero** fp32 module outputs above 65504 — and the engine NaNs
  identically. The NaN is input-dependent, not call-count (the same step-0 inputs run 6× are
  clean; step-2 inputs NaN on the very first call), so the overflow is inside a matmul
  accumulation or softmax, where a per-module hook cannot see it.
- **Activation quantization is not LiteRT's INTEGER path.** `op_input_spec` (int8 activations,
  calibrated on the real oracle inputs) exports fine but NaNs from step 0 — both globally
  (`"*"`, which quantizes the inputs of *every* op) and scoped to `torch.nn.Linear`. It
  appears to emit fake-quant nodes that still execute in fp16.

## Numbers (M4 Max, bf16, vs the fp32 diffusers reference)

| | s/fwd | denoise (8 steps, CFG = 16 forwards) | PSNR |
| --- | --- | --- | --- |
| 256px | 0.36 | 5.8 s | 35.60 dB |
| 512px | 1.12 | 17.9 s | 42.64 dB |
| 1024px | 4.36 | 69.7 s | 42.33 dB |

(Shipped `--io-fp32` graphs. The bf16-boundary variants were ~15 % faster and ~3 dB worse.)

Encoder 0.2 s (2 calls), VAE 0.2–0.8 s. PSNR is **not comparable across prompts**: a
texture-heavy oil-painting prompt scores 27.7 dB while being visually indistinguishable from
the reference — the composition, light and subject match exactly and only high-frequency
detail differs.

`guidance=0` renders fine at 256px and halves the work (16 → 8 forwards).

## Gates

`capture_oracle.py` records the fp32 diffusers run at the transformer boundary (per-step
latent / adaln / velocity for both branches, caption embeds, noise, sigmas). Everything is
teacher-forced against it:

- `parity_dit_torch.py` — the wrapper vs diffusers in fp32: **corr 1.000000**
- `engine_parity_dit.py` — the exported bf16 bundle, all 16 step×branch points: **≥ 0.99972**
- `engine_parity_encoder.py` — penultimate vs the pipeline's caption embeds: **0.999984**
- `pipeline_engine.py` — end-to-end image vs the reference, plus novel prompts and
  resolutions via `ref_image.py`

## Hosting it from Swift (the app tab)

A Swift host cannot fill or read a **bfloat16 `NDArray`** — `CoreAIRuntime.BFloat16` is not
public and a `UInt16` view trips the runtime's element-type check. Since bf16 is the only dtype
this DiT survives, the shipped graphs expose **fp32 boundaries with bf16 weights and bf16
compute** (`export_dit.py --io-fp32`, `export_encoder.py --io-fp32`). The cast costs ~15 %
per forward (0.97 → 1.12 s at 512²) and *raises* fidelity (39.5 → 42.6 dB) — nothing rounds
on the way in.

Three more things keep the host from re-implementing the reference:

- `export_encoder.py --ids` folds `embed_tokens` into the graph, so the app never carries the
  151936×2560 embedding matrix (778 MB bf16).
- `RopeEmbedder` is literally `cat([freqs[i][ids[:,i]] for i in 0..2])` — a **per-axis table
  lookup**. Three tables (392 KB) reproduce it exactly at any resolution and prompt length.
  Coordinates: caption token *i* → `(i+1, 0, 0)`; image token *(h,w)* → `(n_cap+1, h, w)`.
- The timestep MLP ships as a 2 MB `t_embedder` graph.

Host arithmetic that *is* re-implemented (all verified against the model):
patchify feature order is `f = (dy*2 + dx)*16 + c`; the sampler schedule has a closed form,
`σ_i = 3(1−t)/(3(1−t)+t)` with `t = i/(n−1)`, matching the diffusers scheduler to 3e-8 for
every step count and resolution.

⚠️ The VAE graph **bakes the un-scale** (`z/0.3611 + 0.1159`) — feed it the raw latent. Doing
it again on the host costs 18 dB (42.9 → 24.4) and is the one bug the Swift port actually hit.

`ondevice/ZImageRunner` compiles the app's `ZImagePipeline.swift` into a CLI so the Swift host
can be gated against the Python engine on identical noise: images agree to **42.9 dB**.

## Follow-ups

- **iPhone** is blocked on the fp16 NaN. The honest next step is an engine-side depth bisect
  (`--layers 1,2,4,8,…` on fp16, step-2 inputs) to find the first block that NaNs. If the
  toolchain exposes no accumulate-precision control, the exits are bf16 AOT support (the
  bf16 numerics are already proven) or LiteRT-style chunked FP32-compute (a ~24 GB artifact).
- The text encoder graph is fixed at **L=64** chat-templated tokens; longer prompts need
  `export_encoder.py --L 128` (or a dynamic L axis, same `Dim` trick as the DiT — untested).
- Editing / i2i, and `guidance=0` as a fast mode.
