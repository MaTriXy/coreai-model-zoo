# TripoSplat → Core AI (zoo's first 3D)

[VAST-AI/TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) — **single image → 3D Gaussian
splats** (`.ply`/`.splat`), MIT. The zoo's first 3D model: outputs drop straight into a Gaussian-splat
viewer (e.g. Apple RealityKit on visionOS, or [MetalSplatter](https://github.com/scier/MetalSplatter)
on iOS/macOS).

Pure-PyTorch pipeline (no diffusers/CUDA kernels): bg-removal → DINOv3 ViT-H encode + Flux2-VAE encode
→ 20-step flow-matching DiT denoiser → octree probability sampler → Gaussian decoder → splats.

## What runs on Core AI

5 neural nets converted (each gated converted-vs-eager **cos = 1.000000**, fp16):

| net | shape | bundle |
|---|---|---|
| DINOv3 ViT-H encoder | (1,3,1024,1024)→(1,4101,1280) | `dinov3` |
| Flux2-VAE encoder | (1,3,1024,1024)→(1,4096,128) | `vae` |
| DiT denoiser (one step) | latent(1,8192,16)+cam(1,1,5)+t+feat1(1,4101,1280)+feat2(1,4101,128)→latent,cam | `dit` |
| Octree probability decoder | x(1,8192,3)+l(1,)+cond(1,8192,16)→logits(1,8192,8) | `octree` |
| Decode (gs + build_gaussians + .ply activations, baked) | points(1,8192,3)+cond(1,8192,16)→(262144,14) | `decode` |

The flow-matching sampler (`FlowEulerCfgSampler`) and the octree `sample_probs` systematic resampling
stay **host-side** (data-dependent control flow). Scripts: `_conv_*.py` convert+gate each net;
`_conv_fp16.py` makes the half-size fp16 bundles; `_conv_decode.py` bakes build_gaussians + the
Gaussian `.ply`-activation math into one net so the runner just writes raw floats.

## model.py patches (the reusable contribution — see ../../knowledge/conversion-guide.md)

coreai-torch 0.4.0 needed six edits to VAST's `model.py`; all are general gotchas:

1. **float-arg `aten.arange`** → `bad_optional_access` C++ abort. Use int-arg arange (DINOv3 RoPE).
2. **fx `got multiple values for 'mod'`** — submodule called with `mod=` kwarg. Pass positionally.
3. **No complex ops** — rewrote the DiT's complex RoPE (`torch.polar`/`view_as_complex`) as real
   cos/sin math (`apply_rotary_emb`, `RePo3DRotaryEmbedding.forward`).
4. **Constant-folded `sin/cos` of huge args is low-precision** (cos→0.5) — the DiT positional embed
   computed from the fixed Sobol constant was folded wrong; precompute it into a `register_buffer`.
5. **`F.normalize` drops the eps clamp** → near-zero vectors blow up ~1e13; rewrote `MultiHeadRMSNorm`
   as explicit `x*rsqrt(mean(x²)+eps)`. (Emergent only at large seq len — gate by VISUAL/true-scale.)
6. **`prog.optimize()` hangs** on the 24-block/12k-token DiT graph (>90 min) — skip it
   (`convert(optimize=False)`), AOT `coreai-build` optimizes for the device anyway.

Plus: int8 desaturates this model (per-net cos 0.9998 but colors collapse → use **fp16**, which is
GPU-identical to fp32 — gate fp16 on GPU/visual, its CPU cos looks bad but that's a CPU-compute
artifact). Octree decoder: int64 `l` (resolution) input → CoreAIError 3 at runtime, pass it as float32.

## Running it

- **Mac**: `_run_coreai.py` (or `app_backend.py --input <img>`) loads the bundles via coreai.runtime
  (`SpecializationOptions.default()` = GPU; ~1 min/gen at 20 steps, full quality). End-to-end latent
  gate vs torch-DiT: **cos 0.999999**.
- **Mac app / iPhone client**: `apps/` — `TripoSplatMac` (standalone) and `TripoSplatPhone`
  (capture on iPhone → Mac server `server.py` → view splats in
  [MetalSplatter](https://github.com/scier/MetalSplatter) / RealityKit).

## On-device note

Full on-device (iPhone) was **verified infeasible** with this model: DINOv3 ViT-H AOT `.aimodelc`
is ~3.1 GB and the DiT's 12294-token full-attention score matrix alone is ~4.8 GB, both over the
~3.3 GB iOS app memory budget (weight precision doesn't fix the attention working set). Needs
flash-attention conversion / weight streaming. The Mac-link client is the shipped path.
