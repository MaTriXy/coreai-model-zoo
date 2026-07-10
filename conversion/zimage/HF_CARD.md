---
license: apache-2.0
base_model: Tongyi-MAI/Z-Image-Turbo
tags:
  - core-ai
  - coreai
  - text-to-image
  - diffusion-transformer
  - on-device
  - macos
pipeline_tag: text-to-image
library_name: coreai
---

# Z-Image-Turbo — Core AI (macOS)

Alibaba Tongyi-MAI **[Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)**
(6B, Apache-2.0) — a Single-Stream Diffusion Transformer (S3-DiT) — converted to **Core AI**
and generating images entirely on the Mac GPU.

A Qwen3-4B text encoder conditions a 34-block DiT that denoises in 8 FlowMatchEuler steps
with classifier-free guidance; a 16-channel VAE decodes. Photoreal by default.

**One DiT graph covers 256², 512² and 1024², and any prompt length** — both the image-token
and caption axes are dynamic, at a ~5–9 % cost over static shapes.

## What's here

| file | role | size |
| --- | --- | --- |
| `zimage_dit_..._bf16_dyncap_dynimg.aimodel` | the DiT — any resolution, any prompt length | 11 GB |
| `zimage_encoder_seq64_full_bf16.aimodel` | Qwen3 text encoder → **penultimate** hidden | 6.6 GB |
| `zimage_vae_{256,512,1024}_fp32.aimodel` | 16ch VAE decoder (per-size) | 189 MB each |

**bf16, not int8.** On this compute-bound graph weight-only int8 is *slower* than bf16
(2.35 vs 0.89 s/forward at 512²) because it dequantizes back to 16-bit and runs the same
matmul — it only wins on bandwidth-bound shapes and on footprint. bf16 is also what keeps
the port numerically near the fp32 reference.

## Speed & fidelity (M4 Max, vs the fp32 `diffusers` reference)

| | s/forward | denoise (8 steps, CFG = 16 forwards) | PSNR |
| --- | --- | --- | --- |
| 256² | 0.29 | 5.4 s | 41.9 dB |
| 512² | 0.96 | 15.4 s | 39.5 dB |
| 1024² | 4.02 | 64.2 s | 42.4 dB |

Per-step velocity correlation vs the reference is ≥ 0.9997 at every step and both CFG
branches. PSNR is not comparable across prompts: a texture-heavy oil-painting prompt scores
27.7 dB while being visually indistinguishable from the reference.

## Usage

The DiT graph takes host-prepped inputs (patchify, RoPE, pad masks) and returns the
velocity; the sampler loop lives on the host. A complete, runnable reference is
[`conversion/zimage/pipeline_engine.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/zimage/pipeline_engine.py)
(~80 lines):

```python
# per step, for cond and uncond:
#   ins = build_native_inputs(rm, latent, cap)          # patchify + RoPE + pad masks
#   v   = dit(**ins, adaln=t_embedder(t * t_scale))     # Core AI graph
#   vel = unpatchify(v[:, :n_img])
# noise_pred = -(pos + guidance * (pos - neg))          # Z-Image CFG is NEGATED
# latent    += dsigma[s] * noise_pred                   # FlowMatchEuler
# image      = vae(latent)                              # unscale: z/0.3611 + 0.1159
```

Three details each cost a wrong image:

1. the DiT conditions on the encoder's **penultimate** hidden state (`hidden_states[-2]`);
2. the CFG is **negated**: `-(pos + g·(pos − neg))`, not `neg + g·(pos − neg)`;
3. captions are padded to a multiple of **32** with a *learned* pad token that is **real
   attention context** — `n_cap = round_up(L, 32)` must match, and cond/uncond generally
   have different `n_cap` (hence the dynamic caption axis).

## Notes

- **macOS only.** fp16 sends this DiT all-NaN at sampler step 2 (depth-driven, at every
  resolution); bf16 is exact — and `coreai-build compile` refuses a bf16 module, which iOS
  needs for graphs this size. Full analysis in the
  [port notes](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/zimage-port.md).
- The text-encoder graph is fixed at **64 chat-templated tokens** (≈ 35–40 words).
- `guidance=0` skips CFG — half the work, a different composition, still clean at 256².
- Weights are **not redistributed as source**: the graphs are produced from the original
  Apache-2.0 checkpoint by the conversion scripts in the zoo.

License: Apache-2.0 (inherited from Z-Image-Turbo).
