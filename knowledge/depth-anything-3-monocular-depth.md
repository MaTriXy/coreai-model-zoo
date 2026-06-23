# Depth Anything 3 — on-device monocular depth

[Depth Anything 3](https://github.com/ByteDance-Seed/depth-anything-3) (ByteDance, Apache-2.0) is the
zoo's first **depth** model: a DINOv2 ViT backbone + DPT-style dense head that predicts a relative
depth map from one RGB image. DA3 is an "any-view" model (1→N views); fed a **single view (S=1)** it
is a monocular depth estimator. HF: `mlboydaisuke/Depth-Anything-3-CoreAI` (small + base × fp16/fp32).
Conversion: `conversion/export_da3.py`. Sample: `knowledge/scripts/depth_anything_3_sample.py`.

## Architecture / graph

`da3-small` = DINOv2 **ViT-S** (alternating cross-view attention + 2D RoPE + QK-norm + a camera
token) → **DualDPT** head (depth + confidence + a ray aux head). `da3-base` = ViT-B, same shape.
`da3mono-large` = ViT-L + plain **DPT** (depth only, no cross-view). We export only the depth path —
**backbone + head → `depth`, `depth_conf`**; the camera decoder, ray aux head and sky post-processing
are dropped (the ray branch is dead-code-eliminated by `optimize()` because only depth/depth_conf are
named graph outputs).

Why S=1 just works as a static graph: the cross-view **global attention collapses to self-attention**
(s=1), the reference-view reorder is **statically dead** (it needs S ≥ a threshold), and the camera
token is a **fixed parameter** — no data-dependent control flow survives.

Exported as ONE static graph: **`image [1,3,504,504]` RGB raw `[0,1]` → `depth [1,504,504]` (+
`depth_conf`)**. R = 504 = 36×14 matches DA3's default `process_res`; the DINOv2 pos-embed bicubic
interpolation is over fixed sizes so it **folds to a constant at export** (no runtime bicubic).

## ⚠️ The graph normalizes in-graph — feed RAW [0,1] (this cost a day)

The `ExportWrapper` folds ImageNet mean/std into the graph (`x = (image - mean) / std`). **The
runtime input must therefore be raw `[0,1]` RGB.** Feeding an already-ImageNet-normalized tensor
double-normalizes and silently corrupts the depth.

This bug masqueraded as model/engine problems for a long time. A comparison harness that fed
normalized input to the engine produced: a fake **cos ≈ 0.9 engine-vs-torch**, "the engine output
looks noisy", "non-square exports are broken (cos 0.9)", "letterbox padding breaks attention". **All
of it was the double-normalization.** With raw `[0,1]`, the engine is **cos 1.000000 vs torch at any
fixed shape — square AND non-square** (verified on diverse images, relmax ~1e-5 fp32 / ~1e-2 fp16).
Lesson: when an on-device vision graph "looks subtly wrong", first confirm where normalization lives
(in-graph vs host) before blaming the conversion. (CoreAIKit's `DepthEstimator` uses an identity
preprocessor — `ImagePreprocessor(mean: 0, std: 1)` — for exactly this reason.)

## Input contract: square + resize-back (SQUISH), not letterbox

The bundle is a fixed **square** graph. Host: resize the image to 504×504 (cv2 `INTER_AREA`), feed
raw `[0,1]`, run, then **resize the depth back to the original H×W**. Depth is relative, so the brief
aspect squash is recovered by the resize-back.

- **Faithfulness:** squish vs the official DA3 viewer is **mean Pearson r ≈ 0.98** across aspect
  ratios (a square input is **r = 1.000**). That deviation is **within DA3's own resolution
  sensitivity** — its 504-vs-518 outputs already differ by r ≈ 0.975–0.984. So a fixed-square
  deployment is faithful to the model's intrinsic floor, not lossy. **Measure a model's own
  resolution variance before calling a fixed-res port "unfaithful."**
- **Square vs non-square:** the engine is bit-exact at *any* fixed shape, so a per-aspect **non-square**
  bundle (e.g. 504×280 for 16:9) is also possible and is exactly the official preprocessing — useful
  for an exact-match deployment. The single square bundle + resize-back is the simplest contract and
  is what ships.
- Display: the DA3 convention is **inverse-depth → percentile 2–98 normalize → `Spectral` colormap**
  (far = red, near = blue).

## Export patches (numerically identical)

In `conversion/export_da3.py`, applied at import:

1. **RoPE table length baked as a constant.** `RotaryPositionEmbedding2D` sizes its cos/sin table by
   `int(positions.max()) + 1` — a Python int pulled from a traced tensor → a data-dependent guard
   under `torch.export`. The grid is static, so we bake the length.
2. **RoPE / PositionGetter caches made cache-free.** Both memoize tensors into dicts; `torch.export`
   poisons those dicts with **fake** tensors, and a later eager run in the same process then dies
   with `GuardOnDataDependentSymNode`. Recompute every call.
3. **pos-embed UV grid dtype.** The DPT `_add_pos_embed` builds a fp32 sin/cos grid
   (`make_sincos_pos_embed` hard-casts `.float()`); under fp16 it upcasts the feature map and the next
   conv sees Half weights vs float input. Cast it back to the feature dtype.

No GPU-delegate op workarounds were needed (unlike RF-DETR): bilinear upsample, 2D RoPE, SDPA and
ConvTranspose all lower as-is.

## fp16 works — but `.half()`, not autocast

Depth is robust: fp16 holds **cos 1.000000** (≤ ~1% per-pixel). Two rules:
- **`copy.deepcopy(wrapper).to(fp16)`**, never `wrapper.to(fp16)` in place — the caller reuses the
  fp32 module as the verify oracle, and `.to()` mutates in place.
- **No `torch.autocast`** — under autocast LayerNorm stays fp32, so its output collides with the Half
  conv that follows (`Input type (float) and bias type (c10::Half)`). Half the whole model, run Half.

CoreAIKit feeds `.float32(pixels)` regardless; `makeNDArray` converts float32→fp16 from the input
descriptor, so the same Swift path drives the fp16 bundle.

## Matrix (M4 Max GPU, vs the HF eager fp32 reference)

| variant · dtype | params | size | parity | M4 Max GPU |
|---|---|---|---|---|
| small · fp32 | 34.3M | 105 MB | cos 1.000000 (cpu+gpu) | 17.7 ms · 56.5 FPS |
| **small · fp16** | 34.3M | **54 MB** | cos 1.000000, relmax 7e-3 | **15.2 ms · 65.7 FPS** |
| base · fp16/fp32 | 135.4M | 202 / 402 MB | cos 1.000000 | 37.7 / 43.4 ms |
| mono-large · fp32 | 334.2M | 1.34 GB | cos 1.000000 | 118 ms |

`small · fp16` is the on-device hero. App: [`apps/CoreAIDepth`](../apps/CoreAIDepth) (photo + live
camera, iOS + macOS). Card: [`zoo/depth-anything-3.md`](../zoo/depth-anything-3.md).
