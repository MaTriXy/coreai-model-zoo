# AdcSR — on-device diffusion super-resolution

[AdcSR](https://github.com/Guaishou74851/AdcSR) (CVPR 2025) is the zoo's first super-resolution
model: a one-step **diffusion-GAN** ×4 SR. It is the Adversarial Diffusion Compression of
[OSEDiff](https://github.com/cswry/OSEDiff) — a pruned Stable Diffusion 2.1 UNet (~456 M params)
plus a half-size VAE decoder, run in a single forward. Diffusion-grade perceptual quality, small
enough for iPhone, and (unlike the ResShift/SinSR family, which are non-commercial) usable
commercially: the code is Apache-2.0 and the SD-2.1-derived weights carry CreativeML Open RAIL++-M
— the same license under which Apple ships SD CoreML. HF: `mlboydaisuke/AdcSR-CoreAI`.

## Architecture / graph

`Net` (AdcSR `model.py`) takes the diffusers SD-2.1 UNet and surgically: deletes `time_embedding`,
deletes cross-attention (`attn2`/`norm2`), prunes every conv/linear/norm to ×0.75 channels, and
swaps in text/time-free `My*_SD_forward` methods (`forward.py`). The deployed pipeline is
`PixelUnshuffle(2) → pruned UNet → half VAE decoder`, image→image, no prompt/noise/timestep.

Exported as ONE static graph: **`lr [1,3,128,128]` in `[-1,1]` → `sr [1,3,512,512]`** (×4). Only
standard SD ops (conv, group-norm, self-attention `Transformer2DModel`, PixelUnshuffle) — no swin,
so the `coreai-pre-compilation-rewrite` SIGSEGV that bit the SinSR swin port does not apply here.
The single op coreai-torch rejected was `aten.var.correction` from the original color-match's
`.std()`; we removed the in-graph color-match entirely (see below). Conversion: `conversion/export_adcsr.py`.

## Host-side post-processing (CoreAIKit `SuperResolver`)

The graph outputs the **raw** SR. Two things are done on the host:

1. **Tiling.** The graph is a fixed 128→512 tile; `SuperResolver` caps the input long side
   (`maxInputSide=512`, so a full phone photo isn't blown up to a gigapixel result), splits the LR
   into overlapping 128-px windows, runs each, and feather-blends.
2. **Per-image color-match — applied GLOBALLY, once, after stitching.** AdcSR's reference matches
   each channel's mean/std of the SR to the LR. This MUST be global: baking it per-tile in the
   graph divides by a tile's std, which → 0 on uniform tiles (sky/skin/white fur) → pure-white
   square artifacts. The fix is to export the raw model and color-match the whole stitched image.

## fp16 does NOT work — ship fp32

The pruned SD-2.1 UNet is numerically unstable in fp16: attention overflows (the classic SD-2.1
fp16 NaN), and `upcast_attention` has **no effect** because the diffusers SDPA processor ignores
it; group-norm of a low-variance tile also divides by ~0. The result is whole tiles → NaN → black/
gray patches on smooth regions. group-norm-fp32 upcast alone did not fix it. **fp32 (~1.7 GB) is
the shipped precision** — no NaN, output matches the torch reference (cosine 1.000012). With the
512-px input cap + tiling, the per-tile activation is bounded, so fp32 fits iPhone 17 Pro
(~2.7 GB peak < 6 GB).

Note: the Python runtime's *default*-options `AIModel.load` SIGSEGVs in the GPU-delegate JIT
(`CompileForDelegates`) for fp16; loading with an explicit `SpecializationOptions(.gpu)` (which the
Swift `GraphModel` always does) is clean. Irrelevant for fp32, but worth knowing.

## CoreGraphics traps (found on-device)

- **Row stride:** read a `CGContext`'s actual `bytesPerRow` (pass `bytesPerRow: 0`), never assume
  `width*4` — CG pads non-16-aligned widths, and assuming `width*4` shears the image (garbled tiles
  on odd-width photos).
- **No y-flip:** a standard top-down `CGBitmapContext` already maps the image's top to row 0;
  adding a `scaleBy(1,-1)` flip inverts the result vertically.

## App

`apps/CoreAIUpscale` (iOS + macOS, PhotosPicker) and CoreAIKit `Examples/UpscaleDemo`. Both use
`SuperResolver(model: .adcsrX4).upscale(cgImage)`. Verified on iPhone 17 Pro.
