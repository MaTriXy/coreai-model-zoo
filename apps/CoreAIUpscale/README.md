# CoreAIUpscale

On-device **×4 super-resolution** with **AdcSR** (one-step diffusion-GAN, CVPR 2025) on the Core AI
stack — the zoo's first super-resolution app. Pick a photo, get a 4× sharper version on-device.

- **Model:** [`mlboydaisuke/AdcSR-CoreAI`](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI),
  fetched from the Hub on first run. AdcSR = Adversarial Diffusion Compression of OSEDiff: a pruned
  Stable Diffusion 2.1 UNet + half VAE decoder, run in **one step** (no iterative denoising, no
  prompt, no noise).
- **License:** Apache-2.0 (code); weights derived from Stable Diffusion 2.1
  (**CreativeML OpenRAIL++-M** — commercial use permitted, the same license under which Apple
  distributes SD CoreML). Not non-commercial.
- **Size / precision:** fp16, ~870 MB. On the GPU it matches the fp32 reference (cos 1.000008).
- **How it works:** the exported graph upscales one `lr [1,3,128,128] → sr [1,3,512,512]` tile;
  CoreAIKit's `SuperResolver` splits any-size input into overlapping 128-px LR windows, runs each,
  and feather-blends the result.
- **Separate from `CoreAIImageGen`:** that app pins `apple/coreai-models` for the diffusion
  pipeline, while CoreAIKit pulls the `john-rocky/coreai-models` fork — one target can't depend on
  both (the package name `coreai-models` collides). SR needs neither, so it stands alone.

## Build

```sh
xcodegen generate          # in this directory
open CoreAIUpscale.xcodeproj
```

Targets: `CoreAIUpscale` (iOS) and `CoreAIUpscaleMac` (macOS). Build/run on device.
