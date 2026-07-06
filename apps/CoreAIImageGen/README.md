# CoreAIImageGen

A minimal **image-generation** app for Core AI diffusion bundles running on Apple's
official `CoreAIDiffusionPipeline` runtime
([apple/coreai-models](https://github.com/apple/coreai-models)). One SwiftUI codebase,
two targets:

| Target | Platform | Hosted model | Notes |
|---|---|---|---|
| `CoreAIImageGenMac` | macOS 27 | **FLUX.2 klein 4B** (1024) | desktop split-view UI; in-app download |
| `CoreAIImageGen` | iOS 27 | — (load via **Local…**) | runs smaller bundles, e.g. Stable Diffusion 0.9B |

On macOS the app downloads **[FLUX.2 klein 4B](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI)**
(4-step distilled, guidance 1.0). The pipeline type (FLUX.2 / SD3 / SD) is auto-detected from
the bundle's `metadata.json`, mirroring the zoo's `diffusion-runner` reference tool — so any
`coreai.diffusion.export` bundle drops in via **Local…**.

**Why FLUX.2 is macOS-only:** at 4B the peak footprint exceeds a 12 GB iPhone's per-process
memory limit. The transformer's first-run footprint was traced on an iPhone 17 Pro
(`os_proc_available_memory` + `phys_footprint`): the per-process ceiling is ~6.1 GB, and
generation peaks at ~6.5 GB because the text encoder (~1.8 GB) is **not released before the
transformer runs**. AOT-compiling the transformer (`xcrun coreai-build compile … --architecture
h18p`) removes the first-run MPSGraph JIT spike but still lands ~0.4 GB over; the text encoder
fails to AOT-compile (`failedToSpecialize`). So the iOS hosted catalog is empty — the iOS app
loads smaller diffusion bundles via **Local…**.

## Image-to-image

Once a model with a VAE encoder is loaded — FLUX.2 bundles ship one — a **Text → Image /
Image → Image** switch appears. In image-to-image mode you pick a source image and set a
**Strength** (0.1–1.0): the runtime encodes the image through the VAE encoder, blends in noise
up to `strength`, and denoises from there with the prompt as the edit instruction. Lower
strength stays close to the source; higher deviates further. Any input resolution works — the
pipeline resizes to the model's native size. This is a native capability of Apple's
`Flux2Pipeline` (`PipelineConfiguration.startingImage` / `strength`); no model-code port and
no extra download — the FLUX.2 bundle already carries `VAEEncoder.aimodel`.

## In-context editing (Edit tab)

A third **Edit** mode does FLUX.2's native in-context editing — *"add a red hat, keep everything
else"* — where only the instructed change is applied and the subject/pose/background are kept
(unlike SDEdit, which re-renders the frame). Add a second reference and it **combines** them
(*"put the subject from image 1 into the scene from image 2"*). This uses edit-sequence
transformers (`Transformer_edit` / `_2ref`) via `Flux2Pipeline.editImages(referenceImages:)`; the
runtime picks the 1- or 2-reference graph by the number of images. The ~2 GB edit weights are
**fetched on demand** (a *Download edit models* button in the Edit tab) so text-to-image users
don't pay for them up front. Requires the edit-capable runtime — this app pins the
`john-rocky/coreai-models` `flux2-in-context-edit` branch (the stock apple runtime is T2I/img2img
only). Mechanism: [knowledge/flux2-in-context-editing.md](../../knowledge/flux2-in-context-editing.md).

## Build & run

```bash
brew install xcodegen
cd apps/CoreAIImageGen
xcodegen generate
# macOS:
open CoreAIImageGen.xcodeproj          # run the CoreAIImageGenMac scheme (Release)
# iOS (device):
xcodebuild -project CoreAIImageGen.xcodeproj -scheme CoreAIImageGen -configuration Release \
  -sdk iphoneos -destination 'generic/platform=iOS' -derivedDataPath build-ios \
  -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> \
  build-ios/Build/Products/Release-iphoneos/CoreAIImageGen.app
```

Unlike the LLM apps, this one needs **no `coreai-models` patch stack** — the diffusion
runtime runs unmodified. `project.yml` pulls `apple/coreai-models` straight from GitHub
(pinned to the metadata.json-v0.2 revision) rather than the patched repo-root clone.

## Model delivery

On macOS the first launch offers an **in-app download** of the published bundle from Hugging
Face. The `.aimodel` directory bundles + tokenizer stream via the shared
[`AppShared/ModelDownloader`](../AppShared/ModelDownloader.swift) (range-chunked parallel
download, cross-launch resume, atomic placement — a partial bundle never poisons the
content-keyed coreai-cache); the few tiny root files (`metadata.json`, `vae_bn_*.npy`) come
down with a plain resolve GET (the HF tree API only enumerates directories). The screen is
kept awake during the multi-GB transfer so an auto-lock can't suspend it mid-download; for a
big set, stay on Wi-Fi and keep the app foregrounded. On iOS, load a bundle with **Local…**.

## Notes

- Generation runs Apple's `CoreAIDiffusionPipeline` (`Flux2Pipeline` / `SD3Pipeline` /
  `StableDiffusionPipeline`) — there is **no model-code port**: image generation is already
  supported by the stock Core AI stack.
- FLUX.2 / SD3 use the `discreteFlow` scheduler; classic SD uses `dpmSolverMultistep`.
- Reference timing (diffusion-runner): macOS FLUX.2 1024 ≈ 17.4 s @ 4 steps. See the model
  card for details.
