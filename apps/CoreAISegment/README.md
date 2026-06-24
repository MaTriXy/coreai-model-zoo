# CoreAISegment

A minimal cross-platform (macOS + iOS) **text-prompt image segmentation** app for the
[SAM 3](https://huggingface.co/facebook/sam3) Core AI bundle, running on Apple's official
[`CoreAIImageSegmenter`](https://github.com/apple/coreai-models) runtime — **no engine
patch, stock runtime**.

Pick an image, type what to segment (`cat`, `the red car`, `person on the left`), and
SAM 3 returns instance masks, which the app composites back over the image.

<p align="center"><i>Image + "cat" → instance masks, on-device.</i></p>

## What it shows

- The official `coreai.diffusion`-style pipeline for **vision** bundles: one
  `ImageSegmenter(resourcesAt:)` load, one `segment(image:prompt:)` call.
- An open-vocabulary segmenter (text prompt, not a fixed class list) running fully
  on-device via Core AI.

## Model

Downloads [`mlboydaisuke/sam3-CoreAI-official`](https://huggingface.co/mlboydaisuke/sam3-CoreAI-official)
(float16, ~1.7 GB) from the Hub on first launch, cached and resumable across launches via
the shared `AppShared/ModelDownloader`. Or point **Local…** at any exported SAM 3 segmenter
bundle directory (the folder containing `metadata.json`, the `.aimodel`, and `tokenizer/`).

## Build

```bash
brew install xcodegen      # if needed
cd apps/CoreAISegment
xcodegen generate
open CoreAISegment.xcodeproj
```

- **macOS** scheme: `CoreAISegmentMac` (full resolution, loads the hosted JIT bundle directly).
- **iOS** scheme: `CoreAISegment`. **iOS needs the AOT-compiled bundle** — JIT-compiling the
  SAM 3 graph on device overruns the per-process memory budget during MPSGraph constant-folding
  and aborts (`BumpMmapResourceAllocator` bad-alloc). Pre-compile on a Mac and use that bundle:

  ```bash
  xcrun coreai-build compile sam3_float16.aimodel --output out \
      --platform iOS --preferred-compute gpu --architecture h18p --expect-frequent-reshapes
  # → out/sam3_float16.h18p.aimodelc — rename to sam3_float16.aimodel and point metadata.json at it
  ```

  For debugging, sideload the AOT bundle into the app's container so "Download & Load" finds it
  and skips the network fetch:

  ```bash
  xcrun devicectl device copy to --device <id> --domain-type appDataContainer \
      --domain-identifier com.coreai.segment.ios \
      --source <aot-bundle-dir> --destination "Documents/sam3-CoreAI" --remove-existing-content true
  ```

  (`h18p` = iPhone 17 Pro / A19; compile for your device's architecture.)

The app pins `apple/coreai-models` to the revision its build was verified against and
applies **no patch stack** (the segmentation runtime is unmodified upstream).

## CLI equivalent

```bash
# from an apple/coreai-models checkout
swift run -c release image-segmenter \
    --model <downloaded-bundle-dir> --prompt "cat" --image cats.jpg
```
