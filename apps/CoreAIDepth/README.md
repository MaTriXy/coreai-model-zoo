# CoreAIDepth — monocular depth on-device (iOS + macOS)

Depth Anything 3 (the zoo's first depth model) running fully on-device via Core AI. Two modes:

- **Photo** — pick an image from the library, run depth once, see the input next to a colorized
  depth map (DA3 convention: inverse-depth, percentile-normalized, Spectral — far = red, near = blue).
- **Camera** — live depth from the camera at a few FPS (iOS).

It reuses CoreAIKit's `DepthEstimator`, which downloads the small fp16 bundle (~54 MB) from
[`mlboydaisuke/Depth-Anything-3-CoreAI`](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI)
and runs the 504² monocular pipeline (engine cos 1.000000 vs torch; ~15 ms/frame, 65 FPS on an
M4 Max GPU). Model card + conversion: [`models/depth-anything-3/README.md`](../../models/depth-anything-3/README.md).

## Build

```bash
xcodegen generate
# iOS
xcodebuild -project CoreAIDepth.xcodeproj -scheme CoreAIDepth -configuration Release \
  -destination 'platform=iOS,id=<DEVICE_ID>' -allowProvisioningUpdates build
# macOS
xcodebuild -project CoreAIDepth.xcodeproj -scheme CoreAIDepthMac -configuration Release build
```

Depends on the sibling `coreai-kit` checkout (`../../../../coreai-kit`).
