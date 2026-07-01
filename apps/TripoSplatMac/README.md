# TripoSplat · Core AI — macOS app

Pick an image → generate 3D Gaussian splats (all four heavy nets on Apple Core AI) → view in 3D.
Export the `.splat` for [MetalSplatter](https://github.com/scier/MetalSplatter).

Generation is macOS-only (the model is too large for on-device iOS — see
`../../conversion/triposplat/README.md`). To **view** the result on iPhone / iPad / Vision Pro,
open the exported `.splat`/`.ply` in the [MetalSplatter](https://github.com/scier/MetalSplatter)
app; no dedicated viewer app ships here.

The app is a thin SwiftUI front-end; generation runs the verified Core AI pipeline
(`app_backend.py`) as a subprocess. Octree sampling stays eager on host; DINOv3 / Flux2-VAE-enc /
DiT / Gaussian-decoder run on Core AI.

## One-time setup

1. **Stage the runtime** (backend script + converted `.aimodel` bundles + checkpoints, ~9 GB) into
   a stable directory the app can point at:
   ```sh
   ./setup_runtime.sh <SRC>            # stages into ~/TripoSplatRuntime by default
   ```
   `<SRC>` is your TripoSplat runtime dir (from `../../conversion/triposplat`; must contain
   `app_backend.py`, `coreai_out/`, `ckpts/`). Pass a second arg to change the destination.

2. **Generate & open the Xcode project:**
   ```sh
   xcodegen generate
   open TripoSplatMac.xcodeproj
   ```

3. Build & run in Xcode (signing team is already set; sandbox is off so the app can spawn Python).

## Use

- **Choose Image…** → pick any photo (rmbg runs automatically).
- **Steps** 1–50 (20 = quality, 4 = fast smoke). **Generate 3D**.
- Orbit/zoom the point-cloud preview. **Reveal .splat in Finder** to open it in MetalSplatter.

## Settings (if paths differ)

- **Python**: default `~/Code/coreai/coreai-models/.venv/bin/python`
- **Backend dir**: default `~/TripoSplatRuntime` (must contain `app_backend.py`, `coreai_out/`, `ckpts/`)

## Notes

- Bundles are fp32; on the CPU runtime the DiT is ~20 s/step-pair, so a 20-step run takes a few
  minutes. int8/fp16 + on-device AOT is the next speedup.
- The in-app preview is a colored point cloud (fast). For true splat rendering, open the exported
  `.splat` in MetalSplatter.
