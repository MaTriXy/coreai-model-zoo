# CoreAIVideo — LTX-Video 2B text→video on Apple Core AI (macOS)

A SwiftUI Mac app to **type a prompt and get a short video**, generated entirely by Core AI
bundles. The zoo's first video app — the desktop counterpart to [`CoreAIImageGen`](../CoreAIImageGen).

All three heavy nets (T5-XXL text encoder, the flow-matching DiT, the causal video VAE) run as
Core AI `.aimodel` bundles; only LTX's FlowMatch sampler loop runs on host. See the model card
[`models/ltxvideo/README.md`](../../models/ltxvideo/README.md) and recipe [`conversion/ltxvideo/`](../../conversion/ltxvideo).

512×768 · 49 frames · 8 steps → **~14 s on a Mac GPU** (Apple silicon).

## How it works

The app is a thin SwiftUI front-end over a **resident** Python backend (`app_backend.py`):

- `app_backend.py` loads the 3 bundles **once** (prints `READY`), then generates one video per
  `"<seed>\t<prompt>"` line on stdin, streaming `PROGRESS <stage> <i> <n>` → `DONE <mp4>`.
- It does **not** load the 19 GB T5 torch weights (the bundle does that compute) — only the
  tokenizer — so startup is fast and the runtime dir stays ~19.5 GB.
- `Generator.swift` keeps that process alive and surfaces progress; `ContentView.swift` shows the
  prompt box, a progress bar, and plays the result in an AVKit `VideoPlayer`.

## Setup

1. **Convert the bundles** (once) with [`conversion/ltxvideo/`](../../conversion/ltxvideo):
   `_conv_fp16.py dit 512 768 49 256`, `_conv_fp16.py vae 512 768 49`,
   `_conv_fp16.py t5 256 256 25 256 --bf16` → `coreai_out/{dit_fp16,vae_fp16,t5_bf16}.aimodel`.
2. **Stage the runtime**: `./setup_runtime.sh <scratch LTX-Video dir>` → `~/CoreAIVideoRuntime`.
3. **`brew install ffmpeg`** (the backend writes the .mp4 via ffmpeg).
4. **Generate the Xcode project**: `xcodegen generate`, then open `CoreAIVideo.xcodeproj`, build & run.
   - The Swift app only launches/plays — no Core AI Swift dependency, so it builds on plain macOS 14+.
   - Default paths (editable in-app under *Backend paths*): python =
     `~/Code/coreai/coreai-models/.venv/bin/python`, runtime = `~/CoreAIVideoRuntime`.

## Notes

- Fixed at 512×768 × 49 frames (the resolution the staged DiT/VAE bundles were converted at). For
  other resolutions, re-convert DiT+VAE at the new shape and restage (T5 is resolution-independent).
- Sandbox is **off** (the app spawns a subprocess and reads/writes local files) — local/dev use.
- This is the Mac-host pattern (like `TripoSplatMac`): the proven Python pipeline does the work, the
  app is the UI. Full on-device (iPhone) video is a stretch — see the model card.
