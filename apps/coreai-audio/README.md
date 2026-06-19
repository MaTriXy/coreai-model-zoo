# coreai-audio — on-device audio *understanding* (iOS + macOS)

The zoo's first **audio** app: record from the mic, choose a file, or use the demo clip, then ask
*"what do you hear?"* — a local **Qwen2.5-Omni Thinker** describes the **sounds** (events, texture,
emotion), not a transcript. *"I hear a loud hissing sound."* · *"…a continuous sine wave sound."* ·
*"…a series of beeps."* Everything runs on-device; nothing leaves the phone.

Device-verified on **iPhone 17 Pro** (A19 Pro) and **M4 Max**.

## How it works

Two Core AI models + a Swift front end, on the **coreai-pipelined** GPU engine (same static-input
recipe as the VL apps):

- **Audio encoder** (`*_audio_encoder_fp16_k15`, 1.2 GB) — a fixed-shape Whisper-style tower, run
  once per clip: `input_features [1,128,3000]` + `attn_bias [15,1,1,100]` → `audio_embeds [750,2048]`.
- **Text decoder** (`*_thinker_int8lin_n750_s1`, 3.9 GB) — a Qwen2.5-3B decoder; the audio embeds
  ride **one static-input buffer** (`audio_embeds`), and the prompt's `<|AUDIO|>` placeholders carry
  extension ids `vocab + slot` the graph gathers. No rope-shift inputs (TMRoPE collapses to 1-D for
  audio+text → engine-native positions).
- **Mel front end** — Whisper-large-v3 log-mel in Accelerate/vDSP (`AudioMelPreprocessor`), bit-exact
  with the HF feature extractor (gated cos 1.0).

iPhone uses the **AOT** decoder (`.aimodelc`, `coreai-build --platform iOS --architecture h18p`) so
the 3.9 GB graph dodges the on-device JIT jetsam; the AOT weights mmap as clean pages, so it loads
comfortably (≈5.9 GB headroom after load on a 12 GB device). Needs the
`com.apple.developer.kernel.increased-memory-limit` entitlement.

## Run

```sh
cd apps/coreai-audio
xcodegen generate
open coreai-audio.xcodeproj   # set your team, then Run (Release) on iPhone or Mac
```

Models load from `Documents/models/` — download from
[🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI)
(`gpu-pipelined/` for macOS, `ios/` for the iPhone AOT bundle). **Load model** → **Record** /
**Choose…** / **Demo noise** → **Ask**.

Audio understanding here is **Mac + iPhone** (the decoder runs on the GPU engine; an ANE
static-shape rework for lower power is a follow-up). Any clip is decoded to 16 kHz mono, ≤ ~30 s.
