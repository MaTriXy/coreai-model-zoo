# coreai-audio — on-device audio *understanding* (iOS + macOS)

The zoo's first **audio** app: record from the mic, choose a file, or use the demo clip, then ask
*"what do you hear?"* — a local **Qwen2.5-Omni Thinker** describes the **sounds** (events, texture,
emotion), not a transcript. *"I hear a loud hissing sound."* · *"I hear a man speaking in English."*
· *"…a series of beeps."* Everything runs on-device; nothing leaves the phone.

Device-verified on **iPhone 17 Pro** (A19 Pro) and **M4 Max**.

## How it works

Built on **[CoreAIKit](https://github.com/john-rocky/coreai-kit)** (`KitAudioModel`) — a local audio
model behind FoundationModels' `LanguageModel`, on the **coreai-pipelined** GPU engine:

```swift
let model = try await KitAudioModel(model: id)        // downloads decoder + encoder from HF
try await model.attach(samples: pcm16kMono)           // mel → audio encoder → static buffer
let answer = try await LanguageModelSession(model: model).respond(to: "What do you hear?")
```

- **Audio encoder** (Whisper-style, 1.2 GB fp16) — run once per clip → `audio_embeds [750,2048]`.
- **Text decoder** (Qwen2.5-3B, 3.9 GB int8) — the audio embeds ride **one static-input buffer**;
  the prompt's `<|AUDIO|>` placeholders carry extension ids `vocab + slot` the graph gathers. No
  rope-shift inputs (TMRoPE collapses to 1-D for audio+text).
- **Mel front end** — Whisper-large-v3 log-mel in Accelerate/vDSP, bit-exact with the HF feature
  extractor (cos 1.0). Mic capture uses `AVAudioRecorder`; clips are capped to ~6 s for snappy
  on-device prefill.

iPhone downloads the **AOT** decoder (`.aimodelc`) so the 3.9 GB graph dodges the on-device JIT
jetsam (the AOT weights mmap as clean pages → comfortable headroom). Needs the
`com.apple.developer.kernel.increased-memory-limit` entitlement.

## Run

```sh
cd apps/coreai-audio
xcodegen generate
open coreai-audio.xcodeproj   # set your team, then Run (Release) on iPhone or Mac
```

First launch **downloads ~5 GB** from
[🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI)
(`ios/` AOT decoder on iPhone, `gpu-pipelined/` on macOS, shared encoder). **Load model** →
**Record** / **Choose…** / **Demo** → **Ask**.

Audio understanding here is GPU-pipelined (an ANE static-shape rework for lower power is a follow-up).
Any clip is decoded to 16 kHz mono.
