# coreai-audio — on-device audio *understanding* + *transcription* + *speech* (iOS + macOS)

Four tabs, all fully on-device:

- **Understand** — record from the mic, choose a file, or use the demo clip, then ask *"what do you
  hear?"*: a local **Qwen2.5-Omni Thinker** describes the **sounds** (events, texture, emotion), not
  a transcript. *"I hear a loud hissing sound."* · *"I hear a man speaking in English."*
- **Transcribe** — speech → **text**, with a choice of three Core AI ASR models (segmented picker):
  - **Whisper large-v3-turbo** — the Apple-recipe export on the **stock runtime** (fixed-128-token
    decoder window), via CoreAIKit's `KitWhisperModel`. 100 languages, ≤30 s, auto language-detect.
    Downloads from [🤗 whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official)
    on both platforms. See [`knowledge/whisper-asr-fixed-decode.md`](../../knowledge/whisper-asr-fixed-decode.md).
  - **Qwen3-ASR-1.7B** — the zoo's first ASR (AuT encoder + Qwen3 decoder on the pipelined engine),
    via `KitASRModel`. 52 languages, ≤30 s. See [`zoo/qwen3-asr.md`](../../zoo/qwen3-asr.md).
  - **Parakeet-TDT-0.6B** — the zoo's first **transducer / TDT (RNN-T family)** (NVIDIA FastConformer
    encoder + LSTM predictor + joint, 3 graphs driven by a host greedy TDT loop), via
    `KitParakeetModel`. 25 EU languages, ≤30 s; **iPhone 17 Pro 47.9× real-time**. See
    [`zoo/parakeet.md`](../../zoo/parakeet.md).
  - **Diarize — who said what** (toggle): **Streaming Sortformer 4-spk v2** (NVIDIA, CC-BY-4.0) labels
    each speaker turn on Core AI, then the chosen ASR transcribes it → *"Speaker 1 [0.3–4.1s]: …"*.
    A pure host port (NeMo 128-mel + streaming loop + AOSC speaker-cache compression) driving the
    fixed-buffer graph; byte-gated **100% speaker-activity agreement** vs NeMo. See
    [`conversion/sortformer_diar/HANDOFF.md`](../../conversion/sortformer_diar/HANDOFF.md).
- **Voice** — **VoxCPM-0.5B** diffusion text-to-speech (MiniCPM4 LM + LocDiT flow-matching +
  AudioVAE), streaming int8. See [🤗 VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI).
- **Speak** — **Kokoro-82M** (StyleTTS2 + iSTFTNet) text-to-speech on Core AI: pick a voice and a
  phrase, hear it spoken. Three `.aimodel` bundles (predictor / prosody / vocoder) on the CPU
  compute unit + the host DSP (alignment + hn-nsf source) in Swift; ~0.7 s/utterance, magspec-corr
  0.999 vs the PyTorch reference. See [`zoo/kokoro-82m.md`](../../zoo/kokoro-82m.md). The demo
  phrases are phonemized ahead of time (host-side G2P), so this build needs no MLX/espeak; the three
  bundles come from [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI)
  (`KokoroAssets/` ships the voices + tokenizer; drop the `.aimodel` there or in Documents).

Device-verified on **iPhone 17 Pro** (A19 Pro) and **M4 Max** (TTS verified on M4 Max).

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

**Transcribe** rides CoreAIKit's ASR models. Whisper is one stateless graph (no LLM engine) driven
through `GraphModel`; the same `mel_filters.f32` powers it (bit-exact with the HF `mel_filters_128.npy`),
so no extra resource ships:

```swift
let whisper = try await KitWhisperModel(model: .largeV3Turbo)   // downloads .aimodel + tokenizer
let result  = try await whisper.transcribe(samples: pcm16kMono) // -> Transcription(language, text)
// or: let asr = try await KitASRModel(model: .qwen3ASR1_7B); try await asr.transcribe(samples:)
// or: let par = try await KitParakeetModel(model: .parakeetTDT); try await par.transcribe(samples:)
```

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
