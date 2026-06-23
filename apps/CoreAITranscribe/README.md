# CoreAITranscribe

A minimal cross-platform (macOS + iOS) **speech-to-text** app for the
[Whisper large-v3-turbo](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official)
Core AI bundle, running on Apple's **stock** Core AI runtime — no engine patch.

Pick an audio file or record from the mic; the model transcribes one 30 s window on-device.

## What it shows

- A full on-device ASR pipeline on the stock runtime: **log-mel frontend → autoregressive
  decode → detokenize**, no server.
- Driving Whisper's encoder-decoder graph from Swift with a **fixed 128-token decoder
  window** (pad the buffer, read logits at the real last position) so the shape stays
  constant and MPSGraph compiles once (~0.18 s/token on M-series GPU).

## How it works

1. **`WhisperMel.swift`** — audio → log-mel `[1,128,3000]` with Accelerate. Whisper's
   `n_fft=400` isn't an FFT-friendly length, so the 400-point DFT is a matmul against a
   precomputed cos/sin basis; the mel filterbank (`mel_filters_128.npy`, shipped with the
   model) is a second matmul; then `log10` + clamp + normalize.
2. **`TranscribeEngine.swift`** — loads the `.aimodel` via `PreparedModel` (single `main`
   graph → GPU), greedy-decodes the fixed-128 window, and detokenizes with the bundled HF
   Whisper tokenizer (swift-transformers).
3. **`Recorder.swift`** — mic capture to a 16 kHz WAV via `AVAudioRecorder`.

## Model

Downloads [`mlboydaisuke/whisper-large-v3-turbo-CoreAI-official`](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official)
(float16, ~1.5 GB) on first launch (cached, resumable). The bundle ships the `.aimodel`, the
HF tokenizer, and the mel filterbank.

## Build

```bash
brew install xcodegen      # if needed
cd apps/CoreAITranscribe
xcodegen generate
open CoreAITranscribe.xcodeproj
```

- **macOS** scheme: `CoreAITranscribeMac`. **iOS** scheme: `CoreAITranscribe` (iOS bundles
  generally want AOT compilation first — see the model card).
- Pins `apple/coreai-models` to the revision the build was verified against; no patch stack.

The prompt is fixed to English transcription (`<|startoftranscript|><|en|><|transcribe|><|notimestamps|>`);
swap the language token for other languages.
