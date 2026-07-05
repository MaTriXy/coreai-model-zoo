---
license: other
license_name: openmdw-1.1
license_link: https://openmdw.ai/license/
library_name: coreai
pipeline_tag: automatic-speech-recognition
base_model: nvidia/nemotron-3.5-asr-streaming-0.6b
tags: [core-ai, coreaikit, nemotron, streaming, rnn-t, transducer, asr, cache-aware, on-device, apple]
---

# Nemotron 3.5 ASR Streaming 0.6B — Core AI

[`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
(OpenMDW-1.1 — commercial use OK, 600M) converted to **Apple Core AI** `.aimodel` — the first
**STREAMING ASR** in the [zoo](https://github.com/john-rocky/coreai-models-community): live
microphone transcription in **320 ms chunks**, on-device, any audio length (no 30 s bucket).

- **40 locales in ONE checkpoint** — the language is a one-hot graph *input* (`ja-JP`, `en-US`,
  `zh-CN`, … or `auto` for built-in language ID), switchable per session at run time.
- **Punctuation + capitalization built in.**
- **Cache-aware streaming encoder**: the FastConformer's attention KV sliding window (56 frames)
  and causal-conv left context are explicit graph I/O, so each 320 ms chunk is one static-shape
  inference — no re-encoding, constant latency forever.
- Pure-RNNT decode (LSTM predictor + joint) driven by a tiny host greedy loop.

## Graphs (per 320 ms chunk at lookahead 3)

```
mel chunk (25 frames first, then 32)   [host: preemphasis→STFT→slaney mel→log, NO normalization]
 1. stream_pre_first / stream_pre : mel + 3 conv caches → embeds[1,4,1024] + caches   (fp16, 9 MB)
 2. stream_conformer_a : x + neg_mask[1,1,4,60]
                         + k/v_cache[12,8,56,128] + conv_cache[12,1024,8]
                         → x + updated caches                     (layers 0-11, fp16, 605 MB)
 3. stream_conformer_b : x + one_hot[1,128] + neg_mask + caches
                         → enc_proj[1,4,640] + updated caches     (layers 12-23 + prompt fusion
                                                                   + projector, fp16, 615 MB)
 host greedy RNN-T over the 4 new frames:
 4. predict : token[1,1] i32 · h,c[2,1,640] → dec_out[1,640] · h',c'                   (fp32, 61 MB)
 5. joint   : dec_out + enc_frame[1,640] → token_logits[1,13088]                       (fp32, 34 MB)
    blank(13087) advances a frame; a token emits + steps the predictor; 10/frame cap
```

The conformer ships in TWO halves: a single 24-layer AOT bundle (2.4 GB `resources.bin`)
fails to load on-device (instant POSIX-2 from the loader — bisected: identical topology at
1 and 12 layers loads fine), so each half stays ~1.1 GB compiled, for one extra ~1 ms call
per chunk. Platform subtrees: `macos/` (JIT `.aimodel`) and `ios/` (the halves AOT-compiled
to `h18p.aimodelc` — big graphs' on-device JIT aborts; iPhone 17 Pro / A19 Pro).
`ModelID.nemotronASRStreaming` picks the right one automatically.

Gated **token-exact** end-to-end vs the HF streaming reference (99/99 on LibriSpeech, chunked
`use_cache=True` oracle == offline), and again token-exact through Swift **CoreAIKit**
(`KitNemotronModel`, packet-size-independent mel frontend). blank 13087 · vocab 13088 ·
max 10 symbols/frame · 16 kHz.

## Use (CoreAIKit)

```swift
let nemotron = try await KitNemotronModel(model: .nemotronASRStreaming)

// LIVE: feed mic packets as they arrive; the transcript grows while you speak.
let session = try nemotron.makeSession(language: "en-US")   // or "ja-JP", … or "auto"
for await packet in micPackets {                            // 16 kHz mono Float, any packet size
    let partial = try await session.feed(samples: packet)
}
let result = try await session.finish()

// OFFLINE: any-length clip through the same streaming pipeline.
let result = try await nemotron.transcribe(samples: pcm16kMono, language: "en-US")
```

Try it in the zoo's **coreai-audio** app (Transcribe tab → "Nemotron Streaming 0.6B" → **Live**).

## Speed

| | per 320 ms chunk (warm) | real-time factor |
|---|---|---|
| M4 Max (GPU) | ~26 ms | 0.08 (12× real-time) |
| iPhone 17 Pro (GPU, AOT) | ~53 ms | 0.167 (6.0× real-time) |

Load: ~52 s the first time after install (one-time GPU specialization of the two AOT halves), ~4 s cached thereafter.

Streaming latency = the model's lookahead (320 ms at the shipped `lookahead=3`) + chunk compute.
The checkpoint also supports lookahead 0/6/13 (80 ms – 1.12 s); those variants re-export with a
parameter change in the conversion scripts.

## Convert yourself

[`conversion/nemotron_asr/`](https://github.com/john-rocky/coreai-models-community/tree/main/conversion/nemotron_asr)
— streaming oracle (`gen_oracle_streaming.py`, HF chunked `use_cache=True`), cache-explicit
re-author + export (`export_encoder_streaming.py`), token-exact gates (`gate_e2e_streaming.py`,
`gate_mel_swift_streaming.py`).

## License

OpenMDW-1.1 (see `LICENSE`) — the upstream NVIDIA model's license; this conversion redistributes
the weights unchanged in a different serialization.
