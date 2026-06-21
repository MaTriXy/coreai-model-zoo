# CoreAI-Model-Zoo

LLMs converted to Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27) — downloadable, verified
on-device, with the conversion code and a knowledge base. Successor to
[`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

## Models

| Model | Download (`.aimodel`) | License |
|---|---|---|
| **Qwen3.5-0.8B** | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | Apache-2.0 |
| **Qwen3.5-2B** | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | Apache-2.0 |
| **Qwen3.6-35B-A3B** (MoE, Mac-only) | [🤗 Qwen3.6-35B-A3B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI) | Apache-2.0 |
| **Qwen3.6-27B** (dense, Mac-only) | [🤗 Qwen3.6-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI) | Apache-2.0 |
| **GLM-4.7-Flash** (MoE + MLA, Mac-only — zoo's first MLA) | [🤗 GLM-4.7-Flash-CoreAI](https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI) | MIT |
| **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | Gemma |
| **Gemma 4 12B** (dense, Mac-only — custom flash-decode kernel ‡) | [🤗 Gemma-4-12B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI) | Gemma |
| **Gemma 4 31B** (dense, Mac-only — custom flash-decode kernel ‡) | [🤗 Gemma-4-31B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI) | Gemma |
| **LFM2.5-1.2B-Instruct** | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | LFM Open License v1.0 |
| **LFM2.5-8B-A1B** (MoE, custom `gather_qmm` kernel — first iPhone MoE) | [🤗 LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI) | LFM Open License v1.0 |
| **Granite 4.0-H 1B / 350M** | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | Apache-2.0 |
| **Nanbeige4.1-3B** (dense reasoning/agentic, iPhone — 32B-class @ 3.93B) | [🤗 Nanbeige4.1-3B-CoreAI](https://huggingface.co/mlboydaisuke/Nanbeige4.1-3B-CoreAI) | Apache-2.0 |
| **Qwen3-VL** (vision-language) | [🤗 2B](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) · [4B](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI) · [8B](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI) | Apache-2.0 |
| **MiniCPM-V 4.6** (vision-language, sub-2B — strongest tiny VLM) | [🤗 MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI) | Apache-2.0 |
| **Gemma 4 E2B vision (VL)** (image+text) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | Gemma |
| **Qwen2.5-Omni-3B Audio** (audio *understanding* — describes sounds, not a transcript; iPhone + Mac, zoo's first audio model) | [🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) | Apache-2.0 |
| **Kokoro-82M** (text-to-speech — zoo's first TTS; StyleTTS2 + iSTFTNet, 28 English voices, runs on any text) | [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI) | Apache-2.0 |
| **EmbeddingGemma 300M** (text embeddings — on-device RAG / semantic search) | [🤗 embeddinggemma-300m-CoreAI](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI) | Gemma |
| **Qwen3-Embedding 0.6B** (multilingual text embeddings, last-token pooling + MRL) | [🤗 Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI) | Apache-2.0 |
| **Qwen3-Reranker 0.6B** (cross-encoder reranker — yes/no relevance score) | [🤗 Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI) | Apache-2.0 |
| **RF-DETR nano/small/medium/large** (object detection, no NMS) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | Apache-2.0 |
| **RF-DETR-Seg nano→2xlarge** (instance segmentation, 6 sizes) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | Apache-2.0 |

### Decode throughput (tok/s, greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **71.9** | 14.7 | **210** |
| **Qwen3.5-2B** | **29** | — | **161** |
| **LFM2.5-1.2B** | **45.4** | — | **276.5** |
| **Granite 4.0-H 1B** | **36.3** | — | **136.5** |
| **Nanbeige4.1-3B** | **15.9** | — | **114.5** |
| **Gemma 4 E2B** | **30.3** (QAT 30.7) | 6 | **77.0** (QAT 78.9) |
| **Gemma 4 E4B** (official QAT) | **15.1** | — | **55.8** |
| **Gemma 4 E2B VL** (image+text, official QAT) | **25.5** | — | **82.4** |
| **MiniCPM-V 4.6** (vision-language, sub-2B) | **53.4** | — | **224.3** |
| **Qwen3.6-35B-A3B** (MoE, 35B/~3B active, Mac-only) | — | — | **64.9** † |
| **Qwen3.6-27B** (dense, Mac-only) | — | — | **15.9** |
| **GLM-4.7-Flash** (MoE + MLA, 30B/~3B active, Mac-only) | — | — | **52.4** † |
| **Gemma 4 12B** (dense, Mac-only) | — | — | **23** int8 / **33** int4 ‡ |
| **Gemma 4 31B** (dense, Mac-only) | — | — | **17.2** int4 ‡ |

Measured on the iOS 27 / macOS 27 beta, Apple's `coreai-pipelined` GPU engine, zero custom
kernels (ANE column + **†**/**‡** excepted). **†** = MoE bundle using the custom
[`gather_qmm`](knowledge/compute-units-and-authoring.md) Metal kernel (reads only the routed
experts). **‡** = dense bundle whose full/global-attention SDPA is a custom flash-decode Metal
kernel — the stock MPSGraph SDPA crashes on the ≥16-head × 512 Q (a GPU scratch-heap overflow,
[apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27)), so these models are
**unrunnable without it**. Prefill, sizes, per-model caveats, and the Mac-only big models: [`zoo/`](zoo/).

<p align="center">
  <img width="380" alt="CoreAIChat screen recording" src="https://github.com/user-attachments/assets/999dbd95-45b5-468f-b1a8-34112ee3b74d" />
</p>
<p align="center"><i>CoreAIChat (<a href="apps/">apps/</a>) — the zoo's models running on-device on iPhone.</i></p>

## Start here

- **Try the app** (iOS 27 / macOS 27 beta; the model downloads in-app):
  - **Demo app, no build** → Mac: [**.dmg**](https://github.com/john-rocky/coreai-model-zoo/releases/download/mac-v1.0/CoreAI-Zoo-for-Mac.dmg) (notarized, runs the Mac-only bundles) · iPhone: CoreAIChat on TestFlight (coming soon)
  - **Build it** → [`apps/`](apps/) — Xcode 27 beta + xcodegen, the `coreai-models` patch stack + `tokenizer.json`
- **Run a model in your own app** → [`knowledge/swift-runtime.md`](knowledge/swift-runtime.md) + the model card
- **Convert a model** → [`knowledge/conversion-guide.md`](knowledge/conversion-guide.md)
- **Compress** → [`knowledge/compression.md`](knowledge/compression.md)
- **Make it fast** → [`knowledge/custom-metal-kernels.md`](knowledge/custom-metal-kernels.md) · [`knowledge/performance-ceiling.md`](knowledge/performance-ceiling.md)
- **Known beta issue** (in-graph KV-write crash; workarounds + the input-mask escape) → [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](knowledge/coreai-beta-mpsgraph-kvwrite-bug.md) — FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)

## Repository layout

| Dir | What |
|---|---|
| [`zoo/`](zoo/) | Model cards — configurations, sizes, parity, measured throughput. |
| [`knowledge/`](knowledge/) | Verified notes on the framework: conversion, compression, stateful KV, custom Metal kernels, AOT, compute-unit rules, the Swift runtime. |
| [`conversion/`](conversion/) | Re-authored models + convert / verify / compress scripts (PyTorch → `.aimodel`). |
| [`swift/`](swift/) | `CoreAIRunner` — a Swift package that drives `.aimodel` LLM bundles, including architectures beyond the standard runtime. |
| [`apps/`](apps/) | SwiftUI on-device chat apps (iOS 27): CoreAIChat (Gemma 4 E2B GPU/ANE/⚡ + Qwen3.5 / Qwen3.5-2B / LFM2.5 / Granite ⚡pipelined, one picker) + QwenChatFast (Qwen3.5 static kernels) with in-app model download. |

## License

BSD-3-Clause ([`LICENSE`](LICENSE)). Re-authored model code derives from Apple's BSD-3-Clause
`coreai_models` and retains its notices. Model weights follow their own licenses (see each
Hugging Face repo).
