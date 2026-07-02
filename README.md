# CoreAI-Model-Zoo

LLMs converted to Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27) — downloadable, verified
on-device, with the conversion code and a knowledge base. Successor to
[`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

**Run any model on device.** Every row below links a ready-to-build app — in this repo's
[`apps/`](apps) or a [CoreAIKit example](https://github.com/john-rocky/coreai-kit/tree/main/Examples)
(marked ↗). To embed a model in your own app instead, add
[**CoreAIKit**](https://github.com/john-rocky/coreai-kit) and load it by its catalog id.

## Models

| Model | Download (`.aimodel`) | Run in app | License |
|---|---|---|---|
| **Qwen3.5-0.8B** | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| **Qwen3.5-2B** | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| **Qwen3.6-35B-A3B** (MoE, Mac-only) | [🤗 Qwen3.6-35B-A3B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | Apache-2.0 |
| **Qwen3.6-27B** (dense, Mac-only) | [🤗 Qwen3.6-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | Apache-2.0 |
| **GLM-4.7-Flash** (MoE + MLA, Mac-only — zoo's first MLA) | [🤗 GLM-4.7-Flash-CoreAI](https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | MIT |
| **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Gemma |
| **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| **Gemma 4 12B** (dense, Mac-only — custom flash-decode kernel ‡) | [🤗 Gemma-4-12B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | Gemma |
| **Gemma 4 31B** (dense, Mac-only — custom flash-decode kernel ‡) | [🤗 Gemma-4-31B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | Gemma |
| **LFM2.5-1.2B-Instruct** | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | LFM Open License v1.0 |
| **LFM2.5-8B-A1B** (MoE, custom `gather_qmm` kernel — first iPhone MoE) | [🤗 LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | LFM Open License v1.0 |
| **Granite 4.0-H 1B / 350M** | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| **Nanbeige4.1-3B** (dense reasoning/agentic, iPhone — 32B-class @ 3.93B) | [🤗 Nanbeige4.1-3B-CoreAI](https://huggingface.co/mlboydaisuke/Nanbeige4.1-3B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| **MiniCPM5-1B** (1B-class on-device LLM, hybrid Think/No-Think, 128K, OpenBMB) | [🤗 MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| **FastContext-1.0-4B** (repo-exploration agent — first-turn search / multi-turn evidence / file:line citation; Qwen3-4B arch, iPhone GPU, AOT h18p; Microsoft) | [🤗 FastContext-1.0-4B-CoreAI](https://huggingface.co/mlboydaisuke/FastContext-1.0-4B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | MIT |
| **BitCPM-8B** (zoo's first **1.58-bit ternary** LLM — every weight is {-1,0,+1}; MiniCPM4-8B arch, custom 2-bit packed-GEMM Metal kernel; 8B running in ~2.1 GB on iPhone GPU; OpenBMB) | [🤗 BitCPM-8B-CoreAI](https://huggingface.co/mlboydaisuke/BitCPM-8B-CoreAI) | [CoreAIChat](apps/CoreAIChat) ‡ | Apache-2.0 |
| **LLaDA-8B dLLM** (zoo's first **diffusion LLM** — masked-diffusion decode: fills a canvas of `[MASK]` tokens **in parallel**, not left-to-right AR; bidirectional LLaMA-dense 8B, [d3LLM](https://huggingface.co/d3LLM/d3LLM_LLaDA)-distilled; int4 ~4.9 GB, Mac) | [🤗 LLaDA-8B-dLLM-CoreAI](https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | Other |
| **BitVLA** (zoo's first **Vision-Language-Action / robotics** model + first ternary multimodal — image+instruction → 7-DoF robot action; **1.58-bit** BitNet-2B LLM + BitSigLIP vision, shared ternary kernel; runs on iPhone GPU; arXiv 2506.07530) | [🤗 BitVLA-CoreAI](https://huggingface.co/mlboydaisuke/BitVLA-CoreAI) | [CoreAIChat](apps/CoreAIChat) ‡ | MIT |
| **Qwen3-VL** (vision-language) | [🤗 2B](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) · [4B](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI) · [8B](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| **Holo2-4B** (GUI-grounding / computer-use VLM — screenshot + instruction → click coordinates; Qwen3-VL-4B backbone, H Company; zoo's first computer-use model) | [🤗 Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| **MiniCPM-V 4.6** (vision-language, sub-2B — strongest tiny VLM) | [🤗 MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI) | [MiniCPMVisualIntel](apps/MiniCPMVisualIntel) | Apache-2.0 |
| **Gemma 4 E2B vision (VL)** (image+text) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Gemma |
| **Unlimited-OCR** (document OCR → markdown: tables→HTML, formulas→LaTeX; zoo's first doc-OCR — **stock runtime, no patch**, flat-latency R-SWA) | [🤗 Unlimited-OCR-CoreAI](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI) | [CoreAIOCR](apps/CoreAIOCR) | MIT |
| **Qwen2.5-Omni-3B Audio** (audio *understanding* — describes sounds, not a transcript; iPhone + Mac, zoo's first audio model) | [🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) | [coreai-audio](apps/coreai-audio) | Apache-2.0 |
| **Whisper large-v3-turbo** (speech→text — 100 languages, auto-detect; stock runtime, iPhone AOT + Mac) | [🤗 whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official) | [coreai-audio](apps/coreai-audio) | MIT |
| **Qwen3-ASR-1.7B** (speech→text — the zoo's first ASR; AuT encoder + Qwen3 decoder, 52 languages; iPhone + Mac) | [🤗 Qwen3-ASR-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-ASR-1.7B-CoreAI) | [coreai-audio](apps/coreai-audio) | Apache-2.0 |
| **Parakeet-TDT-0.6B** (speech→text — zoo's first **transducer / TDT (RNN-T)**; NVIDIA FastConformer + LSTM predictor + joint, 3 graphs + host greedy loop, 25 EU languages; iPhone 47.9× real-time) | [🤗 Parakeet-TDT-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-CoreAI) | [coreai-audio](apps/coreai-audio) | CC-BY-4.0 |
| **Kokoro-82M** (text-to-speech — zoo's first TTS; StyleTTS2 + iSTFTNet, 28 English voices, runs on any text) | [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI) | [coreai-audio](apps/coreai-audio) | Apache-2.0 |
| **VoxCPM-0.5B** (text-to-speech — diffusion TTS: MiniCPM4 LM + LocDiT flow-matching + AudioVAE; iPhone + Mac, int8 LM) | [🤗 VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI) | [coreai-audio](apps/coreai-audio) | Apache-2.0 |
| **VoxCPM2 2B** (text-to-speech — 2B successor at 48 kHz: MiniCPM4 28L LM + LocDiT-12L flow-matching + 48 kHz AudioVAE; iPhone + Mac, int8 LM) | [🤗 VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI) | [coreai-audio](apps/coreai-audio) | Apache-2.0 |
| **Stable Audio Open Small** (text → **music / audio** — the zoo's first **generative audio**; latent diffusion: T5 encoder + DiT (8-step rectified-flow) + Oobleck VAE, ~11s 44.1 kHz stereo; fp16 ~1 GB, **~0.4 s / 11 s on M4 Max ≈ 30× real-time**; Stability AI + Arm) | [🤗 Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI) | [coreai-audio](apps/coreai-audio) | Stability Community |
| **V-JEPA 2** (ViT-L, SSv2 — the zoo's first **world model**: Meta's self-supervised video encoder (JEPA, predicts in representation space) + action-recognition head, 174 physical-interaction classes; 16-frame clip → action, fp16 ~675 MB, **~160 ms/clip on M4 Max**; Meta AI, MIT) | [🤗 VJEPA2-ViTL-SSv2-CoreAI](https://huggingface.co/mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI) | [coreai-video](apps/coreai-video) | MIT |
| **EmbeddingGemma 300M** (text embeddings — on-device RAG / semantic search) | [🤗 embeddinggemma-300m-CoreAI](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Gemma |
| **Qwen3-Embedding 0.6B** (multilingual text embeddings, last-token pooling + MRL) | [🤗 Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Apache-2.0 |
| **Qwen3-Reranker 0.6B** (cross-encoder reranker — yes/no relevance score) | [🤗 Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Apache-2.0 |
| **RF-DETR nano/small/medium/large** (object detection, no NMS) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| **RF-DETR-Seg nano→2xlarge** (instance segmentation, 6 sizes) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| **AdcSR ×4** (super-resolution — zoo's first; one-step diffusion-GAN, on-device) | [🤗 AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI) | [CoreAIUpscale](apps/CoreAIUpscale) | Apache-2.0 + OpenRAIL++ |
| **Depth Anything 3** (monocular depth — zoo's first depth model; small + base, fp16/fp32) | [🤗 Depth-Anything-3-CoreAI](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI) | [CoreAIDepth](apps/CoreAIDepth) | Apache-2.0 |
| **TripoSplat** (single image → **3D Gaussian splats** — the zoo's first 3D; DINOv3 ViT-H + 20-step flow-matching DiT + octree sampler + Gaussian decoder, Mac GPU ~1 min; `.ply`/`.splat` → RealityKit / [MetalSplatter](https://github.com/scier/MetalSplatter); VAST) | [🤗 TripoSplat-CoreAI](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI) | [TripoSplatMac](apps/TripoSplatMac) | MIT |
| **LTX-Video 2B distilled** (text → **video** — the zoo's first video model; T5-XXL + 8-step flow-matching DiT + causal video VAE, host FlowMatch sampler; 512×768×49f ~14 s Mac GPU; Lightricks) | [🤗 LTX-Video-2B-CoreAI](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI) | [CoreAIVideo](apps/CoreAIVideo) | Other (LTXV) |

▸ **Run in app** — apps in [`apps/`](apps) live in this repo; **↗** links a
[CoreAIKit example app](https://github.com/john-rocky/coreai-kit/tree/main/Examples); **‡** = app
wiring in progress. Full app list: [`apps/README.md`](apps/README.md).

### Decode throughput (tok/s, greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **71.9** | 14.7 | **210** |
| **Qwen3.5-2B** | **29** | — | **161** |
| **LFM2.5-1.2B** | **45.4** | — | **276.5** |
| **Granite 4.0-H 1B** | **36.3** | — | **136.5** |
| **Nanbeige4.1-3B** | **15.9** | — | **114.5** |
| **MiniCPM5-1B** (OpenBMB, int8 — 24/24 exact vs HF) | **66.8** | — | 59.4 |
| **FastContext-1.0-4B** (repo-exploration agent, 4bit — AOT h18p; ANE inference unsupported) | **20.4** | ✗ | — |
| **BitCPM-8B** (1.58-bit ternary, OpenBMB — custom 2-bit packed-GEMM kernel; AOT h18p; ~2.1 GB resident; token-exact 3/3 vs ref) | **17** | ✗ | **62.7** |
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
