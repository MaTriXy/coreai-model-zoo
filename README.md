# CoreAI-Model-Zoo

[![CoreAIKit](https://img.shields.io/github/v/tag/john-rocky/coreai-kit?label=CoreAIKit)](https://github.com/john-rocky/coreai-kit/releases)
[![HF downloads](https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjohn-rocky%2Fcoreai-assets%2Fmain%2Fbadge%2Fhf-downloads.json)](https://huggingface.co/mlboydaisuke)
[![CI](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/ci.yml)
[![Nightly device gate](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml/badge.svg)](https://github.com/john-rocky/coreai-kit/actions/workflows/nightly-gate.yml)

> [!WARNING]
> ## Every converted model in the zoo is currently unusable
>
> They run on iOS/macOS 27 **beta 1**. From **beta 2 onward, all of them fail to load.**
>
> `coreai-torch` 0.4.0 baked PyTorch stack traces into the IR as fused locations. From OS 27
> beta 2 the compiler stopped accepting them, so every asset converted before 2026-07-13 —
> all 83 catalog entries — fails to load:
>
> ```
> error: expected AICode versioned location
> LLVM ERROR: cannot unwrap empty `odiec_module_t`
> ```
>
> Apple has documented this and the call is sound
> ([coreai-torch#37](https://github.com/apple/coreai-torch/issues/37),
> [v0.4.1 release notes](https://github.com/apple/coreai-torch/releases/tag/v0.4.1)):
> *"Reconvert your model using coreai-torch v0.4.1 or later."*
>
> Unfortunately there is no workaround. `coreai-build package` re-emits the asset but leaves
> the IR locations untouched; pinning `coreai-core` back to `1.0.0b1` does not help either, as
> the gate is OS-side. `coreai-build inspect` still reads the asset without complaint, which
> makes it look recoverable — it isn't. The weights are intact. The graph is intact. Only the
> debug metadata is the problem.
>
> Recovery is underway. Every recipe is in [`conversion/`](conversion). Models will return in
> waves, and this banner will go away with the last one.
>
> If this cost you time, I'm sorry.

LLMs converted to Apple **Core AI** (`.aimodel`, iOS 27 / macOS 27) — downloadable, verified
on-device, with the conversion code and a knowledge base. Successor to
[`CoreML-Models`](https://github.com/john-rocky/CoreML-Models).

**The `from_pretrained` of Core AI** — one line, any zoo model, via
[**CoreAIKit**](https://github.com/john-rocky/coreai-kit) (SPM):

```swift
let chat = try await ChatSession(catalog: "qwen3.5-2b")   // downloads once, then cached
let reply = try await chat.respond(to: "What can you do, offline?")
```

Same gesture for every capability — `KitTranscriber(catalog: "whisper-large-v3-turbo")` is
speech-to-text in 3 lines ([card](zoo/whisper-large-v3-turbo.md)). Each model's card carries the
complete copy-paste snippet and its integration checklist. Every row below also links a
ready-to-build app — in this repo's [`apps/`](apps) or a
[CoreAIKit example](https://github.com/john-rocky/coreai-kit/tree/main/Examples) (marked ↗).

Chat models also plug straight into **Apple's FoundationModels API**:
`LanguageModelSession(model: try await KitLanguageModel(model: .qwen3_0_6B))` gives you the
system session — `Tool` calling, `@Generable` guided generation, transcripts — backed by a zoo
model ([how](https://github.com/john-rocky/coreai-kit#works-with-apples-foundationmodels-api)).
Zero-dependency alternative: every bundle loads with Apple's own
`CoreAILanguageModel(resourcesAt:)` as-is; this repo's
[`ZooFMProvider`](swift/Sources/ZooFMProvider) adds streaming tool calling on top (incl. LFM's
native dialect) — engineering notes in [`knowledge/fm-provider.md`](knowledge/fm-provider.md).

## Quickstart — running a model on your device

New here? You'll have a model answering on-device in a few minutes (needs Xcode 27 + a Mac or an
iPhone/iPad on iOS/macOS 27):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/ChatDemo/ChatDemo.xcodeproj   # Run, then pick a model in the picker
```

The app downloads the model on first pick (cached after), then runs it fully offline. **Start
small for the fastest first run:** `Qwen3-0.6B` (454 MB) or `Qwen3.5-2B` on iPhone;
any of the Mac-only rows on a Mac. Prefer the terminal? `swift run chat-cli --model qwen3-0.6b
--prompt "Hello"` from `Examples/ChatDemo`. To drop a model into **your own** app, copy the
snippet from that model's card — it's the same `catalog:` one-liner shown above.

## Models

| Model | Download (`.aimodel`) | Run in app | License |
|---|---|---|---|
| [**Qwen3.5-0.8B**](zoo/qwen3.5.md) | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.5-2B**](zoo/qwen3.5.md) | [🤗 qwen3.5-2B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.6-35B-A3B**](zoo/qwen3.6.md) (MoE, Mac-only) | [🤗 Qwen3.6-35B-A3B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-35B-A3B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Qwen3.6-27B**](zoo/qwen3.6-27b.md) (dense, Mac-only) | [🤗 Qwen3.6-27B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3.6-27B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Ornith-1.0-9B**](zoo/ornith-1.0-9b.md) (zoo's first **agentic-coding** model — self-scaffolding coder, Qwen3.5 arch, DeepReinforce; Mac-only, 48 tok/s int8 / 59 int4) | [🤗 Ornith-1.0-9B-CoreAI](https://huggingface.co/mlboydaisuke/Ornith-1.0-9B-CoreAI) | [CoreAIChatMac](apps/CoreAIChatMac) | MIT |
| [**GLM-4.7-Flash**](zoo/glm-4.7-flash.md) (MoE + MLA, Mac-only — zoo's first MLA) | [🤗 GLM-4.7-Flash-CoreAI](https://huggingface.co/mlboydaisuke/GLM-4.7-Flash-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | MIT |
| **Gemma 4 E2B** (text, incl. official-QAT int4) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| **Gemma 4 E4B** (text, official-QAT int4) | [🤗 gemma-4-E4B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E4B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| [**Gemma 4 12B**](zoo/gemma4-12b.md) (dense, Mac-only — custom flash-decode kernel) | [🤗 Gemma-4-12B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-12B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| [**Gemma 4 31B**](zoo/gemma4-31b.md) (dense, Mac-only — custom flash-decode kernel) | [🤗 Gemma-4-31B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Gemma |
| [**LFM2.5-1.2B-Instruct**](zoo/lfm2.5.md) | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | LFM Open License v1.0 |
| [**LFM2.5-8B-A1B**](zoo/lfm2.5-8b-a1b-moe.md) (MoE, custom `gather_qmm` kernel — first iPhone MoE) | [🤗 LFM2.5-8B-A1B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-8B-A1B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | LFM Open License v1.0 |
| [**Granite 4.0-H 1B / 350M**](zoo/granite-4.0-h.md) | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) (1B) · 350M: [CoreAIChat](apps/CoreAIChat) | Apache-2.0 |
| [**Nanbeige4.1-3B**](zoo/nanbeige4.1-3b.md) (dense reasoning/agentic, iPhone — 32B-class @ 3.93B) | [🤗 Nanbeige4.1-3B-CoreAI](https://huggingface.co/mlboydaisuke/Nanbeige4.1-3B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**MiniCPM5-1B**](zoo/minicpm5-1b.md) (1B-class on-device LLM, hybrid Think/No-Think, 128K, OpenBMB) | [🤗 MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Apache-2.0 |
| [**Youtu-LLM-2B**](zoo/youtu.md) (dense **MLA** — zoo's first **iPhone** MLA & first **dense** MLA; DeepSeek-V2-style latent-KV attention at 1.96B with an absorbed flash-decode kernel, reasoning + agentic; Tencent) | [🤗 Youtu-LLM-2B-CoreAI](https://huggingface.co/mlboydaisuke/Youtu-LLM-2B-CoreAI) | [ChatDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ChatDemo) | Other (youtu-llm) |
| **FastContext-1.0-4B** (repo-exploration agent — first-turn search / multi-turn evidence / file:line citation; Qwen3-4B arch, iPhone GPU, AOT h18p; Microsoft) | [🤗 FastContext-1.0-4B-CoreAI](https://huggingface.co/mlboydaisuke/FastContext-1.0-4B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | MIT |
| **BitCPM-8B** (zoo's first **1.58-bit ternary** LLM — every weight is {-1,0,+1}; MiniCPM4-8B arch, custom 2-bit packed-GEMM Metal kernel; 8B running in ~2.1 GB on iPhone GPU; OpenBMB) | [🤗 BitCPM-8B-CoreAI](https://huggingface.co/mlboydaisuke/BitCPM-8B-CoreAI) | [CoreAIChat](apps/CoreAIChat) ‡ | Apache-2.0 |
| **LLaDA-8B dLLM** (zoo's first **diffusion LLM** — masked-diffusion decode: fills a canvas of `[MASK]` tokens **in parallel**, not left-to-right AR; bidirectional LLaMA-dense 8B, [d3LLM](https://huggingface.co/d3LLM/d3LLM_LLaDA)-distilled; int4 ~4.9 GB, Mac) | [🤗 LLaDA-8B-dLLM-CoreAI](https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI) | [DiffuseChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DiffuseChat) | Other |
| **BitVLA** (zoo's first **Vision-Language-Action / robotics** model + first ternary multimodal — image+instruction → 7-DoF robot action; **1.58-bit** BitNet-2B LLM + BitSigLIP vision, shared ternary kernel; runs on iPhone GPU; arXiv 2506.07530) | [🤗 BitVLA-CoreAI](https://huggingface.co/mlboydaisuke/BitVLA-CoreAI) | [CoreAIChat](apps/CoreAIChat) ‡ | MIT |
| [**Qwen3-VL**](zoo/qwen3-vl.md) (vision-language) | [🤗 2B](https://huggingface.co/mlboydaisuke/Qwen3-VL-2B-CoreAI) · [4B](https://huggingface.co/mlboydaisuke/Qwen3-VL-4B-CoreAI) · [8B](https://huggingface.co/mlboydaisuke/Qwen3-VL-8B-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| **Holo2-4B** (GUI-grounding / computer-use VLM — screenshot + instruction → click coordinates; Qwen3-VL-4B backbone, H Company; zoo's first computer-use model) | [🤗 Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| **MiniCPM-V 4.6** (vision-language, sub-2B — strongest tiny VLM) | [🤗 MiniCPM-V-4.6-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM-V-4.6-CoreAI) | [VLChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/VLChat) | Apache-2.0 |
| **Gemma 4 E2B vision (VL)** (image+text) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | [CoreAIChat](apps/CoreAIChat) | Gemma |
| **Unlimited-OCR** (document OCR → markdown: tables→HTML, formulas→LaTeX; zoo's first doc-OCR — **stock runtime, no patch**, flat-latency R-SWA) | [🤗 Unlimited-OCR-CoreAI](https://huggingface.co/mlboydaisuke/Unlimited-OCR-CoreAI) | [ReadDoc ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc) | MIT |
| [**GLM-OCR**](zoo/glm-ocr.md) (document OCR → Markdown; GLM-4.V small **0.9B**, single-pass, tables→Markdown; iPhone + Mac, ~4 s/page) | [🤗 GLM-OCR-CoreAI](https://huggingface.co/mlboydaisuke/GLM-OCR-CoreAI) | [ReadDoc ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc) | MIT |
| [**MinerU2.5-Pro**](zoo/mineru.md) (whole-page document parsing → structured Markdown; zoo's first **whole-page auto-structuring** — 2-stage layout + per-region recognition in one stock Qwen2-VL **1.2B**, tables→`<table>` HTML; Mac) | [🤗 MinerU2.5-Pro-CoreAI](https://huggingface.co/mlboydaisuke/MinerU2.5-Pro-CoreAI) | [ReadDoc ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ReadDoc) | Apache-2.0 |
| **Qwen2.5-Omni-3B Audio** (audio *understanding* — describes sounds, not a transcript; iPhone + Mac, zoo's first audio model) | [🤗 Qwen2.5-Omni-3B-Audio-CoreAI](https://huggingface.co/mlboydaisuke/Qwen2.5-Omni-3B-Audio-CoreAI) | [AudioChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/AudioChat) | Apache-2.0 |
| [**Whisper large-v3-turbo**](zoo/whisper-large-v3-turbo.md) (speech→text — 100 languages, auto-detect; stock runtime, iPhone AOT + Mac) | [🤗 whisper-large-v3-turbo-CoreAI-official](https://huggingface.co/mlboydaisuke/whisper-large-v3-turbo-CoreAI-official) | [Transcribe ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe) | MIT |
| [**Qwen3-ASR-1.7B**](zoo/qwen3-asr.md) (speech→text — the zoo's first ASR; AuT encoder + Qwen3 decoder, 52 languages; iPhone + Mac) | [🤗 Qwen3-ASR-1.7B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-ASR-1.7B-CoreAI) | [Transcribe ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe) | Apache-2.0 |
| [**Parakeet-TDT-0.6B**](zoo/parakeet.md) (speech→text — zoo's first **transducer / TDT (RNN-T)**; NVIDIA FastConformer + LSTM predictor + joint, 3 graphs + host greedy loop, 25 EU languages; iPhone 47.9× real-time) | [🤗 Parakeet-TDT-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Parakeet-TDT-0.6B-CoreAI) | [Transcribe ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Transcribe) | CC-BY-4.0 |
| **Nemotron 3.5 ASR Streaming 0.6B** (speech→text — the zoo's first **STREAMING ASR**: live-mic transcription in 320 ms chunks, cache-aware FastConformer + pure RNN-T with explicit KV/conv cache I/O; 40 locales in one checkpoint via a run-time language input, punctuation built in, any-length audio) | [🤗 Nemotron-3.5-ASR-Streaming-CoreAI](https://huggingface.co/mlboydaisuke/Nemotron-3.5-ASR-Streaming-CoreAI) | [coreai-audio](apps/coreai-audio) | OpenMDW-1.1 |
| [**Streaming Sortformer 4-spk v2**](zoo/sortformer-diar.md) (speaker diarization — the zoo's first **"who spoke when"**, up to 4 speakers; NeMo core as a Core AI graph + Swift host streaming loop + AOSC speaker-cache compression; pairs with any zoo ASR for a **diarized transcript**; iPhone + Mac, 100% activity-agree vs NeMo) | [🤗 Streaming-Sortformer-Diar-CoreAI](https://huggingface.co/mlboydaisuke/Streaming-Sortformer-Diar-CoreAI) | [coreai-audio](apps/coreai-audio) | CC-BY-4.0 |
| **Kokoro-82M** (text-to-speech — zoo's first TTS; StyleTTS2 + iSTFTNet, 28 English voices, runs on any text) | [🤗 Kokoro-82M-CoreAI](https://huggingface.co/mlboydaisuke/Kokoro-82M-CoreAI) | [Speak ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak) | Apache-2.0 |
| **VoxCPM-0.5B** (text-to-speech — diffusion TTS: MiniCPM4 LM + LocDiT flow-matching + AudioVAE; iPhone + Mac, int8 LM) | [🤗 VoxCPM-0.5B-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM-0.5B-CoreAI) | [Speak ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak) | Apache-2.0 |
| **VoxCPM2 2B** (text-to-speech — 2B successor at 48 kHz: MiniCPM4 28L LM + LocDiT-12L flow-matching + 48 kHz AudioVAE; iPhone + Mac, int8 LM) | [🤗 VoxCPM2-CoreAI](https://huggingface.co/mlboydaisuke/VoxCPM2-CoreAI) | [Speak ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Speak) | Apache-2.0 |
| **Stable Audio Open Small** (text → **music / audio** — the zoo's first **generative audio**; latent diffusion: T5 encoder + DiT (8-step rectified-flow) + Oobleck VAE, ~11s 44.1 kHz stereo; fp16 ~1 GB, **~0.4 s / 11 s on M4 Max ≈ 30× real-time**; Stability AI + Arm) | [🤗 Stable-Audio-Open-Small-CoreAI](https://huggingface.co/mlboydaisuke/Stable-Audio-Open-Small-CoreAI) | [Music ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Music) | Stability Community |
| **V-JEPA 2** (ViT-L, SSv2 — the zoo's first **world model**: Meta's self-supervised video encoder (JEPA, predicts in representation space) + action-recognition head, 174 physical-interaction classes; 16-frame clip → action, fp16 ~675 MB, **~160 ms/clip on M4 Max**; Meta AI, MIT) | [🤗 VJEPA2-ViTL-SSv2-CoreAI](https://huggingface.co/mlboydaisuke/VJEPA2-ViTL-SSv2-CoreAI) | [ActionCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/ActionCamera) | MIT |
| **EmbeddingGemma 300M** (text embeddings — on-device RAG / semantic search) | [🤗 embeddinggemma-300m-CoreAI](https://huggingface.co/mlboydaisuke/embeddinggemma-300m-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Gemma |
| **Qwen3-Embedding 0.6B** (multilingual text embeddings, last-token pooling + MRL) | [🤗 Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Apache-2.0 |
| **Qwen3-Reranker 0.6B** (cross-encoder reranker — yes/no relevance score) | [🤗 Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI) | [DocChat ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocChat) | Apache-2.0 |
| [**ColModernVBERT**](zoo/colmodernvbert.md) (visual document retrieval — late-interaction MaxSim over page *images*, no OCR; zoo's first multi-vector retriever) | [🤗 ColModernVBERT-CoreAI](https://huggingface.co/mlboydaisuke/ColModernVBERT-CoreAI) | [DocSearch ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DocSearch) | MIT |
| [**GLiNER2-PII**](zoo/gliner2-pii.md) (information extraction / NER — the zoo's first **NER / schema-driven extraction** & first **DeBERTa-v3** port; zero-shot PII detection + redaction, any label set at call time; mDeBERTa-v3 fused graph + Swift host collator/decode, iPhone + Mac, byte-identical to GLiNER2) | [🤗 GLiNER2-PII-CoreAI](https://huggingface.co/mlboydaisuke/GLiNER2-PII-CoreAI) | [InfoExtract ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/InfoExtract) | Apache-2.0 |
| [**RF-DETR nano/small/medium/large**](zoo/rf-detr.md) (object detection, no NMS) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| **RF-DETR-Seg nano→2xlarge** (instance segmentation, 6 sizes) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| [**YOLOX-S**](zoo/yolox.md) (object detection — dense anchor-free, host NMS) | [🤗 YOLOX-CoreAI](https://huggingface.co/mlboydaisuke/YOLOX-CoreAI) | [DetectCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DetectCamera) | Apache-2.0 |
| **AdcSR ×4** (super-resolution — zoo's first; one-step diffusion-GAN, on-device) | [🤗 AdcSR-CoreAI](https://huggingface.co/mlboydaisuke/AdcSR-CoreAI) | [UpscaleDemo ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/UpscaleDemo) | Apache-2.0 + OpenRAIL++ |
| [**Depth Anything 3**](zoo/depth-anything-3.md) (monocular depth — zoo's first depth model; small + base, fp16/fp32) | [🤗 Depth-Anything-3-CoreAI](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI) | [DepthCamera ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DepthCamera) | Apache-2.0 |
| **TripoSplat** (single image → **3D Gaussian splats** — the zoo's first 3D; DINOv3 ViT-H + 20-step flow-matching DiT + octree sampler + Gaussian decoder, Mac GPU ~1 min; `.ply`/`.splat` → RealityKit / [MetalSplatter](https://github.com/scier/MetalSplatter); VAST) | [🤗 TripoSplat-CoreAI](https://huggingface.co/mlboydaisuke/TripoSplat-CoreAI) | [TripoSplatMac](apps/TripoSplatMac) | MIT |
| **LTX-Video 2B distilled** (text → **video** — the zoo's first video model; T5-XXL + 8-step flow-matching DiT + causal video VAE, host FlowMatch sampler; 512×768×49f ~14 s Mac GPU; Lightricks) | [🤗 LTX-Video-2B-CoreAI](https://huggingface.co/mlboydaisuke/LTX-Video-2B-CoreAI) | [CoreAIVideo](apps/CoreAIVideo) | Other (LTXV) |
| **FLUX.2 klein 4B** (text → **image** + **in-context editing** — the zoo's first image-generation & editing model; step-distilled flow-matching DiT (4 steps, guidance 1.0) + Qwen3 text encoder, 1024²; native **in-context edit** — add/replace/combine while keeping the subject, unlike strength-based SDEdit — plus **multi-reference** compose, both exported as edit-sequence transformers (output latent T=0 concatenated with reference tokens T=10·i); int4, Mac; Black Forest Labs) | [🤗 FLUX.2-klein-4B-CoreAI](https://huggingface.co/mlboydaisuke/FLUX.2-klein-4B-CoreAI) | [CoreAIImageGen](apps/CoreAIImageGen) | Apache-2.0 |
| [**GLM-Image**](zoo/glm-image.md) (text → **image** — the zoo's first **AR + diffusion hybrid**; a 9B GLM-4 AR model *samples the image as discrete visual prior tokens* like an LLM (~36 tok/s), then a 7B flow-matching DiT denoises conditioned on them + 16ch VAE; composition from the AR, texture from the DiT; 1024² native + 512² fast, int8, Mac; ZhipuAI) | [🤗 GLM-Image-CoreAI](https://huggingface.co/mlboydaisuke/GLM-Image-CoreAI) | [CoreAIImageGen](apps/CoreAIImageGen) | MIT |
| [**Z-Image-Turbo**](zoo/z-image-turbo.md) (text → **image** — a 6B Single-Stream DiT (S3-DiT): Qwen3-4B text encoder → 34-block DiT (8-step FlowMatchEuler + CFG) → 16ch VAE; photoreal by default. **One graph covers 256²/512²/1024² and any prompt length** (dynamic image + caption axes, ~5–9 % cost). bf16 and **near-lossless** — PSNR 42.6 dB vs the fp32 reference; 18 s @512² / 70 s @1024² on M4 Max. fp16 NaNs this model, so Mac-only: AOT will not take a bf16 module; Alibaba Tongyi-MAI) | [🤗 Z-Image-Turbo-CoreAI](https://huggingface.co/mlboydaisuke/Z-Image-Turbo-CoreAI) | [CoreAIImageGen](apps/CoreAIImageGen) | Apache-2.0 |
| [**TimesFM 2.5 200M**](zoo/timesfm.md) (**time-series forecasting** — the zoo's first forecasting foundation model; decoder-only patched transformer, any univariate series → 128-step point + 10-quantile forecast; one stateless graph + host RevIN/flip DSP, fp16 ~463 MB, **~14 ms/forecast** M4 Max / **~25 ms iPhone 17 Pro** device-verified; Google) | [🤗 TimesFM-2.5-200M-CoreAI](https://huggingface.co/mlboydaisuke/TimesFM-2.5-200M-CoreAI) | [Forecast ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples/Forecast) | Apache-2.0 |

▸ **Run in app** — apps in [`apps/`](apps) live in this repo; **↗** links a
[CoreAIKit example app](https://github.com/john-rocky/coreai-kit/tree/main/Examples); **‡** = app
wiring in progress. Full app list: [`apps/README.md`](apps/README.md).

### Built with the zoo

Third-party apps running zoo models. Built something? Open a
[showcase issue](https://github.com/john-rocky/coreai-model-zoo/issues/new?template=showcase.yml)
— a name, a link, and one line is all it takes. *Your app here.*

### Most downloaded

<img src="https://raw.githubusercontent.com/john-rocky/coreai-assets/main/charts/hf-top.svg" alt="Most downloaded zoo models this month" width="700">

*(auto-updated weekly from Hugging Face download counts)*

### Decode throughput (tok/s, greedy; output top-1 exact vs the Hugging Face reference)

| | iPhone 17 Pro · GPU | iPhone 17 Pro · ANE | M4 Max · GPU |
|---|---|---|---|
| **Qwen3.5-0.8B** | **71.9** | 14.7 | **210** |
| **Qwen3.5-2B** | **29** | — | **161** |
| **LFM2.5-1.2B** | **45.4** | — | **276.5** |
| **Granite 4.0-H 1B** | **36.3** | — | **136.5** |
| **Nanbeige4.1-3B** | **15.9** | — | **114.5** |
| **MiniCPM5-1B** (OpenBMB, int8 — 24/24 exact vs HF) | **66.8** | — | 59.4 |
| **Youtu-LLM-2B** (dense MLA, int8 — 16/16 device ≡ Mac ≡ HF) | **~19** (in-app ~24) | — | **102.8** |
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
  - **Demo app, no build** → Mac: [**.dmg**](https://github.com/john-rocky/coreai-model-zoo/releases/download/mac-v1.0/CoreAI-Zoo-for-Mac.dmg) (notarized, runs the Mac-only bundles) · iPhone: [**CoreAIChat on TestFlight**](https://testflight.apple.com/join/bK4P7xby)
  - **Build it** → [`apps/`](apps/) — Xcode 27 beta + xcodegen, the `coreai-models` patch stack + `tokenizer.json`
- **Use a model in your own app** → add [**CoreAIKit**](https://github.com/john-rocky/coreai-kit)
  (SPM) and load the catalog id; the model's card has the complete snippet + a 5-line
  integration checklist (golden example: [`zoo/whisper-large-v3-turbo.md`](zoo/whisper-large-v3-turbo.md)).
  Engine-level deep-dive: [`knowledge/swift-runtime.md`](knowledge/swift-runtime.md)
- **Port a model, end to end** → [**`PORTING.md`**](PORTING.md) — the complete walk from HF
  checkpoint to a verified `.aimodel` on iPhone (oracle → export → gates → device → publish),
  with a vision and an LLM worked example. Start here to contribute a port.
- **Convert a model** (export API + gotchas) → [`knowledge/conversion-guide.md`](knowledge/conversion-guide.md)
- **Compress** → [`knowledge/compression.md`](knowledge/compression.md)
- **Make it fast** → [`knowledge/custom-metal-kernels.md`](knowledge/custom-metal-kernels.md) · [`knowledge/performance-ceiling.md`](knowledge/performance-ceiling.md)
- **Known beta issue** (in-graph KV-write crash; workarounds + the input-mask escape) → [`knowledge/coreai-beta-mpsgraph-kvwrite-bug.md`](knowledge/coreai-beta-mpsgraph-kvwrite-bug.md) — FB23024751 / [apple/coreai-models#5](https://github.com/apple/coreai-models/issues/5)

## Repository layout

| Dir | What |
|---|---|
| [**coreai-kit** ↗](https://github.com/john-rocky/coreai-kit) | (sibling repo) The Swift package that runs this zoo: 1-line `catalog:` APIs (`ChatSession`, `KitTranscriber`, …), model download + cache, and per-kind example apps in [`Examples/`](https://github.com/john-rocky/coreai-kit/tree/main/Examples) — the cards' ▶️ / 💻 doors point there. |
| [`zoo/`](zoo/) | Model cards — configurations, sizes, parity, measured throughput. |
| [`knowledge/`](knowledge/) | Verified notes on the framework: conversion, compression, stateful KV, custom Metal kernels, AOT, compute-unit rules, the Swift runtime. |
| [`conversion/`](conversion/) | Re-authored models + convert / verify / compress scripts (PyTorch → `.aimodel`). |
| [`swift/`](swift/) | `CoreAIRunner` — a Swift package that drives `.aimodel` LLM bundles, including architectures beyond the standard runtime. |
| [`apps/`](apps/) | **Engine showcases** — apps for models that need a hand-tuned backend (custom Metal kernels, patch stack: BitCPM, RWKV-7, LLaDA, …) and the device-verification bench behind the published numbers. Want to *just run a model*? Use the [kit examples ↗](https://github.com/john-rocky/coreai-kit/tree/main/Examples) instead. |

## Contributing

- **Conversion requests** — a model you'd like to see here? [Open an issue](https://github.com/john-rocky/coreai-model-zoo/issues/new) with the Hugging Face link and what you'd use it for.
- **Port one yourself** — [`PORTING.md`](PORTING.md) walks the whole path; PRs welcome.
- **No code needed** — run the Bench tab in [CoreAIChat (TestFlight)](https://testflight.apple.com/join/bK4P7xby) and submit the result: your device becomes a row in [`BENCHMARKS.md`](BENCHMARKS.md).

## License

BSD-3-Clause ([`LICENSE`](LICENSE)). Re-authored model code derives from Apple's BSD-3-Clause
`coreai_models` and retains its notices. Model weights follow their own licenses (see each
Hugging Face repo).
