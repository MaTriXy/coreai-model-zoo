# Zoo — Core AI converted models

Model cards for models converted to Core AI `.aimodel`. **Ready-to-run bundles are on Hugging
Face** (one best verified configuration per platform × compute unit); each card also links the
source checkpoint and the `conversion/` script, plus parity numbers, sizes, and measured
throughput.

| Card | Family | Download | Status |
|---|---|---|---|
| [`qwen3.5.md`](qwen3.5.md) | Qwen3.5 (hybrid linear+full attn) | [🤗 qwen3.5-0.8B-CoreAI](https://huggingface.co/mlboydaisuke/qwen3.5-0.8B-CoreAI) | 0.8B + 2B, top-1 exact vs HF |
| [`gemma4-e2b.md`](gemma4-e2b.md) | Gemma 4 (multimodal; text decoder) | [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | 8/8 exact vs HF |
| [`gemma4-vl.md`](gemma4-vl.md) | Gemma 4 E2B vision (image+text→text, 2nd VLM) | `vl/` in [🤗 gemma-4-E2B-CoreAI](https://huggingface.co/mlboydaisuke/gemma-4-E2B-CoreAI) | margin-ruled exact vs fp32 HF; **82.4 tok/s M4 Max / 25.5 iPhone 17 Pro** (pipelined VLM rider) |
| [`lfm2.5.md`](lfm2.5.md) | LFM2.5 (conv + full-attn hybrid, LiquidAI) | [🤗 LFM2.5-1.2B-CoreAI](https://huggingface.co/mlboydaisuke/LFM2.5-1.2B-CoreAI) | 1.2B, oracle gate 16/16, **276.5 tok/s M4 Max / 44.1–46.6 iPhone (int8 + absmax int8 head)** (pipelined) |
| [`granite-4.0-h.md`](granite-4.0-h.md) | Granite 4.0-H (Mamba2 + attn hybrid, IBM) | [🤗 granite-4.0-h-CoreAI](https://huggingface.co/mlboydaisuke/granite-4.0-h-CoreAI) | 1b + 350m, oracle gate 16/16, **136.5 tok/s M4 Max / 35.4–37.1 iPhone 17 Pro (int8 head)** (pipelined, first SSM-scan rider) |
| [`minicpm5-1b.md`](minicpm5-1b.md) | MiniCPM5-1B (plain Llama dense, OpenBMB) | [🤗 MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) | 1.08B int8, **lossless** (24/24 token-exact vs HF fp32), **66.8 tok/s iPhone 17 Pro** (pipelined, llama→mistral remap) |
| [`fastcontext.md`](fastcontext.md) | FastContext-1.0-4B-SFT (repo-exploration agent, Qwen3-4B arch, Microsoft) | [🤗 FastContext-1.0-4B-CoreAI](https://huggingface.co/mlboydaisuke/FastContext-1.0-4B-CoreAI) | 4B 4bit, parity **23/24 argmax (ppl 1.41)** vs HF, **20.4 tok/s decode / 22.1 prefill iPhone 17 Pro** (AOT h18p GPU; zoo's first stock-arch + first 4B-class iPhone LLM) |
| [`rf-detr.md`](rf-detr.md) | RF-DETR + RF-DETR-Seg (detection / instance segmentation, Roboflow) | [🤗 RF-DETR-CoreAI](https://huggingface.co/mlboydaisuke/RF-DETR-CoreAI) | det ×4 + seg ×6 fp32, gated cpu+gpu (mask IoU 1.000), **8.6–59.1 ms/frame M4 Max GPU** |
| [`depth-anything-3.md`](depth-anything-3.md) | Depth Anything 3 (monocular depth, DINOv2+DPT, ByteDance) — zoo's **first depth model** | [🤗 Depth-Anything-3-CoreAI](https://huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI) | small + base, fp16/fp32, engine cos 1.000000 (cpu+gpu) / vs official mean r 0.98, **54 MB · 65.7 FPS M4 Max GPU (small fp16)** |
| [`qwen3-embedding.md`](qwen3-embedding.md) | Qwen3-Embedding (multilingual text embedder, last-token pooling + MRL, Alibaba) | [🤗 Qwen3-Embedding-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Embedding-0.6B-CoreAI) | 0.6B fp16, torch ladder exact + engine gate cos 0.999998, **25–45 ms/embedding M4 Max GPU** |
| [`qwen3-reranker.md`](qwen3-reranker.md) | Qwen3-Reranker (cross-encoder reranker, yes/no logit score, Alibaba) | [🤗 Qwen3-Reranker-0.6B-CoreAI](https://huggingface.co/mlboydaisuke/Qwen3-Reranker-0.6B-CoreAI) | 0.6B fp16, torch ladder exact (P(yes) Δ=0) + engine gate Δ<5e-4, **45.7 ms/score M4 Max GPU** |
| [`holo2.md`](holo2.md) | Holo2-4B (GUI-grounding / computer-use VLM, Qwen3-VL-4B backbone, H Company) | [🤗 Holo2-4B-CoreAI](https://huggingface.co/mlboydaisuke/Holo2-4B-CoreAI) | 4B int8lin + fp16 vision, parity **vision cos 0.9999 / decoder 16/16** vs fp32 HF; rides the Qwen3-VL pipeline; zoo's **first GUI-grounding / computer-use model** |
| [`colmodernvbert.md`](colmodernvbert.md) | ColModernVBERT (visual document retriever, late-interaction/MaxSim, ModernBERT+SigLIP2) — zoo's **first visual retriever + first late-interaction model** | [🤗 ColModernVBERT-CoreAI](https://huggingface.co/mlboydaisuke/ColModernVBERT-CoreAI) | 250M, query + doc encoders fp16/fp32, engine per-token cosine **1.000000** (fp32) / ≥0.99999 (fp16), MaxSim == `processor.score` exactly, single-tile retrieval 3/3 |
| [`yolox.md`](yolox.md) | YOLOX-S (single-stage anchor-free detector, YOLO-family, Megvii) — zoo's **first YOLO / single-stage detector** (CNN counterpart to RF-DETR; needs host NMS) | [🤗 YOLOX-CoreAI](https://huggingface.co/mlboydaisuke/YOLOX-CoreAI) | 8.97M fp32, gated cpu+gpu (head cos **1.000000**, detections IoU **1.000**), **4.80 ms · 208 FPS M4 Max GPU · ~22 ms iPhone 17 Pro GPU** (device-verified live in DetectCamera) |
| [`parakeet.md`](parakeet.md) | Parakeet-TDT-0.6B (FastConformer transducer / TDT, NVIDIA) — zoo's **first transducer / TDT (RNN-T family)** ASR (3 graphs + host greedy loop, not an LLM) | 🤗 Parakeet-TDT-0.6B-CoreAI *(upload pending)* | 600M, encoder fp16 + predict/joint fp32, **77/77 token-exact e2e** vs HF (GPU enc cos 0.999995); Swift `KitParakeetModel` Mac-validated token-exact; 25 EU langs |

## The matrix (every meaningful platform × compute-unit cell, greedy, top-1 vs HF)

<!-- Mac column RELEASE-VERIFIED 2026-06-10 (R2, ondevice/MACOS_RELEASE_README.md).
     qwen static iOS GPU 27.7 (ctx 2048, release config) = 2026-06-10 RELEASE-build device
     measurement (ctx-256 export measured 30.4).
     gemma4 iOS GPU 22 + ANE 6 = 2026-06-10 hands-on re-measure in the RELEASE chat app
     (int4km monolith; instrumented run 22.5, core 39ms / head 2ms — the earlier 17.7 was the
     AOT-harness number; the Release-confirm TODO is resolved). -->

| | macOS GPU (M4 Max) | iOS GPU | iOS ANE |
|---|---|---|---|
| **Gemma 4 E2B** | ✅ 8/8 · 56.6–59.0 tok/s (int8 kernels) | ✅ 8/8 · **22 tok/s** (int4-k-means kernels) | ✅ 8/8 · 6 tok/s (int8 chunks) |
| **Qwen3.5 0.8B** | ✅ 8/8 · 58.5 (int8 dynamic) | ✅ **27.7** (fp16 static, ctx 2048) / 12.5 (int8 dynamic) | ✅ 14.7 (int8 dynamic); static ✗ this beta (fp16 SSM recurrence) |

macOS ANE is intentionally out of scope (the runtime auto-prefers GPU on Mac for these
structures, and the Mac GPU dominates it anyway).

Parity is measured against the Hugging Face eager reference (cosine + top-1 argmax on a fixed
prompt): conversion on macOS, then re-verified end-to-end on-device (iPhone 17 Pro, iOS 27 beta).
Device numbers are int8, greedy, prompt "What is the capital of France?" / "The capital of France
is" → "Paris".
