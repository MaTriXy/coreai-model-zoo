# MiniCPM5-1B — Core AI

[🤗 mlboydaisuke/MiniCPM5-1B-CoreAI](https://huggingface.co/mlboydaisuke/MiniCPM5-1B-CoreAI) · Apache-2.0 · base [openbmb/MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)

OpenBMB's 1.08B on-device LLM (hybrid Think / No-Think reasoning, 128K context, 1B-class open-source SOTA), converted to Apple **Core AI** and running fully on-device on iPhone via the pipelined engine.

<!-- gen-cards:use-it begin id=minicpm5-1b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
<!-- gen-cards:use-it end -->

## On-device (iPhone 17 Pro, A19 Pro — `PipelinedBench`, random 128-tok prompt, greedy)

| bundle | decode | prefill | quality | size |
|---|---:|---:|---|---:|
| **`int8/`** (ship) | **66.8 tok/s** | 68.0 tok/s | **lossless** (24/24 token-exact vs HF fp32) | **1.0 GB** |

int8 is **~2.2× faster than fp16** on iPhone (decode is memory-bandwidth-bound → half the weight read ≈ double throughput) at **no quality cost**. So int8 strictly dominates fp16 here.

## Conversion

- **`llama → mistral` remap** — MiniCPM5-1B is a plain `LlamaForCausalLM`; the stock exporter has no `llama` graph family, but the Mistral builder is architecturally identical (GQA, no qkv bias, no qk-norm, explicit `head_dim`). One-line remap in the model registry.
- **int8** — weight-only symmetric per-channel (absmax, no clipping; SDPA/RoPE/RMSNorm full precision) via `coreai.llm.export … --compression-config` with a `quantization_config` (coreai-opt torch pre-export). Same recipe family as the zoo's `sym8`.
- **Chat EOS** — base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073); the bundle's tokenizer `eos_token` is set to `<|im_end|>` (as Qwen ships) so generation halts cleanly.
- **Dynamic-shape bundle** → the pipelined engine (the iPhone path). Runs unchanged on macOS and iOS.

## Run

In the zoo's **CoreAIChat** app (Model → "MiniCPM5 1B"), or via Foundation Models:

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: int8BundleURL)
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "Explain on-device AI in one sentence."))
```
