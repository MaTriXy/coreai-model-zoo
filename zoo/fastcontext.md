# FastContext-1.0-4B-SFT — Core AI

> **Deprecated 2026-07-21** — Microsoft removed the upstream weights from Hugging Face
> (2026-06-30, no explanation); the shipped 0.4.0-era artifact cannot be rebuilt without them.
> The HF repo is kept for reference.

[🤗 mlboydaisuke/FastContext-1.0-4B-CoreAI](https://huggingface.co/mlboydaisuke/FastContext-1.0-4B-CoreAI) · MIT · base [microsoft/FastContext-1.0-4B-SFT](https://huggingface.co/microsoft/FastContext-1.0-4B-SFT)

Microsoft's long-context **repository-exploration agent** — a Qwen3-4B-Instruct backbone SFT'd on
exploration traces (broad first-turn search, multi-turn evidence gathering, precise file:line
citation), converted to Apple **Core AI** and running fully on-device on iPhone. The zoo's **first
stock-architecture LLM** (byte-identical Qwen3-4B → no re-authored decoder) and **first 4B-class
iPhone text model**.

## On-device (iPhone 17 Pro, A19 Pro — CoreAIChat pipelined GPU engine, greedy)

| bundle | decode | prefill | quality | size | load |
|---|---:|---:|---|---:|---:|
| **`gpu/`** (ship, AOT h18p) | **20.4 tok/s** | **22.1 tok/s** | parity **23/24 argmax**, ppl 1.41 vs HF fp16 | 2.1 GB | 8 s cold / 1 s warm |

Resident footprint ~0.3–0.6 GB (weights memory-mapped). The single parity flip is a near-tie
quant-noise position; greedy continuation is token-identical to HF on the probe prompt.

## Conversion

- **Stock — no re-authoring.** FastContext is a plain `Qwen3ForCausalLM` byte-identical to
  `Qwen/Qwen3-4B`, so it rides the stock `coreai_models` `qwen3` graph unchanged (GQA, q/k-norm,
  tied embeddings — all handled). The only additions are a `model_registry.py` short-name preset
  (`fastcontext-4b`, macOS 4bit + iOS palettized) and an `export/metadata.py` entry:
  `uv run coreai.llm.export fastcontext-4b --platform macOS`.
- **4-bit linear-INT4** (macOS dynamic export) — the proven on-device GPU compression.
- **AOT, not a portable IR.** A 4B graph can't specialize on-device (the GPU specializer exhausts
  device scratch disk; an iOS-tagged palettized IR's ANE path dies at inference). So the shipped
  `gpu/` bundle is **AOT-compiled** for the h18p GPU
  (`xcrun coreai-build compile … --preferred-compute gpu --architecture h18p` → `.aimodelc`),
  the same approach as the Gemma-4B zoo bundle. See
  [`../knowledge/aot-and-specialization.md`](../knowledge/aot-and-specialization.md).
- **Device class:** h18p = iPhone 17 / 18 class (a 4B needs the RAM anyway). ANE is unsupported —
  the ANE bundle static-loads (31 regions) but inference fails (`ANECompilerService` Code=4097).
- **Chat:** ChatML template, eos `<|im_end|>` (151645).

## Run

In the zoo's **CoreAIChat** app (Model → "FastContext 4B"), or via Foundation Models:

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: gpuBundleURL)
let session = LanguageModelSession(model: model)
print(try await session.respond(to:
    "Find where JWT token expiration is validated in an unfamiliar Python service — list the exact ripgrep queries to run first."))
```

It plays to its training: instead of a generic answer, it hands you a numbered set of `ripgrep`
queries + filename patterns to locate the code — the on-device repo-exploration agent.
