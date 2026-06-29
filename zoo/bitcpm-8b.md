# BitCPM-8B (1.58-bit ternary) — Core AI

[🤗 mlboydaisuke/BitCPM-8B-CoreAI](https://huggingface.co/mlboydaisuke/BitCPM-8B-CoreAI) · Apache-2.0 · base [openbmb/BitCPM-CANN-8B](https://huggingface.co/openbmb/BitCPM-CANN-8B)

The zoo's **first 1.58-bit ternary LLM** and **first sub-int8 packed-GEMM Metal kernel**, running
fully on-device on **iPhone** through Apple **Core AI**. `BitCPM-CANN-8B` is OpenBMB's MiniCPM4-8B
architecture **quantization-aware trained to ternary** — every transformer weight is just
**{-1, 0, +1}**. The result: an 8B model at a **3–4B-class footprint and decode speed**, with
8B-class quality (95.7–97.2% of full precision, OpenBMB).

## On-device (iPhone 17 Pro, A19 Pro — CoreAIChat pipelined GPU engine, greedy)

| bundle | decode | prefill | resident | load |
|---|---:|---:|---:|---:|
| **`gpu-pipelined/`** (ship, AOT h18p) | **17 tok/s** | **13 tok/s** | **~2.1 GB** | 9 s cold |

Headroom ~4.3 GB, no jetsam. An int4 8B needs ~5–6 GB resident — the 2-bit ternary weight stream is
the lever. 17 tok/s decode puts an 8B model in the zoo's **3–4B speed class** (Nanbeige-3B 15.9,
FastContext-4B 20.4, Gemma-E4B 15.1).

## Mac (M4 Max GPU, pipelined engine, greedy)

| | decode | numerics |
|---|---:|---|
| **fp16-activation ternary** | **62.7 tok/s** | engine **token-identical** to the torch ternary reference — 3/3 probe prompts, greedy |

## Conversion

- **Architecture is known.** `BitCPM-CANN-8B` is the MiniCPM4-8B decoder (32 layers, hidden 4096,
  GQA 32/2 head_dim 128, LongRoPE, SwiGLU, RMSNorm) with MiniCPM **mup** scalars (embed ×12, residual
  ×`scale_depth/√L`=0.247, logits ÷`hidden/dim_model_base`=16, untied head). All of that is the zoo's
  existing MiniCPM4 path — the **only new piece is the ternary GEMM kernel**.
- **Ternary weights = TQ2_0.** The published gguf stores 2 bits/weight: per 256-element block along
  the reduction axis, a code in {-1,0,+1} × one fp16 scale. The **224 transformer linears**
  (q/k/v/o + gate/up/down × 32) go ternary; the **embedding (Q4_K)** and **LM head (Q6_K)** stay
  higher-precision (BitNet practice — those layers are sensitive).
- **The kernel** (`bitcpm_ternary_metal.py`, zoo's first sub-int8 packed GEMM): a simpler sibling of
  the int4 k-means matvec — **16 ternary codes packed per uint32**, dequant is just `(code−1)` in
  {-1,0,+1}, sign-accumulated against the fp16 activation with the per-256-block scale applied
  per-lane, **no codebook**. Decode is `M=1` (one row); 32 lanes × 16 codes = a 512-K block.
- **The export gotcha** ⚙️: the `M=1` kernel can't take a multi-token prefill. A dynamic-`input_ids`
  export lets prefill run `S>1`, and MPSGraph then refuses to lower it
  (`mps_spi.copy_discarding_constraints … must have tensor constraints`). Pin `input_ids` to a static
  `[1,1]` **S=1 contract** (`--static-ids`, prefill via `COREAI_CHUNK_THRESHOLD=1`) and it compiles,
  AOT-survives, and runs on the iPhone GPU.
- **AOT, not a portable IR.** An 8B graph can't specialize on-device, so the shipped bundle is
  **AOT-compiled** for the h18p GPU (`xcrun coreai-build compile … --preferred-compute gpu
  --architecture h18p` → `.aimodelc`). A custom-Metal-kernel graph survives AOT (bit-identical
  outputs). See [`../knowledge/bitcpm-ternary-1.58bit.md`](../knowledge/bitcpm-ternary-1.58bit.md) and
  [`../knowledge/aot-and-specialization.md`](../knowledge/aot-and-specialization.md).
- **Chat:** ChatML template, eos `<|im_end|>` (73440).

Conversion scripts: [`../conversion/export_bitcpm8b_decode_pipelined.py`](../conversion/export_bitcpm8b_decode_pipelined.py)
(+ `conversion/bitcpm/` for the gguf reference, oracle gate, and engine gate).

## Run

In the zoo's **CoreAIChat** app (Model → "BitCPM-8B 1.58bit"), or via Foundation Models:

```swift
import FoundationModels
import CoreAILanguageModels
let model = try await CoreAILanguageModel(resourcesAt: bundleURL)
let session = LanguageModelSession(model: model)
print(try await session.respond(to: "The capital of France is"))   // -> "Paris."
```

## Why ternary, now

MLX got fast on Apple Silicon in 2026 (M5 neural accelerators; MLX-Swift overtaking llama.cpp on
decode). The durable edge for Core AI isn't matching MLX on a Mac — it's a kernel MLX **doesn't have**
(its quantization is 4/8-bit affine; there is no 2-bit ternary GEMM), on a device MLX **doesn't ship
to**. 1.58-bit on iPhone is both.
