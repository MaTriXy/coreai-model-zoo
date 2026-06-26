# MiniCPM5-1B — Core AI conversion notes (reusable techniques)

MiniCPM5-1B (OpenBMB, Apache-2.0) is a plain `LlamaForCausalLM` (1.08B, GQA 16:2, RoPE θ=5e6,
RMSNorm, SiLU, explicit `head_dim` 128, untied head, vocab 130560, 128K, hybrid Think/No-Think).
The port produced no model-specific code — it's three reusable levers worth keeping.

## 1. Plain-Llama → the stock exporter via a `llama → mistral` remap

The stock `coreai.llm.export` graph registry has families for qwen2/qwen3/gemma/mistral/… but **no
`llama`**. A plain `LlamaForCausalLM` is architecturally identical to the **Mistral** builder minus
the sliding window: GQA + RoPE + RMSNorm + SiLU, **no qkv bias** (qwen2 has it), **no qk-norm**
(qwen3 has it), and the Mistral builder already honors an explicit `config.head_dim`. So:

```python
# coreai_models/models/registry.py — MODEL_TYPE_REMAPPING
"llama": "mistral",
```

is a one-line unlock for *any* plain-Llama checkpoint. Unregistered HF ids also need
`--experimental --compute-precision float16`. Validate with greedy parity vs HF (token-exact).

## 2. Clean weight-only INT8 without a custom decode-pipelined export

The macOS `--compression int8` preset is **iOS-palettization-only** (`AssertionError: palettization
is only supported for iOS variant`), and the zoo's int8 LLM bundles come from per-model
`export_*_decode_pipelined.py` scripts (none exists for plain Llama). But `coreai.llm.export
--compression-config <yaml>` accepts a **`quantization_config`** (macOS torch-pre-export via
coreai-opt `quantize_pytorch_model`) — write **symmetric per-channel int8, absmax (NO clipping;
clipping craters the big-vocab LM head — absmax keeps it lossless), SDPA/RoPE/RMSNorm excluded**
(see `conversion/minicpm5_int8sym.yaml`). coreai-opt has only PTQ/palettization/pruning — **no
GPTQ/AWQ** — so symmetric-per-channel int8 is the clean ceiling; int4 hits the non-QAT cliff.

## 3. Ship a DYNAMIC-shape bundle for the iPhone (pipelined engine), not a static iOS export

`EngineFactory.autoDetectVariant`: **dynamic structure → pipelined engine** / **chunkedStatic →
staticShape engine**. A `coreai.llm.export --platform iOS` static bundle is detected as
chunkedStatic and routed to the staticShape engine, which expects `extend_*` / `load_embeddings`
multi-graph functions an FM-format bundle doesn't provide → `NSPOSIXError 2` at engine-create on
device. The **dynamic** FM-format bundle (the macOS default export) routes to the **pipelined
engine** and runs unchanged on both macOS and iPhone. So for the CoreAIChat / pipelined path, ship
the dynamic bundle (sideload to `Documents/models/<name>`; `LanguageBundle` + `EngineFactory`
load it like any pipelined model). Cold first-load is one-time (~45 s JIT spec); the cache persists
→ warm loads ~2–5 s, so AOT (`.aimodelc`) is unnecessary.

Chat EOS: base `eos_token` is `</s>`, but the chat template ends turns with `<|im_end|>` (130073) —
set the bundle's tokenizer `eos_token` to `<|im_end|>` (as Qwen ships) or generation never halts.

## Result

iPhone 17 Pro (`PipelinedBench`): **int8 decode 66.8 / prefill 68.0 tok/s, 24/24 token-exact vs HF
fp32 (lossless), 1.0 GB** — ~2.2× fp16 (decode is bandwidth-bound → half the weight read ≈ double
throughput) at no quality cost. 🤗 `mlboydaisuke/MiniCPM5-1B-CoreAI`.
