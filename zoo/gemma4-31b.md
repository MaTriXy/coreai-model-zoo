# Gemma 4 31B (text decoder) — Core AI

Gemma 4 31B **dense** text decoder on the pipelined-engine fast path, ported
**directly from Google's QAT release**
[`google/gemma-4-31B-it-qat-q4_0-unquantized`](https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-unquantized)
(bf16 weights *trained* for q4_0 rounding — per-block-32 absmax symmetric int4, the
`int4lin --lin-sym` recipe). The **frontier dense** companion to the 12B: same clean
`gemma4` architecture scaled to 60 layers / 5376 hidden, unblocked on Core AI by the **same
custom flash-decode SDPA kernel** as the 12B.

**⬇️ Converted `.aimodel` bundle:
[mlboydaisuke/Gemma-4-31B-CoreAI](https://huggingface.co/mlboydaisuke/Gemma-4-31B-CoreAI)**
— `gpu-pipelined/` decode-only **int4linsym** (19 GB, Mac-only), with the higher-occupancy
flash-decode kernel (`_g8`).

> **Same engine blocker, same bypass as the 12B.** The full (global) attention layers have a
> 16-byte-KB-class Q tensor (here **32 heads × 512** = 32 KB fp16) that overflows MPSGraph's GPU
> decode scratch heap — the stock SDPA crashes at the first token
> ([apple/coreai-models#27](https://github.com/apple/coreai-models/issues/27)). `--metal-sdpa`
> swaps the full layers' SDPA for a custom Metal flash-decode kernel, so the 31B runs on the stock
> pipelined engine. **The 31B has 4 global KV heads** (vs the 12B's 1), so the kernel does block
> GQA over the unified cache (validated bit-exact vs ground-truth GQA).

<!-- gen-cards:use-it begin id=gemma-4-31b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
<!-- gen-cards:use-it end -->

## Architecture (config + checkpoint verified) — clean dense, no PLE

Top `model_type: gemma4`; text `gemma4_text`. Like the 12B and unlike the on-device E2B/E4B, it
is a **clean dense interleaved-attention Gemma decoder** — NO PLE / AltUp / Laurel / MoE /
KV-sharing / double-wide-MLP.

- **60 layers**, hidden 5376, intermediate 21504, **32 attention heads**, vocab 262144, softcap 30.0.
- **5:1 sliding:full** (`layer_types`; full attention every 6th layer).
- **Dual head_dim**: sliding 256 / full `global_head_dim` 512.
- **Dual KV-head count via `attention_k_eq_v: true`**: sliding layers = `num_key_value_heads` **16**
  with a real `v_proj`; full layers = `num_global_key_value_heads` **4**, no `v_proj`, value = the
  raw `k_proj` output followed by a scale-free V RMSNorm.
- Attention scale 1.0 (QK-norm), per-head Q/K RMSNorm + scale-free V RMSNorm, learned per-layer
  scalar, dual RoPE (sliding θ 1e4 / full θ 1e6 proportional), tied embeddings.

It reuses the **12B dense overlay verbatim** (`gemma4_dense_text.py` + `gemma4_dense_pipelined.py`
+ `gemma4_dense_metal_sdpa.py`) — the config drives the layer/head counts and the cache replication
is `repeat_interleave` so the block GQA is correct for the 31B's 4 global heads (a no-op for the
12B's 1). Both attention shapes ride ONE growing KV pair (`[60,1,16,S,512]`); the bundle loads on
the **stock pipelined engine — no engine patch** (2 states). In-graph embed + tied head + softcap.

## Verification

Engine token-match (greedy on the fp32-oracle prompt, `_smoke/engine_tokenmatch_gemma4_12b.py` —
the Gemma tokenizer + chat template are shared with the 12B): the int4 bundle answers the
capital-of-France prompt correctly (`Paris`). The kernel itself is numerically bit-exact vs the
composite SDPA / ground-truth GQA (eager parity, both the 12B's 1-global-head and the 31B's
4-global-head configs).

## Throughput (M4 Max, `--metal-sdpa --split-g 8`, `-p 128 -g 256 -n 3`)

| bundle | decode | prefill | size |
|---|---|---|---|
| `int4linsym_msdpa_g8` | **17.2 tok/s** | 22.1 tok/s | 19 GB |

A frontier 31B dense at int4 is bandwidth-bound, so decode (17.2 tok/s) is in the public Q4 Mac
range (Ollama ~18-24) — the win is "Core AI runs a frontier dense model that the stock engine
**cannot**." The custom flash-decode kernel (necessary to run at all) adds some overhead vs the raw
bandwidth ceiling. Public Q4 figures of 40-50 tok/s use Gemma 4's MTP speculative drafters, which
Core AI does not have — compare plain-decode to plain-decode.

## Reproduce

```
cd coreai-models
.venv/bin/python ../coreai-models-community/conversion/export_gemma4_12b_decode_pipelined.py \
    int4lin --lin-sym --metal-sdpa --hf-id google/gemma-4-31B-it-qat-q4_0-unquantized
# gate (run SOLO — parallel python-GPU = MTL4CommandQueueError):
.venv/bin/python ../_smoke/engine_tokenmatch_gemma4_12b.py \
    exports/gemma4_31b_qat_decode_int4linsym_msdpa_g8
# bench:
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/gemma4_31b_qat_decode_int4linsym_msdpa_g8 -p 128 -g 256 -n 3
```
