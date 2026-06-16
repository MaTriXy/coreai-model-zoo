# Qwen3-Coder-Next 80B-A3B (text decoder) — Core AI

**The local coder.** Source: `Qwen/Qwen3-Coder-Next` — an 80B-parameter sparse MoE that
activates only **~3B parameters per token**, the agentic-coding sibling of
[Qwen3.6-35B-A3B](qwen3.6.md). Same `qwen3_5_moe`-family hybrid decoder (3:1
**GatedDeltaNet** linear-attention / **gated full attention**, GVA 32 value over 16 key heads,
partial mRoPE, a sigmoid-gated shared expert), scaled to the zoo's largest expert grid:

- **48 layers, all MoE** (`decoder_sparse_step=1`); full attention every 4th layer
  (`[3,7,…,47]`), GatedDeltaNet on the other 36;
- **512 routed experts, top-10** per token (`moe_intermediate_size` 512) + one shared expert;
- hidden 2048, head_dim 256, 16 query / 2 KV heads, untied 151936-vocab `lm_head`.

## The `gather_qmm` win — 512 experts, top-10 → a 51× over-read removed

Stock MoE decode dequantizes **all 512 experts** every layer even though routing reads only 10
— a **51× over-read**, the highest of any MoE that fits a Mac. The custom
[`gather_qmm`](../knowledge/compute-units-and-authoring.md) Metal kernel reads **only the 10
routed experts** (`QP[w,n,e]`, `e=IDX[slot]`), the same kernel shipped for
[LFM2.5-8B-A1B](lfm2.5-8b-a1b-moe.md) / [Qwen3.6-35B-A3B](qwen3.6.md) /
[GLM-4.7-Flash](glm-4.7-flash.md). `sym8` (symmetric-linear per-block-32 int8) is the clean
floor for a bf16-release model (no official QAT-int4).

**Streaming export** solves the RAM blocker: the model is **~159 GB in fp16 > 128 GB host**, so
it never fully materializes — the routed experts are read one MoE layer at a time, immediately
sym8-quantized into the kernel buffers, and the fp16 freed before the next layer (peak ~99 GB).

**⬇️ Converted `.aimodel` bundle:** `qwen3_coder_next_decode_sym8_gather/` (**79 GB**, full
LanguageBundle incl. tokenizer; decode-only loop-free for the
[pipelined engine](../knowledge/pipelined-engine.md)).

## Measured (macOS 27 beta, M4 Max 128 GB, release `llm-benchmark`, `COREAI_CHUNK_THRESHOLD=1`)

| config | bundle | prefill tok/s | decode tok/s | quality |
|---|---:|---:|---:|---|
| **sym8 gather + untied absmax int8 head (`sym8 --head-sym`) = SHIP** | **79 GB** | ~24 | **~24** (warm 27) | end-to-end correct; sym8 expert fidelity cos 0.99996 |

**Quality** — an 80B fp16 token-oracle is RAM-infeasible (159 GB), so quality is gated
end-to-end instead:

- **Correct generation (engine greedy).** *"The capital of France is"* → *"Paris. The capital of
  Germany is Berlin. The capital of Italy is Rome. The capital of Spain is Madrid."* The coder
  prompt `def first_primes(n):` greedy-completes to a correct docstring, a correct
  `>>> first_primes(5) → [2, 3, 5, 7, 11]` doctest, and a correct algorithm. A 48-layer wiring
  error would garble this — the full forward (RoPE, GatedDeltaNet, MoE routing, attention, head)
  is sound.
- **sym8 is essentially lossless on these experts.** Over 432 sampled routed experts (9 layers ×
  48), the fp16-vs-sym8 SwiGLU output cosine is **mean 0.99996 / min 0.99995** (0 of 432 below
  0.999), rel-err 0.89 %. The kernel is bit-exact to the sym8 dequant; the quantizer is the
  shipped int8-linear recipe proven clean on the three smaller gather ports.

## Speed — a memory-bandwidth wall, not the model

~24 tok/s warm is set by reading **~3.46 GB per token** (int8) out of a **79 GB working set that
does not stay cache-resident** on a 128 GB Mac. It is *not* GatedDeltaNet-bound: a 4-layer
truncation of this exact bundle decodes at **529 tok/s** (GDN compute and dispatch are fast), and
the 48-layer model is ~1.8× slower **per layer** — the classic single-stream cold-weight wall
(4-layer hot ≈ 77 % of peak BW, 48-layer cold ≈ 20 %). The engine itself is competitive — the
zoo's [Qwen3.6-35B-A3B](qwen3.6.md) on the same kernel does **64.9 tok/s**, matching MLX 8-bit on
M4 Max; this model just carries 2.3× more cold weight (35→79 GB). It lands above the dense
[Qwen3.6-27B](qwen3.6-27b.md) (15.9) and [Gemma 4 31B](gemma4-31b.md) (17.2), at the 80B-on-Mac
ceiling. **Mac-only:** 79 GB is far past the iPhone jetsam limit.

## How to reproduce

```bash
cd coreai-models   # with the qwen3_next overlay (models/macos/qwen3_next.py, see ../conversion)
# stream-convert (routed experts sym8-quantized one layer at a time; fp16 never fully materialized)
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_coder_next_metal_decode_pipelined.py \
    sym8 --head-sym --hf-id Qwen/Qwen3-Coder-Next
# bench
COREAI_CHUNK_THRESHOLD=1 .build/release/llm-benchmark \
    --model exports/qwen3_coder_next_decode_sym8_gather -p 128 -g 512 -n 4
```

Model overlay: `models/macos/qwen3_next.py` reuses the `qwen3_5_moe` model classes verbatim — it
is a **loader/config port, not a new architecture**. The only Coder-Next-specific code is the flat
config reader (layer types derived from `full_attention_interval`), the per-expert streaming loader
(`moe_metal_streaming.py`), and the **GatedDeltaNet `in_proj_qkvz`/`in_proj_ba` de-interleave**:
the upstream checkpoint packs those projections HF-canonically (grouped by key head), while the
reused `qwen3_5` body has separate contiguous `in_proj_{qkv,z,b,a}` — a pure row reorganization,
verified bit-exact against `fix_query_key_value_ordering`. **Decode-only loop-free** because the
GatedDeltaNet `while_loop` doesn't lower on the GPU delegate.
