# BitVLA (1.58-bit Vision-Language-Action) — Core AI

[🤗 mlboydaisuke/BitVLA-CoreAI](https://huggingface.co/mlboydaisuke/BitVLA-CoreAI) · MIT · base [lxsy/bitvla-bf16](https://huggingface.co/lxsy/bitvla-bf16) · paper [arXiv:2506.07530](https://arxiv.org/abs/2506.07530)

The zoo's **first Vision-Language-Action model** (its first robotics model) and **first ternary
multimodal** — running fully on-device on **iPhone** through Apple **Core AI**. BitVLA takes an
**image + a natural-language instruction** and predicts a **7-DoF robot end-effector action**
(Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper) — OpenVLA-style discrete action tokens. Every transformer
weight, in **both** the language model **and** the vision tower, is **1.58-bit ternary** ({-1, 0, +1}),
so the whole policy is ~32× smaller than a full-precision VLA (OpenVLA-7.5B ≈ 15 GB) and fits on a
phone. The LLM's per-layer linears run the **custom 2-bit packed-ternary Metal kernel** (shared with
[BitCPM-8B](bitcpm-8b.md)) on the iPhone GPU.

## On-device (iPhone 17 Pro, A19 Pro — Core AI GPU, greedy)

One image + instruction → 7-DoF action:

| stage | cold (first run) | warm |
|---|---:|---:|
| vision encode (BitSigLIP-SO400M, 256 tokens) | 2.7 s | **0.13 s** |
| LLM prefill (≈308 positions = 256 image + text, M=1 loop) | 11.9 s | **8.8 s** |
| action decode (7 tokens, ternary kernel) | 0.30 s | **0.26 s** |

Resident ≈ 2 GB, headroom ~6.4 GB, no jetsam. The first run pays the GPU specialization; it caches.

## Parity (on-device vs the official model)

Same image + `"pick up the remote"`, action-token argmax over the 256-bin head, then BOUNDS-Q99
un-normalization (`bridge_orig`):

| | Δx | Δy | Δz | Δroll | Δpitch | Δyaw | gripper |
|---|---:|---:|---:|---:|---:|---:|---:|
| **BitVLA on iPhone** | 0.028 | -0.000 | 0.040 | 0.081 | -0.092 | -0.207 | 0.996 |
| official (fork transformers, Mac) | 0.028 | 0.003 | 0.040 | 0.081 | -0.092 | -0.207 | 0.996 |

**6/7 action tokens identical**; the 7-DoF action is effectively the official model's (the one
differing dim is ~0 either way — a near-boundary bin flip). The vision tower + projector match the
official image embeddings at **per-token cosine 0.999**.

## Architecture

- **LLM = BitNet b1.58 2B4T** (microsoft/bitnet-b1.58-2B-4T, MIT): 30 layers, hidden 2560, FFN 6912,
  GQA 20/5 head_dim 128, **ReLU² FFN**, **SubLN** (extra attn/ffn sub-norms), tied-free LM head,
  RoPE θ500000, LLaMA-3 tokenizer. W1.58-A8: per-**tensor** absmean ternary weight + per-token int8
  activation.
- **Vision = BitSigLIP-SO400M** (siglip_vision_model, ~400M): 26 layers, hidden 1152, FFN 4304,
  16 heads, patch 14 / 224 px → **256 patch tokens**, gelu-tanh; all attn/MLP linears ternary
  (vit_weight_bits 1), conv patch-embed + position-embed + LayerNorm stay fp16. Feature = last
  encoder layer (no post-LN).
- **Connector** = 2-layer MLP (1152→2560→2560, gelu), fp16.
- **Action** = OpenVLA discrete: the 256 vision embeds are spliced into the LLM sequence; it
  autoregressively generates 7 action tokens from the **256-token tail of the vocab**; each maps to a
  uniform bin in [-1,1] then un-normalizes via the OXE `norm_stats` (27-dataset mix) to continuous
  7-DoF. Base `bitvla-bf16` is the **autoregressive** OXE policy (the LIBERO fine-tunes use OFT
  bi-attention instead).

## Conversion

- **Ternary kernel reuse, generalized.** The BitCPM 2-bit packed matvec needed `K % 512 == 0` /
  `N % 32 == 0`, which BitVLA's dims break (down_proj K=6912; every SigLIP linear K∈{1152,4304}; fc1
  N=4304). `bitnet_ternary_metal.py` generalizes it: **arbitrary K** (K%16, per-lane tail guard),
  **N padded** to 32, and a **per-tensor (per-row) scale** for BitNet's absmean (vs BitCPM's
  per-256-block). The `BitLinearMetal` wrapper applies the A8 activation quant before the kernel, so
  it equals `F.linear(ActQuant(x), WeightQuant(W))` by construction.
- **inputs_embeds, not input_ids.** The LLM graph takes `inputs_embeds[1,1,2560]` so the host can
  splice the 256 projected vision embeds; the embedding table stays host-side. Static-ids S=1
  contract (M=1 kernel is decode-only); prefill is a position-by-position loop.
- **Action-head slice.** The model only ever emits the 256 tail tokens, so the LM head is sliced to
  those 256 rows — 656 MB → 1.3 MB, and decode argmax is constrained to valid action bins.
- **Vision compute = fp16 activations.** The in-graph A8 `act_quant` (per-token round/amax) stalls
  the iPhone GPU; dropping it to fp16 activations (ternary weights still baked) keeps parity (cos
  0.997) and runs fast. Ternary ⊂ int8, so the weights are carried losslessly.
- **AOT + on-device load.** The custom Metal kernel **cannot JIT on device** (the on-device compiler
  errors); it must be **AOT-compiled** (`xcrun coreai-build compile … --architecture h18p` →
  `.aimodelc`), and the dynamic-shape LLM `.aimodelc` is loaded low-level with
  `expectFrequentReshapes = false`. See [`../knowledge/bitvla-1.58bit-vla.md`](../knowledge/bitvla-1.58bit-vla.md).

Conversion scripts: [`../conversion/export_bitvla_llm_decode_pipelined.py`](../conversion/export_bitvla_llm_decode_pipelined.py),
[`../conversion/export_bitvla_vision.py`](../conversion/export_bitvla_vision.py) (+ `conversion/bitvla/`
for the torch reference, the official-model oracle, and the CPU/engine gates).

## Run

On-device in the zoo's **CoreAIChat** app (the BitVLA sheet): pick an image + a robot instruction
→ predict → 7-DoF action. Or drive the two `.aimodel`s directly: `pixel_values → vision → 256
embeds → [host splice] → LLM S=1 loop → 7 action tokens → bins → un-normalize`.

## Why a ternary VLA on iPhone

VLA / robotics policies don't exist in Apple's stock models or in MLX, and ternary VLA otherwise runs
only on bitnet.cpp (CPU). 1.58-bit makes a 7-DoF manipulation policy small enough to run on the phone
GPU — the durable Core AI edge (a kernel MLX lacks, on a device MLX doesn't ship to), now for robotics.
