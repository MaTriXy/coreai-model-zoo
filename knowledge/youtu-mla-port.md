# Youtu-LLM-2B — dense MLA on iPhone (Core AI port notes)

Verified engineering notes from porting [`tencent/Youtu-LLM-2B`](https://huggingface.co/tencent/Youtu-LLM-2B)
(dense DeepSeek-V2/V3 Multi-head Latent Attention, 1.96B) to Core AI. This is the zoo's
**first MLA model that runs on iPhone** and its **first dense MLA** — [GLM-4.7-Flash](../zoo/glm-4.7-flash.md)
brought MLA but as a 30B Mac-only MoE. The whole port is mostly reuse of the GLM MLA path.

## The reuse: dense MLA == GLM's MLA minus the MoE

Youtu's attention (`YoutuAttention`) is **bit-identical to `glm4_moe_lite`'s MLA**. The only
structural differences from GLM-4.7-Flash are:

- **MoE FFN → dense `MLP`** (SwiGLU `intermediate 6144`) on *every* layer (GLM: 64-expert MoE
  on layers 1–46). This deletes all the router / `SwitchGLU` / `gather_qmm` machinery.
- **weight-tied lm_head** (GLM's is untied at the checkpoint root).
- dims: heads 16, `q_lora_rank` 1536, `qk_nope` 128, `v_head_dim` 128, `qk_head_dim` 192, 32
  layers, `rms_norm_eps` 1e-6, RoPE θ=1.6e6.

So `models/macos/youtu.py` is `glm4_moe_lite.py` with the MoE decoder swapped for a dense
`MLP` and a simpler weight loader (HF names == authored names, no expert stacking, no MTP skip).
The absorbed-MLA decode (`youtu_absorbed.py`) *imports* GLM's `glm4_moe_lite_absorbed`
(`Glm4MoeLiteMLAAbsorbed`, `AbsorbedKVCache`, `build_absorbed_decode_state`, the stateful
wrapper) and just re-points it at the Youtu model — those classes are fully config-driven and
duck-type on `.model.layers[i].self_attn` + the config attrs.

### Two things that made the reuse exact

1. **Interleaved decoupled RoPE is the same function.** HF's `apply_rotary_pos_emb_interleave`
   (even/odd pair form) and `glm4_moe_lite`'s `apply_rope_interleave` (de-interleave →
   rotate-half) are algebraically identical; the cos/sin are built over `rd = qk_rope_head_dim`
   (64) with `config.head_dim = qk_rope_head_dim`. Verified by the fp32 gate (0 flips) — the
   convention is load-bearing (the non-interleaved path shifts attention scores by ~25).
2. **The absorbed-MLA flash-decode Metal kernel already bakes Youtu's config.** `mla_metal_sdpa.py`
   was written for `kv_lora` 512 / `qk_rope` 64 with the scale hard-coded per model — and it
   explicitly lists `192**-0.5` (**DeepSeek-V2-Lite**) as a supported scale. Youtu's
   `qk_head_dim` = 128+64 = **192** and its **asymmetric K(576)≠V(512)** shape (v_head_dim 128 ≠
   qk 192) *is* the DeepSeek-V2-Lite shape the kernel was designed for. Register bounds
   (`_MAX_EPTL` = 512/32 = 16, `_MAX_EPTR` = 64/32 = 2) fit Youtu exactly. Zero kernel changes.

The absorbed form is also why a *dense 2B MLA fits iPhone*: it caches only the compressed latent
`[512]` + shared rope key `[64]` (2×`[288]` halves) per token, not a full per-head K/V — a tiny
KV even at 128K context.

## Gates (all PASS)

- **fp32 authored == HF fp32 oracle, token-exact.** Naive/materialized: prefill per-position
  cosine **1.000002**, greedy 16-tok **0 flips**. Absorbed (latent-cache stateful decode): same
  cosine, **0 flips** — the `W_UK`/`W_UV` factorization (sliced from `kv_b_proj`) + `AbsorbedKVCache`
  are correct.
- **int8 on the real GPU engine == oracle, byte-for-byte, on BOTH platforms.** `llm-runner` on
  M4 Max and PipelinedBench on iPhone 17 Pro both reproduce the oracle greedy 16/16:
  `" Paris.Okay, so the user is talking about the capital of France. They"`.

## Findings worth keeping

- **int4-linear (blk32) hits the int4 cliff.** 1.5 GB / 123 tok/s on Mac, but greedy **forks at
  token 2** vs the oracle (`" Paris, truer than ever"` vs `" Paris.Okay…"`) — well before the
  ~17-token GPU/CPU fp16 fork, so a real quality loss, not noise. **int8 (body + head) is the
  ship floor.** A smaller bundle needs int4-**k-means** (outlier-robust) or QAT, not linear RTN.
- **`split_g` is a non-lever for this model.** Mac decode g8 106.8 / g16 107.0 / g32 107.8 tok/s
  (<1% spread). A 2B int8 decode is bandwidth-bound on the per-token weight read; the absorbed-MLA
  kernel dispatch is a tiny fraction of the step, so its occupancy knob barely moves the total.
- **The custom MLA kernel JIT-cold-specializes fine on the A19 GPU** — no AOT needed on the
  PipelinedBench/CoreAIChat `.aimodel` path (engine ready 38.5 s cold for the 2.2 GB bundle,
  0 ANE_region = GPU-placed). An AOT `.aimodelc` (h18p) also compiles cleanly if wanted.
- **Chat template**: DeepSeek-style `<|begin_of_text|><|User|>…<|Assistant|>` (thinking mode on
  → `<think>…</think>` then the answer). `<|User|>`=128236, `<|Assistant|>`=128237 are atomic
  special tokens; `add_bos_token` is false, so the BOS is explicit. swift-transformers'
  `applyChatTemplate` renders it correctly (verified: kit `ChatSession(catalog:"youtu-llm-2b")`
  → "The capital of France is Paris."), and the kit surfaces the `<think>` block as `.thinking`
  events so the returned content is the clean answer. Turn-end stop: eos `<|end_of_text|>` 128001
  (+ `<|eot_id|>` 128009).

## Reproduce

```bash
cd coreai-models   # with the youtu overlay (../conversion)
.venv/bin/python ../coreai-models-community/conversion/export_youtu_decode_pipelined.py \
    --hf-id tencent/Youtu-LLM-2B --split-g 8         # int8 body+head, absorbed+msdpa; ~2.2 GB
```

Overlay: `models/macos/youtu.py` (+ `youtu_absorbed.py`). See the [pipelined engine notes](pipelined-engine.md)
for the decode-only static-`[1,1]` run contract, and the [GLM-4.7-Flash card](../zoo/glm-4.7-flash.md)
for the shared MLA / absorbed-cache machinery.
