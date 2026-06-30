# Diffusion LLMs (dLLM) on Core AI — port notes

A masked **diffusion** LLM is not autoregressive. Instead of writing one token left-to-right with a
KV cache, it runs a **bidirectional** forward over a fixed-length canvas of `[MASK]` tokens and
**unmasks the most-confident positions in parallel**, repeating until the canvas resolves. Ported
here as **LLaDA-8B** (base `GSAI-ML/LLaDA-8B-Instruct`, distilled `d3LLM/d3LLM_LLaDA`).

## Shape of the port

1. **Backbone overlay** (`conversion/dllm/llada.py`) — a fixed-shape, exportable LLaMA-dense 8B, but:
   - **bidirectional** SDPA (`is_causal=False`, no mask) — the whole point; the `LLaDALlamaBlock`
     name is misleading.
   - **no KV cache** — every denoising step is a full forward over the whole canvas `S`.
   - RoPE θ=500000 (baked), RMSNorm `weight*x` (not Gemma's `1+w`), MHA 32×128 (no GQA), no qk-norm,
     SwiGLU intermediate **12288** (config `mlp_hidden_size`; `activation_type="silu"` so `ff_proj`
     and `up_proj` are each full-width — NOT half), lm_head `ff_out` (weight_tying False).
   - Gated cos≈1.0 vs the official `LLaDAModelLM` (`gate_llada_torch.py`), per-layer + logits.
2. **Decode loop** (`generate_llada.py`) — the host side: semi-AR blocks; each step unmask the masked
   positions with **lowest entropy** (always ≥1, plus any below `threshold`). Token-exact vs the
   official `generate` at temperature 0 (`gate_llada_decode.py`).
3. **Export** (`export_llada.py`) — one static bundle `main(input_ids[1,S] int32) → logits[1,S,vocab]`,
   bidirectional, no KV. VoxCPM2 idiom (externalize-drop SDPA+RoPE, `export_to_coreai`,
   `quantize_pytorch_model`). int4 per-block-32 body + int8 head ≈ 4.9 GB.
4. **Host** — the same loop in Swift (`apps/CoreAIChatMac` `LLaDAEngine`): load the `.aimodel`,
   tokenize, run the denoising loop, stream the canvas. `metadata.json` carries the diffusion knobs
   (`seq`, `block_size`, `threshold`) so they retune without a recompile.

## Lessons (the non-obvious ones)

- **Exact-token-match is the WRONG metric for a diffusion LM.** Small logit noise reroutes the
  denoising path to a *different but equally valid paraphrase*, so token-match drops while the answer
  stays correct. We almost rejected int4 over this (PTQ int4 showed 20/64 token-match) — the decoded
  TEXT was correct the whole time. **Judge by output text, not token ids.**
- **int4 is viable** (not a cliff here). int8 is lossless; int4 per-block-32 (engine
  `symmetric_with_clipping`) keeps answers correct. Head fp16 vs int8 made no quality difference.
- **The speed ceiling is the no-KV full-canvas forward** (~185 ms/forward @ S=128, ~linear in S). The
  entropy `threshold` is a near-free knob — it only changes the step count (NFE), not ms/forward:
  0.5→NFE19, 1.0→NFE11/~38 tok/s, 1.5→NFE8/~53 tok/s, ~2.5 starts to degrade. Distillation already
  cuts NFE hard (~8 tokens/forward). The real lever is a **delayed-KV-cache decode** (d3LLM's
  `generate_multi_block_kv_cache`: cache committed blocks, forward only the active region) — needs a
  KV-state bundle re-export; not yet done.
- **Keep `block_size` small (32).** Larger blocks denoise more of the canvas in parallel (a more
  striking "whole-canvas fill" visual) but **garble coherence** (a count comes out `11,22,2,3…`).
  LLaDA's semi-AR blocks are what hold the output together — so the left-to-right "tail" fill is
  correct, and the parallelism lives *within* each block.
- **Canvas `S` bounds the answer** (no KV ⇒ prompt+answer must fit in `S`). S=128 ≈ 80-token answers
  (truncates long ones and breaks multi-turn — the whole history must fit); S=256 ≈ 210 tokens. The
  host must reserve gen room and drop old turns, or a long history collapses the gen budget to a
  sliver (1-token "answers").
- **Demo prompts:** the parallel fill is only visible when the model emits a **direct structured
  answer with no preamble** — `List the planets / months separated by commas`, `Count 1 to 20`, or a
  short arithmetic word problem (the equation digits fill out of order: `48 + ░4 =░7░ → 72`).
  `explain` / `show your work` / `write a function` trigger a chatty preamble that eats the canvas and
  buries the structured part.
