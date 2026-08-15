---
license: apache-2.0
base_model: meta-models/Muse-Glimmer-30B
tags:
  - coreai
  - aimodel
  - apple-silicon
  - on-device
  - muse-glimmer
pipeline_tag: text-generation
---

# Muse-Glimmer-30B (text decoder) — Apple Core AI (`.aimodel`)

**Meta's Muse-Glimmer-30B converted to Apple's Core AI** (the Core ML successor announced at
WWDC26), ready to run on macOS 27. This is the **text decoder** of their 30B agentic VLM —
the perception encoder is not included.

**It decodes faster than Meta's own on-device Apple-GPU build of the same model.**

| | weights | decode tok/s | prompt tok/s |
| --- | ---: | ---: | ---: |
| ExecuTorch `k-quant-17G-128K-text-solo-metal` *(Meta's published figure)* | 17.9 GB | 23.7 | — |
| **this bundle — Core AI `int4hu`** | **16.35 GB** | **26.69** | **269.0** |

**+12.6% decode on 8.7% fewer bytes**, on Apple's stock `coreai-pipelined` GPU engine with no
custom Metal kernels.

*MacBook Pro M4 Max (40-core GPU, 128 GB, macOS 27.0 26A5406e), 512 prompt / 1024 generation /
3 trials, `llm-benchmark`. Meta's figure is theirs, not a re-measurement: batch 1, greedy,
averaged over a prompt set they do not publish. Their M4 Max is the same 546 GB/s bin —
23.7 × 17.9 GB is 424 GB/s of traffic, which the 410 GB/s bin cannot produce. Their `metal`
backend is MLX-native, per their own README.*

Decode barely moves with context — 27.46 tok/s at 128 prompt tokens, 26.73 at 2048 — which is
what 39 of 52 layers capped at a 2048-token window should look like.

**Both artifacts run on the same machine**, same prompts, greedy, batch 1, 192 new tokens,
interleaved A/B/A/B with a 45 s cooldown between every run (block-ordering is not safe here —
ExecuTorch decayed 23.5 → 17.4 tok/s inside a single block, so whoever runs second inherits a
hot GPU):

| prompt | ExecuTorch | Core AI |
| --- | ---: | ---: |
| 1 (code + explanation) | 19.3 / 19.7 | *16.2* (cold) / **27.5** |
| 2 (step-by-step reasoning) | 24.0 / 23.9 | **27.3 / 27.4** |
| 3 (long-context tradeoff) | *1 token, stopped* | **27.7 / 27.7** |

Core AI holds 27.3–27.7 across every prompt and round. ExecuTorch ranges 19.3–24.0; at its own
best prompt it reproduces Meta's published 23.7 almost exactly, and Core AI is +14% there.
Prompt 1 round 1 for Core AI is a cold-cache artifact (ExecuTorch had just filled the page
cache with 17.9 GB) and is discarded on the strength of round 2; on prompt 3 the ExecuTorch
runner emitted one token despite `--ignore_eos=true`, which is a runner behaviour and is not
counted as a win.

**Not claimed here:** Meta's **DFlash** speculative figure (37.8 tok/s) is a different weight
class; it runs a separate 5.1 GB block-diffusion drafter that predicts 16 tokens per pass, and
nothing in this bundle speculates.

## The architecture, and what the port had to do about it

52 layers, hidden 6656, GQA 32 q / 2 kv heads (head_dim 128), SwiGLU intermediate 19968, vocab
202 048 with an **untied** lm_head, 131 072 context — **27.855 B** parameters in the text tower
alone. Gemma-3 shaped, with four things that are not:

* `sliding(2048) × 3 + full × 1` layer pattern where the **full layers are NoPE** —
  `layer_rope_theta[i] == 0` marks them, and only the sliding layers carry rotary.
* A **sigmoid output gate** on every attention layer, reading the same pre-attention hidden
  states as Q/K/V — so the re-authored module folds all four into one `qkvg_proj` (4 GEMMs → 1).
* **Weight-less** RMSNorm on Q/K and on the embedding output.
* **Two epsilons** across the sandwich norms (1e-5 pre, 1e-8 post), and logits pre-scaled by
  `output_multiplier` before a tanh softcap at 20.

`qk_scale_factor` (3.87) is folded into the SDPA scale rather than applied to Q — attention is
`softmax((aQ)·Kᵀ/√d)`, `a` is a scalar and rotary is a rotation, so it is algebraically
identical and saves a full-width multiply per layer.

## Gates

| stage | result |
| --- | --- |
| Authoring vs HF `transformers` 5.15, fp32, real weights, 8 layers (2 NoPE) | cos **1.0000** at embedding, every layer, final norm and logits; top-1 match; **0** argmax flips |
| Full-size key binding (52 layers) | 471/471 parameters bound; 0 missing / 0 unexpected / 0 shape mismatch |
| This bundle vs the fp16 oracle, `coreai_gate.py` | **PASS — token-for-token** over 24 generated tokens |

## Run it

```bash
llm-benchmark --model gpu-pipelined/muse_glimmer_30b_decode_int4hu_block32_sym \
              -p 512 -g 1024 -n 3

llm-runner --model gpu-pipelined/muse_glimmer_30b_decode_int4hu_block32_sym \
           --prompt "..." --temperature 0.0 \
           --inference-engine-variant coreai-pipelined --warmup off
```

Needs ~17 GB of unified memory for the weights plus the KV cache (52 KB/token; 0.44 GB at the
8192 context this bundle is exported for). **Mac only** — no quantization brings 16.35 GB to an
iPhone.

## Layout

```
gpu-pipelined/muse_glimmer_30b_decode_int4hu_block32_sym/
    muse_glimmer_30b_decode_int4hu_block32_sym.aimodel/
    metadata.json
    tokenizer/          (incl. chat_template.jinja, copied verbatim)
config.json             the source config, so zoo_verify can compare against it
LICENSE                 carried from the source repo (Apache-2.0)
```

Recipe, export script and the port's findings:
[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo) —
`models/muse-glimmer-30b/`, `conversion/export_muse_glimmer_decode_pipelined.py`,
`knowledge/muse-glimmer-port.md`.
