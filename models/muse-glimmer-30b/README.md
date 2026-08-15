# Muse-Glimmer-30B (text decoder) — Core AI

Meta's 30B agentic VLM (`meta-models/Muse-Glimmer-30B`, Apache-2.0), text tower ported to
Core AI. **52 layers**, hidden 6656, GQA **32 q / 2 kv heads, head_dim 128**, SwiGLU
intermediate 19968, **vocab 202048 (untied lm_head)**, 131072 context — with four things that
are not the usual dense transformer: a `sliding(2048) × 3 + full × 1` layer pattern where the
**full layers are NoPE**, a **sigmoid output gate** on every attention layer, **weight-less**
RMSNorm on Q/K and on the embedding output, and **two epsilons** across the sandwich norms
(1e-5 pre, 1e-8 post). Logits are pre-scaled by `output_multiplier` then tanh-softcapped at 20.

**27.855 B** parameters in the text tower alone. Mac-only: int4 lands at 16.35 GB, which no
quantization brings to an iPhone.

## It is faster than Meta's own on-device build

Meta ships official **ExecuTorch Metal `.pte`** artifacts for this model and publishes their
speed. Their `metal` backend is MLX-native ("`metal` = Apple Silicon (MLX)" — their README),
so this is Core AI against MLX-inside-ExecuTorch on hardware they name.

| | weights | decode tok/s | prompt tok/s |
| --- | ---: | ---: | ---: |
| ExecuTorch `k-quant-17G-128K-text-solo-metal` *(Meta's published figure)* | 17.9 GB | 23.7 | — |
| **Core AI `int4hu` decode-pipelined (ship)** | **16.35 GB** | **26.69** | **269.0** |
| Core AI `int4hu` `--static-ids` variant | 16.35 GB | 27.09 | 28.9 |

**+12.6% decode on 8.7% fewer bytes**, on the stock pipelined engine with no custom kernels.

*Measured on MacBook Pro M4 Max (40-core GPU, 128 GB, macOS 27.0 26A5406e), 512 prompt / 1024
generation / 3 trials, `llm-benchmark`. Trial spread is negligible (the static-ids variant
gives 27.436 / 27.469 / 27.468 at 128p/256g). Meta's number is theirs, not a re-measurement:
batch 1, greedy, averaged over a prompt set they do not publish. Their M4 Max is the same
546 GB/s bin — 23.7 × 17.9 GB is 424 GB/s of traffic, which the 410 GB/s bin cannot produce.*

The two variants are the same weights and the same decode graph; only the `input_ids`
dimension differs. Fixing it at [1, 1] (`--static-ids`) buys **1.5% decode** and costs **9.3×
prefill**, because every step — prompt included — then runs at S=1. On a Mac that is the wrong
trade, so the dynamic-ids bundle is the ship artifact; the static one is kept because it is
the shape an iPhone-class deployment would need, and it is what the decode figure is cleanest
on. At the short protocol (128p/256g) the ship bundle gives 27.39 decode / 233.4 prompt.

Decode barely moves with context: 27.46 at 128 prompt tokens, 26.73 at 2048 (−2.7%), which is
what 39 of 52 layers capped at a 2048-token window should look like.

### Same machine, head to head

Meta's figure is theirs; this is both artifacts run here, same prompts, greedy, batch 1, 192
new tokens, **interleaved A/B/A/B with a 45 s cooldown between every run**. Interleaving is not
optional — a first attempt that ran one block per side had ExecuTorch decaying 23.5 → 17.4
tok/s inside its own block, and the side that went second inherited a hot GPU.

| prompt | ExecuTorch | Core AI |
| --- | ---: | ---: |
| 1 (code + explanation) | 19.3 / 19.7 | *16.2* (cold) / **27.5** |
| 2 (step-by-step reasoning) | 24.0 / 23.9 | **27.3 / 27.4** |
| 3 (long-context tradeoff) | *1 token, stopped* | **27.7 / 27.7** |

Core AI lands on **27.3–27.7 across every prompt and round**, matching its isolated
`llm-benchmark` figure. ExecuTorch ranges 19.3–24.0, prompt-dependent; at its own best prompt
it reproduces Meta's published 23.7 almost exactly, and Core AI is **+14%** there and **+40%**
on prompt 1.

Three things that do not favour this port and are stated anyway: prompt 1 round 1 for Core AI
(16.2) is a cold-cache artifact — ExecuTorch had just filled the page cache with 17.9 GB — and
is discarded on the strength of round 2, not hidden. On prompt 3 the ExecuTorch runner emitted
a single token despite `--ignore_eos=true`; that is a runner behaviour, not a speed loss, and
is not counted as a win. And ExecuTorch's ~20% prompt-to-prompt variance is unexplained here —
decode on a dense model should not depend on generated content, and Core AI's does not.

### Raw MLX is the third arm, and it changes what the ExecuTorch result means

ExecuTorch's `metal` backend is MLX-native, so beating it could mean beating MLX or beating
the wrapper around MLX. Only raw MLX separates the two. Same machine, same prompts, greedy,
192 tokens, interleaved CA/ET/MLX with a 45 s cooldown between every run:

| | p1 r1 | p1 r2 | p2 r1 | p2 r2 | mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Core AI** `int4hu`, 16.35 GB | 27.5 | 27.1 | 27.5 | 27.6 | **27.43** |
| **MLX** `mlx-community/…-4bit`, 18 GB | 27.18 | 27.38 | 27.50 | 27.37 | **27.36** |
| **ExecuTorch** `k-quant-17G…metal`, 17.9 GB | 24.0 | 23.8 | 24.1 | 24.1 | **24.00** |

**Core AI and raw MLX are indistinguishable (+0.3%). Both beat Meta's own build by ~14%.**

So the honest reading is not "Core AI is fast here" — it is that **Meta's shipped on-device
artifact leaves ~14% on the table against the runtime it is built on**. Core AI matching MLX
at 27.9 B dense is what [`coreai-vs-mlx-speed.md`](../../knowledge/coreai-vs-mlx-speed.md)
already predicts: Core AI ≥ MLX on small dense, converging to a tie as the model grows and
MLX's 4-bit byte advantage cashes in. This is the largest dense point on that curve so far,
and it lands on the tie.

Prompt processing is not matched here and no claim is made from it (Core AI 269 tok/s at 512
prompt tokens, MLX 128–136 at 77–81 — different lengths, different batching).

**Not claimed.** Their published quality figure (1.0% degradation across 15 benchmarks for the
17G quant) has no matched counterpart here; this port has a token-exact gate and read
generations, not a benchmark suite.

## Speculative decoding: 1.3–2.0×, lossless, and it needs no drafter

Meta's **DFlash** figure (37.8 tok/s) buys speculation with a separate **5.1 GB** block-diffusion
drafter — a 31% surcharge on a 16.35 GB artifact. The same lever is available here for **zero
extra bytes**: this decode graph already runs its head on every position and takes a dynamic
`input_ids`, so K drafted tokens can be verified in one forward, and an **n-gram (prompt-lookup)
drafter needs no weights at all**. Same bundle, no re-export — the only new thing is a host loop.

256 generated tokens, greedy, batch 1, best draft length per workload:

| workload | spec off | spec on | | vs DFlash 37.8 |
| --- | ---: | ---: | ---: | ---: |
| free chat | 27.31 | **36.65** | 1.34× | 0.97× |
| code rewrite | 27.37 | **53.71** | 1.96× | **1.42×** |
| tool calling (ATEM, 3 tools) | 27.25 | **50.19** | 1.84× | **1.33×** |

Every committed token is the target model's own greedy argmax, so this changes speed and
nothing else — **46 A/B runs, 46/46 byte-identical** to the same loop with drafting switched
off. `spec off` is that loop, run seconds before each `on` run so a warming GPU can only work
against the ratio, and it reproduces the shipped engine's output character-for-character at
27.2 vs 27.6 tok/s.

Three things that do not favour this and are stated anyway. **It is workload-bound**: n-gram
pays where the continuation is already in the context, and these three prompts are not a
distribution — Meta's 37.8 is an average over a prompt set they do not publish, so the
comparison is directional, not matched. **It decays with generation length**: at 512 tokens
instead of 256, code falls to 1.40× and free chat to 1.10× (**30.10 tok/s — below their
number**) as the model leaves the phase where it is restating its input. Tool calling is the
exception and holds at 1.85×, because the ATEM protocol keeps echoing prompt values for the
whole turn. **And the draft length is not a free parameter**: verify cost on this bundle is a
staircase, not a slope — S ≤ 3 is free (the forward is bandwidth-bound on 16.35 GB), S = 4…8
costs ~1.47×, S ≥ 9 costs ~2.3× — so K=8 turns free chat into a 5% *loss* while K=2 makes it a
1.34× win with the same drafter. Method, the full sweep, and the tuning rule that falls out of
it are in [`knowledge/spec-decode-ngram-dense.md`](../../knowledge/spec-decode-ngram-dense.md).

## Gates

| stage | result |
| --- | --- |
| Authoring vs HF `transformers` 5.15, fp32, real weights, 8 layers (2 NoPE) | cos **1.0000** at embedding, every layer, final norm and logits; top-1 match; **0** argmax flips |
| Full-size key binding (52 layers) | 471/471 parameters bound, 0 missing / 0 unexpected / 0 shape mismatches |
| Exported int4hu bundle vs fp16 authored oracle, `coreai_gate.py` | **PASS — token-for-token** over 24 generated tokens |
| Generation read (greedy, int4hu) | coherent; derives O(n²) vs O(n·w) correctly in the ATEM reasoning channel |

## Reproduce

```bash
cd coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_muse_glimmer_decode_pipelined.py \
    int4hu --head-sym

llm-benchmark --model exports/muse_glimmer_30b_decode_int4hu_block32_sym -p 512 -g 1024 -n 3
```

Add `--static-ids` for the S=1 variant; it then wants `COREAI_CHUNK_THRESHOLD=1` at runtime.

Two traps in this port generalize beyond it, and are written up in
[`knowledge/muse-glimmer-port.md`](../../knowledge/muse-glimmer-port.md): the loader's
`_mutate_state_dict` never runs on the shared slice (a nested text tower with an untied head
loads to garbage), and `to_empty()` silently leaves non-persistent buffers uninitialized, so
an oracle built that way runs on garbage RoPE frequencies and accuses a correct port.

## Not done

- **No published bundle.** The `.aimodel` exists locally only; publishing weights is the
  maintainer's call.
- **No vision tower.** The 2.5 B perception encoder is dropped; this is the text decoder.
- **No iPhone anything.** 16.35 GB.
