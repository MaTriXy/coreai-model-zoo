# Muse-Glimmer-30B — port knowledge

`meta-models/Muse-Glimmer-30B` (Apache-2.0, plain unmodified licence text plus a separate
non-binding usage policy) is Meta's 30B image-text-to-text VLM. This note covers the **text
decoder** port. Two of its findings are not Muse-specific and will bite the next port that
looks like this one: the loader trap in §3 catches any multimodal checkpoint with an
**untied** head, and the oracle trap in §4 catches anyone reading per-layer activations out
of transformers.

**Read §0 first** — and then read it again, because the EDGE answer changed once the port was
measured. The port was taken on knowing it failed the gate as written; it then beat the
vendor's own on-device Apple-GPU build, which is a different EDGE sentence entirely.

## 0. EDGE: written off as Mac-only, then measured against the vendor's own build

### 0a. Why it was written off

`PORTING.md` §2 asks for one sentence naming what the Core AI port does that stock Apple +
MLX do not. Before measuring, there wasn't one:

* **Size puts it on Mac only.** The text tower is **27.855 B** params. fp16 is 55.7 GB; int4
  lands at ~16 GB. Against an iPhone ceiling of ~6 GB, no quantization closes that.
* **MLX already has it, converted, at every bit-width** — `mlx-community` publishes
  4/5/6/8-bit plus mxfp4 and nvfp4; the 4-bit alone has ~18 k downloads.
* **The measured ceiling for large dense on this stack is MLX parity**
  ([`coreai-vs-mlx-speed.md`](coreai-vs-mlx-speed.md)).

### 0b. What measuring found

Meta ships **official ExecuTorch Metal `.pte` builds** of this model and publishes their
speed. Their `metal` backend is **MLX-native** (repo README: "`metal` = Apple Silicon (MLX)"),
so this is Core AI against MLX-inside-ExecuTorch, on hardware they name.

| | weights | decode tok/s, M4 Max |
| --- | ---: | ---: |
| ExecuTorch `k-quant-17G-128K-text-solo-metal` (Meta's published number) | 17.9 GB | 23.7 |
| **Core AI `int4hu` decode-pipelined** | **16.35 GB** | **27.09** |

**+14.3% on decode, on 8.7% fewer bytes**, stock pipelined engine, no custom kernels. 512
prompt / 1024 generation / 3 trials; the short protocol (128p/256g) gives 27.46. Trial spread
is negligible (27.436 / 27.469 / 27.468). Their M4 Max is the same 546 GB/s bin as the one
used here — 23.7 tok/s × 17.9 GB is 424 GB/s of traffic, which the 410 GB/s bin cannot
produce.

So the EDGE sentence exists after all, and it is not the one the gate was looking for: *Core
AI runs Meta's 30B faster on an Apple GPU than Meta's own on-device build of it.* Worth
keeping as a rule — **"MLX already runs it" is a prediction, not a measurement**, and a
vendor's shipped on-device artifact is a target you can actually beat.

Still open, and not claimed: their **DFlash** speculative number (37.8 tok/s) is a different
weight class — and note that both rivals *can* speculate (mlx-vlm ships
`--draft-kind {dflash,eagle3,mtp}`), so a speculative row here needs all three measured.

Quality is no longer open: **GSM8K over the same 100 questions gives Core AI 98, ExecuTorch
97, MLX 95** — three apart, which n=100 cannot resolve. The lightest artifact of the three
did not pay for its size in accuracy, which is the objection the byte column invites. See §9.

### 0c. What still holds, and what the landscape now looks like

Mac-only stands: nothing about beating ExecuTorch puts a 16 GB artifact on an iPhone. The
sizes, for the record — 52 × 483.9 M layers plus two untied 1.345 B embedding matrices:
fp16 55.7 GB, int8 31.3 GB, int4 ~16.4 GB.

Worth noting beyond this port: **a vendor now ships official Apple-GPU on-device artifacts
directly** (text-only 17.9 GB through text+image+DFlash 23.8 GB, plus a separate 5.1 GB
DFlash drafter). The cards in this repo were written when "the vendor ships CUDA, we ship
Apple" was a safe assumption. It no longer is — but as §0b shows, the artifact existing is
not the same as it being the fastest one.

## 1. Architecture — Gemma-3 shaped, with four things that are not

Closest shipped port is `gemma3_text.py` (sandwich norms, sliding/full interleave). The
deltas, all read out of `transformers` 5.15 `modeling_muse_glimmer.py` and confirmed against
the weight map, not inferred:

| | Muse-Glimmer | Gemma-3 |
| --- | --- | --- |
| layer pattern | `sliding(2048) × 3 + full × 1`, 13 groups over 52 layers | 5:1 |
| **full layers** | **NoPE** — `layer_rope_theta[i] == 0`, and HF hands those layers `position_embeddings=None` | RoPE with a second theta |
| Q/K norm | **weight-less** RMSNorm (no learned scale), then Q × `qk_scale_factor` 3.87 | learned scale |
| attention output | **× `sigmoid(gate_proj(x))`** — a fifth projection reading the pre-attention hidden states | none |
| embedding | weight-less RMSNorm on the output (HF keeps it outside the matrix so the DFlash drafter can embed without it) | × `sqrt(hidden)` |
| sandwich norm eps | **two** — `rms_norm_eps` 1e-5 on the pre-norms, `post_norm_eps` 1e-8 on the post-norms | one |
| logits | `20 · tanh(logits · 0.19611 / 20)` — `output_multiplier` pre-scale then Gemma-style softcap | softcap only |

Two of these simplify in the re-authored module, both algebraically exact:

* **`qk_scale_factor` folds into the SDPA scale.** Attention is
  `softmax((αQ)·Kᵀ/√d)`; α is a scalar and rotary is a rotation, so `α/√d` as the SDPA scale
  is identical and removes a full-width multiply per layer.
* **`gate_proj` fuses into Q/K/V.** It reads the same pre-attention hidden states, so the
  module ships one `qkvg_proj` (4 GEMMs → 1). Fusion is along the *output* axis, so
  per-block-32 quantization along the input axis is unaffected, and each output row keeps
  its own scale — mixing Q/K/V/gate rows in one matrix is quantization-neutral.

## 2. The head is untied and it is 2.7 GB

`tie_word_embeddings: false`, vocab 202 048 × 6656. `lm_head.weight` is a real, separate
tensor at the checkpoint root. That drives two decisions: the `*hu` quantization modes exist
at all, and they use plain `symmetric` (absmax) via `--head-sym` — the big-vocab-head rule
from [`int8-head-and-decode-measurement.md`](int8-head-and-decode-measurement.md).

## 3. Trap: `_mutate_state_dict` never runs on the shared slice

**This is the one to remember.** `BaseForCausalLM.from_hf_memory_efficient` splits the
checkpoint into per-layer slices and one shared dict, and calls `_mutate_state_dict` **only
on the per-layer slices**. Which keys are "per-layer" is decided by matching
`^model\.layers\.(\d+)\.` against the key *after* `hf_state_dict_prefix` is stripped.

Muse-Glimmer keys the text tower as `model.language_model.layers.N.*`. The obvious prefixes
both fail, silently:

* `hf_state_dict_prefix = ""` → nothing matches the layer regex → **every** tensor lands in
  the shared dict → the Q/K/V/gate fusion never runs → `qkvg_proj` is never populated → the
  model loads, runs, and emits garbage. The expensive failure mode, exactly as `AGENTS.md`
  warns.
* `hf_state_dict_prefix = "model.language_model."` → stripped keys read `layers.N.*`, which
  also misses the regex. Same outcome.

The prefix that works is **`"model.language_"`** — stripping it leaves `model.layers.N.*`
(streaming path, mutation runs), `model.embed_tokens.weight` and `model.norm.weight` (shared,
and already on their real module paths), and skips `model.vision_tower.*` without loading it.

It also skips `lm_head.weight`, which sits outside any text prefix. The model class overrides
`from_hf_memory_efficient` to load that one tensor explicitly after the parent returns. An
untied head that is silently never loaded is the same garbage-that-looks-like-success
failure — worth a `strict` check rather than trust.

*Generalization:* any multimodal checkpoint whose text tower is nested **and** whose head is
untied hits both halves of this. Check the prefix against the layer regex, and check that
the head is inside the prefix, before trusting a load.

## 4. Trap: `hidden_states[-1]` is the final norm, not the last layer

Reading per-layer activations from `output_hidden_states=True` for a layer-by-layer cosine
comparison: `hidden_states[0]` is the embedding output and `hidden_states[i+1]` is layer
*i*'s output — **except the last entry**, which transformers replaces with the post-`norm`
output. The last decoder layer's raw output is not exposed.

The signature is a gate that reports **cos 1.000000 on every layer, cos 1.000000 on the
logits, and cos ≈ −0.08 on the final layer**. That is arithmetically impossible for a real
defect (a broken last layer cannot produce matching logits) and is the tell: compare the last
entry against your own final-norm output, not your last block.

## 4b. Trap: `to_empty()` silently destroys the oracle's RoPE

The real-weight gate failed on the first run — cos 0.990 at layer 0 decaying to
0.953 by layer 6, logits 0.80, 9 argmax flips out of 14. Every parameter was
already proven **bit-identical** to the checkpoint, so the loader was not at
fault; the arithmetic matched a transcribed reference at cos 1.0, so the module
was not at fault either. **The oracle was wrong.**

Building the reference the memory-efficient way —

```python
with torch.device("meta"):
    model = MuseGlimmerTextModel(config)
model = model.to_empty(device="cpu")
model.load_state_dict(state_dict, strict=False)
```

— leaves every **non-persistent buffer** as uninitialized memory. `load_state_dict`
cannot restore them because, being non-persistent, they are not in the state dict
at all. `MuseGlimmerTextRotaryEmbedding.inv_freq` is one, so the oracle ran with
garbage rotary frequencies and looked entirely plausible while doing it.

The tell is in the per-layer curve: **the NoPE layers do not degrade.** Layer 2 →
layer 3 (the first `full_attention`, no rotary) went 0.9737 → 0.9742, while every
sliding layer lost another half percent. A defect that skips exactly the layers
which skip RoPE is a RoPE defect — and since the port's own NoPE routing was
under test, it is easy to read that curve backwards and start "fixing" the port.

Fix: rebuild the buffers on a real device after `to_empty` —
`model.rotary_emb = MuseGlimmerTextRotaryEmbedding(config)` — and print
`inv_freq[:3]` so the oracle asserts its own soundness. For theta 500 000 /
head_dim 128 the first three are `1.0, 0.8146, 0.6636`; all-zeros or denormals
mean the buffer never got initialized.

*Generalization:* `to_empty()` + `load_state_dict` is only safe for a module whose
buffers are all persistent. Grep the modeling file for `persistent=False` before
trusting an oracle built that way — the failure is silent, and it accuses the port.

## 5. Oracle: build a transformers-5.15 venv

`muse_glimmer` landed in transformers 5.15; the pinned export venv is 4.57 and has neither
the config nor the modeling code. Same answer as
[qwen3.8-27b](qwen3.8-27b-port.md): a throwaway `uv venv` with `transformers==5.15.0` for the
oracle, and an `AutoConfig` shim (`muse_glimmer_config_shim.py`) so the 4.57 export venv can
still parse `config.json`.

One consequence reaches the bundle: `tokenizer_config.json` declares
`tokenizer_class: TokenizersBackend`, a transformers-5 concept 4.57 cannot instantiate. Call
`save_tokenizer(..., via_transformers=False)` so the raw files (including
`chat_template.jinja`) are copied verbatim instead of re-serialized.

## 6. Authoring gate

Structural parity was proven **before** the 60 GB checkpoint finished downloading, by
building a shrunken model (8 layers, 256 hidden, `sliding_window=6` so the window actually
bites at S=24) with seeded random weights under the 5.15 venv and loading that state dict
into the re-authored module. Norm weights are randomized rather than left at their zero init,
so a swapped pre/post norm or a wrong eps cannot hide.

Result: **cos 1.0000 at the embedding, every layer, the final norm, and the logits; 0 argmax
flips over 24 positions.** This covers the fused QKVG layout, the gate ordering, the folded
`qk_scale_factor`, weight-less Q/K norm, NoPE routing, sliding-window semantics, dual-eps
sandwich norms, and the softcap.

Doing it this way is worth copying: an authoring bug found against a 3 M-parameter random
model costs seconds to iterate, and the download is not on the critical path. It also pays a
second time — when the real-weight gate later failed, the random-weight PASS was what made
"the oracle is wrong" a live hypothesis instead of an excuse (§4b).

The real-weight gate (8 layers, fp32, covering two NoPE layers) then returns **cos 1.000 at
the embedding, every layer, the final norm and the logits, top-1 match, 0 argmax flips**.

## 7. Environment: the Swift side needs the beta Xcode, and breaks on every seed bump

Rebuilding `llm-runner` / `llm-benchmark` is a prerequisite for any measurement, and both
were dead in dyld on this machine:

* The release Xcode's SDK has **no `CoreAI` module at all** — the framework exists on the
  running OS but ships no `.swiftmodule`, so it cannot be compiled against. Core AI work
  needs `DEVELOPER_DIR` pointed at the beta Xcode. There is no way to route around this from
  the system framework.
* Then the FM beta ABI churn: Xcode 27 beta 5 **dropped the `capabilities:` argument label**
  from `LanguageModelCapabilities.init`. A binary built before that seed dies at load with
  `Symbol not found: …LanguageModelCapabilitiesV12capabilities…`. One-character fix, but it
  strands every prebuilt tool until someone rebuilds.

## 8. The generation budget is charged whether or not it is used

Measured on the shipped bundle: `llm-runner`'s wall time is

```
wall ≈ 5 s + 0.037 × max_tokens
```

**independent of how many tokens are actually generated.** Same prompt, same 532-token
answer, four budgets: 600 → 26.9 s, 900 → 37.9 s, 1200 → 49.0 s, 1800 → 71.6 s. All four
fit that line within 1%. The stop token halts *output*, not the decode loop — declaring
`<|eot|>` (§3b) fixed the garbage, and cost 9% of the time.

ExecuTorch's `solo_runner` does not do this: a 2048 budget that produces 219 tokens costs
219 tokens' worth of time.

This is invisible in the summary, which reports a healthy 27.4 tok/s regardless — that
figure is honest for the *decode*, and was confirmed against wall clock by varying the
trial count (`llm-benchmark -n 1/2/3`: marginal 14.6 s per trial against 14.55 s predicted).
It simply does not include the budget tax.

**Consequences for anyone benchmarking a Core AI bundle:**

* Set `--max-tokens` to what the task needs, not to a safe ceiling. A 9-hour GSM8K run
  became 3.5 hours from this alone.
* If you must allow a large budget, **run two passes** — a low budget, then re-run only
  the answers that hit it. On this model that changed the GSM8K score from 87 to 98,
  because a truncated answer scores as a wrong number rather than as a blank.
* Do not read a wall-clock/report discrepancy as an inflated tok/s. It is the budget.

*Getting here cost seven wrong hypotheses — model load, generation-length variance, the
tokenizer, `verify()` hashing the asset, chunk threshold, the sampler path, KV-cache
growth. Each was plausible and each was theory ahead of measurement. What settled it was
changing one variable at a time and taking wall clock, which is also the only step that
needed no GPU.*

## 9. Quality: the lightest artifact, and it does not pay for it

| | weights | GSM8K, 100 questions |
| --- | ---: | ---: |
| **Core AI** `int4hu` | **16.35 GB** | **98** |
| ExecuTorch `k-quant-17G` | 17.9 GB | 97 |
| MLX 4-bit | 18 GB | 95 |

Greedy, same questions, scored by Yardstick's `scripts/parity_gsm8k.py` — **that file was
not edited**; the two arms it lacks were added around it (ExecuTorch through Meta's own
`solo_runner`; MLX through `mlx_vlm`, because `mlx_lm` raises `Model type muse_glimmer not
supported`). Three questions apart is not resolvable at n=100: read it as "no arm is
meaningfully worse", not as a win.

**The two-pass budget is not optional here, and it is the transferable part.** At a single
budget of 700 Core AI scores **87**; 32 answers were truncated. Re-running only those at
2048 gives **98**. A truncated answer is not blank — the extractor falls back to "the last
number in the text", so it scores a number from the middle of the reasoning, wrong most
times and right by accident sometimes. Truncation rates differ per arm (32 / 26 / 14), so
one fixed budget silently penalises whichever arm reasons longer, and a quality table
becomes a verbosity table.

Residue, stated rather than dropped: 3 / 2 / 1 questions still hit 2048 and are scored from
truncated output. Core AI and MLX also emit visibly longer answers than ExecuTorch for the
same questions (median 532 / 453 / 326 tokens) — unexplained, and invisible in the score.
