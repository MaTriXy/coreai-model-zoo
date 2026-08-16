# Shieldstral-1.0-3B — port knowledge

Mistral's 3B safety model. Ported as a **stateless classifier graph**, not a decoder:
`conversion/export_shieldstral.py`, gates in `_smoke/*shieldstral*`, suite in
`conversion/_shieldstral_suite.py`.

## A model that answers with one token should not ship as a decoder

Shieldstral's answer is the next-token distribution at the last prompt token, restricted to
`yes` (13059) and `no` (2649). Everything a decoder bundle exists to do — KV cache, position
bookkeeping, a sampling loop, a 131 072-way head — is dead weight for that. So the whole tail is
baked in:

```
(input_ids [1,S], attention_mask [1,S]) -> probs [1,2] = softmax([no, yes])
```

gather the last real token's hidden state (mask-based; right padding is invisible under the
causal mask), then apply **two rows of the tied embedding** as the head. The full head is
131 072 × 3072 = 805 MB of fp16 that a classifier never reads, and it costs nothing to not export
it. Same archetype as `export_qwen3_reranker.py`; the two of them are the reason to keep that
shape written down.

## `ministral3` is Mistral + YARN, and that is measurable in ten minutes

The conversion venv is transformers **4.57.6**, which cannot load this checkpoint at all:

- the tokenizer declares `tokenizer_class: TokenizersBackend` → `AutoTokenizer` raises;
- `AutoModelForCausalLM` rejects `Mistral3Config`;
- the zoo's overlay shim registers `Ministral3TextConfig` with `AutoConfig` only, so it gets you
  a config and no model.

transformers **git main** (5.16.0.dev0) knows `ministral3` natively, YARN intact — so that is
where the oracle lives, and the port's whole premise became: 4.57.6's `MistralModel`, handed this
checkpoint's `rope_parameters` as `rope_scaling`, *is* `ministral3`.

That is exactly the kind of premise not to assume — a mis-scaled rope still produces fluent
logits and plausible probabilities. `_smoke/test_shieldstral_torch_ladder.py` measures it:
**cos 1.000000 on last-position logits and |ΔP| = 0.00000 on all nine cases**. 4.57.6's yarn init
reads the same knobs (`beta_fast`, `beta_slow`, `mscale`, `truncate`,
`original_max_position_embeddings`); the checkpoint's extra `llama_4_scaling_beta` is not
consumed by either, and `type` is a v5 duplicate of `rope_type`. Weight names map 1:1
(`language_model.model.*` → `MistralModel.*`), and there is no `lm_head` key — it is tied.

**The general move**: when the export venv is a version behind the checkpoint, don't force the
config and hope. Build the oracle in the venv that supports it natively, then measure the
substitute against it. The measurement is cheaper than the debugging it replaces.

## Build the 4D mask yourself

`create_causal_mask` returns a 4D mask untouched ("it can also be an already prepared 4D mask, in
which case it is returned as-is"), so the module builds `causal & pad → 0 / finfo.min` and hands
it over. Padding semantics stay visible in the code that depends on them, and there is no
version-dependent mask-construction path inside the export.

## Host prompt: flatten the template, then prove it

The chat template renders exactly `<s>[SYSTEM_PROMPT]{sys}[/SYSTEM_PROMPT][INST]{user}[/INST]`,
and the body is `<Instruct>: …\n\n<Query>: …\n\n<Document>: …`. The exporter hardcodes that as
`PREFIX`/`SUFFIX` (the phone has no Jinja) and then **gates the flattening against the oracle's
own ids** — 9/9 bit-identical, Japanese included. Encode with `add_special_tokens=False`: `<s>`
is in the template text and this tokenizer's post-processor does not prepend one, so letting it
add specials gives you two.

## What the measurements said

M4 Max, `S=512`: fp16 6.88 GB / 230.5 ms, int8lin 4.04 GB / 253.8 ms, int4lin 2.53 GB / 232.5 ms
— all **9/9 verdicts** against fp32.

- **Quantization buys size, not speed.** fp16 is the fastest of the three; int8's dequant makes
  it the slowest. A single forward over a padded grid is compute-bound, which inverts the whole
  intuition built on decode loops.
- **The cost is the grid, not the text.** Same weights, same numerics, 73.6 ms at S=128 vs
  232.5 ms at S=512, for cases that are 83–96 tokens either way.
- **int4's error lands where the model was unsure.** Worst |ΔP| 0.030, all of it on the one case
  fp32 didn't saturate (JA violence, 0.9011 → 0.9315); everything else ≤ 0.005. The fp16 baseline
  row is what makes that readable: it puts the graph's own noise floor at 0.00056.

The practical consequence for anyone using it: **tune the threshold against the bundle you ship.**

iPhone 17 Pro (`ios-h18p`, 2.336 GiB `resources.bin`): 9/9 at both grids, 624.7 ms (S=512) and
371.9 ms (S=256) per verdict, engine ready in 10.9 / 5.3 s. The phone's probabilities match the
Mac's **to four decimals**, so the |ΔP| column is the same number on both — what int4 costs this
model is a property of the weights, not of where they run. The phone is 2.7-3.0x slower than the
M4 Max, and the grid is still the bigger lever.

## Environment

Oracle: `~/code/litertlm-convert/.venv-vl0930-t515` (transformers git main). Export, gates and
everything else: the normal conversion venv. The two share
`conversion/_shieldstral_suite.py` so they cannot disagree about what was scored.

## Left on the table

The checkpoint carries a Pixtral vision tower (`image_size` 1540, `spatial_merge_size` 2,
`multi_modal_projector` + `patch_merger`) for image moderation. Not ported — this bundle is text
only.
