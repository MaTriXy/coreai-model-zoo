---
license: apache-2.0
base_model: mistralai/Shieldstral-1.0-3B
tags:
  - coreai
  - aimodel
  - apple-silicon
  - on-device
  - safety
  - moderation
  - classifier
  - mistral
pipeline_tag: text-classification
---

# Shieldstral-1.0-3B — Apple Core AI (`.aimodel`)

**Mistral's 3B safety model converted to Apple's Core AI** (the Core ML successor announced at
WWDC26), for macOS 27 and iOS 27. Twelve languages, Apache-2.0.

The policy is a string in your code, not a fixed taxonomy: the host writes an Instruction ("Flag
self-harm promotion; do not flag help-seeking or support resources") and a Query, hands over the
content, and gets back a probability.

**It ships as a classifier, not a decoder.** Shieldstral answers by putting mass on `yes` or `no`
at the last prompt token, so the whole tail is baked into the graph:

```
(input_ids [1,S] int32, attention_mask [1,S] int32) -> probs [1,2] = softmax([no, yes])
```

One `.aimodel` forward is one verdict. No KV cache, no decode loop, no sampling, and no
131 072-way head — two rows of the tied embedding are the head, which is 805 MB of fp16 a
classifier never reads.

> Requires macOS 27 / iOS 27 (Core AI ships with the OS). Conversion code, gates and knowledge base:
> **[coreai-model-zoo](https://github.com/john-rocky/coreai-model-zoo)**.

## Bundles

| path | size | verdict latency (M4 Max) | numerics |
|---|---:|---:|---|
| `gpu-classify/…_int4lin_s512` | 2.53 GB | **232.5 ms** | **9/9** verdicts vs fp32, worst \|ΔP\| 0.030 |
| `gpu-classify/…_int4lin_s256` | 2.53 GB | **123.6 ms** | 9/9, numerics identical to S=512 |

macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`, median of 10 warm forwards,
engine ready in ~2 s.

**No iPhone bundle yet.** The AOT build for `h18p` is 2.336 GiB — under the 2.39 GiB that another
model in this zoo loads on an iPhone 17 Pro — so it is expected to fit, and *expected to fit* is
not a measurement. It ships when a phone has run it.

**Two measurements worth knowing before you pick a bundle.**

**Quantization buys size, not speed.** At the same grid, fp16 runs 230.5 ms and int8lin 253.8 ms
against int4lin's 232.5 — one forward over a padded grid is compute-bound, so shrinking weights
moves 6.88 GB to 2.53 GB and leaves the clock alone. That is the inverse of the decode loop,
where int4 is the main speed lever. Only int4lin is published because the larger bundles are not
better at anything.

**The cost of a verdict is the grid, not the text.** Both bundles hold the same weights and
produce the same probabilities; S=256 is 1.9× faster because it computes half as much padding.
Pick the grid from the longest document you will actually moderate — the scaffolding alone is
~60 tokens, so S=256 leaves ~196 for the document and S=512 leaves ~450.

## Verdicts

Nine cases, four policies, EN + JA — four of them near-misses that share a topic with a flagged
case, because a model that only separates the easy pairs is a keyword filter with extra steps.

| flagged | fp32 | int4 | | not flagged | fp32 | int4 |
|---|---:|---:|---|---|---:|---:|
| violence (EN) | 0.9972 | 0.9988 | | sourdough recipe | 0.0000 | 0.0000 |
| violence (**JA**) | 0.9011 | 0.9315 | | park recommendation (**JA**) | 0.0001 | 0.0001 |
| weapon-making | 0.9919 | 0.9967 | | chemical **safety** question | 0.0001 | 0.0001 |
| doxxing request | 1.0000 | 1.0000 | | refusal to dox | 0.0003 | 0.0004 |
| | | | | help-seeking | 0.0001 | 0.0002 |

Every verdict survives int4. What moves is the probability, and only on the case fp32 did not
already saturate (JA violence). The fp16 bundle's own noise floor is 0.00056, which is what makes
int4's 0.030 readable as real. **Tune any threshold against the bundle you ship, not against
fp32.**

## Host contract

Everything outside the forward is yours, and all of it is in `reference.json` next to the bundle:

```
PREFIX = "<s>[SYSTEM_PROMPT]" + SYSTEM + "[/SYSTEM_PROMPT][INST]"
BODY   = "<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
SUFFIX = "[/INST]"
```

- encode with **`add_special_tokens=False`** — `<s>` is in the template text and this tokenizer's
  post-processor does not add one, so letting it add specials gives you two;
- **right-pad** to the grid with `pad_token_id` 11, mask `1 × real + 0 × pad`. Under the causal
  mask the last real token never sees the padding, which is why S=128 and S=512 agree exactly;
- read `probs[1]` = P(violation).

`SYSTEM` is fixed (it ships in `reference.json`); `Instruct`, `Query` and `Document` are yours.

**Not included:** the checkpoint's Pixtral vision tower (`image_size` 1540). Text only.

## Using it from Swift

```swift
let guard = try await SafetyClassifier(model: .shieldstral3B)   // .shieldstral3BShort for S=256
let verdict = try await guard.check(message, policy: .selfHarm)
```

[coreai-kit](https://github.com/john-rocky/coreai-kit)'s `SafetyClassifier` owns the scaffolding,
the padding and the threshold. `reference.json` in each bundle ships the nine gated cases with
their fp32 probabilities, so any host — Swift, Python, yours — can check its own prompt
construction rather than trusting it.

## Converting it yourself

The conversion venv here is transformers 4.57.6, which cannot load this checkpoint at all — the
tokenizer declares `TokenizersBackend`, `AutoModelForCausalLM` rejects `Mistral3Config`. The
oracle therefore runs on transformers **git main**, which knows `ministral3` natively, and the
export is built on the claim that `ministral3` is Mistral + YARN (4.57.6's `MistralModel` handed
this config's `rope_parameters` as `rope_scaling`).

That claim is **measured, not assumed** — cos 1.000000 on last-position logits and |ΔP| = 0.00000
across all nine cases — because a mis-scaled rope still emits fluent logits and plausible
probabilities. See
[`_smoke/test_shieldstral_torch_ladder.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/_smoke/test_shieldstral_torch_ladder.py),
[`conversion/export_shieldstral.py`](https://github.com/john-rocky/coreai-model-zoo/blob/main/conversion/export_shieldstral.py)
and
[`knowledge/shieldstral-port.md`](https://github.com/john-rocky/coreai-model-zoo/blob/main/knowledge/shieldstral-port.md).

## License

Apache-2.0, carried from
[`mistralai/Shieldstral-1.0-3B`](https://huggingface.co/mistralai/Shieldstral-1.0-3B) (revision
`003ec7e2b0bab5f0e6307edbaf186fa5822b76f5`). Not affiliated with Apple or Mistral AI.
