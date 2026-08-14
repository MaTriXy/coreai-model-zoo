# Shieldstral-1.0-3B (safety classifier) — Core AI

**A moderation model whose policy you write at call time, as one `.aimodel` forward.** A Core AI
port of [`mistralai/Shieldstral-1.0-3B`](https://huggingface.co/mistralai/Shieldstral-1.0-3B):
text + policy → P(violation), Apache-2.0, twelve languages including Japanese.

Apple's stock answer here is `SensitiveContentAnalysis` — a fixed policy over images. This is the
other shape: the app states the rule in plain language ("Flag self-harm promotion; do not flag
help-seeking or support resources") and gets a probability back, so the policy is a string in
your code rather than a model you have to retrain.

**It ships as a classifier, not a decoder.** Shieldstral answers by putting mass on `yes` or `no`
at the last prompt token, so the whole tail is baked into the graph — gather the last real
token's hidden state, apply *two rows* of the tied embedding as the head, softmax:

```
(input_ids [1,S] int32, attention_mask [1,S] int32) -> probs [1,2] = softmax([no, yes])
```

No KV cache, no decode loop, no sampling, and nothing for the host to get wrong after the
forward. Dropping the head to two rows is also 805 MB of fp16 that a classifier never reads.

[`knowledge/shieldstral-port.md`](../../knowledge/shieldstral-port.md) ·
[`conversion/export_shieldstral.py`](../../conversion/export_shieldstral.py)

## Numbers

**M4 Max**, macOS 27.0 (26A5378n), Xcode 27.0 (27A5218g), `coreai-torch 0.4.1`, `S=512` grid:

| bundle | size | verdict latency | verdicts vs fp32 | worst \|ΔP\| |
|---|---:|---:|---|---:|
| `classify_fp16_s512` | 6.88 GB | 230.5 ms | **9/9** | 0.00056 |
| `classify_int8lin_s512` | 4.04 GB | 253.8 ms | **9/9** | 0.00220 |
| **`classify_int4lin_s512` (ship)** | **2.53 GB** | **232.5 ms** | **9/9** | 0.03037 |
| **`classify_int4lin_s256` (ship)** | **2.53 GB** | **123.6 ms** | **9/9** | 0.03037 |
| `classify_int4lin_s128` | 2.53 GB | 73.6 ms | **9/9** | 0.03037 |

Median of 10 warm forwards; engine ready in ~2 s.

**No iPhone row.** The `h18p` AOT build is 2.336 GiB, under the 2.39 GiB North-Micro-Vision loads
on an iPhone 17 Pro, so it is expected to fit — and that is a prediction, not a measurement. The
gate exists (`PB_SHIELD` in PipelinedBench, fixture in `_smoke/shieldstral_ref/`); the phone was
on Wi-Fi rather than USB and the tunnel would not hold a run.

**Two things those rows say that are easy to miss.**

**Quantization buys size, not speed.** fp16 is the *fastest* of the three at the same grid
(230.5 ms vs int8's 253.8). One forward over a 512-token grid is compute-bound, so shrinking the
weights moves the artifact from 6.88 GB to 2.53 GB and leaves the clock where it was — the exact
opposite of the decode loop, where bandwidth is the whole game and int4 is the main lever.

**The cost is the grid, not the text.** Every case in the suite is 83–96 tokens, and the same
cases take 73.6 / 123.6 / 232.5 ms at S=128 / 256 / 512 for numerics identical to five decimal
places. Padding is computed, not skipped. So pick the grid from the longest document you will
actually moderate — the weights are the same 2.53 GB either way, and the prompt scaffolding alone
is ~60 tokens, which is why S=128 (leaving ~68 for the document) is measured here but not
shipped.

## Gates

Nine cases, four policies, EN + JA. Four of them are near-misses that share a topic with a
flagged case — a weapons **safety** question, a refusal to dox, someone asking to talk to
somebody about a hard time. A moderation model that only separates the easy pairs is a keyword
filter with extra steps.

| gate | what it proves |
|---|---|
| [`test_shieldstral_torch_ladder.py`](../../_smoke/test_shieldstral_torch_ladder.py) | `ministral3` **is** Mistral + YARN: cos **1.000000** on last-position logits, \|ΔP\| = 0.00000, all 9 |
| exporter's host-prompt gate | the flattened `PREFIX/SUFFIX` reproduces the chat template's ids **bit-identically**, all 9 (JA included) |
| exporter's fp32 wrapper gate | the padded-grid graph equals the oracle's own scoring, worst \|ΔP\| **0.00000** |
| [`test_shieldstral_aimodel_gate.py`](../../_smoke/test_shieldstral_aimodel_gate.py) | the exported bundle's verdicts and probabilities, per case, against fp32 |

The oracle ([`_smoke/shieldstral_ref.py`](../../_smoke/shieldstral_ref.py)) is HF's native
`ministral3` on **transformers git main**, and fp32 gets all nine cases right on its own:

| case | P(violation) | | case | P(violation) |
|---|---:|---|---|---:|
| violence (EN) | 0.9972 | | benign (EN) | 0.0000 |
| violence (**JA**) | 0.9011 | | benign (**JA**) | 0.0001 |
| weapon-making | 0.9919 | | chemical **safety** question | 0.0001 |
| doxxing request | 1.0000 | | refusal to dox | 0.0003 |
| | | | help-seeking | 0.0001 |

## What compression costs

**Every verdict survives int4** — 9/9, including all four near-misses. What moves is the
probability, and only on the one case fp32 did not already saturate: the Japanese violence
prompt, 0.9011 → 0.9033 at int8 → 0.9315 at int4. Every other case moves by ≤ 0.005.

That is the pattern worth carrying: the fp16 baseline row exists to say the 0.00056 floor is the
graph's own fp16 noise, so int4's 0.030 is real and it is concentrated where the model was least
sure. A threshold at 0.5 doesn't notice; a threshold at 0.9 would. **If you tune a threshold,
tune it against the bundle you ship, not against fp32.**

Three int4 verdicts in this family of ports do not line up with size — LFM2.5-VL-450M craters,
LFM2.5-VL-3B doesn't move, North-Micro-Vision craters — and this one doesn't move either. Read
the outputs of the model in front of you.

## Host contract

The graph sees ids and a mask; everything else is the host's, and all of it is in
`reference.json` next to the bundle.

```
PREFIX = "<s>[SYSTEM_PROMPT]" + SYSTEM + "[/SYSTEM_PROMPT][INST]"
BODY   = "<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"
SUFFIX = "[/INST]"
```

- encode with **`add_special_tokens=False`** — `<s>` is in the template text and the tokenizer's
  post-processor does not add one;
- **right-pad** to the grid with `pad_token_id` 11 and set the mask to `1 × real + 0 × pad`. The
  causal mask means the last real token never sees the padding, which is why the numerics are
  identical at S=128 and S=512;
- read `probs[1]`. `Instruct` is the policy, `Query` is the question asked of the document, and
  `Document` is the content under review — the model is trained on this shape, and the four
  policies in the gate suite are examples, not a fixed list.

**Not ported:** the checkpoint's Pixtral vision tower (`image_size` 1540), so image moderation is
not in this bundle. Text only.

## Using it from Swift

`SafetyClassifier` in [coreai-kit](https://github.com/john-rocky/coreai-kit) owns the prompt
scaffolding, the padding and the threshold:

```swift
let guard = try await SafetyClassifier(model: .shieldstral3B)   // or .shieldstral3BShort (S=256)
let verdict = try await guard.check(message, policy: .selfHarm)
if verdict.violates { … }                                       // verdict.probability is the number to log
```

The four presets (`.violence`, `.weapons`, `.privacy`, `.selfHarm`) are the policies this gate
suite scored, so they are the ones with measured behaviour — but a `Policy` is just two strings,
and writing your own is the point of the model.

Cross-runtime parity is tested against the bundle's own `reference.json`, which ships the nine
cases and their fp32 probabilities: the Swift host reproduces the Python host's numbers
(0.9988 / 0.9315 / 0.9967 …), which is what catches a scaffolding mistake — a second `<s>`,
padding on the wrong side, a dropped `[/INST]` — none of which look wrong in the output.

## Convert / verify

```bash
# oracle — transformers GIT MAIN only (4.57.6 cannot load this checkpoint at all)
~/code/litertlm-convert/.venv-vl0930-t515/bin/python _smoke/shieldstral_ref.py

# the claim the whole port rests on, measured
python _smoke/test_shieldstral_torch_ladder.py

# export (host-prompt + fp32 wrapper gates run inside)
python conversion/export_shieldstral.py int4lin --seq-len 512
python conversion/export_shieldstral.py int4lin --seq-len 256

# engine gate + latency
python _smoke/test_shieldstral_aimodel_gate.py --mode int4lin --bench 10
```
