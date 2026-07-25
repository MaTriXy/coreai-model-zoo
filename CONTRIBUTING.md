# Contributing a model port

External ports are welcome — the zoo's job is the **standard and the gate**, not a single
author's queue. A model you port ships to every CoreAIKit app as one line of Swift
(`ChatSession(catalog: "your-model")`), revision-pinned and nightly-verified, **with your
name on the card, the README row, and the release notes**.

## What gets accepted

Any model, any modality, if it clears three bars:

1. **License** — the upstream license must permit redistributing converted weights (Apache-2.0,
   MIT, BSD, most Gemma/LFM-style community licenses are fine; note the license in the card and
   ship any required LICENSE file inside the HF repo).
2. **Parity** — the ship gate the zoo applies to its own ports: teacher-forced / oracle top-1
   parity against the fp32 reference implementation (HF `transformers` or the official repo),
   plus a greedy-rollout sanity check. See [`PORTING.md`](PORTING.md) §gates and
   [`knowledge/evaluations-framework.md`](knowledge/evaluations-framework.md).
3. **Real hardware** — measured on an Apple silicon Mac at minimum (tok/s for LLMs, RTF for
   audio); iPhone numbers if you publish an iOS variant. Debug builds don't count — measure
   Release.

## Toolchain requirement

Export with **coreai-core ≥ 1.0.0b2**. Bundles exported with earlier wheels are rejected by the
Xcode 27 beta 3+ SDK loader (`Failed to convert to versioned IR` — tracked as FB23666783); the
zoo's own pre-b2 artifacts are being migrated for the same reason.

## Process

1. **Open (or claim) a [model request](../../issues/new?template=model-request.yml)** so work
   isn't duplicated — say you're porting it yourself. Maintainer support happens in the thread.
2. **Port it** — [`PORTING.md`](PORTING.md) is the runbook, [`knowledge/`](knowledge/) has
   per-architecture recipes (start with
   [`knowledge/README.md`](knowledge/README.md)). Ask early when something looks
   architecture-specific; it usually is.
3. **Host the bundle on your own Hugging Face account** — that's fine and encouraged (your
   models, your name). The CoreAIKit catalog pins a specific revision hash, so apps get the
   exact verified bytes regardless of where the repo lives.
4. **Open a PR here** with: `models/<model-id>/README.md` (the card — copy an existing one's
   structure), `models/<model-id>/recipe.toml` (the exact configuration that produced the bundle
   you published — `python3 conversion/zoo_convert.py show <name>` must print a complete
   command), the conversion script under `conversion/`, and the gate outputs (parity numbers +
   measured speed, environment noted). `python3 conversion/zoo_verify.py <your-hf-repo>` should
   report no FAIL.
5. **Review + enrollment** — after review, the model is enrolled in the
   [coreai-kit](https://github.com/john-rocky/coreai-kit) catalog with its revision pin (plus
   engine/runtime glue if it's a new capability kind), and the card gets its generated
   "Use it" block.

## Also welcome without a full port

- **Benchmark rows** from your device — the
  [bench-result issue template](../../issues/new?template=bench-result.yml) (the app measures,
  you paste).
- **Knowledge fixes** — corrections or additions to `knowledge/` notes, especially where a beta
  changed behavior.
- **Bug reports** with a catalog id and device/OS —
  [bug template](../../issues/new?template=bug-report.yml).
