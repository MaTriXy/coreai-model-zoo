---
name: reproduce-a-zoo-model
description: Use this skill to rebuild, verify, or run a model from the Core AI model zoo (coreai-model-zoo) — any request like "convert Qwen3.5 for Core AI", "reproduce the gemma-4-E2B bundle", "which zoo model should I use for on-device chat / OCR / TTS / embeddings", "check whether this .aimodel was published correctly", or "what did this SDK beta break". Also triggers on models/<name>/recipe.toml, zoo_convert.py, zoo_verify.py, and .aimodel bundles published under mlboydaisuke on Hugging Face.
---

# Reproduce a zoo model

The zoo publishes ~70 Core AI repos and records, for each, the exact configuration that
produced the published bundle. This skill is how you use that record: pick a model, rebuild
its bundle with one command, and check the result against the model it came from.

**Related**: `Skill("coreai-skills:working-with-coreai")` (Apple's own skill — the Core AI
toolchain itself: TorchConverter, coreai-build, the runtime) | `PORTING.md` in this repo (how
to port a *new* model, which is a different job).

---

## Start here

`models/index.json` is the machine-readable catalog. Read it first — do not grep the tree.

```bash
python3 -c "import json;d=json.load(open('models/index.json'));print(len(d['models']),'families')"
```

Each entry gives the family, its card, and one recipe per published bundle:

```json
{"family": "qwen3.5",
 "card": "models/qwen3.5/README.md",
 "recipes": [{"name": "qwen3.5-0.8b", "status": "verified",
              "hf_repo": "mlboydaisuke/qwen3.5-0.8B-CoreAI",
              "bundle": "gpu-pipelined/qwen3_5_0_8b_decode_int8hu_block32_sym",
              "run": "python3 conversion/zoo_convert.py run qwen3.5-0.8b"}]}
```

`models/_INVENTORY.md` is the same data for humans, plus 30-day downloads and the current
verification verdicts. Use it to choose *which* model, then come back here to run it.

---

## Reproduce a published bundle

```bash
python3 conversion/zoo_convert.py list                   # recipes that can be run as recorded
python3 conversion/zoo_convert.py show  qwen3.5-0.8b     # command + every prerequisite
python3 conversion/zoo_convert.py doctor                 # is this interpreter wired up?
python3 conversion/zoo_convert.py run   qwen3.5-0.8b     # export
```

`show` prints four kinds of prerequisite. Read them before running — the export succeeds
without the run-time ones and the bundle then misbehaves inside the app:

| Line | Means |
| --- | --- |
| `overlay` | the interpreter needs `coreai_models` with `conversion/overlay/` applied. `doctor` checks it. |
| `needs` | something the export cannot run without: a checkpoint download, a gather-table dump, a package patch |
| `runtime` | what the **app** needs to run the resulting bundle: an engine patch, an environment variable |
| `device` | the AOT compile step for the iPhone bundle |

Some ports need none of that. `show` prints a `uv` line when the script declares its own
dependencies inline (PEP 723) — depth, detection, embeddings, TTS, time series and similar:

```bash
uv run conversion/export_da3.py --variant small --dtype float16 --res 504
```

**Set up the interpreter once** — needed only for the exporters that import re-authored model
code, which is what `doctor` is checking:

```bash
git clone https://github.com/apple/coreai-models.git
git -C coreai-models checkout "$(awk -F': *' '/^commit:/{print $2}' conversion/overlay/BASE)"
python3 conversion/overlay/apply.py ./coreai-models
cd coreai-models && python3 -m venv .venv && . .venv/bin/activate && pip install -e python/
```

Paths never hardcode a home directory. `python3 conversion/_paths.py` prints where downloads,
exports and the Hugging Face cache resolve; `ZOO_WORK_ROOT`, `ZOO_EXPORTS`, `ZOO_CODE_ROOT` and
`HF_HUB_CACHE` move them.

### Do not expect byte-identity

A rebuilt bundle will not hash-match the published one, and that is not a failure: running the
same recipe twice on the same machine produces different bytes too (measured: `main.mlirb`
differing by 7 bytes, `main.hash` entirely). Judge a reproduction by the gates the script runs
and by `zoo_verify.py`, never by a checksum.

### When a recipe is `unverified`

`zoo_convert.py run` refuses, and prints the exact question it cannot answer — usually "was
`--head-sym` passed?". The repository does not record which configuration produced the
published bundle, so running the recipe yields *a* bundle, not *the* bundle, and the difference
is invisible afterwards.

**Do not guess the missing argument.** Either ask the owner, or use `--force` and state clearly
in your output that the result may not match what was published.

---

## Check that a bundle is correct

```bash
python3 conversion/zoo_verify.py mlboydaisuke/Gemma-4-12B-CoreAI     # one repo
python3 conversion/zoo_verify.py --all --json models/_VERIFY.json    # whole catalog, minutes
```

Tier 1 needs no oracle, no device and no weights: it reads the bundle's `metadata.json` and
tokenizer straight from Hugging Face and compares them against the **source repository the
bundle itself names**. Four checks — eos/bos, chat template, context length, declared
precision.

Read the verdicts precisely:

- `PASS` — agrees with its source.
- `DIFF` — deviates from its source with no recorded reason. Not automatically a bug: swapping
  `eos_token` for the turn terminator is a real ship-time decision. It becomes correct by being
  *recorded* in `models/<family>/verify.toml`, after which an unexplained deviation fails.
- `FAIL` — wrong on its own terms (e.g. the source ships a chat template and the bundle ships
  none, so a host cannot format prompts).
- `skipped` — the check could not run. Never report a skipped check as a pass.

This is also the "what did this SDK beta break?" tool: rerun `--all` and diff `_VERIFY.json`.

---

## Run a bundle

Bundles are plain Core AI assets. Swift hosts load them with `GraphModel` /
`AIModel.load`; `swift/` in this repo has the runner package and
`knowledge/swift-runtime.md` the engine anatomy. Many models are also enrolled in
**CoreAIKit**, where one line runs them (`CoreAI.summarize(text, options: .model("qwen3.5-2b"))`);
the card's "Use it" block shows the exact call, and `models/index.json` records which families
have one.

Two hazards worth stating outright, both from real incidents:

- **Never run an iOS-compiled bundle on a Mac.** It can wedge the GPU stack and force a reboot.
- **Benchmark with a self-test entrypoint, not through a chat UI.** Numbers measured through UI
  are not comparable to anything.

---

## What not to do

- Do not edit `models/_INVENTORY.md`, `models/index.json` or `models/_VERIFY.json` by hand —
  they are generated (`scripts/gen_inventory.py`, `conversion/zoo_verify.py`).
- Do not change export hyperparameters while reproducing. A recipe reproduces the published
  bundle; improving it is a separate, owner-approved change.
- Do not delete bundles, oracles or export outputs. Several are the only copy in existence.
- Do not push to Hugging Face, post, or open PRs against `apple/*` on the owner's behalf.
