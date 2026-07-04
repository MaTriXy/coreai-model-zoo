# coreai_models python overlay

The zoo's conversion scripts import re-authored model definitions and export-pipeline hooks
that Apple's `coreai-models` does not ship (it takes no PRs and does not register newer
models). This directory packages those additions so the zoo is **self-contained**: anyone can
reproduce the conversion environment from a pinned upstream checkout instead of needing our
working tree.

## Contents

- `BASE` — pinned upstream repo + commit the overlay applies to.
- `patches/python-overlay.patch` — edits to **tracked** upstream files: export pipeline
  (`export/{bundle,macos,metadata,pipeline,presets}.py`), registries
  (`model_registry.py`, `models/registry.py`), primitives
  (`primitives/{macos,ios}/{cache,rope}.py`), and small model fixes (`mistral`, `base`).
- `files/` — **new** files, mostly `models/{macos,ios}/*.py`: the re-authored decoders
  (qwen3.5 / qwen3.6-MoE / gemma4 / GLM-4.7 / LFM2.5 / LLaDA / BitCPM / RWKV-7 / Zaya /
  MiniCPM / omni / VL towers, …) plus Metal-kernel variants. Includes research variants that
  are assets but not ship configs (e.g. `gemma4_metal_mlp_fp4.py`, `*_int2.py`,
  `gemma4_mtp_drafter.py`) — the zoo cards say which export is the shipping one.
- `apply.py` — applies patch + files onto a pinned checkout (verifies the base commit).
- `regen.sh` — regenerates patch + files from a live checkout (run after new porting work).

## Use

```sh
git clone https://github.com/apple/coreai-models.git
git -C coreai-models checkout "$(awk -F': *' '/^commit:/{print $2}' BASE)"
python3 apply.py ./coreai-models
cd coreai-models && python3 -m venv .venv && . .venv/bin/activate && pip install -e python/
```

Then run any `conversion/export_*.py` with that venv (per-script flags: see
[`../README.md`](../README.md) and the model's zoo card).

## Scope

Python package only (`python/src/coreai_models/**`). The Swift engine changes that some zoo
apps need are **not** captured here — they live as commits on the public fork
[john-rocky/coreai-models](https://github.com/john-rocky/coreai-models) (e.g. `trimKVCache`,
pipelined-engine stop fix) and as per-app patches like
[`../../apps/coreai-pipelined-extra-states.patch`](../../apps/coreai-pipelined-extra-states.patch);
in-flight engine experiments are tracked by their stream state docs, not frozen here.

## License

The patch modifies and the new files derive idioms from Apple's `coreai-models`
(see upstream LICENSE); they are provided under the same terms as this repository's LICENSE.
