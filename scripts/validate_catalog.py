#!/usr/bin/env python3
"""Check that the catalog is internally consistent — no Mac, no GPU, no weights.

The zoo's claim is that every published bundle carries the recipe that produced it. That
claim is only worth anything if the recipes actually resolve: a recipe naming a script that
was renamed, or a card that was deleted, is a broken promise that looks fine in a diff.
Nothing here re-runs a conversion or touches a model — it reads the repository and asks
whether it describes itself truthfully, which is exactly the part a CI runner can do.

    python3 scripts/validate_catalog.py           # check everything
    python3 scripts/validate_catalog.py --quiet   # only failures

Checked per recipe entry:
  - required keys are present, and the entry names *some* way to reproduce it
  - `script` (or each step's script) exists under conversion/
  - `card` exists in the model's directory
  - `status` is a value the tooling understands, and `unverified` carries `open_questions`
  - `runtime_patches` point at files that exist
  - `hf_repo` looks like `owner/name`

And per model directory: a README card exists, and index.json agrees with the directories on
disk.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "models"
CONVERSION = REPO / "conversion"
STATUSES = {"verified", "unverified"}
HF_REPO = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")

failures: list[str] = []
checked = 0


def fail(where: str, msg: str) -> None:
    failures.append(f"{where}: {msg}")


def check_script(where: str, script: str) -> None:
    # `script` may name a subdirectory, e.g. "dllm/export_llada.py".
    if not (CONVERSION / script).is_file():
        fail(where, f"script not found: conversion/{script}")


def check_entry(model_dir: Path, name: str, step: dict) -> None:
    global checked
    checked += 1
    where = f"{model_dir.name}/recipe.toml [{name}]"

    for key in ("card", "hf_repo", "status"):
        if key not in step:
            fail(where, f"missing required key `{key}`")

    status = step.get("status")
    if status is not None and status not in STATUSES:
        fail(where, f"status {status!r} is not one of {sorted(STATUSES)}")
    if status == "unverified" and not step.get("open_questions"):
        fail(where, "status is `unverified` but the entry records no `open_questions` — "
                    "an unverified recipe must say what it cannot answer")

    repo = step.get("hf_repo")
    if repo is not None and not HF_REPO.match(repo):
        fail(where, f"hf_repo {repo!r} is not owner/name")

    card = step.get("card")
    if card is not None and not (model_dir / card).is_file():
        fail(where, f"card not found: models/{model_dir.name}/{card}")

    # A `verified` recipe must record some reproduction path: a single script, a multi-step
    # build, or an explicit command. `unverified` is the documented state for "the repository
    # does not record what produced this bundle" — it is allowed to have none, provided it
    # says so in open_questions (checked above) rather than looking complete.
    if "script" in step:
        check_script(where, step["script"])
    elif "steps" in step:
        for i, sub in enumerate(step["steps"]):
            if isinstance(sub, dict) and "script" in sub:
                check_script(f"{where} step {i}", sub["script"])
    elif "command" not in step and status != "unverified":
        fail(where, "records no `script`, `steps`, or `command` — nothing to reproduce")

    for patch in step.get("runtime_patches", []):
        if not (REPO / patch).is_file():
            fail(where, f"runtime_patch not found: {patch}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true", help="print only failures")
    args = ap.parse_args()

    model_dirs = sorted(p for p in MODELS.iterdir() if p.is_dir())
    for d in model_dirs:
        if not (d / "README.md").is_file():
            fail(d.name, "no README.md — every model directory is a card")
        recipe = d / "recipe.toml"
        if not recipe.is_file():
            continue
        try:
            parsed = tomllib.loads(recipe.read_text())
        except tomllib.TOMLDecodeError as e:
            fail(f"{d.name}/recipe.toml", f"does not parse: {e}")
            continue
        for name, step in parsed.items():
            if isinstance(step, dict):
                check_entry(d, name, step)

    index_path = MODELS / "index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            indexed = {e["family"] for e in index["models"]}
            on_disk = {d.name for d in model_dirs}
            if missing := on_disk - indexed:
                fail("models/index.json", f"model directories missing from the index: "
                                          f"{sorted(missing)} — run scripts/gen_inventory.py")
            if extra := indexed - on_disk:
                fail("models/index.json", f"indexed models with no directory: {sorted(extra)}")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            fail("models/index.json", f"unreadable: {e}")

    if not args.quiet:
        print(f"checked {len(model_dirs)} model directories, {checked} recipe entries")
    if failures:
        print(f"\n{len(failures)} problem(s):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        sys.exit(1)
    if not args.quiet:
        print("catalog is self-consistent")


if __name__ == "__main__":
    main()
