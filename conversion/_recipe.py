#!/usr/bin/env python3
"""Read a recipe entry the way the tooling does — shared so it is read one way.

The source checkpoint a bundle was converted from is the single most useful fact about it,
and most recipes do not state it: `--hf-id` appears in `args` only when the recipe *deviates*
from the exporter's default, so the usual answer lives in the export script.

This resolves it, and reports how. Deliberately not written into `recipe.toml`: copying an
exporter default into a second file creates two sources of truth that drift silently, which
is the failure mode this repository has already paid for once. Generated artifacts
(`models/index.json`) carry the resolved value instead, so it cannot go stale.
"""
from __future__ import annotations

import re
from pathlib import Path

CONVERSION = Path(__file__).resolve().parent

# Ways an export script names its source checkpoint, most explicit first. Matched against the
# source text rather than by importing the module, which would drag in torch to read a string.
_PATTERNS = [
    re.compile(r'--hf-id"\s*,\s*default\s*=\s*"([^"]+/[^"]+)"'),
    re.compile(r'^\s*(?:MODEL_ID|HF_ID|REPO_ID|HF_REPO|MODEL_REPO)\s*=\s*"([^"]+/[^"]+)"', re.M),
    re.compile(r'from_pretrained\(\s*"([^"]+/[^"]+)"'),
    re.compile(r'snapshot_download\(\s*(?:repo_id\s*=\s*)?"([^"]+/[^"]+)"'),
]


def flag_value(args: list[str], name: str) -> str | None:
    """Read `--name value` out of a recipe's argv."""
    if name not in args:
        return None
    i = args.index(name) + 1
    return args[i] if i < len(args) else None


def script_source_model(script: str) -> str | None:
    """The checkpoint an export script uses when the recipe doesn't override it."""
    # `script` may name a subdirectory, e.g. "dllm/export_llada.py".
    path = CONVERSION / script if script else None
    if path is None or not path.is_file():
        return None
    text = path.read_text()
    for pattern in _PATTERNS:
        if m := pattern.search(text):
            return m.group(1)
    return None


def source_model(step: dict) -> tuple[str | None, str | None]:
    """`(checkpoint, how it was determined)` for one recipe entry.

    `how` is "recipe" when the entry states it and "exporter" when it came from the script's
    own default — worth keeping, because the second is only as pinned as the script is.
    """
    args = [str(a) for a in step.get("args", [])]
    if explicit := flag_value(args, "--hf-id"):
        return explicit, "recipe"
    if inferred := script_source_model(str(step.get("script", ""))):
        return inferred, "exporter"
    return None, None


def revision(step: dict) -> str | None:
    """The upstream checkpoint revision, when the recipe pins one."""
    return flag_value([str(a) for a in step.get("args", [])], "--revision")
