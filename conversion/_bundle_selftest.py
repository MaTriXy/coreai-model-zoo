#!/usr/bin/env python3
"""Prove `_bundle.py` still emits what each driver emitted before it was shared.

`_bundle_goldens.json` holds the exact `metadata.json` every driver produced when it owned
a private copy of `write_bundle_metadata`, and the exact `lm_head` spec every driver's
`head_quant_spec` returned, recorded from the tree at commit 6932324 — before the extraction.
The drivers here have already published bundles, so the bar for sharing the code was that
not one byte of output moves. This checks that, and it checks it against the call sites *on
disk*: it parses each driver, evaluates its real `write_bundle_metadata(...)` /
`head_quant_spec(...)` expression against stub arguments, and compares. A driver whose call
site is edited into disagreement fails here rather than at the end of a six-hour export.

What it does not cover: `save_tokenizer`, whose output is whatever the Hub returns. Its one
deliberate behaviour change is described in `_bundle.py`.

    python3 conversion/_bundle_selftest.py
"""
from __future__ import annotations

import ast
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _bundle import head_quant_spec, write_bundle_metadata  # noqa: E402

HERE = Path(__file__).resolve().parent
DATE = re.compile(r'"date": "[^"]*"')
GRID = [("block32", False), ("block32", True), ("block16", False),
        ("block8", True), ("perchan", False), ("perchan", True)]

NAME, HF, CTX, VOCAB = "qwen3-6-35b-a3b_int8hu", "Qwen/Qwen3.6-35B-A3B", 8192, 151936


class Cfg:
    vocab_size = VOCAB


class Args:
    """Stands in for the driver's parsed arguments, with the values the goldens used."""

    def __init__(self, **kw):
        self.hf_id, self.max_ctx, self.s = HF, CTX, 4
        self.mode, self.revision = "int8hu", None
        self.weights_source = "gemma-4-E2B-it-qat-mobile-transformers"
        self.head_quant, self.head_sym = "block32", False
        self.__dict__.update(kw)


# Drivers whose metadata depends on a flag are evaluated once per value it can take; the
# label is the golden's key. Everything else is evaluated once per call site, in order.
LABELS: dict[str, list[tuple[str, dict]]] = {
    "export_nanbeige41_decode_pipelined.py": [("fp16", {"mode": "fp16", "revision": None}),
                                              ("int8hu", {"mode": "int8hu",
                                                          "revision": "abc123"})],
    "export_nemotron_h_decode_pipelined.py": [("fp16", {"mode": "fp16"}),
                                              ("int8hu", {"mode": "int8hu"})],
    "export_glm_ocr_pipelined.py": [("llm", {}), ("prefill", {})],
    "export_mineru_pipelined.py": [("llm", {}), ("prefill", {})],
    "export_qwen3_vl_pipelined.py": [("llm", {}), ("prefill", {})],
}
DEFAULT_LABEL = {"export_gemma4_pf_pipelined.py": "pf",
                 "export_qwen3_5_verify_pipelined.py": "verify",
                 "export_gemma4_mixedbit_verify_pipelined.py": "verify"}


def calls(src: str, fn: str) -> list[ast.Call]:
    return [n for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fn]


def run_meta(call: ast.Call, args: Args) -> str:
    d = Path(tempfile.mkdtemp())
    ns = {"write_bundle_metadata": write_bundle_metadata, "out_dir": d, "name": NAME,
          "vname": NAME, "args": args, "cfg": Cfg(), "HF_ID": "Zyphra/ZAYA1-8B"}
    eval(compile(ast.Expression(call), "<call>", "eval"), ns)
    out = (d / "metadata.json").read_text()
    shutil.rmtree(d)
    return DATE.sub('"date": "FROZEN"', out)


def main() -> int:
    goldens = json.loads((HERE / "_bundle_goldens.json").read_text())
    bad = checked = 0

    for key, want in sorted(goldens["metadata"].items()):
        driver, _, label = key.partition("::")
        src = (HERE / driver).read_text()
        found = calls(src, "write_bundle_metadata")
        labels = LABELS.get(driver)
        if labels:                       # flag-dependent, or one call site per label
            per_call = len(found) == len(labels)
            idx = [l for l, _ in labels].index(label)
            call = found[idx if per_call else 0]
            args = Args(**dict(labels)[label])
        else:
            if len(found) != 1:
                print(f"FAIL {key}: {len(found)} call sites, expected 1")
                bad += 1
                continue
            call, args = found[0], Args()
            if DEFAULT_LABEL.get(driver, "decode") != label:
                print(f"FAIL {key}: unexpected label")
                bad += 1
                continue
        checked += 1
        got = run_meta(call, args)
        if got != want:
            bad += 1
            import difflib
            print(f"\nFAIL {key}")
            print("\n".join(difflib.unified_diff(want.splitlines(), got.splitlines(),
                                                 "golden", "on disk", lineterm="", n=1)))

    for driver, want in sorted(goldens["head_quant"].items()):
        src = (HERE / driver).read_text()
        found = calls(src, "head_quant_spec")
        if not found:
            print(f"FAIL {driver}: head_quant_spec call site gone")
            bad += 1
            continue
        expr = compile(ast.Expression(found[0]), "<call>", "eval")
        got = set()
        for gran, sym in GRID:
            ns = {"head_quant_spec": head_quant_spec,
                  "args": Args(head_quant=gran, head_sym=sym)}
            got.add(json.dumps(eval(expr, ns), sort_keys=True))
        checked += 1
        if got != {json.dumps(v, sort_keys=True) for v in want}:
            bad += 1
            print(f"FAIL {driver}: head_quant_spec output set moved")

    print(f"\n{checked} checked, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
