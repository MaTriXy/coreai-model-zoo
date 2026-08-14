#!/usr/bin/env python3
"""Does the exported Shieldstral classifier reach the same verdicts as fp32?

One forward per case, straight through the `.aimodel`, against the oracle in
`shieldstral_3b_suite_ref.npz`. Two things are measured and they are not the same
question:

* **sides** — does each case land on the correct side of 0.5? This is the product
  gate. A moderation model that flips a verdict under compression is broken in the
  only way users can see.
* **|dP|** — how far the probability moved. A verdict that survives at 0.51 is not
  the same artifact as one that survives at 0.99, and only this number says which.

The near-miss cases (a weapons *safety* question, a refusal to dox, help-seeking)
are where compression shows up first, so they are printed with everything else
rather than summarized away.

Run (conversion venv, GPU, _GPU_LOCK held):
    ../coreai-models/.venv/bin/python _smoke/test_shieldstral_aimodel_gate.py \
        --mode int8lin --bench 10
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import exports_dir, hf_snapshot  # noqa: E402
from _shieldstral_suite import SUITE  # noqa: E402
from export_shieldstral import render_prompt  # noqa: E402

DEFAULT_REF = Path(__file__).parent / "shieldstral_3b_suite_ref.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="mistralai/Shieldstral-1.0-3B")
    ap.add_argument("--mode", default="int8lin")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--exports", default=None)
    ap.add_argument("--asset", default=None, help="explicit .aimodel/.aimodelc path")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--bench", type=int, default=0, help="timed forwards after the gate")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    snap = Path(hf_snapshot(args.hf_id))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    pad_id = int(json.loads((snap / "config.json").read_text())["text_config"]["pad_token_id"])

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_classify_{args.mode}_s{args.seq_len}"
    root = Path(args.exports) if args.exports else exports_dir()
    if args.asset:
        asset = Path(args.asset)
    else:
        aotc = root / f"{name}_aotc" / f"{name}.h16c.aimodelc"
        asset = aotc if aotc.exists() else root / name / f"{name}.aimodel"

    ref = np.load(args.ref, allow_pickle=True)
    t0 = time.perf_counter()
    model = await maybe_await(rt.AIModel.load(str(asset), rt.SpecializationOptions.default()))
    fn = await maybe_await(model.load_function(model.function_names[0]))
    print(f"{asset.name} ready in {time.perf_counter() - t0:.1f}s\n")

    print(f"{'case':26s} {'P(fp32)':>8s} {'P(bundle)':>10s} {'|dP|':>8s}  verdict")
    sides_ok = worst = 0
    for i, (instr, query, doc, want_unsafe, label) in enumerate(SUITE):
        ids = tok.encode(render_prompt(instr, query, doc), add_special_tokens=False).ids
        real = len(ids)
        padded = np.array([ids + [pad_id] * (args.seq_len - real)], dtype=np.int32)
        mask = np.array([[1] * real + [0] * (args.seq_len - real)], dtype=np.int32)
        out = await maybe_await(fn(inputs={
            "input_ids": rt.NDArray(padded), "attention_mask": rt.NDArray(mask)}))
        p = float(np.asarray(out["probs"].numpy())[0, 1])
        p_ref = float(ref[f"case{i}_p"])
        side = p > 0.5
        ok = side == want_unsafe
        sides_ok += ok
        worst = max(worst, abs(p - p_ref))
        print(f"{label:26s} {p_ref:8.4f} {p:10.4f} {abs(p - p_ref):8.5f}  "
              f"{'UNSAFE' if side else 'safe':6s} {'OK' if ok else '** WRONG SIDE **'}")

    print(f"\n{sides_ok}/{len(SUITE)} verdicts match fp32 | worst |dP| {worst:.5f}")

    if args.bench:
        ids = tok.encode(render_prompt(*SUITE[0][:3]), add_special_tokens=False).ids
        padded = np.array([ids + [pad_id] * (args.seq_len - len(ids))], dtype=np.int32)
        mask = np.array([[1] * len(ids) + [0] * (args.seq_len - len(ids))], dtype=np.int32)
        inputs = {"input_ids": rt.NDArray(padded), "attention_mask": rt.NDArray(mask)}
        await maybe_await(fn(inputs=inputs))  # warm
        ts = []
        for _ in range(args.bench):
            t = time.perf_counter()
            await maybe_await(fn(inputs=inputs))
            ts.append((time.perf_counter() - t) * 1e3)
        print(f"verdict latency: median {np.median(ts):.1f} ms "
              f"(min {min(ts):.1f} / max {max(ts):.1f}, n={args.bench}, S={args.seq_len})")

    return 0 if sides_ok == len(SUITE) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
