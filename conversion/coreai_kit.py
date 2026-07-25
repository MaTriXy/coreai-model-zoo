"""coreai_kit — minimal helpers to convert PyTorch models to Core AI `.aimodel`
bundles and verify them against torch eager.

Verified on macOS 26.0 / Python 3.12 with coreai-torch 0.4.0 + coreai-core 1.0.0b1.
Conversion AND the Python runtime both run on macOS 26 (a toy model round-trips to
~9e-8 vs torch); only on-device iOS / Swift + the AOT `coreai-build` step need
Xcode 27 / iOS 27.

API quirks codified here so callers don't re-discover them:
  - save_asset() wants a pathlib.Path (not str) and refuses to overwrite -> rmtree first.
  - AIModel.load(...) and InferenceFunction.__call__(...) are async; load_function() is sync.
  - model.function_names is a list attribute, not a method.
"""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import numpy as np
import torch

from coreai_torch import TorchConverter, get_decomp_table
import coreai.runtime as rt


def convert(model, example_inputs, input_names, output_names, out_path, *, optimize=True):
    """Export `model` and convert it to a Core AI `.aimodel` bundle at `out_path`.

    example_inputs: tuple matching forward(*example_inputs).
    Returns the output Path.
    """
    model = model.eval()
    ep = torch.export.export(model, tuple(example_inputs)).run_decompositions(get_decomp_table())
    prog = (
        TorchConverter()
        .add_exported_program(
            exported_program=ep,
            input_names=list(input_names),
            output_names=list(output_names),
        )
        .to_coreai()
    )
    if optimize:
        prog.optimize()
    out_path = Path(out_path)
    shutil.rmtree(out_path, ignore_errors=True)  # save_asset refuses to overwrite an existing bundle
    prog.save_asset(out_path, rt.AIModelAssetMetadata())
    return out_path


async def _run_async(path, feed, compute, function):
    opt = rt.SpecializationOptions.cpu_only() if compute == "cpu" else None
    model = await rt.AIModel.load(Path(path), opt)
    fn = model.load_function(function)  # sync
    feed_nd = {k: rt.NDArray(np.asarray(v)) for k, v in feed.items()}
    res = await fn(feed_nd)  # InferenceFunction.__call__ is async -> dict[str, NDArray]
    return {k: v.numpy() for k, v in res.items()}


def run(path, feed, *, compute="cpu", function="main"):
    """Run a `.aimodel`. feed: {input_name: array}. Returns {output_name: np.ndarray}.

    compute: "cpu" (deterministic, cpu_only) or "default" (runtime picks the unit).
    """
    return asyncio.run(_run_async(path, feed, compute, function))


def verify(model, example_inputs, input_names, output_names, out_path, *, compute="cpu", atol=1e-3):
    """Convert, run on Core AI, and compare to torch eager.

    Returns (ok: bool, maxdiff: float, outputs: dict[str, np.ndarray]).
    """
    model = model.eval()
    with torch.no_grad():
        ref = model(*example_inputs)
    refs = list(ref) if isinstance(ref, (list, tuple)) else [ref]
    refs = [r.detach().numpy() for r in refs]

    convert(model, example_inputs, input_names, output_names, out_path)
    feed = {n: ex.numpy() for n, ex in zip(input_names, example_inputs)}
    got = run(out_path, feed, compute=compute)

    got_list = [got[n] for n in output_names]
    maxdiff = max(float(np.abs(g - r).max()) for g, r in zip(got_list, refs))
    return maxdiff <= atol, maxdiff, got


if __name__ == "__main__":
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(16, 32)
            self.l2 = nn.Linear(32, 8)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    torch.manual_seed(0)
    ok, maxdiff, _ = verify(
        Toy(), (torch.randn(1, 16),), ["x"], ["y"],
        "/Users/majimadaisuke/Code/coreai/_smoke/toy.aimodel",
    )
    print(f"self-test: ok={ok} maxdiff={maxdiff:.2e}")
