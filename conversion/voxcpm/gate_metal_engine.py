# Community port — NOT an Apple model.
"""Bit-exactness gate: run the STOCK fp16 decode bundle vs the fp16-METAL-kernel decode bundle on the
Core AI engine with identical seeded inputs + identically-grown KV, compare hidden states per step.
The kernel is fp32-accumulating F.linear, so it must match the stock matmul to fp16 precision.

  python gate_metal_engine.py [--which base|res] [--steps 12]
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parent / "artifacts"
NKV, HD, CL, H = 2, 64, 512, 1024


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


async def run(model_dir, steps, seed):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = await rt.AIModel.load(str(model_dir / f"{model_dir.name}.aimodel"), gpu)
    fn = m.load_function("main")
    NL = 24 if "base" in model_dir.name else 6
    state = {
        "keyCache": rt.NDArray(np.zeros((NL, 1, NKV, CL, HD), dtype=np.float16)),
        "valueCache": rt.NDArray(np.zeros((NL, 1, NKV, CL, HD), dtype=np.float16)),
    }
    rng = np.random.default_rng(seed)
    outs = []
    for i in range(steps):
        e = rng.standard_normal((1, 1, H)).astype(np.float16) * 0.3
        p = np.array([i], dtype=np.int32)
        res = await fn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(e)),
                               "pos": rt.NDArray(np.ascontiguousarray(p))}, state=state)
        outs.append(res["hidden"].numpy().copy())
    return outs


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "res"])
    ap.add_argument("--steps", type=int, default=12)
    a = ap.parse_args()
    stock = ART / f"voxcpm_{a.which}_fp16_decode_cl{CL}"
    metal = ART / f"voxcpm_{a.which}_fp16metal_decode_cl{CL}"
    so = await run(stock, a.steps, seed=0)
    mo = await run(metal, a.steps, seed=0)
    cs = [cos(s, m) for s, m in zip(so, mo)]
    for i, c in enumerate(cs):
        print(f"  decode[{i:2d}] stock vs metal cos={c:.6f} {'OK' if c >= 0.999 else 'FAIL'}")
    lo = min(cs)
    print(f"\n>>> {a.which}: min cos(stock, metal) = {lo:.6f} -> {'BIT-EXACT (PASS)' if lo >= 0.999 else 'FAIL'}")
    sys.exit(0 if lo >= 0.999 else 1)


if __name__ == "__main__":
    asyncio.run(main())
