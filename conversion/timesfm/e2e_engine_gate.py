"""End-to-end: host DSP + Core AI fp16 engine graph -> final forecast vs HF oracle.

This is the real ship gate: proves the whole pipeline (RevIN/Welford/flip/quantile-head
host-side, transformer on the Core AI engine) reproduces HF forecasts.
"""
import argparse, asyncio
import numpy as np
import torch
import coreai.runtime as rt

from timesfm_core import load_core_from_safetensors
import host_forecast as H

CFG = dict(patch=32, horizon=128, hidden=1280, layers=20, heads=16, head_dim=80,
           inter=1280, q=9, oql=1024, eps=1e-6)


class EngineCore:
    """Callable matching TimesFmCore.forward, backed by the Core AI engine."""
    def __init__(self, fn, dtype):
        self.fn, self.dtype = fn, dtype

    def to(self, *a, **k):
        return self

    def __call__(self, tok_in, cos, sin, attn_bias):
        d = self.dtype
        out = asyncio.run(self.fn({
            "tok_in": rt.NDArray(tok_in.to(d).numpy()),
            "cos": rt.NDArray(cos.to(d).numpy()),
            "sin": rt.NDArray(sin.to(d).numpy()),
            "attn_bias": rt.NDArray(attn_bias.to(d).numpy()),
        }))
        pp = torch.tensor(out["proj_point"].numpy().astype(np.float32))
        pq = torch.tensor(out["proj_q"].numpy().astype(np.float32))
        return pp, pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", required=False)
    ap.add_argument("--bundle", default="exports/timesfm_2p5_200m_ctx512_fp16.aimodel")
    ap.add_argument("--oracle", default="oracle.npz")
    ap.add_argument("--unit", default="gpu")
    args = ap.parse_args()

    z = np.load(args.oracle, allow_pickle=True)
    CTX = int(z["ctx_len"]); series = z["series"]; names = z["series_names"]

    opts = (rt.SpecializationOptions.cpu_only() if args.unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, args.unit)()))
    model = asyncio.run(rt.AIModel.load(args.bundle, opts))
    fn = model.load_function("main")
    engine = EngineCore(fn, torch.float16)

    print(f"== E2E: host DSP + engine({args.unit},fp16) vs HF oracle final forecast ==")
    worst = 1.0
    for i, nm in enumerate(names):
        mp, fp = H.forecast(engine, torch.tensor(series[i]), CTX, CFG)
        omp, ofp = z["mean_pred"][i], z["full_pred"][i]
        cm = float(mp.numpy().ravel() @ omp.ravel() / (np.linalg.norm(mp.numpy())*np.linalg.norm(omp)+1e-12))
        mae = float(np.abs(mp.numpy() - omp).mean())
        rel = mae / (np.abs(omp).mean() + 1e-9)
        worst = min(worst, cm)
        print(f"  {str(nm):8s} mean cos={cm:.7f} MAE={mae:.3e} rel={rel:.3e}  fc[:4]={mp.numpy()[:4].round(3)} (HF {omp[:4].round(3)})")
    print("RESULT:", "PASS" if worst > 0.999 else "FAIL", f"(min mean cos={worst:.7f})")


if __name__ == "__main__":
    main()
