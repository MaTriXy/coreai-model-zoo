"""Gate the DiT flow head overlay vs the torch oracle (stage d of the ladder).

Replays the exact DiT inputs the oracle captured (x, timesteps, attn_mask, pos_ids,
g_cond, +duration for meanflow) through DiTOverlay and asserts cosine >= 0.999 on the
velocity output vs dit.out0. Runs both the soar (flow_matching, CFG batch-2) and the
mf (meanflow, batch-1, +duration) checkpoints.

Run (repo root):
  W=/private/tmp/.../scratchpad/dots_tts
  $W/venv/bin/python conversion/dots_tts/gate_dit.py --soar $W/weights/dots.tts-soar --mf $W/weights/dots.tts-mf
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

from torch_overlays import DiTOverlay


def cos(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def t(z, k):
    return torch.from_numpy(z[k]).to(torch.float32)


def run_one(src, npz_path, mode, artifacts):
    cfg = json.loads((Path(src) / "config.json").read_text())["DiT"]
    z = np.load(Path(artifacts) / npz_path)
    sd = {k: v for k, v in load_file(str(Path(src) / "model.safetensors")).items()
          if k.startswith("velocity_field_predictor.")}
    m = DiTOverlay(cfg, mode=mode).to(torch.float32).eval()
    m.load_upstream(sd)
    kw = dict(
        x=t(z, "dit.kw_x"),
        timesteps=t(z, "dit.kw_timesteps"),
        attn_mask=t(z, "dit.kw_attn_mask"),
        pos_ids=t(z, "dit.kw_pos_ids"),
        g_cond=t(z, "dit.kw_g_cond"),
    )
    if "dit.kw_duration" in z.files:
        kw["duration"] = t(z, "dit.kw_duration")
    with torch.no_grad():
        out = m(**kw)
    c = cos(out.numpy(), z["dit.out0"])
    mae = float(np.abs(out.numpy() - z["dit.out0"]).max())
    print(f"  [{mode:13s}] x{tuple(kw['x'].shape)} -> v{tuple(out.shape)}  "
          f"cos={c:.6f}  max|err|={mae:.3e}  {'PASS ✅' if c >= 0.999 else 'FAIL ❌'}")
    return c >= 0.999


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soar", required=True)
    ap.add_argument("--mf", default=None)
    ap.add_argument("--artifacts", default="conversion/dots_tts/artifacts")
    args = ap.parse_args()
    print("=== GATE d: DiT flow head ===")
    ok = run_one(args.soar, "oracle_ref.npz", "flow_matching", args.artifacts)
    if args.mf:
        ok = run_one(args.mf, "oracle_ref_mf.npz", "meanflow", args.artifacts) and ok
    print("  RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
