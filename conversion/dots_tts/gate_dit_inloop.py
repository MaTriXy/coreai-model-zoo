"""In-loop gate for the DiT overlay (stage d, strengthened).

gate_dit.py checks only the FIRST DiT call. This drives the real upstream generate()
and, on EVERY velocity_field_predictor call (all solver steps, all patches, both CFG
branches), runs the DiTOverlay on the identical inputs and records the cosine. The
minimum cosine over the whole run is the true drop-in fidelity of the overlay inside
the acoustic loop.

Run (repo root):
  W=/private/tmp/.../scratchpad/dots_tts
  PYTHONPATH="$W/_shims:$W/dots.tts/src" $W/venv/bin/python \
      conversion/dots_tts/gate_dit_inloop.py --src $W/weights/dots.tts-soar --mode flow_matching --num-steps 10
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", default="flow_matching", choices=["flow_matching", "meanflow"])
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--text", default="Hello from Core A I.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = json.loads((Path(args.src) / "config.json").read_text())["DiT"]
    sd = {k: v for k, v in load_file(str(Path(args.src) / "model.safetensors")).items()
          if k.startswith("velocity_field_predictor.")}
    overlay = DiTOverlay(cfg, mode=args.mode).to(torch.float32).eval()
    overlay.load_upstream(sd)

    torch.manual_seed(args.seed)
    from dots_tts.runtime import DotsTtsRuntime
    rt = DotsTtsRuntime.from_pretrained(args.src, precision="float32")
    dit = rt.model.core.velocity_field_predictor
    _orig = dit.forward

    stats = {"n": 0, "min_cos": 1.0, "max_err": 0.0}

    def spy(x, timesteps, duration=None, mask=None, attn_mask=None, g_cond=None, **kw):
        out = _orig(x=x, timesteps=timesteps, duration=duration, mask=mask,
                    attn_mask=attn_mask, g_cond=g_cond, **kw)
        with torch.no_grad():
            ov = overlay(x=x, timesteps=timesteps, attn_mask=attn_mask,
                         pos_ids=kw.get("pos_ids"), g_cond=g_cond, duration=duration)
        c = cos(ov.numpy(), out.detach().numpy())
        stats["n"] += 1
        stats["min_cos"] = min(stats["min_cos"], c)
        stats["max_err"] = max(stats["max_err"], float(np.abs(ov.numpy() - out.detach().numpy()).max()))
        return out

    dit.forward = spy
    rt.generate(text=args.text, num_steps=args.num_steps)
    dit.forward = _orig

    print(f"=== GATE d (in-loop): DiT overlay across the whole {args.mode} run ===")
    print(f"  DiT calls = {stats['n']}   min cos = {stats['min_cos']:.6f}   max|err| = {stats['max_err']:.3e}")
    ok = stats["min_cos"] >= 0.999
    print("  RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
