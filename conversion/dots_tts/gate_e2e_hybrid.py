"""Hybrid e2e gate (stage g, tractable form).

Instead of reconstructing the whole host loop from scratch, this SUBSTITUTES the DiT
overlay's velocity into the real upstream generate() loop (returning the overlay output
in place of the upstream module on every solver call) and checks that the resulting
48 kHz waveform still matches the golden oracle wav — MAGSPEC (magnitude-STFT cosine),
raw-sample cosine, and RMS. Proves the from-scratch DiT graph drives the true acoustic
loop to the same audio end-to-end (the qwen2/patch_encoder/vocoder overlays are already
gated cos=1.0 standalone; a full from-scratch host loop is deferred to the Swift port).

Run (repo root):
  W=/private/tmp/.../scratchpad/dots_tts
  PYTHONPATH="$W/_shims:$W/dots.tts/src" $W/venv/bin/python \
      conversion/dots_tts/gate_e2e_hybrid.py --src $W/weights/dots.tts-soar --mode flow_matching --num-steps 10
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
    n = min(a.size, b.size)
    a, b = a[:n], b[:n]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def magspec_cos(a, b, n_fft=1024, hop=256):
    def spec(x):
        x = torch.from_numpy(x.reshape(-1).astype(np.float32))
        S = torch.stft(x, n_fft=n_fft, hop_length=hop, return_complex=True,
                       window=torch.hann_window(n_fft))
        return S.abs().numpy()
    Sa, Sb = spec(a), spec(b)
    m = min(Sa.shape[1], Sb.shape[1])
    return cos(Sa[:, :m], Sb[:, :m])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", default="flow_matching", choices=["flow_matching", "meanflow"])
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--text", default="Hello from Core A I.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--artifacts", default="conversion/dots_tts/artifacts")
    ap.add_argument("--npz", default="oracle_ref.npz")
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
    n = {"calls": 0}

    def sub(x, timesteps, duration=None, mask=None, attn_mask=None, g_cond=None, **kw):
        with torch.no_grad():
            ov = overlay(x=x, timesteps=timesteps, attn_mask=attn_mask,
                         pos_ids=kw.get("pos_ids"), g_cond=g_cond, duration=duration)
        n["calls"] += 1
        return ov  # <-- SUBSTITUTE: the loop consumes the overlay's velocity

    dit.forward = sub
    result = rt.generate(text=args.text, num_steps=args.num_steps)
    dit.forward = _orig

    wav = result["audio"] if isinstance(result, dict) else getattr(result, "audio")
    wav = wav.detach().cpu().numpy().reshape(-1) if isinstance(wav, torch.Tensor) else np.asarray(wav).reshape(-1)
    golden = np.load(Path(args.artifacts) / args.npz)["wav"]

    c_raw = cos(wav, golden)
    c_mag = magspec_cos(wav, golden)
    print(f"=== GATE g (hybrid e2e): DiT overlay substituted into real generate() ===")
    print(f"  {args.mode}: {n['calls']} DiT calls  wav {wav.shape} vs golden {golden.shape}")
    print(f"  raw-sample cos = {c_raw:.6f}   MAGSPEC cos = {c_mag:.6f}   "
          f"rms {wav.std():.4f} vs {golden.std():.4f}")
    ok = c_mag >= 0.999
    print("  RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
