# Community port — NOT an Apple model.
"""Torch parity gate for the VoxCPM2 AudioVAE decoder (48kHz, 1920x, SR-conditioned).

My exportable overlay (`audio_vae_v2.AudioVAEDecoderV2`, weight_norm folded, SR-cond baked for the fixed
48kHz output) vs the OFFICIAL `voxref.modules.audiovae.AudioVAEV2` decoder (loaded from audiovae.pth,
use_noise_block=False = deterministic). Identical random latents -> compare the waveform.

Pass = cos >= 0.999 (deterministic vocoder; v1's was bit-exact 1.000001).

  coreai-models/.venv/bin/python gate_v2_vae_torch.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref_v2"))

from audio_vae_v2 import load_audio_vae_v2, OUT_SR  # noqa: E402
from voxref.modules.audiovae import AudioVAEV2, AudioVAEConfigV2  # noqa: E402

DTYPE = torch.float32


def snap() -> str:
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    vae_sd = torch.load(snap() + "/audiovae.pth", map_location="cpu", weights_only=True)
    vae_sd = vae_sd.get("state_dict", vae_sd)

    # official decoder (config from config.json's audio_vae_config = defaults that match)
    off = AudioVAEV2(config=AudioVAEConfigV2()).to(DTYPE).eval()
    miss, unexp = off.load_state_dict(
        {k[len("audio_vae."):] if k.startswith("audio_vae.") else k: v for k, v in vae_sd.items()},
        strict=False,
    )
    miss = [m for m in miss if "encoder." not in m]  # we only exercise the decoder
    if miss:
        raise RuntimeError(f"official vae decoder unloaded: {miss[:6]}")

    my = load_audio_vae_v2(vae_sd)

    torch.manual_seed(0)
    T = 8
    z = torch.randn(1, 64, T, dtype=DTYPE)

    with torch.inference_mode():
        o = off.decode(z.clone())                       # [1,1,1920T], sr_cond defaults to 48000
        m = my(z.clone())
    c = cos(m, o)
    md = (m - o).abs().max().item()
    exp_len = T * int(np.prod([8, 6, 5, 2, 2, 2]))
    print(f"[vae] out_sr={OUT_SR}  off={tuple(o.shape)} my={tuple(m.shape)}  (expect len {exp_len})")
    print(f">>> vae wav cos={c:.6f}  max|Δ|={md:.6e}  ->  {'GATE PASS' if c >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if c >= 0.999 else 1)


if __name__ == "__main__":
    main()
