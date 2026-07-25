# Community port — NOT an Apple model.
"""Torch cross-check: the static-KV patch_encoder (patch_encoder.py) must reproduce the upstream
oracle decode_patch fixture (emb + conv_tail) cos>=0.999. Closes the KV/downsample reformulation
before the engine export.

  PYTHONPATH=. <coreai-venv>/bin/python gate_patchenc_static.py --src <weights/dots.tts-soar>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from patch_encoder import build_kv_state, load_patch_encoder

ART = Path(__file__).resolve().parent / "artifacts"


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--npz", default="oracle_ref.npz")
    a = ap.parse_args()
    DT = torch.float32

    sd = load_file(str(Path(a.src) / "model.safetensors"))
    cfg_json = json.loads((Path(a.src) / "config.json").read_text())
    z = np.load(ART / a.npz)
    buf = z["patch_encoder.in_layer_caches_0_0"].shape[2]  # 1000

    m, cfg = load_patch_encoder(sd, cfg_json, buf, DT)

    latent = torch.from_numpy(z["patch_encoder.in_latent_patch"]).to(DT)
    conv_tail = torch.from_numpy(z["patch_encoder.in_conv_tail"]).to(DT)
    positions = torch.from_numpy(z["patch_encoder.in_positions"]).to(torch.int64)
    pos0 = int(positions[0].item())
    print(f"  positions={positions.tolist()} (pos0={pos0}, out_ds_rate={cfg.out_ds_rate}, buf={buf})")

    # seed the stacked KV cache from the oracle's per-layer PRE-write caches
    kc, vc = build_kv_state(cfg, buf, DT)
    for i in range(cfg.n_layers):
        kc[i, 0] = torch.from_numpy(z[f"patch_encoder.in_layer_caches_{i}_0"]).to(DT)[0]
        vc[i, 0] = torch.from_numpy(z[f"patch_encoder.in_layer_caches_{i}_1"]).to(DT)[0]

    with torch.inference_mode():
        emb, new_tail = m(latent, conv_tail, torch.tensor([pos0], dtype=torch.int32), kc, vc)

    c_emb = cos(emb.numpy(), z["patch_encoder.out_embedding"])
    c_tail = cos(new_tail.numpy(), z["patch_encoder.out_conv_tail"])
    print(f"  emb {tuple(emb.shape)} cos={c_emb:.6f}   conv_tail cos={c_tail:.6f}")
    ok = c_emb >= 0.999 and c_tail >= 0.999
    print(">>>", "PATCH-ENC STATIC GATE PASS" if ok else "PATCH-ENC STATIC GATE FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
