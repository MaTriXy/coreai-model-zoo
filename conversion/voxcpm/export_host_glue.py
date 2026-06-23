# Community port — NOT an Apple model.
"""Dump the small host-side glue weights the Swift/CoreAIKit VoxCPM host runs outside the engine:
the token-embedding table (prefill lookup) + the per-frame projections/FSQ/stop-head matmuls.

Output: artifacts/voxcpm_host_glue/{manifest.json, <tensor>.bin}. Raw little-endian; embed is fp16
(150MB, prefill-only), the per-frame matmul weights are fp32 (for Accelerate cblas). The Swift host
mmaps these and runs: dit = lm2dit(lm_h)+res2dit(res_h); FSQ = out_proj(round(tanh(in_proj(h))*9)/9);
stop = argmax(stop_head(silu(stop_proj(lm_h)))). enc_to_lm_proj is folded into the feat_encoder bundle.

  python export_host_glue.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parent / "artifacts"

# (state_dict key, output name, numpy dtype). embed fp16 (prefill-only, big); matmuls fp32 (Accelerate).
SPECS = [
    ("base_lm.embed_tokens.weight", "embed_tokens", np.float16),
    ("lm_to_dit_proj.weight", "lm_to_dit_w", np.float32),
    ("lm_to_dit_proj.bias", "lm_to_dit_b", np.float32),
    ("res_to_dit_proj.weight", "res_to_dit_w", np.float32),
    ("res_to_dit_proj.bias", "res_to_dit_b", np.float32),
    ("fsq_layer.in_proj.weight", "fsq_in_w", np.float32),
    ("fsq_layer.in_proj.bias", "fsq_in_b", np.float32),
    ("fsq_layer.out_proj.weight", "fsq_out_w", np.float32),
    ("fsq_layer.out_proj.bias", "fsq_out_b", np.float32),
    ("stop_proj.weight", "stop_proj_w", np.float32),
    ("stop_proj.bias", "stop_proj_b", np.float32),
    ("stop_head.weight", "stop_head_w", np.float32),
]


def main():
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)

    out = ART / "voxcpm_host_glue"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, name, dt in SPECS:
        t = sd[key].to(torch.float32).numpy().astype(dt)
        t = np.ascontiguousarray(t)
        (out / f"{name}.bin").write_bytes(t.tobytes())
        manifest[name] = {"dtype": "float16" if dt == np.float16 else "float32",
                          "shape": list(t.shape), "bytes": int(t.nbytes)}
        print(f"  {name:14s} {str(list(t.shape)):16s} {manifest[name]['dtype']:8s} {t.nbytes/1e6:7.2f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(v["bytes"] for v in manifest.values())
    print(f"-> {out}  ({total/1e6:.1f} MB total, {len(manifest)} tensors)")


if __name__ == "__main__":
    main()
