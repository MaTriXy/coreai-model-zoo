# Community port — NOT an Apple model.
"""Dump the VoxCPM2 (2B) host-side glue weights the Swift host runs outside the engine: the token-embed
table (prefill lookup) + per-frame projections / FSQ / stop-head / the NEW fusion_concat_proj.

v2 deltas vs v1 (export_host_glue.py): hidden 2048 (v1 1024); FSQ latent 512 (v1 256); NEW
`fusion_concat_proj` [2048,4096] (residual input = fusion(cat(fsq(lm_h), curr_embed)); v1 just added);
`enc_to_lm_proj` is FOLDED into the feat_encoder bundle (FeatEncWrap), so it's NOT dumped here.

Swift host per frame: dit = cat(lm2dit(lm_h), res2dit(res_h)) [2048]; FSQ = out(round(tanh(in(h))*9)/9);
stop = argmax(stop_head(silu(stop_proj(lm_h)))); res_in = fusion(cat(fsq(lm_h), curr_embed)).

Output: artifacts/voxcpm2_host_glue/{manifest.json, <tensor>.bin} (embed fp16, matmuls fp32).

  python export_host_glue_v2.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch

ART = Path(__file__).resolve().parent / "artifacts"

SPECS = [
    ("base_lm.embed_tokens.weight", "embed_tokens", np.float16),   # [73448,2048] prefill-only, big
    ("lm_to_dit_proj.weight", "lm_to_dit_w", np.float32),          # [1024,2048]
    ("lm_to_dit_proj.bias", "lm_to_dit_b", np.float32),
    ("res_to_dit_proj.weight", "res_to_dit_w", np.float32),        # [1024,2048]
    ("res_to_dit_proj.bias", "res_to_dit_b", np.float32),
    ("fusion_concat_proj.weight", "fusion_w", np.float32),         # [2048,4096]  NEW vs v1
    ("fusion_concat_proj.bias", "fusion_b", np.float32),
    ("fsq_layer.in_proj.weight", "fsq_in_w", np.float32),          # [512,2048]
    ("fsq_layer.in_proj.bias", "fsq_in_b", np.float32),
    ("fsq_layer.out_proj.weight", "fsq_out_w", np.float32),        # [2048,512]
    ("fsq_layer.out_proj.bias", "fsq_out_b", np.float32),
    ("stop_proj.weight", "stop_proj_w", np.float32),               # [2048,2048]
    ("stop_proj.bias", "stop_proj_b", np.float32),
    ("stop_head.weight", "stop_head_w", np.float32),               # [2,2048]
]


def main():
    from safetensors.torch import load_file
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]
    sd = load_file(snap + "/model.safetensors")

    out = ART / "voxcpm2_host_glue"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for key, name, dt in SPECS:
        t = np.ascontiguousarray(sd[key].to(torch.float32).numpy().astype(dt))
        (out / f"{name}.bin").write_bytes(t.tobytes())
        manifest[name] = {"dtype": "float16" if dt == np.float16 else "float32",
                          "shape": list(t.shape), "bytes": int(t.nbytes)}
        print(f"  {name:14s} {str(list(t.shape)):16s} {manifest[name]['dtype']:8s} {t.nbytes/1e6:7.2f} MB")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(v["bytes"] for v in manifest.values())
    print(f"-> {out}  ({total/1e6:.1f} MB total, {len(manifest)} tensors)")


if __name__ == "__main__":
    main()
