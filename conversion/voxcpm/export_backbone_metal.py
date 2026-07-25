# Community port — NOT an Apple model.
"""Export a VoxCPM MiniCPM4 backbone DECODE bundle (q=1) with the bandwidth-dominant Linears running
the fp16 "simd" matvec Metal kernel (lossless launch/occupancy fix; see minicpm4_metal.py).

Decode-only (the shipped path is prefill-via-decode, all q=1). Bit-exact vs the stock fp16 decode
bundle up to fp accumulation (kernel = fp32-accumulating F.linear).

  python export_backbone_metal.py [--which base|res] [--cache-len 512] [--no-attn]
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels  # noqa: E402
from minicpm4 import build_kv_state, load_backbone  # noqa: E402
from minicpm4_metal import metalize_minicpm4  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
_DROP = {"scaled_dot_product_attention", "rope"}
# Same externalization as the stock export: keep RMSNorm/GatherMM external, decompose SDPA (explicit
# mask) + rope (baked cos/sin). Pass explicitly (with-kernels defaults to () = disabled).
_EXT = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]


class DecodeWrap(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, inputs_embeds, pos, k_cache, v_cache):
        return self.bb.decode(inputs_embeds, pos, k_cache, v_cache)


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    print(f"saving {aim} ...", flush=True)
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


def _load_v2_sd():
    import json
    from safetensors.torch import load_file
    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]
    lm = json.load(open(snap + "/config.json"))["lm_config"]
    return load_file(snap + "/model.safetensors"), lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "res"])
    ap.add_argument("--variant", default="v1", choices=["v1", "v2"])
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--no-attn", action="store_true", help="metalize MLP only (skip attn q/o)")
    a = ap.parse_args()
    DT = torch.float16

    if a.variant == "v1":
        H = 1024
        snap = sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
        ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
        sd = ck.get("state_dict", ck)
        if a.which == "base":
            bb = load_backbone(sd, "base_lm.", 24, 73448, a.cache_len, DT)
        else:
            bb = load_backbone(sd, "residual_lm.", 6, 0, a.cache_len, DT)
    else:  # v2 (VoxCPM2 2B)
        sd, lm = _load_v2_sd()
        H = lm["hidden_size"]
        ov = dict(hidden_size=lm["hidden_size"], intermediate_size=lm["intermediate_size"],
                  num_attention_heads=lm["num_attention_heads"],
                  num_key_value_heads=lm["num_key_value_heads"], head_dim=lm["kv_channels"],
                  short_factor=lm["rope_scaling"]["short_factor"])
        if a.which == "base":
            bb = load_backbone(sd, "base_lm.", 28, 73448, a.cache_len, DT, **ov)
        else:
            bb = load_backbone(sd, "residual_lm.", 8, 0, a.cache_len, DT, no_rope=True, **ov)
    bb = bb.to(DT).eval()

    kernel = metalize_minicpm4(bb, do_attn=not a.no_attn)
    print(f"[{a.which}] metalized (attn={'no' if a.no_attn else 'yes'}); exporting DECODE q=1 ...", flush=True)

    kc, vc = build_kv_state(bb.cfg, a.cache_len, DT)
    dec_ref = {
        "inputs_embeds": torch.zeros(1, 1, H, dtype=DT),
        "pos": torch.tensor([0], dtype=torch.int32),
        "k_cache": kc.clone(), "v_cache": vc.clone(),
    }
    prog = export_to_coreai_with_kernels(
        DecodeWrap(bb).eval(), dec_ref, custom_kernels=[kernel],
        input_names=("inputs_embeds", "pos"), output_names=("hidden",),
        state_names=("keyCache", "valueCache"), externalize_modules=_EXT)
    pref = "voxcpm2" if a.variant == "v2" else "voxcpm"
    ddir = ART / f"{pref}_{a.which}_fp16metal_decode_cl{a.cache_len}"
    print(f"  -> {ddir} ({_du(_save(prog, ddir))})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
