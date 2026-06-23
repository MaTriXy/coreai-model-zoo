# Community port — NOT an Apple model.
"""Export a VoxCPM MiniCPM4 backbone (base_lm or residual_lm) as TWO static Core AI bundles:
prefill (q_len=T baked) + decode (q=1, scalar pos). inputs_embeds -> hidden; shared KV state
(keyCache/valueCache) threaded host-side. Mirrors conversion/qwen3_asr/export_static.py.

SDPA externalization is DROPPED (engine-native SDPA can't take our explicit causal-buffer mask);
rope is already decomposed (baked cos/sin), so no rope op to externalize.

  python export_backbone.py [--which base|res] [--mode fp16] [--prefill-len 9] [--cache-len 512]
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
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from minicpm4 import build_kv_state, load_backbone  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
_DROP = {"scaled_dot_product_attention", "rope"}
# This build hardcodes externalize_modules=_EXTERNALIZE_SPECS; drop SDPA (explicit mask) + rope
# (baked cos/sin) at the module level so the export decomposes them. RMSNorm/GatherMM stay external.
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]


class PrefillWrap(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, inputs_embeds, k_cache, v_cache):
        return self.bb.prefill(inputs_embeds, k_cache, v_cache)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "res"])
    ap.add_argument("--mode", default="fp16", choices=["fp16"])
    ap.add_argument("--prefill-len", type=int, default=9)
    ap.add_argument("--cache-len", type=int, default=512)
    a = ap.parse_args()
    DT = torch.float16
    H = 1024

    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    ck = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = ck.get("state_dict", ck)
    if a.which == "base":
        bb = load_backbone(sd, "base_lm.", 24, 73448, a.cache_len, DT)
    else:
        bb = load_backbone(sd, "residual_lm.", 6, 0, a.cache_len, DT)
    bb = bb.to(DT).eval()

    kc, vc = build_kv_state(bb.cfg, a.cache_len, DT)
    ART.mkdir(parents=True, exist_ok=True)
    T = a.prefill_len

    # prefill bundle (q_len=T)
    print(f"[{a.which}] export PREFILL (q_len={T}) ...", flush=True)
    pref_ref = {
        "inputs_embeds": torch.zeros(1, T, H, dtype=DT),
        "k_cache": kc.clone(), "v_cache": vc.clone(),
    }
    prog_p = export_to_coreai(
        PrefillWrap(bb).eval(), pref_ref, dynamic_shapes=None,
        input_names=("inputs_embeds",), output_names=("hidden",),
        state_names=("keyCache", "valueCache"))
    pdir = ART / f"voxcpm_{a.which}_{a.mode}_prefill_t{T}"
    print(f"  -> {pdir} ({_du(_save(prog_p, pdir))})", flush=True)

    # decode bundle (q=1, scalar pos)
    print(f"[{a.which}] export DECODE (q=1, cache_len={a.cache_len}) ...", flush=True)
    dec_ref = {
        "inputs_embeds": torch.zeros(1, 1, H, dtype=DT),
        "pos": torch.tensor([T], dtype=torch.int32),
        "k_cache": kc.clone(), "v_cache": vc.clone(),
    }
    prog_d = export_to_coreai(
        DecodeWrap(bb).eval(), dec_ref, dynamic_shapes=None,
        input_names=("inputs_embeds", "pos"), output_names=("hidden",),
        state_names=("keyCache", "valueCache"))
    ddir = ART / f"voxcpm_{a.which}_{a.mode}_decode_cl{a.cache_len}"
    print(f"  -> {ddir} ({_du(_save(prog_d, ddir))})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
