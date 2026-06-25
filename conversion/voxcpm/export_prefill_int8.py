# Community port — NOT an Apple model.
"""Export an int8 weight-only PREFILL bundle (q=T batched) for a VoxCPM MiniCPM4 backbone.
Companion to export_backbone_int8.py (decode) — gives int8 the same fast batched prefill the fp16
path has, so int8 gets fp16-level TTFB at ~half the RAM. int8 quant matches the int8 decode KV.

  python export_prefill_int8.py --which base|res [--prefill-len 32]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.export.compression import quantize_pytorch_model  # noqa: E402
from minicpm4 import build_kv_state, load_backbone  # noqa: E402

_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
ART = HERE / "artifacts"
DT, H, CL = torch.float16, 1024, 512
INT8_CFG = {"execution_mode": "eager", "global_config": {
    "op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric",
                                 "granularity": {"type": "per_channel", "axis": 0}}},
    "op_input_spec": None, "op_output_spec": None}}


class PrefillWrap(nn.Module):
    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, inputs_embeds, k_cache, v_cache):
        return self.bb.prefill(inputs_embeds, k_cache, v_cache)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="base", choices=["base", "res"])
    ap.add_argument("--prefill-len", type=int, default=32)
    a = ap.parse_args()
    T = a.prefill_len

    snap = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM-0.5B/snapshots/*")))[-1]
    sd = torch.load(snap + "/pytorch_model.bin", map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    nl, pref, vocab = (24, "base_lm.", 73448) if a.which == "base" else (6, "residual_lm.", 0)
    bb = load_backbone(sd, pref, nl, vocab, CL, DT).to(DT).eval()
    kc, vc = build_kv_state(bb.cfg, CL, DT)

    inp = (torch.zeros(1, T, H, dtype=DT), kc.clone(), vc.clone())
    dyn = {"inputs_embeds": None, "k_cache": None, "v_cache": None}
    print(f"[{a.which}] int8 quantize PREFILL (q={T}) ...", flush=True)
    qm = quantize_pytorch_model(PrefillWrap(bb).eval(), inp, dyn, dict(INT8_CFG))

    ref = {"inputs_embeds": torch.zeros(1, T, H, dtype=DT), "k_cache": kc.clone(), "v_cache": vc.clone()}
    prog = export_to_coreai(qm, ref, dynamic_shapes=None, input_names=("inputs_embeds",),
                            output_names=("hidden",), state_names=("keyCache", "valueCache"))
    prog.optimize()
    out = ART / f"voxcpm_{a.which}_int8_prefill_t{T}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    import coreai.runtime as rt
    aim = out / f"{out.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    du = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
    print(f"saved {aim} ({du})\nDONE", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
