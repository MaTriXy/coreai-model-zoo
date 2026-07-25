"""Export the MiniCPM-V-4.6 VISION encoder (.aimodel) + self-gate vs the fp32 oracle.

Fixed single-slice grid (32×32 patches @448px → 64 visual tokens). The grid is baked as a
python constant so the window-index / argsort / bucketized pos-ids fold to constants and lower
cleanly to the Core AI GPU delegate. Reuses the parity-validated `_smoke/minicpmv46_vision.py`
math.

Modes:
  fp16    — vision ship dtype (default historically), ~1.0GB.
  int8lin — per-block-16 symmetric int8 on the SigLIP Linear layers (LayerNorm / patch-Conv2d /
            pos-Embedding stay fp16). ~0.5GB → halves the vision-encode weight bandwidth, the
            dominant VLM TTFT term. block_size 16 (not 32) because the SigLIP MLP intermediate
            (4304) is not divisible by 32.

Run (GPU; _GPU_LOCK held):
    coreai-models/.venv/bin/python ../coreai-models-community/conversion/export_minicpmv46_vision.py [fp16|int8lin]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from _paths import exports_dir, hf_snapshot, smoke_dir

sys.path.insert(0, str(smoke_dir()))
from minicpmv46_vision import MiniCPMV46Vision  # noqa: E402

import coreai.runtime as rt  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

DTYPE = torch.float16
GRID = 32
SNAP = hf_snapshot("openbmb/MiniCPM-V-4.6")
REF = str(smoke_dir() / "minicpmv46_ref.npz")
EXPORTS = exports_dir()


def vision_quant_config() -> dict:
    """int8 per-block-16 symmetric on Linear weights; norms/conv/embedding stay fp16.

    block_size 16 (not the LLM's 32) because the SigLIP MLP intermediate dim 4304 is not
    divisible by 32 (4304/16 = 269). All other Linear in_features (1152, 4608, 17216) divide 16.
    """
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": "int8", "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 16, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            # keep these fp16 — only nn.Linear gets quantized
            "torch.nn.modules.normalization.LayerNorm": None,
            "torch.nn.modules.conv.Conv2d": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
    }


class VisionExport(nn.Module):
    """pixel_values [1,3,448,448] -> image_features [64,1024], grid baked to 32."""

    def __init__(self, core: MiniCPMV46Vision):
        super().__init__()
        self.core = core

    def forward(self, pixel_values):
        x, gh2, gw2 = self.core.vision_tower(pixel_values, GRID, GRID)  # post-insert grid 16
        return self.core.merger(x, gh2, gw2)                             # [64,1024]


def load_vision(model: MiniCPMV46Vision) -> None:
    sd = {}
    with safe_open(glob.glob(SNAP + "/model.safetensors")[0], framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith("model.vision_tower.") or k.startswith("model.merger."):
                sd[k[len("model."):]] = f.get_tensor(k).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)


async def gate(oracle, out_dir: Path) -> bool:
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"[gate] loading {aimodel.name} on GPU ...", flush=True)
    m = await rt.AIModel.load(
        str(aimodel),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    z = np.load(REF)
    pv = z["pixel_values"].astype(np.float16)
    res = await asyncio.wait_for(fn(inputs={"pixel_values": rt.NDArray(np.ascontiguousarray(pv))}),
                                 timeout=300)
    feats = torch.from_numpy(res["image_features"].numpy().astype(np.float32))
    o = torch.from_numpy(oracle)
    pertok = torch.nn.functional.cosine_similarity(feats, o, dim=-1)
    print(f"[gate] shape {tuple(feats.shape)} per-token cos mean {pertok.mean():.5f} "
          f"min {pertok.min():.5f} maxabs {(feats - o).abs().max():.4f}")
    return pertok.mean().item() > 0.99 and pertok.min().item() > 0.98


def main() -> None:
    ap = argparse.ArgumentParser()
    # default int8lin = the shipped config (~0.6 GB, half fp16; cos 0.9998). fp16 = exact reference.
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "int8lin"])
    args = ap.parse_args()

    out_dir = EXPORTS / ("minicpmv46_vision" if args.mode == "fp16"
                         else f"minicpmv46_vision_{args.mode}")

    core = MiniCPMV46Vision().to(DTYPE).eval()
    load_vision(core)
    core.bake_constants(GRID)  # constant pos-ids / window-index / inverse → no bucketize/argsort in graph
    model = VisionExport(core).eval()

    z = np.load(REF)
    pv = torch.from_numpy(z["pixel_values"]).to(DTYPE)

    # eager sanity (export-wrapper math == oracle)
    with torch.no_grad():
        eager = model(pv).float()
    o = z["image_features"]
    c = torch.nn.functional.cosine_similarity(eager, torch.from_numpy(o), dim=-1).mean().item()
    print(f"[eager] export-wrapper cos vs oracle {c:.5f} (fp16)")

    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("[quant] linear int8 per-block-16 (norms/conv/embed stay fp16) ...", flush=True)
        model = quantize_pytorch_model(model, (pv,), None, vision_quant_config())

    print(f"[export] vision ({args.mode}) -> Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model, {"pixel_values": pv}, dynamic_shapes=None,
        input_names=("pixel_values",), output_names=("image_features",),
        state_names=None, externalize_modules=[])
    prog.optimize()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"[save] {aimodel}", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    meta = {
        "metadata_version": "0.2", "kind": "vision-encoder", "name": out_dir.name,
        "assets": {"main": f"{out_dir.name}.aimodel"},
        "vision": {"input": "pixel_values[1,3,448,448]", "output": "image_features[64,1024]",
                   "grid": GRID, "dtype": args.mode},
        "source": {"model_definition": "torch", "hf_model_id": "openbmb/MiniCPM-V-4.6"},
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    ok = asyncio.run(gate(o, out_dir))
    print(f"\n{'✅ PASS' if ok else '❌ FAIL'} — vision .aimodel ({args.mode}) "
          f"{'matches' if ok else 'DIVERGES from'} oracle")


if __name__ == "__main__":
    main()
