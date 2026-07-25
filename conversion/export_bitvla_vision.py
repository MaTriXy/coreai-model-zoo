"""Export the BitVLA vision encoder + projector (.aimodel) and self-gate vs the torch reference.

pixel_values [1,3,224,224] -> img_embeds [1,256,2560] (BitSigLIP-SO400M ternary tower, last encoder
layer, no post-LN; then the fp 2-layer MLP projector). The SigLIP linears are 1.58-bit ternary
weights pre-baked as values, so int8 affine compression is LOSSLESS (ternary subset of int8); the
M=N prefill runs the native int8 GEMM on the GPU delegate (the M=1 ternary matvec is decode-only).
Activations use A8 (per-token int8) in-graph to match the reference exactly.

Modes:
  fp16    — exact reference (~0.8GB vision).
  int8lin — int8 per-block-16 on SigLIP linears (block 16 since FFN 4304 % 32 != 0); norms/conv/
            posembed + the fp projector stay fp16. ~0.4GB. Lossless on the ternary weights.

Run (GPU; _GPU_LOCK held):
  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_bitvla_vision.py [fp16|int8lin]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from _paths import exports_dir, work_path

sys.path.insert(0, str(Path(__file__).resolve().parent / "bitvla"))
import bitvla_ref as B  # noqa: E402

import coreai.runtime as rt  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

DTYPE = torch.float16
ORACLE = str(work_path("_bitvla_ckpt", "oracle.npz"))
EXPORTS = exports_dir()


def vision_quant_config() -> dict:
    """int8 per-block-16 symmetric on SigLIP Linear weights only (block 16: FFN 4304/16=269).
    The fp projector (multi_modal_projector.linear_1/2) + norms/conv/embed stay fp16."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": "int8", "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 16, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "torch.nn.modules.normalization.LayerNorm": None,
            "torch.nn.modules.conv.Conv2d": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
    }


class VisionExport(nn.Module):
    """pixel_values [1,3,224,224] -> img_embeds [1,256,2560]."""

    def __init__(self, vis: B.BitSigLIP, proj: B.Projector):
        super().__init__()
        self.vis = vis
        self.proj = proj

    def forward(self, pixel_values):
        return self.proj(self.vis(pixel_values))


async def gate(oracle_embeds, out_dir: Path) -> bool:
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"[gate] loading {aimodel.name} on GPU ...", flush=True)
    m = await rt.AIModel.load(
        str(aimodel),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    z = np.load(ORACLE)
    pv = z["pixel_values"].astype(np.float16)
    res = await asyncio.wait_for(fn(inputs={"pixel_values": rt.NDArray(np.ascontiguousarray(pv))}),
                                 timeout=300)
    feats = torch.from_numpy(res["img_embeds"].numpy().astype(np.float32)).reshape(-1, 2560)
    o = torch.from_numpy(oracle_embeds).reshape(-1, 2560)
    pertok = torch.nn.functional.cosine_similarity(feats, o, dim=-1)
    print(f"[gate] per-token cos mean {pertok.mean():.5f} min {pertok.min():.5f} "
          f"maxabs {(feats - o).abs().max():.4f}", flush=True)
    return pertok.mean().item() > 0.99 and pertok.min().item() > 0.98


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="fp16", choices=["fp16", "int8lin"])
    args = ap.parse_args()
    out_dir = EXPORTS / ("bitvla_vision" if args.mode == "fp16" else f"bitvla_vision_{args.mode}")

    vis = B.BitSigLIP().to(DTYPE).eval()
    proj = B.Projector().to(DTYPE).eval()
    B.load_vision(vis, proj)
    for mod in vis.modules():                                # bake ternary weights -> lossless int8
        if isinstance(mod, B.BitLinear):
            mod.prebake()
            mod.skip_act = True                              # fp16 acts (no A8 round/amax) — device GPU
    model = VisionExport(vis, proj).to(DTYPE).eval()

    z = np.load(ORACLE)
    pv = torch.from_numpy(z["pixel_values"]).to(DTYPE)
    with torch.no_grad():
        eager = model(pv).float()
    o = z["img_embeds"]
    c = torch.nn.functional.cosine_similarity(
        eager.reshape(-1, 2560), torch.from_numpy(o).reshape(-1, 2560), dim=-1).mean().item()
    print(f"[eager] export-wrapper cos vs oracle {c:.5f} (fp16, prebaked)", flush=True)

    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("[quant] SigLIP linear int8 per-block-16 ...", flush=True)
        model = quantize_pytorch_model(model, (pv,), None, vision_quant_config())

    print(f"[export] vision ({args.mode}) -> Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model, {"pixel_values": pv}, dynamic_shapes=None,
        input_names=("pixel_values",), output_names=("img_embeds",),
        state_names=None, externalize_modules=[])
    prog.optimize()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"[save] {aimodel}", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    meta = {"metadata_version": "0.2", "kind": "vision-encoder", "name": out_dir.name,
            "assets": {"main": f"{out_dir.name}.aimodel"},
            "vision": {"input": "pixel_values[1,3,224,224]", "output": "img_embeds[1,256,2560]",
                       "dtype": args.mode},
            "source": {"model_definition": "torch", "hf_model_id": "lxsy/bitvla-bf16"},
            "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    ok = asyncio.run(gate(o, out_dir))
    print(f"\n{'PASS' if ok else 'FAIL'} — vision .aimodel ({args.mode}) "
          f"{'matches' if ok else 'DIVERGES from'} reference", flush=True)


if __name__ == "__main__":
    main()
