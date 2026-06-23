"""Export the Unlimited-OCR DeepEncoder (.aimodel) + self-gate vs the fp32 oracle.

DeepEncoder (frozen, DeepSeek-OCR derived): image[1,3,640,640]
  -> sam_model(ImageEncoderViT) [1,1024,10,10]
  -> vision_model(VitModel/CLIP, conditioned on sam) [1,101,1024]
  -> bridge cat(clip[:,1:], sam.flatten(2).permute, -1) [1,100,2048]
  -> projector(MlpProjector 2048->1280) [1,100,1280]  = 100 visual tokens.
Fixed 640px input -> static shapes -> SAM window/rel-pos + CLIP pos-interp fold to constants.
Arrangement (image_newline / view_seperator / tiling) is Swift-side, out of graph.

Run (coreai venv):
    coreai-models/.venv/bin/python export_vision.py        # fp16 (ship)
    OCR_VDTYPE=fp32 coreai-models/.venv/bin/python export_vision.py
"""
from __future__ import annotations
import asyncio, glob, json, os, shutil, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from torch import nn

HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpt"
sys.path.insert(0, str(CKPT))
import deepencoder as _de  # noqa: E402
from deepencoder import build_sam_vit_b, build_clip_l, MlpProjector  # noqa: E402
try:
    from addict import Dict as ADict
except Exception:
    from easydict import EasyDict as ADict

# --- Bake the bicubic pos-embed interpolations to constants (fixed input size) ---
# get_abs_pos (CLIP) and get_abs_pos_sam (SAM) each run once on a frozen param with a
# constant target size -> the interpolated pos-embed is a compile-time constant. Folding
# it removes aten._upsample_bicubic2d_aa (unsupported in the Core AI dialect). Cache is
# populated by the eager pre-run, then returned as a constant during torch.export trace.
_POS_CACHE = {}
def _bake_pos(orig, name):
    def wrap(abs_pos, tgt_size):
        key = (name, int(tgt_size))
        if key not in _POS_CACHE:
            _POS_CACHE[key] = orig(abs_pos, tgt_size).detach()
        return _POS_CACHE[key]
    return wrap
_de.get_abs_pos = _bake_pos(_de.get_abs_pos, "clip")
_de.get_abs_pos_sam = _bake_pos(_de.get_abs_pos_sam, "sam")

import coreai.runtime as rt  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402

DTYPE = torch.float16 if os.environ.get("OCR_VDTYPE", "fp16") == "fp16" else torch.float32
REF = HERE / "out/_vision_oracle/vision_tensors.npz"
OUT = HERE / "out/_vision_export"


class VisionExport(nn.Module):
    """image [1,3,640,640] -> visual_tokens [1,100,1280] (DeepEncoder + bridge + projector)."""

    def __init__(self, sam, vis, proj):
        super().__init__()
        self.sam_model, self.vision_model, self.projector = sam, vis, proj

    def forward(self, image):
        sam = self.sam_model(image)                              # [1,1024,10,10]
        clip = self.vision_model(image, sam)                     # [1,101,1024]
        sam_flat = sam.flatten(2).permute(0, 2, 1)               # [1,100,1024]
        bridge = torch.cat([clip[:, 1:], sam_flat], dim=-1)      # [1,100,2048]
        return self.projector(bridge)                            # [1,100,1280]


def build() -> VisionExport:
    sam = build_sam_vit_b().to(DTYPE).eval()
    vis = build_clip_l().to(DTYPE).eval()
    proj = MlpProjector(ADict(projector_type="linear", input_dim=2048, n_embed=1280)).to(DTYPE).eval()
    st = glob.glob(str(CKPT / "model-*.safetensors"))[0]
    bk = {"model.sam_model.": {}, "model.vision_model.": {}, "model.projector.": {}}
    with safe_open(st, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            for pref, d in bk.items():
                if k.startswith(pref):
                    d[k[len(pref):]] = f.get_tensor(k).to(DTYPE); break
    sam.load_state_dict(bk["model.sam_model."], strict=False, assign=True)
    vis.load_state_dict(bk["model.vision_model."], strict=False, assign=True)
    proj.load_state_dict(bk["model.projector."], strict=False, assign=True)
    return VisionExport(sam, vis, proj).eval()


async def gate(oracle: np.ndarray, img: np.ndarray) -> bool:
    aimodel = OUT / f"{OUT.name}.aimodel"
    print(f"[gate] loading {aimodel.name} on GPU ...", flush=True)
    m = await rt.AIModel.load(
        str(aimodel),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    np_dt = np.float16 if DTYPE == torch.float16 else np.float32
    res = await asyncio.wait_for(
        fn(inputs={"image": rt.NDArray(np.ascontiguousarray(img.astype(np_dt)))}), timeout=600)
    feats = torch.from_numpy(res["visual_tokens"].numpy().astype(np.float32))
    o = torch.from_numpy(oracle.astype(np.float32))
    pertok = torch.nn.functional.cosine_similarity(feats, o, dim=-1)
    print(f"[gate] shape {tuple(feats.shape)}  per-token cos mean {pertok.mean():.5f} "
          f"min {pertok.min():.5f}  max|Δ| {(feats - o).abs().max():.4f}")
    return pertok.mean().item() > 0.99 and pertok.min().item() > 0.98


def main() -> None:
    model = build()
    z = np.load(REF)
    img = z["sam_in0"].astype(np.float32)                # [1,3,640,640]
    pv = torch.from_numpy(img).to(DTYPE)
    o = z["proj_out"]                                    # [1,100,1280] fp32 oracle

    with torch.no_grad():
        eager = model(pv).float()
    c = torch.nn.functional.cosine_similarity(eager, torch.from_numpy(o), dim=-1)
    print(f"[eager] export-wrapper vs oracle: cos mean {c.mean():.6f} min {c.min():.6f} "
          f"max|Δ| {(eager - torch.from_numpy(o)).abs().max():.3e}  (dtype={DTYPE})", flush=True)

    print("[export] DeepEncoder -> Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model, {"image": pv}, dynamic_shapes=None,
        input_names=("image",), output_names=("visual_tokens",),
        state_names=None, externalize_modules=[])
    prog.optimize()

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    aimodel = OUT / f"{OUT.name}.aimodel"
    print(f"[save] {aimodel}", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    (OUT / "metadata.json").write_text(json.dumps({
        "metadata_version": "0.2", "kind": "vision-encoder", "name": OUT.name,
        "assets": {"main": f"{OUT.name}.aimodel"},
        "vision": {"input": "image[1,3,640,640]", "output": "visual_tokens[1,100,1280]",
                   "dtype": str(DTYPE).split(".")[-1]},
        "source": {"model_definition": "torch(deepencoder.py)", "hf_model_id": "baidu/Unlimited-OCR"},
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }, indent=2))

    ok = asyncio.run(gate(o, img))
    print(f"\n{'✅ PASS' if ok else '❌ FAIL'} — DeepEncoder .aimodel {'matches' if ok else 'DIVERGES from'} oracle")


if __name__ == "__main__":
    main()
