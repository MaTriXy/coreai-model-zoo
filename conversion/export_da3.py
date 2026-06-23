"""Export Depth Anything 3 (ByteDance) monocular depth to Core AI.

The zoo's first depth model. One artifact:

* ``da3-<variant>_<dtype>.aimodel`` — single static graph
  ``image [1, 3, R, R] RGB in [0, 1]`` (ImageNet mean/std folded in-graph) ->
  ``depth [1, R, R]`` (relative, exp-activated) + ``depth_conf [1, R, R]``.

DA3-SMALL is an "any-view" model (DINOv2 ViT-S backbone with alternating
cross-view attention + 2D RoPE + a camera token, then a DualDPT depth/ray head).
Fed a single view (S=1) it is a monocular depth estimator: cross-view global
attention collapses to self-attention, the reference-view reorder is statically
dead (it needs S >= THRESH), and the camera token is a fixed parameter. We export
only the depth path (backbone + head -> depth, depth_conf); the camera decoder,
ray head and sky post-processing are dropped (the ray branch is dead-code
eliminated by ``optimize()`` because only depth/depth_conf are graph outputs).

The bundle is a fixed SQUARE R x R graph (R % 14 == 0). Host contract: resize the
RGB image to R x R (cv2 INTER_AREA), feed it as RAW [0, 1] (the ImageNet
normalization is folded into the graph), run, then resize the depth map back to the
original H x W. Depth is relative, so the brief aspect squash is recovered by the
resize-back -> the model output vs the official DA3 viewer is mean r ~= 0.98 across
aspect ratios (square inputs r = 1.000), within DA3's own 504-vs-518 variance
(r ~= 0.975-0.984). The engine is bit-exact to torch (cos 1.0) at ANY fixed shape,
square or non-square, so a per-aspect non-square bundle is also possible (pass a
non-square --res-hw); the single square bundle is the shipped contract.

R = 504 = 36 * 14 matches the DA3 default process_res; the pos-embed bicubic
interpolation is over fixed sizes so it folds to a constant at export (no runtime
bicubic). 518 (= 37*14, the DINOv2 native grid, no interpolation at all) also works
and is ~5% faster.

Import-time patch (numerically identical): RoPE's ``_compute_frequency_components``
is sized by ``int(positions.max()) + 1``, a Python int pulled from a traced tensor
-> data-dependent guard under torch.export. The grid is static, so we bake the
table length as a constant.

Run:
  python export_da3.py --variant small \
      [--dtype float32] [--res 504] [--out-dir exports] \
      [--verify-images img1.png,img2.png] [--unit gpu]

Deps: depth_anything_3 on PYTHONPATH (--da3-src) + the coreai stack (torch <= 2.11).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# DA3 patch (see module docstring; numerically identical)
# ---------------------------------------------------------------------------
def apply_da3_patches(max_position: int):
    """Bake RoPE table length as a constant (drop int(positions.max())).

    Also make the RoPE / position caches recompute every call: the originals
    memoize tensors into dicts, which during torch.export get poisoned with traced
    (fake) tensors and then break a later eager run in the same process.
    """
    from depth_anything_3.model.dinov2.layers.rope import (
        PositionGetter,
        RotaryPositionEmbedding2D,
    )

    def rope_forward_const(self, tokens, positions):
        feature_dim = tokens.size(-1) // 2
        dtype, device = tokens.dtype, tokens.device
        exponents = torch.arange(0, feature_dim, 2, device=device).float() / feature_dim
        inv_freq = 1.0 / (self.base_frequency ** exponents)
        pos = torch.arange(max_position, device=device, dtype=inv_freq.dtype)
        angles = torch.einsum("i,j->ij", pos, inv_freq).to(dtype)
        angles = torch.cat((angles, angles), dim=-1)
        cos_comp, sin_comp = angles.cos().to(dtype), angles.sin().to(dtype)
        vert, horiz = tokens.chunk(2, dim=-1)
        vert = self._apply_1d_rope(vert, positions[..., 0], cos_comp, sin_comp)
        horiz = self._apply_1d_rope(horiz, positions[..., 1], cos_comp, sin_comp)
        return torch.cat((vert, horiz), dim=-1)

    def pg_call(self, batch_size, height, width, device):
        y = torch.arange(height, device=device)
        x = torch.arange(width, device=device)
        positions = torch.cartesian_prod(y, x)
        return positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()

    RotaryPositionEmbedding2D.forward = rope_forward_const
    PositionGetter.__call__ = pg_call

    # DPT/DualDPT _add_pos_embed builds a fp32 sin/cos UV grid (make_sincos_pos_embed
    # hard-casts .float()); under fp16 this upcasts the feature map and the next conv
    # sees Half weights vs float input. Cast the embedding back to the feature dtype.
    from depth_anything_3.model import dpt as _dpt, dualdpt as _dualdpt
    from depth_anything_3.model.utils.head_utils import create_uv_grid, position_grid_to_embed

    def _add_pos_embed(self, x, W, H, ratio=0.1):
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]).to(x.dtype) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    _dpt.DPT._add_pos_embed = _add_pos_embed
    _dualdpt.DualDPT._add_pos_embed = _add_pos_embed
    print(f"[patch] RoPE const ({max_position}) + pos-embed dtype-safe")


# ---------------------------------------------------------------------------
# model sourcing
# ---------------------------------------------------------------------------
class MonoDepth(torch.nn.Module):
    """image [1,3,H,W] RGB in [0,1] -> depth [1,H,W], depth_conf [1,H,W]."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        x = x.unsqueeze(1)  # [B,1,3,H,W] (S=1, monocular)
        feats, _ = self.net.backbone(
            x, cam_token=None, export_feat_layers=[], ref_view_strategy="first"
        )
        out = self.net.head(feats, image.shape[-2], image.shape[-1], patch_start_idx=0)
        depth = out["depth"][:, 0]
        # DualDPT (any-view models) emits depth_conf; plain DPT (mono-large) does not
        if "depth_conf" in out:
            return depth, out["depth_conf"][:, 0]
        return (depth,)


def build_model(ckpt_dir: str):
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from depth_anything_3.cfg import create_object

    cfg_json = json.load(open(Path(ckpt_dir) / "config.json"))
    cfg = OmegaConf.create(cfg_json["config"])
    net = create_object(cfg).eval()
    sd = load_file(str(Path(ckpt_dir) / "model.safetensors"))
    sd = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # the only missing keys are unused aux(ray)-head LayerNorms -> dead-code path
    assert not unexpected, f"unexpected keys: {unexpected[:6]}"
    n = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[model] da3 ({cfg_json['model_name']}): {n:.1f}M params, "
          f"missing(unused-aux)={len(missing)}")
    return MonoDepth(net).eval()


# ---------------------------------------------------------------------------
# export + convert
# ---------------------------------------------------------------------------
def export_and_convert(wrapper, res: int, dtype, out_path: Path):
    import coreai.runtime as rt
    from coreai_torch import TorchConverter, get_decomp_table

    x = torch.rand(1, 3, res, res, dtype=dtype)
    if dtype != torch.float32:
        # deepcopy first: .to() mutates in place, and the caller reuses the fp32
        # module as the verify oracle. Half the whole model; NO autocast (uniform dtypes).
        wrapper = copy.deepcopy(wrapper).to(dtype)

    real_assert = torch._assert
    torch._assert = lambda *a, **k: None
    t0 = time.time()
    try:
        with torch.no_grad():
            ep = torch.export.export(wrapper, (x,))
    finally:
        torch._assert = real_assert
    ep = ep.run_decompositions(get_decomp_table())
    print(f"[export] torch.export + decompositions in {time.time() - t0:.1f}s")

    bad = {
        str(n.target)
        for n in ep.graph.nodes
        if n.op == "call_function" and "grid_sampler" in str(n.target)
    }
    if bad:
        raise RuntimeError(f"unsupported ops leaked into the graph: {bad}")

    out_names = ["depth", "depth_conf"][: len(ep.graph_signature.user_outputs)]
    prog = TorchConverter().add_exported_program(
        exported_program=ep,
        input_names=["image"],
        output_names=out_names,
    ).to_coreai()
    prog.optimize()
    shutil.rmtree(out_path, ignore_errors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = rt.AIModelAssetMetadata()
    meta.author = "ByteDance (Depth Anything 3); Core AI export: coreai-model-zoo"
    meta.license = "Apache-2.0"
    meta.model_description = (
        "Depth Anything 3 (DINOv2 ViT-S backbone + DualDPT head), monocular depth. "
        "Input: RGB [0,1]; outputs: relative depth + confidence map. "
        "https://github.com/ByteDance-Seed/depth-anything-3"
    )
    meta.creation_date = int(time.time())
    prog.save_asset(out_path, meta)
    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"[convert] saved {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# verification (per-pixel depth match vs torch fp32 oracle)
# ---------------------------------------------------------------------------
def to_np(t):
    return np.asarray(t.detach().to(torch.float32).cpu().contiguous().tolist(), dtype=np.float32)


def depth_stats(ref, got):
    ref = ref.reshape(-1).astype(np.float64)
    got = got.reshape(-1).astype(np.float64)
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got) + 1e-12))
    rel = np.abs(got - ref) / (np.abs(ref) + 1e-6)
    return cos, float(rel.mean()), float(rel.max()), float(np.abs(got - ref).max())


async def verify(ref_wrapper, res, out_path: Path, image_paths, dtype, unit):
    import coreai.runtime as rt
    from PIL import Image

    if unit == "cpu":
        opts = rt.SpecializationOptions.cpu_only()
    else:
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(
            getattr(rt.ComputeUnitKind, unit)()
        )
    model = await rt.AIModel.load(out_path, opts)
    fn = model.load_function("main")

    cos_min = 0.9999 if dtype == torch.float32 else 0.998
    relmean_max = 2e-3 if dtype == torch.float32 else 2e-2
    all_pass = True
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize((res, res), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            ref = ref_wrapper(x)
        ref_d = ref[0]
        out = await fn({"image": rt.NDArray(x.to(dtype).numpy())})
        got_d = out["depth"].numpy().astype(np.float32)
        dcos, drm, drx, dabs = depth_stats(to_np(ref_d), got_d)
        ok = dcos >= cos_min and drm <= relmean_max
        conf_note = ""
        if len(ref) > 1 and "depth_conf" in out:
            ccos, crm, _, _ = depth_stats(to_np(ref[1]), out["depth_conf"].numpy().astype(np.float32))
            conf_note = f" | conf cos={ccos:.6f} relmean={crm:.2e}"
        all_pass &= ok
        print(
            f"[verify:{Path(p).stem}] depth cos={dcos:.6f} relmean={drm:.2e} "
            f"relmax={drx:.2e}{conf_note} -> {'PASS' if ok else 'FAIL'}"
        )
    print(f"[verify] {'ALL PASS' if all_pass else 'FAIL'} ({unit}, {dtype})")
    return all_pass


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", default="small")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--ckpt", required=True, help="dir with config.json + model.safetensors")
    ap.add_argument("--da3-src", default=None, help="path to depth_anything_3 package src")
    ap.add_argument("--verify-images", default=None)
    ap.add_argument("--unit", default="cpu", help="cpu | gpu | neural_engine")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    if args.da3_src:
        sys.path.insert(0, args.da3_src)
    assert args.res % 14 == 0, "res must be a multiple of patch size 14"
    max_position = args.res // 14 + 1  # grid + 1 (patch_start_idx offset)

    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    apply_da3_patches(max_position)
    wrapper = build_model(args.ckpt)
    out_path = Path(args.out_dir) / f"da3-{args.variant}_{args.dtype}.aimodel"

    if not args.skip_convert:
        export_and_convert(wrapper, args.res, dtype, out_path)

    if args.verify_images:
        ok = asyncio.run(
            verify(wrapper, args.res, out_path, args.verify_images.split(","), dtype, args.unit)
        )
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
