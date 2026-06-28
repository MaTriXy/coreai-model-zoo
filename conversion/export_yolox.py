"""Export YOLOX (Megvii) object detection to Core AI — the zoo's first anchor-free
single-stage detector (companion to the DETR-family RF-DETR).

One artifact per variant:

* ``yolox-<variant>_<dtype>.aimodel`` — single static graph
  ``image [1, 3, S, S] float32`` (YOLOX-native preprocessing already applied:
  BGR, 0-255, letterboxed with pad value 114, top-left aligned — NO /255, NO
  mean/std for the non-legacy checkpoints) ->
  ``preds [1, A, 85]`` where A = (S/8)^2 + (S/16)^2 + (S/32)^2 anchors and the
  85 columns are ``[cx, cy, w, h, obj, cls_0 .. cls_79]``. The box is already
  grid+stride DECODED to pixel coordinates in the S-space letterbox, and obj/cls
  are already SIGMOID-ed — all in-graph (``decode_in_inference=True``).

Unlike the DETR family (RF-DETR), YOLOX is a dense anchor-free detector, so the
host post-processing is the classic ``score = obj * cls`` threshold + per-class
NMS, then unletterbox the surviving boxes back to original-image pixels.

Variants (depth, width, depthwise, input):
  nano 416 (dw) · tiny 416 · s 640 · m 640 · l 640 · x 640
All share the CSPDarknet(Focus stem + SPP) backbone -> PAFPN neck -> decoupled
YOLOXHead. ``s`` is the canonical 9M-param / 40.5 mAP baseline.

fp32 is the default ship dtype (detection has no bandwidth-bound decode loop, so
weight bytes barely matter and fp32 gates bit-clean on cpu AND gpu); ``--dtype
float16`` exists for experiments.

This script needs the upstream YOLOX repo on the path and a checkpoint:
    git clone https://github.com/Megvii-BaseDetection/YOLOX
    curl -L -o yolox_s.pth \
      https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
A tiny ``cv2`` stub is injected so we can import ``yolox.models`` without the
(unused at export time) OpenCV dependency — letterboxing here is pure
numpy/Pillow.

Numerics gate (``--verify-image``): the converted ``.aimodel`` is compared to
the torch-fp32 reference two ways on a real image —
  1. raw-tensor cosine + max|delta| over the whole [1, A, 85] head output;
  2. set-based detection match after the full host post-process (score>conf,
     per-class NMS): every confident reference detection must have a same-class
     partner with IoU >= 0.6 and score within tol in the .aimodel detections.
Run on ``--unit cpu`` and ``--unit gpu``.

Run (GPU; _GPU_LOCK held):
    coreai-models/.venv/bin/python \
      ../coreai-models-community/conversion/export_yolox.py \
      --variant s --yolox-repo /path/to/YOLOX --weights /path/to/yolox_s.pth \
      [--dtype float32] [--out-dir exports] \
      [--verify-image cats.jpg] [--unit gpu]

Deps: the coreai stack (coreai-core + coreai-torch, torch <= 2.11) + loguru
(yolox.models imports it). No OpenCV needed.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

# variant -> (depth, width, depthwise, input_size). COCO = 80 classes.
VARIANTS = {
    "nano": (0.33, 0.25, True, 416),
    "tiny": (0.33, 0.375, False, 416),
    "s": (0.33, 0.50, False, 640),
    "m": (0.67, 0.75, False, 640),
    "l": (1.00, 1.00, False, 640),
    "x": (1.33, 1.25, False, 640),
}
NUM_CLASSES = 80
STRIDES = (8, 16, 32)


# ---------------------------------------------------------------------------
# model sourcing (replicates Exp.get_model() without importing yolox.exp,
# which drags in cv2 via yolox.utils.demo_utils)
# ---------------------------------------------------------------------------
def build_yolox(variant: str, repo: Path, weights: Path):
    if "cv2" not in sys.modules:
        sys.modules["cv2"] = types.ModuleType("cv2")  # unused at export time
    sys.path.insert(0, str(repo))
    from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

    depth, width, depthwise, size = VARIANTS[variant]
    in_ch = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_ch, depthwise=depthwise, act="silu")
    head = YOLOXHead(NUM_CLASSES, width, in_channels=in_ch, depthwise=depthwise, act="silu")
    model = YOLOX(backbone, head).eval()

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    assert model.head.decode_in_inference, "need in-graph grid+stride decode"

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    anchors = sum((size // s) ** 2 for s in STRIDES)
    print(f"[model] yolox-{variant}: input={size}, params={n_params:.2f}M, anchors={anchors}")
    return model, size


# ---------------------------------------------------------------------------
# export + convert
# ---------------------------------------------------------------------------
def export_and_convert(model, size: int, dtype, out_path: Path, variant: str):
    import coreai.runtime as rt
    from coreai_torch import TorchConverter, get_decomp_table

    if dtype != torch.float32:
        model = model.to(dtype)
    x = torch.rand(1, 3, size, size, dtype=dtype)

    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(model, (x,))
    ep = ep.run_decompositions(get_decomp_table())
    print(f"[export] torch.export + decompositions in {time.time() - t0:.1f}s")

    prog = (
        TorchConverter()
        .add_exported_program(
            exported_program=ep,
            input_names=["image"],
            output_names=["preds"],
        )
        .to_coreai()
    )
    prog.optimize()

    shutil.rmtree(out_path, ignore_errors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = rt.AIModelAssetMetadata()
    meta.author = "Megvii (YOLOX); Core AI export: coreai-model-zoo"
    meta.license = "Apache-2.0"
    meta.model_description = (
        f"YOLOX-{variant} anchor-free single-stage detector (CSPDarknet + PAFPN + "
        "decoupled head). Input: BGR 0-255 letterboxed [1,3,S,S]; output preds "
        "[1,A,85] = decoded cxcywh pixels + obj + 80 sigmoid class scores. Host "
        "does score=obj*cls threshold + per-class NMS. https://github.com/Megvii-BaseDetection/YOLOX"
    )
    meta.creation_date = int(time.time())
    prog.save_asset(out_path, meta)
    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"[convert] saved {out_path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# host preprocessing (YOLOX-native letterbox; pure numpy/Pillow, no cv2)
# ---------------------------------------------------------------------------
def letterbox(image_path: str, size: int):
    """Replicate yolox.data.data_augment.preproc for a non-legacy checkpoint:
    BGR, scale to fit by the min ratio, top-left paste into a 114-filled S x S
    canvas, CHW float32. Returns (tensor[1,3,S,S], ratio, (orig_h, orig_w))."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    ow, oh = img.size
    r = min(size / oh, size / ow)
    nw, nh = int(ow * r), int(oh * r)
    resized = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32)  # RGB HWC
    bgr = resized[:, :, ::-1]  # YOLOX consumes BGR
    canvas = np.full((size, size, 3), 114.0, dtype=np.float32)
    canvas[:nh, :nw] = bgr
    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])  # [1,3,S,S]
    return torch.from_numpy(chw), r, (oh, ow)


# ---------------------------------------------------------------------------
# host post-process: decoded preds [1,A,85] -> kept (xyxy, score, cls)
# ---------------------------------------------------------------------------
def postprocess(preds: np.ndarray, conf_thr=0.25, nms_thr=0.45):
    from torchvision.ops import batched_nms

    p = preds[0]  # [A, 85]
    boxes = p[:, :4]
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    obj = p[:, 4]
    cls = p[:, 5:]
    cls_id = cls.argmax(axis=1)
    score = obj * cls[np.arange(cls.shape[0]), cls_id]
    keep = score > conf_thr
    if not keep.any():
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), dtype=int)
    xyxy, score, cls_id = xyxy[keep], score[keep], cls_id[keep]
    k = batched_nms(
        torch.from_numpy(xyxy).float(),
        torch.from_numpy(score).float(),
        torch.from_numpy(cls_id),
        nms_thr,
    ).numpy()
    return xyxy[k], score[k], cls_id[k]


def _iou_xyxy(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_sets(ref, got, score_tol, iou_thr=0.6):
    """ref/got = (boxes, scores, cls). Every confident ref detection must find an
    unused same-class got detection with IoU>=iou_thr and score within tol."""
    rb, rs, rc = ref
    gb, gs, gc = got
    used, matched, worst_iou = set(), 0, 1.0
    for i in range(len(rs)):
        best_j, best_iou = -1, 0.0
        for j in range(len(gs)):
            if j in used or gc[j] != rc[i] or abs(gs[j] - rs[i]) > score_tol:
                continue
            iou = _iou_xyxy(rb[i], gb[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0 and best_iou >= iou_thr:
            used.add(best_j)
            matched += 1
            worst_iou = min(worst_iou, best_iou)
    return matched, len(rs), worst_iou


async def verify(model, size, out_path: Path, image_path: str, dtype, unit):
    import coreai.runtime as rt

    if unit == "cpu":
        opts = rt.SpecializationOptions.cpu_only()
    else:
        opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(
            getattr(rt.ComputeUnitKind, unit)()
        )
    aimodel = await rt.AIModel.load(out_path, opts)
    fn = aimodel.load_function("main")

    x, _, _ = letterbox(image_path, size)
    # cast the reference to the gate dtype regardless of --skip-convert (export
    # mutates the module in place, but a skip-convert run would otherwise leave
    # the fp16 reference as fp32 -> "expected Half but found Float")
    ref_model = model if dtype == torch.float32 else model.to(dtype)
    with torch.no_grad():
        ref = ref_model(x if dtype == torch.float32 else x.to(dtype)).float().numpy()
    out = await fn({"image": rt.NDArray(x.to(dtype).numpy())})
    got = out["preds"].numpy().astype(np.float32)

    # 1) raw head-tensor parity (NMS-independent signal)
    a, b = ref.reshape(-1), got.reshape(-1)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    maxd = float(np.abs(a - b).max())

    # 2) end-to-end detection set match
    ref_det = postprocess(ref)
    got_det = postprocess(got)
    score_tol = 2e-2 if (dtype == torch.float32) else 5e-2
    matched, n, worst_iou = match_sets(ref_det, got_det, score_tol)
    # near-tie slack: dense detectors swap a duplicate box rank under fp noise
    ok = (cos > 0.9999 if dtype == torch.float32 else cos > 0.999) and (
        matched == n or (n - matched <= 1 and matched >= 0.9 * max(n, 1))
    )
    print(
        f"[verify:{unit}/{dtype}] cos={cos:.6f} max|d|={maxd:.2e} | "
        f"det-set {matched}/{n} (ref) worst-IoU={worst_iou:.3f} "
        f"| ref={len(ref_det[1])} got={len(got_det[1])} dets -> {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--variant", choices=list(VARIANTS), default="s")
    ap.add_argument("--yolox-repo", required=True, help="path to a YOLOX checkout")
    ap.add_argument("--weights", required=True, help="path to yolox_<v>.pth")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--verify-image", default=None, help="real image for the numerics gate")
    ap.add_argument("--unit", default="cpu", help="verify unit: cpu | gpu | neural_engine")
    ap.add_argument("--skip-convert", action="store_true", help="verify an existing artifact")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16}[args.dtype]
    model, size = build_yolox(args.variant, Path(args.yolox_repo), Path(args.weights))
    out_path = Path(args.out_dir) / f"yolox-{args.variant}_{args.dtype}.aimodel"

    if not args.skip_convert:
        export_and_convert(model, size, dtype, out_path, args.variant)

    if args.verify_image:
        ok = asyncio.run(verify(model, size, out_path, args.verify_image, dtype, args.unit))
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
