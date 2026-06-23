# Minimal Depth Anything 3 sample: run a da3 .aimodel on one image, save a colorized depth map.
#
# The graph folds ImageNet normalization in-graph, so the input is RAW [0,1] RGB resized to the
# square graph size (504). Depth is resized back to the original aspect for display.
#
# Usage:
#   python depth_anything_3_sample.py da3-small_float16.aimodel photo.jpg depth.png [--unit gpu]
#
# Bundle: huggingface.co/mlboydaisuke/Depth-Anything-3-CoreAI (small/da3-small_float16.aimodel)
import asyncio
import sys

import cv2
import numpy as np
import coreai.runtime as rt


def colorize(depth: np.ndarray) -> np.ndarray:
    """DA3 convention: inverse-depth, percentile 2-98 normalize, Spectral-ish (far=red, near=blue)."""
    d = depth.astype(np.float64)
    m = d > 1e-6
    disp = np.where(m, 1.0 / np.where(m, d, 1.0), 0.0)
    lo, hi = np.percentile(disp[m], 2), np.percentile(disp[m], 98)
    t = np.clip((disp - lo) / (hi - lo + 1e-9), 0, 1)
    stops = np.array([[0.84, 0.24, 0.31], [0.99, 0.68, 0.38], [1.0, 1.0, 0.75],
                      [0.40, 0.76, 0.65], [0.20, 0.34, 0.65]])
    x = t * (len(stops) - 1)
    i = np.clip(x.astype(int), 0, len(stops) - 2)
    f = (x - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    return (rgb * 255).astype(np.uint8)


async def main(model_path: str, image_path: str, out_path: str, unit: str):
    opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(
                getattr(rt.ComputeUnitKind, unit)()))
    model = await rt.AIModel.load(model_path, opts)
    fn = model.load_function("main")

    side = 504  # da3 square graph size
    im = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    h, w = im.shape[:2]
    sq = cv2.resize(im, (side, side), interpolation=cv2.INTER_AREA)

    # RAW [0,1], NCHW — the graph normalizes internally. Match the bundle dtype.
    dtype = np.float16 if "float16" in model_path else np.float32
    x = (sq.astype(dtype) / 255.0).transpose(2, 0, 1)[None]
    depth = (await fn({"image": rt.NDArray(x)}))["depth"].numpy().reshape(side, side).astype(np.float32)

    vis = colorize(cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR))
    cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"saved {out_path}  ({w}x{h}, depth range {depth.min():.2f}..{depth.max():.2f})")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    unit = next((a.split("=")[-1] for a in sys.argv if a.startswith("--unit")), "gpu")
    if "--unit" in sys.argv:
        unit = sys.argv[sys.argv.index("--unit") + 1]
    asyncio.run(main(args[0], args[1], args[2] if len(args) > 2 else "depth.png", unit))
