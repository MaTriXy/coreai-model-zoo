#!/usr/bin/env python3
"""North-Micro-Vision host image preprocessing in NumPy — the spec the app reproduces.

The Core AI bundle bakes ONE patch grid (32x32 patches = a 512x512 canvas ->
2x2 merge -> 256 tokens), so the host's whole job is:

    RGB uint8 [H,W,3] -> resize 512x512 -> /255 -> (x-0.5)/0.5
                      -> patchify 16x16  -> patches [1024, 1536] float32

Two things differ from the LFM2.5-VL host in this same directory, and mixing
them up produces a model that runs and describes a different picture:

  * **The patch layout is Qwen-VL's, not NaFlex's.** Inside one patch vector the
    order is `[C][T][py][px]` — channel-major, with the still frame duplicated
    across `temporal_patch_size = 2` — and the patches themselves are
    **block-major**: the 2x2 merge group is contiguous, so the tower's merger
    sees four neighbours in a row. (LFM2.5-VL is channel-LAST and row-major.)
    Hence `patch_dim = 3 * 2 * 16 * 16 = 1536`.
  * **The tower is native-resolution.** The processor picks a grid from the
    image's own aspect ratio inside a pixel budget (`shortest_edge` 65 536 =
    256^2, `longest_edge` 16 777 216 = 4096^2), so the 640x480 fixture keeps its
    native 30x40 patch grid. Baking a square grid is the export's choice, not
    the model's; it stretches non-square images, and no gate here can see that
    cost because the fixed-grid oracle is captured through the same stretch.

Resampling is `resample: 3` (PIL BICUBIC), whose negative lobes ring past both
ends on hard edges — Pillow clips because it writes uint8, so this clips too.
Rescale 1/255 then (x-0.5)/0.5; mean and std are 0.5 on all three channels.

Run the gate:
    ../coreai-models/.venv/bin/python _smoke/northmv_preprocess.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from lfm25vl_preprocess import BICUBIC, resize_antialias  # noqa: E402

TILE = 512
PATCH = 16
MERGE = 2
TEMPORAL = 2
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_REF = Path(__file__).parent / "north_micro_vision_instruct_ref_512x512.npz"


def patchify_qwen(
    image: np.ndarray, patch: int = PATCH, merge: int = MERGE, temporal: int = TEMPORAL
) -> np.ndarray:
    """[H,W,C] -> [n_patch, C*T*patch*patch] in Qwen-VL's block-major order.

    Mirrors `Qwen2VLImageProcessor`: patches are visited merge-block by
    merge-block, and each patch vector is [C][T][py][px] with the still frame
    repeated `temporal` times.
    """
    h, w, c = image.shape
    if h % (patch * merge) or w % (patch * merge):
        raise ValueError(f"{h}x{w} not divisible by patch*merge {patch * merge}")
    ph, pw = h // patch, w // patch
    # [ph, pw, py, px, c]
    grid = image.reshape(ph, patch, pw, patch, c).transpose(0, 2, 1, 3, 4)
    # block-major over the 2x2 merge groups
    grid = grid.reshape(ph // merge, merge, pw // merge, merge, patch, patch, c)
    grid = grid.transpose(0, 2, 1, 3, 4, 5, 6)          # [bh, bw, mh, mw, py, px, c]
    grid = grid.reshape(-1, patch, patch, c)             # [n_patch, py, px, c]
    # per patch: [c][t][py][px]
    out = np.repeat(grid.transpose(0, 3, 1, 2)[:, :, None], temporal, axis=2)
    return out.reshape(out.shape[0], -1)


def preprocess(image: np.ndarray, tile: int = TILE, patch: int = PATCH) -> np.ndarray:
    """RGB uint8 [H,W,3] -> patches [(tile/patch)^2, 3*2*patch*patch] float32."""
    resized = resize_antialias(image, tile, tile, BICUBIC)
    x = (resized / 255.0 - IMAGE_MEAN) / IMAGE_STD
    return patchify_qwen(x).astype(np.float32)


def _fixture_image() -> np.ndarray:
    import requests
    from PIL import Image

    img = Image.open(requests.get(IMAGE_URL, stream=True, timeout=60).raw).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--tol", type=float, default=0.999)
    args = ap.parse_args()

    from PIL import Image

    src = _fixture_image()
    ref = np.load(args.ref)
    print(f"fixture {src.shape[1]}x{src.shape[0]} | oracle {args.ref}")
    failures: list[str] = []

    def check(name: str, got: np.ndarray, want: np.ndarray, tol: float, scale: str) -> None:
        g = np.asarray(got, dtype=np.float64).reshape(-1)
        w = np.asarray(want, dtype=np.float64).reshape(-1)
        c = float(g @ w / (np.linalg.norm(g) * np.linalg.norm(w)))
        mx = float(np.abs(g - w).max())
        ok = c >= tol
        print(f"  {'PASS' if ok else 'FAIL'} {name:28s} cos {c:.6f}  max|d| {mx:.3e} ({scale})")
        if not ok:
            failures.append(name)

    print("\nA. resize 640x480 -> 512x512 (BICUBIC)")
    pil = np.asarray(Image.fromarray(src).resize((TILE, TILE), BICUBIC), dtype=np.float64)
    check("resize vs PIL BICUBIC", resize_antialias(src, TILE, TILE, BICUBIC), pil,
          0.99999, "0-255")

    print("\nB. host path -> patches, vs the oracle the bundle is gated at")
    want = ref["pixel_values"]
    got = preprocess(src)
    print(f"  patches {got.shape} vs oracle {want.shape}")
    if got.shape != want.shape:
        print("  FAIL shape mismatch"); failures.append("shape")
    else:
        check("pixel_values", got, want, args.tol, "normalized")
        via_pil = patchify_qwen((pil / 255.0 - IMAGE_MEAN) / IMAGE_STD).astype(np.float32)
        check("pixel_values (PIL resize)", via_pil, want, args.tol, "normalized")

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
