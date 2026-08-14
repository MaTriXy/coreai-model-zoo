#!/usr/bin/env python3
"""LFM2.5-VL host image preprocessing in NumPy — the spec the app-side code
reproduces, gated against the HF processor before any Swift exists.

The Core AI bundle bakes ONE patch grid (32x32 = a single 512x512 tile ->
256 tokens = the checkpoint's max_image_tokens), so the host's whole job is:

    RGB uint8 [H,W,3] -> resize 512x512 -> /255 -> (x-0.5)/0.5
                      -> patchify 16x16  -> patches [1024, 768] float32

Two things here are easy to get wrong and silent when wrong:

  * The PATCH LAYOUT. HF's ``convert_image_to_patches`` permutes
    (b, C, ph, P, pw, P) -> (b, ph, pw, P, P, C), so inside one 768-vector the
    CHANNEL is the fastest axis and the patch is row-major: [y][x][c]. Writing
    the natural [c][y][x] instead still produces a 768-vector, still runs, and
    still emits fluent text about the wrong image.

  * The RESIZE FILTER. The processor resizes with ``resample: 2`` (PIL
    BILINEAR), which for a downscale is an ANTIALIASED triangle filter whose
    support grows with the reduction factor — not the 2x2 bilinear tap that
    every GPU "bilinear" gives you. A 2x2-tap resize of a 640x480 photo drops
    most of the information the tower is reading. ``resize_bilinear_antialias``
    below is that filter, written out; it is the part a Swift/CoreImage host
    has to match, and the part to check first when device output degrades.

The aspect ratio is NOT preserved: a fixed square grid means a non-square image
is stretched. That is the cost of baking the grid — the NaFlex checkpoint would
otherwise pick a per-image grid — and it is what the fixed-grid oracle
(`lfm25vl_ref.py --resize 512x512`) is captured through, so the gate measures
the shipped path rather than a nicer one.

Run the gate (any interpreter with numpy + Pillow; add --hf-check on a
transformers>=5 interpreter to compare against the processor itself):
    python3 _smoke/lfm25vl_preprocess.py
    ~/code/litertlm-convert/.venv-vl093/bin/python _smoke/lfm25vl_preprocess.py --hf-check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

TILE = 512
PATCH = 16
IMAGE_MEAN = 0.5
IMAGE_STD = 0.5
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
DEFAULT_REF = Path(__file__).parent / "lfm2_5_vl_450m_ref_512x512.npz"


# PIL resample ids, as they appear in `processor_config.json` ("resample").
BILINEAR, BICUBIC = 2, 3
_SUPPORT = {BILINEAR: 1.0, BICUBIC: 2.0}


def _kernel(x: np.ndarray, resample: int) -> np.ndarray:
    """Pillow's filter weight at distance ``x`` (already divided by filterscale)."""
    x = np.abs(x)
    if resample == BILINEAR:
        return np.clip(1.0 - x, 0.0, None)
    if resample == BICUBIC:
        # Pillow's bicubic, a = -0.5 (Catmull-Rom).
        a = -0.5
        w = np.zeros_like(x)
        m1 = x < 1.0
        w[m1] = ((a + 2.0) * x[m1] - (a + 3.0)) * x[m1] * x[m1] + 1.0
        m2 = (x >= 1.0) & (x < 2.0)
        w[m2] = (((x[m2] - 5.0) * x[m2] + 8.0) * x[m2] - 4.0) * a
        return w
    raise ValueError(f"unsupported resample {resample} (2 = BILINEAR, 3 = BICUBIC)")


def _filter_weights(in_size: int, out_size: int, resample: int):
    """Pillow's resample weights for one axis.

    Returns (starts [out_size], weights [out_size, k]) zero-padded to a common
    width. Downscaling widens the filter (``filterscale = in/out``), which is
    what makes it antialiased; upscaling keeps the base support and reduces to
    the familiar 2-tap (bilinear) / 4-tap (bicubic) interpolation.
    """
    scale = in_size / out_size
    filterscale = max(1.0, scale)
    support = _SUPPORT[resample] * filterscale
    kmax = int(np.ceil(support) * 2) + 1

    starts = np.zeros(out_size, dtype=np.int64)
    weights = np.zeros((out_size, kmax), dtype=np.float64)
    for i in range(out_size):
        center = (i + 0.5) * scale
        xmin = int(max(0, np.floor(center - support + 0.5)))
        xmax = int(min(in_size, np.ceil(center + support + 0.5)))
        xs = np.arange(xmin, xmax)
        w = _kernel((xs + 0.5 - center) / filterscale, resample)
        total = w.sum()
        if total > 0:
            w = w / total
        starts[i] = xmin
        weights[i, : xs.size] = w
    return starts, weights


def _resample_axis(img: np.ndarray, out_size: int, axis: int, resample: int) -> np.ndarray:
    """Apply the filter along one axis of a float [H,W,C] image."""
    img = np.moveaxis(img, axis, 0)
    starts, weights = _filter_weights(img.shape[0], out_size, resample)
    kmax = weights.shape[1]
    padded = np.concatenate([img, np.zeros((kmax,) + img.shape[1:], img.dtype)], axis=0)
    idx = starts[:, None] + np.arange(kmax)[None, :]          # [out, k]
    taps = padded[idx]                                        # [out, k, ...]
    out = np.einsum("okyc,ok->oyc", taps, weights.astype(img.dtype))
    return np.moveaxis(out, 0, axis)


def resize_antialias(
    image: np.ndarray, out_h: int, out_w: int, resample: int = BILINEAR
) -> np.ndarray:
    """PIL-equivalent antialiased resize of an [H,W,3] image, in float64.

    Separable: rows then columns, same as Pillow. Pillow itself rounds the
    weights into 8-bit fixed point for uint8 images, so this agrees to well
    under one 0-255 level rather than bit-exactly.

    ``resample`` is the checkpoint's own `processor_config.json` value: the
    450M ships 2 (BILINEAR), the 3B ships 3 (BICUBIC). Feeding one model the
    other's filter is a silent quality loss, not an error.
    """
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError(f"expected [H,W,C], got {x.shape}")
    x = _resample_axis(x, out_h, axis=0, resample=resample)
    x = _resample_axis(x, out_w, axis=1, resample=resample)
    # Pillow writes uint8, so it CLIPS. Bicubic's negative lobes ring past both
    # ends on hard edges -- without this the two agree to cos 0.999997 but differ
    # by up to 15 levels at the ringing pixels, and only on the 3B (bicubic).
    return np.clip(x, 0.0, 255.0)


def patchify(image: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """[H,W,C] -> [num_patches, patch*patch*C], HF's (ph, pw, py, px, c) order."""
    h, w, c = image.shape
    if h % patch or w % patch:
        raise ValueError(f"{h}x{w} not divisible by patch {patch}")
    ph, pw = h // patch, w // patch
    out = image.reshape(ph, patch, pw, patch, c).transpose(0, 2, 1, 3, 4)
    return out.reshape(ph * pw, patch * patch * c)


def preprocess(
    image: np.ndarray, tile: int = TILE, patch: int = PATCH, resample: int = BILINEAR
) -> np.ndarray:
    """RGB uint8 [H,W,3] -> patches [(tile/patch)^2, patch*patch*3] float32."""
    resized = resize_antialias(image, tile, tile, resample)
    x = resized / 255.0
    x = (x - IMAGE_MEAN) / IMAGE_STD
    return patchify(x, patch).astype(np.float32)


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #
def _processor_resample(hf_id: str) -> int:
    """The checkpoint's own resample id — never guess it, the family disagrees."""
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
    from _paths import hf_snapshot  # noqa: E402

    with open(Path(hf_snapshot(hf_id)) / "processor_config.json") as f:
        return int(json.load(f)["image_processor"]["resample"])


def _fixture_image() -> np.ndarray:
    import requests
    from PIL import Image

    img = Image.open(requests.get(IMAGE_URL, stream=True, timeout=60).raw).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument(
        "--resample",
        type=int,
        default=None,
        help="PIL resample id; default = read it from the checkpoint's processor_config.json",
    )
    ap.add_argument(
        "--hf-check",
        action="store_true",
        help="also run the real Lfm2VlProcessor (needs transformers>=5)",
    )
    args = ap.parse_args()

    from PIL import Image

    ref = np.load(args.ref)
    hf_id = str(ref["_meta_hf_id"])
    resample = args.resample if args.resample is not None else _processor_resample(hf_id)
    src = _fixture_image()
    print(f"fixture {src.shape[1]}x{src.shape[0]} from {IMAGE_URL}")
    print(f"model {hf_id} | resample {resample} "
          f"({'BILINEAR' if resample == BILINEAR else 'BICUBIC'})")
    failures: list[str] = []

    def check(name: str, got: np.ndarray, want: np.ndarray, tol: float, scale: str) -> None:
        g = np.asarray(got, dtype=np.float64).reshape(-1)
        w = np.asarray(want, dtype=np.float64).reshape(-1)
        c = float(g @ w / (np.linalg.norm(g) * np.linalg.norm(w)))
        mx = float(np.abs(g - w).max())
        ok = c >= tol
        print(f"  {'PASS' if ok else 'FAIL'} {name:26s} cos {c:.6f}  max|d| {mx:.3e} ({scale})")
        if not ok:
            failures.append(name)

    # A. the resize filter, against Pillow's own implementation
    print("\nA. resize 640x480 -> 512x512")
    pil = np.asarray(
        Image.fromarray(src).resize((TILE, TILE), resample), dtype=np.float64
    )
    mine = resize_antialias(src, TILE, TILE, resample)
    check(f"resize vs PIL {resample}", mine, pil, 0.99999, "0-255")

    # B. the whole host path, against the fixed-grid oracle's pixel_values
    print("\nB. host path -> patches, vs the oracle the bundle is gated at")
    want = ref["pixel_values"][0]
    got = preprocess(src, resample=resample)
    print(f"  patches {got.shape} vs oracle {want.shape}")
    if got.shape != want.shape:
        print("  FAIL shape mismatch")
        failures.append("shape")
    else:
        check("pixel_values", got, want, args.tol, "normalized")
        # The oracle's own input was Pillow-resized, so isolating Pillow's
        # fixed-point rounding tells us how much of any gap is the filter.
        via_pil = patchify((pil / 255.0 - IMAGE_MEAN) / IMAGE_STD).astype(np.float32)
        check("pixel_values (PIL resize)", via_pil, want, args.tol, "normalized")

    # C. optional: the real processor, on the pre-resized fixture
    if args.hf_check:
        print("\nC. vs Lfm2VlProcessor")
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(hf_id)
        square = Image.fromarray(src).resize((TILE, TILE), resample)
        out = proc.image_processor(images=[square], return_tensors="np")
        hf = out["pixel_values"][0]
        check("pixel_values vs processor", preprocess(src, resample=resample), hf,
              args.tol, "normalized")
        n_real = int(out["pixel_attention_mask"][0].sum())
        shape = tuple(int(v) for v in out["spatial_shapes"][0])
        print(f"  processor grid {shape}, {n_real} real patches (bundle bakes 32x32 = 1024)")
        if shape != (TILE // PATCH, TILE // PATCH) or n_real != (TILE // PATCH) ** 2:
            failures.append("processor grid")

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
