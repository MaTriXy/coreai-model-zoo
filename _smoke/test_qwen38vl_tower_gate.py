#!/usr/bin/env python3
"""Gate: the authored Qwen3.5 vision tower vs the HF oracle, per suite case.

Two stages, both against `_smoke/qwen38vl_tower_fp32_ref.npz` — the HF tower
run in FP32 on the suite's own patches (the tower alone is 458M, so fp32 is
cheap even though the full-model oracle had to be bf16). The bf16 suite embeds
are NOT the target: HF-fp32 vs HF-bf16 already differs by min-row cos 0.9929,
so gating against bf16 would only measure oracle noise (isolated 2026-08-15,
identical cos values from the authored tower and HF-fp32 vs the bf16 dump):

  torch   — the re-authored fp32 tower (qwen3_5_vision.py), eager CPU.
            Proves the wiring: patch-linear collapse, block-major positional
            constants, bilinear pos-embed, 2D rope, merger. cos >= 0.999.
  aimodel — the exported fp16 .aimodel on the GPU delegate. Same bar.
            (GPU run: hold _GPU_LOCK, python-GPU SOLO.)

Run (coreai-models/.venv):
    ../coreai-models/.venv/bin/python _smoke/test_qwen38vl_tower_gate.py            # torch
    ../coreai-models/.venv/bin/python _smoke/test_qwen38vl_tower_gate.py --stage aimodel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

DEFAULT_SUITE = Path(__file__).parent / "qwen38vl_suite_512.npz"
FP32_REF = Path(__file__).parent / "qwen38vl_tower_fp32_ref.npz"
HF_ID = "Qwen/Qwen3.8-27B"
COS_BAR = 0.999


def case_cos(got: np.ndarray, want: np.ndarray) -> tuple[float, float]:
    g = got.astype(np.float64).reshape(-1)
    w = want.astype(np.float64).reshape(-1)
    c = float(g @ w / (np.linalg.norm(g) * np.linalg.norm(w)))
    row = (got.astype(np.float64) * want.astype(np.float64)).sum(-1) / (
        np.linalg.norm(got.astype(np.float64), axis=-1)
        * np.linalg.norm(want.astype(np.float64), axis=-1)
    )
    return c, float(row.min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="torch", choices=["torch", "aimodel"])
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--asset", default=None,
                    help="aimodel stage: path to the vision .aimodel "
                         "(default exports/qwen3_8_27b_vision_fp16/...)")
    args = ap.parse_args()

    suite = np.load(args.suite)
    ref = np.load(str(FP32_REF))
    n_cases = int(suite["_meta_cases"])
    # dedupe: the tower is per-image; cases share images
    img_cases = {}
    for case in range(n_cases):
        img_cases.setdefault(int(suite[f"case{case}_image_idx"]), case)
    print(f"suite: {n_cases} cases, {len(img_cases)} unique images | stage {args.stage}")

    fails = []
    if args.stage == "torch":
        import torch

        from coreai_models.models.macos.qwen3_5_vision import Qwen3_5VisionEncoder

        vis = Qwen3_5VisionEncoder.from_hf(HF_ID, target_dtype=torch.float32)
        for img, case in sorted(img_cases.items()):
            patches = torch.from_numpy(suite[f"case{case}_patches"]).float()
            with torch.no_grad():
                got = vis(patches).numpy()
            c, rmin = case_cos(got, ref[f"image{img}_embeds_fp32"])
            ok = c >= COS_BAR and rmin >= COS_BAR
            print(f"  {'PASS' if ok else 'FAIL'} image{img}: cos {c:.6f} min-row {rmin:.6f}")
            if not ok:
                fails.append(img)
    else:
        import asyncio
        import inspect

        import coreai.runtime as rt

        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
        from _paths import exports_dir  # noqa: E402

        asset = Path(args.asset) if args.asset else (
            exports_dir() / "qwen3_8_27b_vision_fp16" / "qwen3_8_27b_vision_fp16.aimodel")

        async def run() -> None:
            async def maybe(x):
                return await x if inspect.isawaitable(x) else x

            m = await maybe(rt.AIModel.load(str(asset), rt.SpecializationOptions.default()))
            fn = await maybe(m.load_function(m.function_names[0]))
            for img, case in sorted(img_cases.items()):
                patches = np.ascontiguousarray(
                    suite[f"case{case}_patches"].astype(np.float16))
                out = await maybe(fn(inputs={"patches": rt.NDArray(patches)}))
                got = np.asarray(out["image_embeds"].numpy()).astype(np.float32)
                c, rmin = case_cos(got, ref[f"image{img}_embeds_fp32"])
                ok = c >= COS_BAR and rmin >= COS_BAR
                print(f"  {'PASS' if ok else 'FAIL'} image{img}: cos {c:.6f} "
                      f"min-row {rmin:.6f}")
                if not ok:
                    fails.append(img)

        asyncio.run(run())

    print("ALL PASS" if not fails else f"FAILED images: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(main())
