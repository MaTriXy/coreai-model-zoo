#!/usr/bin/env python3
"""Build the PipelinedBench device fixture for North-Micro-Vision.

Emits the four-input (deepstack) VL shape PipelinedBench's `vl: true` path
reads — the same layout the Qwen3-VL fixture uses, since this checkpoint shares
that tower:

    image_embeds.f16.bin       [256, 2048] fp16, from the EXPORTED tower
    deepstack_embeds.f16.bin   [768, 2048] fp16, same run
    vl_ref.json                prompt ids (V+slot), expected ids, rope shift

The embeds come from the exported bundle rather than the fp32 oracle because on
device that is what feeds the decoder; a fixture built from fp32 would gate a
chain nobody runs.

Run (coreai-models/.venv, GPU, _GPU_LOCK held):
    ../coreai-models/.venv/bin/python _smoke/dump_northmv_device_ref.py
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import sys
from pathlib import Path

import numpy as np

import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import exports_dir, hf_snapshot  # noqa: E402

from coreai_models.models.macos.cohere_compass import (  # noqa: E402
    cohere_compass_config_from_dict,
)

DEFAULT_REF = Path(__file__).parent / "north_micro_vision_instruct_ref_512x512.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="CohereLabs/North-Micro-Vision-Instruct")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--out", default=str(Path(__file__).parent / "northmv_ref"))
    ap.add_argument("--expected", type=int, default=24)
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    root = exports_dir()
    vis_name = f"{short}_vision_fp16"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = np.load(args.ref)
    _, ph, pw = (int(v) for v in ref["image_grid_thw"][0])
    grid = ph // 2
    n_tokens = grid * (pw // 2)

    with open(Path(hf_snapshot(args.hf_id)) / "config.json") as f:
        raw = json.load(f)
    cfg = cohere_compass_config_from_dict(raw)
    image_token_id = int(raw["image_token_id"])

    opts = rt.SpecializationOptions.default()
    vm = await maybe_await(rt.AIModel.load(str(root / vis_name / f"{vis_name}.aimodel"), opts))
    vfn = await maybe_await(vm.load_function(vm.function_names[0]))
    patches = np.ascontiguousarray(ref["pixel_values"].astype(np.float16))
    res = await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
    embeds = np.asarray(res["image_embeds"].numpy()).astype(np.float16)
    deep = np.asarray(res["deepstack_embeds"].numpy()).astype(np.float16)
    print(f"image_embeds {embeds.shape} deepstack {deep.shape} from {vis_name}")

    (out / "image_embeds.f16.bin").write_bytes(np.ascontiguousarray(embeds).tobytes())
    (out / "deepstack_embeds.f16.bin").write_bytes(np.ascontiguousarray(deep).tobytes())

    ids = ref["input_ids"][0].astype(np.int64).copy()
    img_pos = np.nonzero(ids == image_token_id)[0]
    if img_pos.size != n_tokens:
        raise SystemExit(f"{img_pos.size} placeholders vs {n_tokens} tokens")
    img_start = int(img_pos[0])
    ids[img_pos] = cfg.vocab_size + np.arange(n_tokens)
    meta = {
        "prompt_ids": [int(v) for v in ids],
        "expected_ids": [int(v) for v in ref["gen_ids"][: args.expected]],
        "shift_start": img_start + n_tokens,
        "shift_amount": n_tokens - grid,
        "hidden": int(embeds.shape[1]),
        "tokens": int(n_tokens),
        "hf_id": args.hf_id,
        "vision_bundle": vis_name,
    }
    (out / "vl_ref.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} (shift {meta['shift_start']}/{meta['shift_amount']}, "
          f"{len(meta['prompt_ids'])} prompt ids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
