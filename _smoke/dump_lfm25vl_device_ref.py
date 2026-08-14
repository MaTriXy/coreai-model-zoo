#!/usr/bin/env python3
"""Build the PipelinedBench device-gate fixture for LFM2.5-VL-450M.

The device gate has to check the IMAGE path, not just decode speed, so the app
needs the same three things the Mac gate uses: the vision bundle's own
`image_embeds` as a raw fp16 buffer, the prompt with its `<image>` ids rewritten
to extension ids `V + slot`, and the greedy continuation to match against.

`image_embeds` comes from the EXPORTED vision bundle rather than from the fp32
oracle on purpose — on device the tower's own output is what feeds the decoder,
so a fixture built from fp32 would be gating a chain that never runs.

Writes (default `_smoke/lfm25vl_ref/`):
    image_embeds.f16.bin   [256, 1024] fp16, row-major (what the app binds)
    vl_ref.json            prompt ids, expected ids, shapes, provenance

and prints the two Swift `ModelSpec` arrays to paste into PipelinedBench.

Run (in coreai-models/.venv, GPU, _GPU_LOCK held):
    ../coreai-models/.venv/bin/python _smoke/dump_lfm25vl_device_ref.py
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

from coreai_models.models.macos.lfm2_vl import lfm2_vl_configs_from_dict  # noqa: E402

DEFAULT_REF = Path(__file__).parent / "lfm2_5_vl_450m_ref_512x512.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


def swift_array(name: str, ids, per_line: int = 12) -> str:
    rows = [
        ", ".join(str(int(v)) for v in ids[i:i + per_line])
        for i in range(0, len(ids), per_line)
    ]
    body = ",\n                ".join(rows)
    return f"            {name}: [\n                {body},\n            ],"


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-VL-450M")
    ap.add_argument("--vision-mode", default="fp16")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--out", default=str(Path(__file__).parent / "lfm25vl_ref"))
    ap.add_argument("--expected", type=int, default=24,
                    help="greedy tokens to check on device (the app compares this many)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    vis_name = f"{short}_vision_{args.vision_mode}"
    root = exports_dir()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = np.load(args.ref)
    with open(Path(hf_snapshot(args.hf_id)) / "config.json") as f:
        raw = json.load(f)
    _, cfg = lfm2_vl_configs_from_dict(raw)
    image_token_id = int(raw["image_token_id"])

    opts = rt.SpecializationOptions.default()
    vm = await maybe_await(
        rt.AIModel.load(str(root / vis_name / f"{vis_name}.aimodel"), opts))
    vfn = await maybe_await(vm.load_function(vm.function_names[0]))
    patches = np.ascontiguousarray(ref["pixel_values"][0].astype(np.float16))
    res = await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
    embeds = np.asarray(res["image_embeds"].numpy()).astype(np.float16)
    n_img, hidden = embeds.shape
    print(f"image_embeds {embeds.shape} from {vis_name}")

    (out / "image_embeds.f16.bin").write_bytes(
        np.ascontiguousarray(embeds).tobytes())

    ids = ref["input_ids"][0].astype(np.int64).copy()
    img_pos = np.nonzero(ids == image_token_id)[0]
    if img_pos.size != n_img:
        raise SystemExit(f"{img_pos.size} placeholders vs {n_img} image tokens")
    ids[img_pos] = cfg.vocab_size + np.arange(n_img)
    expected = ref["gen_ids"][: args.expected].astype(np.int64)

    meta = {
        "hf_id": args.hf_id,
        "vision_bundle": vis_name,
        "image_tokens": int(n_img),
        "hidden": int(hidden),
        "vocab_size": int(cfg.vocab_size),
        "image_token_id": image_token_id,
        "prompt_ids": [int(v) for v in ids],
        "expected_ids": [int(v) for v in expected],
        "fixture": str(ref["_meta_image_url"]),
        "prompt": str(ref["_meta_prompt"]),
    }
    (out / "vl_ref.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out}/image_embeds.f16.bin ({embeds.nbytes} bytes) + vl_ref.json")

    print("\n--- paste into PipelinedBench ModelSpec ---")
    print(swift_array("oraclePrompt", ids))
    print(swift_array("oracleExpected", expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
