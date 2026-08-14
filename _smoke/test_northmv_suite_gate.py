#!/usr/bin/env python3
"""Token-exactness of the exported North-Micro-Vision bundles over the multi-case oracle.

`test_northmv_aimodel_gate.py` proves the chain is wired right on one fixture.
This answers the other question — does the compressed decoder say the same
things as fp32? — over `_smoke/north_micro_vision_instruct_suite_512.npz`
(9 image x prompt cases), counting greedy tokens per case.

Compare bundles by cases-exact, and always against an **fp16 baseline** rather
than fp32 alone: greedy decoding turns any near-tie into a different tail, so an
uncompressed bundle diverging on a case is the control, not a bug.

Run (in coreai-models/.venv, GPU, _GPU_LOCK held):
    ../coreai-models/.venv/bin/python _smoke/test_northmv_suite_gate.py --mode int8lin
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

KV_SEQ = 2048
DEFAULT_SUITE = Path(__file__).parent / "north_micro_vision_instruct_suite_512.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="CohereLabs/North-Micro-Vision-Instruct")
    ap.add_argument("--mode", default="int8lin")
    ap.add_argument("--exports", default=None)
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--decoder-asset", default=None)
    ap.add_argument("--show-text", action="store_true",
                    help="decode both continuations for every case that diverges — a count "
                         "says how much, only the text says whether it matters")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    root = Path(args.exports) if args.exports else exports_dir()
    vis_name = f"{short}_vision_fp16"
    dec_name = f"{short}_decode_{args.mode}"

    suite = np.load(args.suite)
    n_cases = int(suite["_meta_cases"])
    tile = int(suite["_meta_tile"])
    grid = tile // 16 // 2  # merged grid side
    n_tokens = grid * grid
    print(f"suite {args.suite}: {n_cases} cases, tile {tile} -> {n_tokens} image tokens")

    with open(Path(hf_snapshot(args.hf_id)) / "config.json") as f:
        raw = json.load(f)
    cfg = cohere_compass_config_from_dict(raw)
    image_token_id = int(raw["image_token_id"])

    opts = rt.SpecializationOptions.default()
    vm = await maybe_await(rt.AIModel.load(str(root / vis_name / f"{vis_name}.aimodel"), opts))
    vfn = await maybe_await(vm.load_function(vm.function_names[0]))

    if args.decoder_asset:
        dec_asset = Path(args.decoder_asset)
    else:
        aotc = root / f"{dec_name}_aotc" / f"{dec_name}.h16c.aimodelc"
        dec_asset = aotc if aotc.exists() else root / dec_name / f"{dec_name}.aimodel"
    dm = await maybe_await(rt.AIModel.load(str(dec_asset), opts))
    dfn = await maybe_await(dm.load_function(dm.function_names[0]))
    print(f"vision {vis_name} | decoder {dec_asset.name}\n")

    decode = None
    if args.show_text:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(root / dec_name / "tokenizer" / "tokenizer.json"))
        decode = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=True)  # noqa: E731

    kshape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, KV_SEQ, cfg.head_dim)
    total_tok = total_match = exact_cases = 0
    for case in range(n_cases):
        patches = np.ascontiguousarray(suite[f"case{case}_patches"].astype(np.float16))
        out = await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
        embeds = np.asarray(out["image_embeds"].numpy()).astype(np.float16)
        deep = np.asarray(out["deepstack_embeds"].numpy()).astype(np.float16)

        ids = suite[f"case{case}_ids"].astype(np.int64).copy()
        img_pos = np.nonzero(ids == image_token_id)[0]
        if img_pos.size != n_tokens:
            print(f"  case {case}: FAIL {img_pos.size} placeholders vs {n_tokens} tokens")
            continue
        img_start = int(img_pos[0])
        ids[img_pos] = cfg.vocab_size + np.arange(n_tokens)

        state = {
            "keyCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
            "valueCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
        }
        ie = rt.NDArray(np.ascontiguousarray(embeds))
        ds = rt.NDArray(np.ascontiguousarray(deep))
        ss = rt.NDArray(np.array([img_start + n_tokens], dtype=np.int32))
        sa = rt.NDArray(np.array([n_tokens - grid], dtype=np.int32))

        async def step(token: int, pos: int, state=state, ie=ie, ds=ds, ss=ss, sa=sa):
            res = await maybe_await(dfn(inputs={
                "input_ids": rt.NDArray(np.array([[token]], dtype=np.int32)),
                "position_ids": rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None]),
                "image_embeds": ie, "deepstack_embeds": ds,
                "rope_shift_start": ss, "rope_shift_amount": sa,
            }, state=state))
            return np.asarray(res["logits"].numpy())[0, -1]

        logits = None
        for i, token in enumerate(ids.tolist()):
            logits = await step(int(token), i)

        want = suite[f"case{case}_gen"].astype(np.int64)
        got = [int(logits.argmax())]
        for k in range(1, want.size):
            logits = await step(got[-1], ids.size + k - 1)
            got.append(int(logits.argmax()))
        got_arr = np.array(got, dtype=np.int64)

        match = int((got_arr == want).sum())
        first_bad = None if match == want.size else int(np.argmax(got_arr != want))
        total_tok += want.size
        total_match += match
        exact_cases += int(match == want.size)
        tag = "PASS" if match == want.size else f"FAIL @{first_bad}"
        print(f"  case {case}: {match}/{want.size} {tag}")
        if decode is not None and first_bad is not None:
            print(f"      fp32: {decode(want)!r}")
            print(f"    bundle: {decode(got_arr)!r}")

    print(f"\n{exact_cases}/{n_cases} cases token-exact, "
          f"{total_match}/{total_tok} tokens "
          f"({100 * total_match / max(total_tok, 1):.1f}%)")
    return 0 if exact_cases == n_cases else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
