#!/usr/bin/env python3
"""Token-exactness of the EXPORTED LFM2.5-VL bundles over the multi-case oracle.

`test_lfm25vl_aimodel_gate.py` proves the chain is wired right on one fixture.
This answers the other question — does the compressed decoder say the same
things as fp32? — by running the engine over every case in
`_smoke/lfm2_5_vl_450m_suite_512.npz` (see `lfm25vl_suite_ref.py`) and counting
greedy tokens that match, per case.

Cosine on a logits vector is deliberately NOT the criterion here: it is a
single-position summary that hides the argmax flips that change what a VLM
says. Compare bundles by "cases fully exact / tokens matched" instead.

Run (in coreai-models/.venv, GPU, _GPU_LOCK held; AOT-compiled decoder assumed
-- see --decoder-asset in test_lfm25vl_aimodel_gate.py):
    ../coreai-models/.venv/bin/python _smoke/test_lfm25vl_suite_gate.py --mode int8lin
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

KV_SEQ = 2048
DEFAULT_SUITE = Path(__file__).parent / "lfm2_5_vl_450m_suite_512.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-VL-450M")
    ap.add_argument("--mode", default="int8lin")
    ap.add_argument("--vision-mode", default="fp16")
    ap.add_argument("--exports", default=None)
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--decoder-asset", default=None)
    ap.add_argument(
        "--show-text",
        action="store_true",
        help="decode and print both continuations for every case that diverges. "
        "A token count says HOW MUCH a bundle disagrees; only the text says "
        "whether the disagreement matters (a synonym at a near-tie vs a wrong "
        "answer), and greedy decoding makes every token after the first "
        "divergence differ by construction.",
    )
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    root = Path(args.exports) if args.exports else exports_dir()
    vis_name = f"{short}_vision_{args.vision_mode}"
    dec_name = f"{short}_decode_{args.mode}"

    suite = np.load(args.suite)
    n_cases = int(suite["_meta_cases"])
    print(f"suite {args.suite}: {n_cases} cases, tile {int(suite['_meta_tile'])}")

    with open(Path(hf_snapshot(args.hf_id)) / "config.json") as f:
        raw = json.load(f)
    _, cfg = lfm2_vl_configs_from_dict(raw)
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

    kshape = (cfg.num_full_layers, 1, cfg.num_key_value_heads, KV_SEQ, cfg.head_dim)
    cshape = (cfg.num_conv_layers, 1, cfg.hidden_size, cfg.conv_state_width)

    decode = None
    if args.show_text:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(root / dec_name / "tokenizer" / "tokenizer.json"))
        decode = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=True)  # noqa: E731

    total_tok = total_match = exact_cases = 0
    for case in range(n_cases):
        patches = np.ascontiguousarray(suite[f"case{case}_patches"].astype(np.float16))
        out = await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
        embeds = np.asarray(out["image_embeds"].numpy()).astype(np.float16)
        n_img = embeds.shape[0]

        ids = suite[f"case{case}_ids"].astype(np.int64).copy()
        img_pos = np.nonzero(ids == image_token_id)[0]
        if img_pos.size != n_img:
            print(f"case {case}: FAIL {img_pos.size} placeholders vs {n_img} tokens")
            continue
        ids[img_pos] = cfg.vocab_size + np.arange(n_img)

        # Fresh state per case: the engine's states are mutated in place.
        state = {
            "keyCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
            "valueCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
            "convState": rt.NDArray(np.zeros(cshape, dtype=np.float16)),
        }
        ie = rt.NDArray(np.ascontiguousarray(embeds))

        async def step(token: int, pos: int, state=state, ie=ie) -> np.ndarray:
            inputs = {
                "input_ids": rt.NDArray(np.array([[token]], dtype=np.int32)),
                "position_ids": rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None]),
                "image_embeds": ie,
            }
            res = await maybe_await(dfn(inputs=inputs, state=state))
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

    print(
        f"\n{exact_cases}/{n_cases} cases token-exact, "
        f"{total_match}/{total_tok} tokens ({100 * total_match / max(total_tok, 1):.1f}%)"
    )
    return 0 if exact_cases == n_cases else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
