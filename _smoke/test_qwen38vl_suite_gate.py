#!/usr/bin/env python3
"""Token-exactness of the EXPORTED Qwen3.8-27B vision path over the oracle suite.

Runs the ENTIRE shipped chain per case, on the python runtime:

    uint8 image -> NumPy preprocess (qwen38vl_preprocess) -> vision .aimodel
    -> embed splice + host mRoPE planes (qwen38vl_host)
    -> embeddings decoder .aimodel ("prefill" S=32 chunks + "main" S=1)
    -> greedy tokens, compared against the bf16 HF oracle's

Cosine is deliberately NOT the criterion (single-position summary; hides argmax
flips) — cases fully exact / tokens matched, the family rule. Greedy makes
every token after a first divergence differ by construction, so --show-text
prints both continuations to judge whether a divergence matters.

Also reports the measured card numbers: tower ms, prefill tok/s, decode tok/s.

Run (coreai-models/.venv, GPU, _GPU_LOCK held, python-GPU SOLO):
    ../coreai-models/.venv/bin/python _smoke/test_qwen38vl_suite_gate.py
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
import time
from pathlib import Path

import numpy as np

import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import exports_dir  # noqa: E402

from qwen38vl_host import mrope_positions, splice_embeds  # noqa: E402
from qwen38vl_preprocess import preprocess  # noqa: E402

DEFAULT_SUITE = Path(__file__).parent / "qwen38vl_suite_512.npz"
KV_SEQ = 2048
PF = 32
# hybrid state shapes (Qwen3.8-27B text config)
N_FULL, N_LIN = 16, 48
N_KV_HEADS, HEAD_DIM = 4, 256
CONV_DIM, CONV_W = 2 * 16 * 128 + 48 * 128, 3
NUM_V, DK, DV = 48, 128, 128


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--vision", default=None)
    ap.add_argument("--decoder", default=None)
    ap.add_argument("--mode", default="int8hu_block32_sym")
    ap.add_argument("--show-text", action="store_true")
    ap.add_argument("--gpu-pref", action="store_true",
                    help="load the DECODER with preferred=GPU specialization "
                         "(ANERegionFormationPass asserts on the multifunction "
                         "prefill graph under default options)")
    ap.add_argument("--decoder-asset", default=None,
                    help="explicit decoder asset path (e.g. an AOT .aimodelc)")
    args = ap.parse_args()

    root = exports_dir()
    vis_dir = Path(args.vision) if args.vision else root / "qwen3_8_27b_vision_fp16"
    dec_dir = (Path(args.decoder) if args.decoder
               else root / f"qwen3_8_27b_vl_decode_{args.mode}_pf{PF}")

    suite = np.load(args.suite)
    n_cases = int(suite["_meta_cases"])
    print(f"suite {args.suite}: {n_cases} cases")

    opts = rt.SpecializationOptions.default()
    vm = await maybe(rt.AIModel.load(str(vis_dir / f"{vis_dir.name}.aimodel"), opts))
    vfn = await maybe(vm.load_function(vm.function_names[0]))
    dec_opts = (rt.SpecializationOptions.from_preferred_compute_unit_kind(
        rt.ComputeUnitKind.gpu()) if args.gpu_pref else opts)
    dec_asset = (Path(args.decoder_asset) if args.decoder_asset
                 else dec_dir / f"{dec_dir.name}.aimodel")
    dm = await maybe(rt.AIModel.load(str(dec_asset), dec_opts))
    fnames = list(dm.function_names)
    dfn = await maybe(dm.load_function("main"))
    pfn = await maybe(dm.load_function("prefill")) if "prefill" in fnames else None
    print(f"vision {vis_dir.name} | decoder {dec_dir.name} functions {fnames}")

    from safetensors.numpy import load_file

    embed_table = load_file(
        str(dec_dir / "embed_tokens.safetensors"))["embed_tokens.weight"]
    print(f"embed table {embed_table.shape} {embed_table.dtype}")

    decode_txt = None
    if args.show_text:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(dec_dir / "tokenizer" / "tokenizer.json"))
        decode_txt = lambda ids: tok.decode(  # noqa: E731
            [int(i) for i in ids], skip_special_tokens=True)

    kshape = (N_FULL, 1, N_KV_HEADS, KV_SEQ, HEAD_DIM)
    cshape = (N_LIN, 1, CONV_DIM, CONV_W)
    rshape = (N_LIN, 1, NUM_V, DK, DV)

    total_tok = total_match = exact_cases = 0
    t_tower = t_prefill_tok = t_prefill_s = t_dec_tok = t_dec_s = 0.0
    for case in range(n_cases):
        img = suite[f"image{int(suite[f'case{case}_image_idx'])}_u8"]
        t0 = time.perf_counter()
        patches = preprocess(img).astype(np.float16)  # NumPy host path, NOT HF
        out = await maybe(vfn(inputs={"patches": rt.NDArray(
            np.ascontiguousarray(patches))}))
        embeds_img = np.asarray(out["image_embeds"].numpy()).astype(np.float16)
        t_tower += time.perf_counter() - t0

        ids = suite[f"case{case}_ids"].astype(np.int64)
        grid = tuple(int(v) for v in suite[f"case{case}_grid_thw"][0])
        pos, delta = mrope_positions(ids, [grid])
        embeds = splice_embeds(ids, embed_table, embeds_img)
        S = ids.size

        state = {
            "keyCache": rt.NDArray(np.zeros(kshape, np.float16)),
            "valueCache": rt.NDArray(np.zeros(kshape, np.float16)),
            "convState": rt.NDArray(np.zeros(cshape, np.float16)),
            "recState": rt.NDArray(np.zeros(rshape, np.float16)),
        }

        async def call(fn, x, ramp_len, p3, state=state):
            inputs = {
                "inputs_embeds": rt.NDArray(np.ascontiguousarray(x[None])),
                "position_ids": rt.NDArray(np.arange(ramp_len, dtype=np.int32)[None]),
                "pos_t": rt.NDArray(np.ascontiguousarray(p3[0:1].astype(np.int32))),
                "pos_h": rt.NDArray(np.ascontiguousarray(p3[1:2].astype(np.int32))),
                "pos_w": rt.NDArray(np.ascontiguousarray(p3[2:3].astype(np.int32))),
            }
            res = await maybe(fn(inputs=inputs, state=state))
            return np.asarray(res["logits"].numpy())[0, -1]

        # chunked prefill: S=PF chunks through "prefill", remainder S=1 via "main"
        t0 = time.perf_counter()
        row = None
        o = 0
        while o + PF <= S and pfn is not None:
            row = await call(pfn, embeds[o:o + PF], o + PF, pos[:, o:o + PF])
            o += PF
        while o < S:
            row = await call(dfn, embeds[o:o + 1], o + 1, pos[:, o:o + 1])
            o += 1
        t_prefill_s += time.perf_counter() - t0
        t_prefill_tok += S

        want = suite[f"case{case}_gen"].astype(np.int64)
        got = [int(row.argmax())]
        t0 = time.perf_counter()
        for k in range(1, want.size):
            p3 = np.full((3, 1), S + k - 1 + delta, dtype=np.int32)
            row = await call(dfn, embed_table[got[-1]][None].copy(), S + k, p3)
            got.append(int(row.argmax()))
        t_dec_s += time.perf_counter() - t0
        t_dec_tok += want.size - 1
        got_arr = np.array(got, dtype=np.int64)

        match = int((got_arr == want).sum())
        first_bad = None if match == want.size else int(np.argmax(got_arr != want))
        total_tok += want.size
        total_match += match
        exact_cases += int(match == want.size)
        tag = "PASS" if match == want.size else f"FAIL @{first_bad}"
        print(f"  case {case}: {match}/{want.size} {tag}")
        if decode_txt is not None and first_bad is not None:
            print(f"    oracle: {decode_txt(want)!r}")
            print(f"    bundle: {decode_txt(got_arr)!r}")

    print(f"\n{exact_cases}/{n_cases} cases token-exact, "
          f"{total_match}/{total_tok} tokens "
          f"({100 * total_match / max(total_tok, 1):.1f}%)")
    print(f"measured: tower {1000 * t_tower / n_cases:.0f} ms/img | "
          f"prefill {t_prefill_tok / t_prefill_s:.1f} tok/s | "
          f"decode {t_dec_tok / t_dec_s:.1f} tok/s")
    return 0 if exact_cases == n_cases else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
