#!/usr/bin/env python3
"""Gate: chunked prefill must agree with S=1-only prefill on real images.

This is the regression gate for the pf32 lesson (2026-08-15): the in-graph
fp16 doubling-inverse in the chunked GDN scan overflows CONTENT-DEPENDENTLY —
a chunk size can pass an oracle suite and still collapse to "!" spam on the
next real photo (weak-decay image spans; reproduced on two Pexels photos at
PF=32, one of which only failed through the app's CGContext resize). An oracle
is not needed to catch that class: the S=1 path (single-step scan, no inverse)
is the in-family reference, and greedy tokens from both prefills must agree.

Per image: run the FULL chain (NumPy preprocess -> tower -> splice/mRoPE ->
decoder) twice — once prefilling through the bundle's "prefill" chunks +
S=1 remainder, once through S=1 steps only — then greedy-decode N tokens from
each and compare. FAIL on: non-finite logits, a degenerate run (any token
repeated >= 8 times in a row), or token disagreement beyond near-tie noise
(> 25% of positions).

Run (coreai-models/.venv, GPU, _GPU_LOCK held):
    ../coreai-models/.venv/bin/python _smoke/test_qwen38vl_chunk_consistency.py \
        [--decoder-asset <.aimodelc>] [--images img1.jpg img2.jpg ...]
Default images: the 3 suite tiles in _smoke/qwen38vl_images/.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import sys
from pathlib import Path

import numpy as np

import coreai.runtime as rt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "conversion"))
from _paths import exports_dir  # noqa: E402

from qwen38vl_host import mrope_positions, splice_embeds  # noqa: E402
from qwen38vl_preprocess import preprocess  # noqa: E402

KV = 2048
N_GEN = 48
SYS = ("Reasoning effort is set to low. Please think carefully through the task, "
       "validate key assumptions, consider plausible alternatives, and prioritize "
       "correctness, consistency, and clarity in the final answer.")
PROMPT = "What is in this image?"


async def maybe(x):
    return await x if inspect.isawaitable(x) else x


def degenerate(tokens: list[int]) -> bool:
    run = 1
    for a, b in zip(tokens, tokens[1:]):
        run = run + 1 if a == b else 1
        if run >= 8:
            return True
    return False


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", default=None, help="decoder bundle dir (tokenizer+embeds)")
    ap.add_argument("--decoder-asset", default=None, help=".aimodel(c) to load")
    ap.add_argument("--pf", type=int, default=None, help="chunk size (default: from bundle name)")
    ap.add_argument("--images", nargs="*", default=None)
    args = ap.parse_args()

    from PIL import Image
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    root = exports_dir()
    dec_dir = Path(args.decoder) if args.decoder else next(
        p for p in sorted(root.glob("qwen3_8_27b_vl_decode_int8hu_block32_sym_pf*"))
        if p.is_dir() and not p.name.endswith("_aotc"))
    pf = args.pf or int(dec_dir.name.rsplit("_pf", 1)[1])
    asset = Path(args.decoder_asset) if args.decoder_asset else (
        dec_dir.parent / f"{dec_dir.name}_aotc" / f"{dec_dir.name}.h16c.aimodelc")
    images = args.images or sorted(
        str(p) for p in (Path(__file__).parent / "qwen38vl_images").glob("*_512.png"))
    print(f"decoder {asset.name} | PF {pf} | {len(images)} images")

    opts = rt.SpecializationOptions.default()
    vm = await maybe(rt.AIModel.load(
        str(root / "qwen3_8_27b_vision_fp16/qwen3_8_27b_vision_fp16.aimodel"), opts))
    vfn = await maybe(vm.load_function("main"))
    dm = await maybe(rt.AIModel.load(str(asset), opts))
    dfn = await maybe(dm.load_function("main"))
    pfn = await maybe(dm.load_function("prefill"))
    table = load_file(str(dec_dir / "embed_tokens.safetensors"))["embed_tokens.weight"]
    tok = Tokenizer.from_file(str(dec_dir / "tokenizer/tokenizer.json"))

    text = (f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n<|vision_start|>"
            + "<|image_pad|>" * 256
            + f"<|vision_end|>{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n")
    ids = np.array(tok.encode(text).ids, np.int64)
    pos, delta = mrope_positions(ids, [(1, 32, 32)])
    S = ids.size

    fails = []
    for img_path in images:
        img = np.asarray(Image.open(img_path).convert("RGB"), np.uint8)
        patches = preprocess(img).astype(np.float16)
        out = await maybe(vfn(inputs={"patches": rt.NDArray(
            np.ascontiguousarray(patches))}))
        emb_img = np.asarray(out["image_embeds"].numpy()).astype(np.float16)
        embeds = splice_embeds(ids, table, emb_img)

        async def run(chunked: bool) -> tuple[list[int], bool]:
            state = {
                "keyCache": rt.NDArray(np.zeros((16, 1, 4, KV, 256), np.float16)),
                "valueCache": rt.NDArray(np.zeros((16, 1, 4, KV, 256), np.float16)),
                "convState": rt.NDArray(np.zeros((48, 1, 10240, 3), np.float16)),
                "recState": rt.NDArray(np.zeros((48, 1, 48, 128, 128), np.float16)),
            }
            async def call(fn, x, ramp, p3):
                res = await maybe(fn(inputs={
                    "inputs_embeds": rt.NDArray(np.ascontiguousarray(x[None])),
                    "position_ids": rt.NDArray(np.arange(ramp, dtype=np.int32)[None]),
                    "pos_t": rt.NDArray(np.ascontiguousarray(p3[0:1].astype(np.int32))),
                    "pos_h": rt.NDArray(np.ascontiguousarray(p3[1:2].astype(np.int32))),
                    "pos_w": rt.NDArray(np.ascontiguousarray(p3[2:3].astype(np.int32))),
                }, state=state))
                return np.asarray(res["logits"].numpy())[0, -1]
            row = None
            o = 0
            if chunked:
                while o + pf <= S:
                    row = await call(pfn, embeds[o:o + pf], o + pf, pos[:, o:o + pf])
                    o += pf
            while o < S:
                row = await call(dfn, embeds[o:o + 1], o + 1, pos[:, o:o + 1])
                o += 1
            finite = bool(np.isfinite(row).all())
            gen = [int(row.argmax())]
            for k in range(1, N_GEN):
                if gen[-1] in (248044, 248046):
                    break
                p3 = np.full((3, 1), S + k - 1 + delta, np.int32)
                row = await call(dfn, table[gen[-1]][None].copy(), S + k, p3)
                gen.append(int(row.argmax()))
            return gen, finite

        g_chunk, fin_c = await run(chunked=True)
        g_step, fin_s = await run(chunked=False)
        n = min(len(g_chunk), len(g_step))
        agree = sum(a == b for a, b in zip(g_chunk, g_step)) / max(n, 1)
        bad = (not fin_c or degenerate(g_chunk) or agree < 0.75)
        name = Path(img_path).name
        print(f"  {'FAIL' if bad else 'PASS'} {name}: finite {fin_c}/{fin_s} "
              f"degenerate {degenerate(g_chunk)} agree {agree:.2f}")
        if bad:
            print(f"    chunk: {tok.decode(g_chunk, skip_special_tokens=True)[:120]!r}")
            print(f"    step : {tok.decode(g_step, skip_special_tokens=True)[:120]!r}")
            fails.append(name)

    print("ALL PASS" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
