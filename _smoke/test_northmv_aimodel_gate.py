#!/usr/bin/env python3
"""Engine gate for the exported North-Micro-Vision bundles — the .aimodel, not the torch.

Runs both graphs through `coreai.runtime` at the baked 16x16 merged grid and
checks them against `_smoke/north_micro_vision_instruct_ref_512x512.npz`:

  A. VISION — the tower's `image_embeds` (and the deepstack rows it feeds the
     decoder's first three layers) vs the fp32 oracle.

  B. DECODER — driven token by token with the ENGINE's own vision output bound
     to the static inputs, so a vision error cannot be masked by an oracle
     input. Prompt ids are rewritten to extension ids `V + slot`, and the
     rope-shift pair carries the Qwen3-VL contract this checkpoint shares.
     Token-exact against the oracle's greedy ids, or it failed.

Run (in coreai-models/.venv, GPU, _GPU_LOCK held; AOT-compiled decoder assumed —
a raw .aimodel re-specializes per prompt length and thrashes):
    ../coreai-models/.venv/bin/python _smoke/test_northmv_aimodel_gate.py --mode int8lin
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
DEFAULT_REF = Path(__file__).parent / "north_micro_vision_instruct_ref_512x512.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


def cos(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="CohereLabs/North-Micro-Vision-Instruct")
    ap.add_argument("--mode", default="int8lin")
    ap.add_argument("--exports", default=None)
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--decoder-asset", default=None)
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument("--bench-vision", type=int, default=0, metavar="N")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    root = Path(args.exports) if args.exports else exports_dir()
    vis_name = f"{short}_vision_fp16"
    dec_name = f"{short}_decode_{args.mode}"

    ref = np.load(args.ref)
    t, ph, pw = (int(v) for v in ref["image_grid_thw"][0])
    grid_h, grid_w = ph // 2, pw // 2
    n_tokens = grid_h * grid_w
    print(f"oracle {args.ref} | grid {ph}x{pw} patches -> {n_tokens} tokens")

    with open(Path(hf_snapshot(args.hf_id)) / "config.json") as f:
        raw = json.load(f)
    cfg = cohere_compass_config_from_dict(raw)
    image_token_id = int(raw["image_token_id"])
    failures: list[str] = []

    opts = rt.SpecializationOptions.default()

    # ---------------- A. vision --------------------------------------------
    print(f"\nA. vision engine ({vis_name})")
    vm = await maybe_await(rt.AIModel.load(str(root / vis_name / f"{vis_name}.aimodel"), opts))
    vfn = await maybe_await(vm.load_function(vm.function_names[0]))
    print(f"  fn {vfn.desc.name} in {vfn.desc.input_names} out {vfn.desc.output_names}")

    patches = np.ascontiguousarray(ref["pixel_values"].astype(np.float16))
    out = await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
    image_embeds = np.asarray(out["image_embeds"].numpy())
    deepstack_embeds = np.asarray(out["deepstack_embeds"].numpy())
    c = cos(image_embeds, ref["image_features"])
    print(f"  {'PASS' if c >= args.tol else 'FAIL'} image_embeds {image_embeds.shape} "
          f"cos {c:.6f}")
    if c < args.tol:
        failures.append("image_embeds")
    print(f"  deepstack_embeds {deepstack_embeds.shape} (not in the oracle; gated via the chain)")

    if args.bench_vision:
        import time

        times = []
        for _ in range(args.bench_vision):
            t0 = time.perf_counter()
            await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
            times.append((time.perf_counter() - t0) * 1e3)
        times.sort()
        print(f"  encode {args.bench_vision}x: median {times[len(times) // 2]:.1f} ms "
              f"(min {times[0]:.1f}, max {times[-1]:.1f})")

    if args.skip_decoder:
        print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
        return 0 if not failures else 1

    # ---------------- B. decoder -------------------------------------------
    if args.decoder_asset:
        dec_asset = Path(args.decoder_asset)
    else:
        aotc = root / f"{dec_name}_aotc" / f"{dec_name}.h16c.aimodelc"
        dec_asset = aotc if aotc.exists() else root / dec_name / f"{dec_name}.aimodel"
    print(f"\nB. decoder engine ({dec_asset.name})")
    if dec_asset.suffix == ".aimodel":
        print("  WARNING: raw .aimodel -- expect per-length re-JIT thrash")
    dm = await maybe_await(rt.AIModel.load(str(dec_asset), opts))
    dfn = await maybe_await(dm.load_function(dm.function_names[0]))
    print(f"  fn {dfn.desc.name} in {dfn.desc.input_names} state {dfn.desc.state_names}")

    ids = ref["input_ids"][0].astype(np.int64).copy()
    img_pos = np.nonzero(ids == image_token_id)[0]
    if img_pos.size != n_tokens:
        print(f"  FAIL {img_pos.size} placeholders vs {n_tokens} tokens")
        return 1
    img_start = int(img_pos[0])
    ids[img_pos] = cfg.vocab_size + np.arange(n_tokens)

    kshape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, KV_SEQ, cfg.head_dim)
    state = {
        "keyCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
        "valueCache": rt.NDArray(np.zeros(kshape, dtype=np.float16)),
    }
    ie = rt.NDArray(np.ascontiguousarray(image_embeds.astype(np.float16)))
    ds = rt.NDArray(np.ascontiguousarray(deepstack_embeds.astype(np.float16)))
    ss = rt.NDArray(np.array([img_start + n_tokens], dtype=np.int32))
    sa = rt.NDArray(np.array([n_tokens - max(grid_h, grid_w)], dtype=np.int32))

    async def step(token: int, pos: int) -> np.ndarray:
        inputs = {
            "input_ids": rt.NDArray(np.array([[token]], dtype=np.int32)),
            "position_ids": rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None]),
            "image_embeds": ie,
            "deepstack_embeds": ds,
            "rope_shift_start": ss,
            "rope_shift_amount": sa,
        }
        res = await maybe_await(dfn(inputs=inputs, state=state))
        return np.asarray(res["logits"].numpy())[0, -1]

    print(f"  prefilling {ids.size} tokens as S=1 steps ...")
    logits = None
    for i, token in enumerate(ids.tolist()):
        logits = await step(int(token), i)

    c = cos(logits, ref["logits_last"])
    print(f"  {'PASS' if c >= args.tol else 'FAIL'} logits_last cos {c:.6f}")
    if c < args.tol:
        failures.append("logits_last")

    want = ref["gen_ids"].astype(np.int64)
    got = [int(logits.argmax())]
    for k in range(1, want.size):
        logits = await step(got[-1], ids.size + k - 1)
        got.append(int(logits.argmax()))
    got_arr = np.array(got, dtype=np.int64)
    match = int((got_arr == want).sum())
    exact = match == want.size
    print(f"  {'PASS' if exact else 'FAIL'} greedy {match}/{want.size} token-exact")
    if not exact:
        first = int(np.argmax(got_arr != want))
        print(f"    first divergence at {first}: got {got_arr[first]} want {want[first]}")
        failures.append("gen_ids")

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
