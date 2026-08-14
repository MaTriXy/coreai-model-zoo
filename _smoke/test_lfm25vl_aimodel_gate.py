#!/usr/bin/env python3
"""Engine gate for the exported LFM2.5-VL bundles — the .aimodel, not the torch.

Runs both graphs through `coreai.runtime` on the fixed 32x32 grid and checks
them against the fp32 oracle `_smoke/lfm2_5_vl_450m_ref_512x512.npz`:

  A. VISION — the exported tower+projector on the oracle's pixel_values,
     ``image_embeds`` vs the oracle's image_features (cos >= --tol).

  B. DECODER — the S=1 decode bundle driven token by token: the prompt's
     ``<image>`` ids rewritten to extension ids ``V + slot``, the ENGINE's own
     image_embeds (stage A's output, not the oracle's) bound to the static
     input, then 48 greedy steps compared to the oracle's gen_ids. Token-exact
     or it failed -- cosine on logits hides exactly the argmax flips that make
     a VLM describe a different picture.

Feeding stage A's output into stage B is deliberate: it gates the CHAIN, the
way the app runs it, so a vision error cannot be masked by an oracle input.

Run (in coreai-models/.venv, GPU, with the _GPU_LOCK held):
    ../coreai-models/.venv/bin/python _smoke/test_lfm25vl_aimodel_gate.py \
        [--mode int8lin] [--vision-mode fp16] [--exports DIR]
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

KV_SEQ = 2048  # TRACE_KV_CACHE_SEQ_LEN
DEFAULT_REF = Path(__file__).parent / "lfm2_5_vl_450m_ref_512x512.npz"


async def maybe_await(x):
    return await x if inspect.isawaitable(x) else x


def cos(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-VL-450M")
    ap.add_argument("--mode", default="int8lin")
    ap.add_argument("--vision-mode", default="fp16")
    ap.add_argument("--exports", default=None)
    ap.add_argument(
        "--decoder-asset",
        default=None,
        help="path to the decoder asset. DEFAULT: the AOT-compiled "
        "<name>_aotc/<name>.h16c.aimodelc if it exists, else the raw .aimodel. "
        "Driving the raw .aimodel from python re-JIT-specializes the graph for "
        "every prompt length and thrashes (multi-GB scratch, no output); "
        "AOT with --expect-frequent-reshapes is the fix. Build it with: "
        "xcrun coreai-build compile <name>.aimodel --platform macOS "
        "--preferred-compute gpu --expect-frequent-reshapes --architecture h16c "
        "--output <name>_aotc",
    )
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument(
        "--bench-vision",
        type=int,
        default=0,
        metavar="N",
        help="after gating, time N vision encodes and report the median. The tower "
        "is a fixed-shape graph, so the python runtime times it honestly (no "
        "re-specialization per call); it is the dominant time-to-first-token term.",
    )
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    root = Path(args.exports) if args.exports else exports_dir()
    vis_name = f"{short}_vision_{args.vision_mode}"
    dec_name = f"{short}_decode_{args.mode}"

    ref = np.load(args.ref)
    grid = tuple(int(v) for v in ref["spatial_shapes"][0])
    print(f"oracle {args.ref} (grid {grid[0]}x{grid[1]})")

    snap = hf_snapshot(args.hf_id)
    with open(Path(snap) / "config.json") as f:
        raw = json.load(f)
    _, text_cfg = lfm2_vl_configs_from_dict(raw)
    image_token_id = int(raw["image_token_id"])
    failures: list[str] = []

    # ---------------- A. vision -------------------------------------------
    print(f"\nA. vision engine ({vis_name})")
    opts = rt.SpecializationOptions.default()
    vm = await maybe_await(rt.AIModel.load(str(root / vis_name / f"{vis_name}.aimodel"), opts))
    vfn = await maybe_await(vm.load_function(vm.function_names[0]))
    print(f"  fn {vfn.desc.name} in {vfn.desc.input_names} out {vfn.desc.output_names}")

    patches = np.ascontiguousarray(ref["pixel_values"][0].astype(np.float16))
    out = await maybe_await(vfn(inputs={"patches": rt.NDArray(patches)}))
    image_embeds = np.asarray(out["image_embeds"].numpy())
    want = ref["image_features"].reshape(-1, ref["image_features"].shape[-1])
    c = cos(image_embeds, want)
    print(f"  {'PASS' if c >= args.tol else 'FAIL'} image_embeds {image_embeds.shape} "
          f"cos {c:.6f}  max|d| {np.abs(image_embeds.astype(np.float64) - want).max():.3e}")
    if c < args.tol:
        failures.append("image_embeds")

    # ---------------- B. decoder ------------------------------------------
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

    if args.decoder_asset:
        dec_asset = Path(args.decoder_asset)
    else:
        aotc = root / f"{dec_name}_aotc" / f"{dec_name}.h16c.aimodelc"
        dec_asset = aotc if aotc.exists() else root / dec_name / f"{dec_name}.aimodel"
    print(f"\nB. decoder engine ({dec_asset.name})")
    if dec_asset.suffix == ".aimodel":
        print("  WARNING: raw .aimodel -- expect per-length re-JIT thrash; see --decoder-asset")
    dm = await maybe_await(rt.AIModel.load(str(dec_asset), opts))
    dfn = await maybe_await(dm.load_function(dm.function_names[0]))
    print(f"  fn {dfn.desc.name} in {dfn.desc.input_names} state {dfn.desc.state_names}")

    n_img = image_embeds.shape[0]
    ids = ref["input_ids"][0].astype(np.int64).copy()
    img_pos = np.nonzero(ids == image_token_id)[0]
    if img_pos.size != n_img:
        print(f"  FAIL prompt has {img_pos.size} placeholders, encoder made {n_img} tokens")
        return 1
    ids[img_pos] = text_cfg.vocab_size + np.arange(n_img)

    cfg = text_cfg
    kshape = (cfg.num_full_layers, 1, cfg.num_key_value_heads, KV_SEQ, cfg.head_dim)
    key = rt.NDArray(np.zeros(kshape, dtype=np.float16))
    val = rt.NDArray(np.zeros(kshape, dtype=np.float16))
    conv = rt.NDArray(
        np.zeros((cfg.num_conv_layers, 1, cfg.hidden_size, cfg.conv_state_width),
                 dtype=np.float16)
    )
    ie = rt.NDArray(np.ascontiguousarray(image_embeds.astype(np.float16)))
    state = {"keyCache": key, "valueCache": val, "convState": conv}

    async def step(token: int, pos: int) -> np.ndarray:
        inputs = {
            "input_ids": rt.NDArray(np.array([[token]], dtype=np.int32)),
            "position_ids": rt.NDArray(np.arange(pos + 1, dtype=np.int32)[None]),
            "image_embeds": ie,
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

    want_ids = ref["gen_ids"].astype(np.int64)
    got: list[int] = [int(logits.argmax())]
    for step_i in range(1, args.max_new_tokens):
        logits = await step(got[-1], ids.size + step_i - 1)
        got.append(int(logits.argmax()))

    got_arr = np.array(got, dtype=np.int64)
    n_match = int((got_arr == want_ids[: got_arr.size]).sum())
    exact = n_match == got_arr.size
    print(f"  {'PASS' if exact else 'FAIL'} greedy {n_match}/{got_arr.size} token-exact")
    if not exact:
        first = int(np.argmax(got_arr != want_ids[: got_arr.size]))
        print(f"    first divergence at {first}: got {got_arr[first]} want {want_ids[first]}")
        failures.append("gen_ids")

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
