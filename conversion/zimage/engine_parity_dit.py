"""Teacher-forced parity of the EXPORTED Core AI DiT bundle vs the fp32 oracle.

For each step/branch, build the DiT graph inputs from the oracle latent + caption
+ adaln (host prep via zimage_host), run the Core AI engine, unpatchify, and
compare the velocity to the pipeline's recorded output. This is the real
numerical gate for the full-depth exported graph (fp16 or int8lin).

  python engine_parity_dit.py <bundle.aimodel> [--dtype bf16]
"""
import argparse
import asyncio
import json
import os

import numpy as np
import torch

import coreai.runtime as rt
from zimage_host import build_native_inputs, unpatchify_velocity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")
DTYPE = torch.float16  # overridden by --dtype


def load(name, shape):
    return torch.from_numpy(np.fromfile(os.path.join(OUT, f"{name}.f32"), "<f4")).reshape(shape).float()


def corr(a, b):
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def nd(a):
    return rt.NDArray(a.detach().cpu().to(DTYPE).contiguous())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = ap.parse_args()
    global DTYPE
    DTYPE = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    build_inputs = build_native_inputs
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    lat, steps, branches = meta["lat"], meta["steps"], meta["branches"]
    Lc, Lu = meta["cap_cond_L"], meta["cap_uncond_L"]

    from diffusers import ZImageTransformer2DModel
    print("[engine-parity] loading transformer (fp32, host prep) ...", flush=True)
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()
    cap_cond = load("cap_cond", (1, Lc, 2560))[0]
    cap_uncond = load("cap_uncond", (1, Lu, 2560))[0]

    fn = (await rt.AIModel.load(args.bundle, rt.SpecializationOptions.default())).load_function("main")
    order = ("img_tokens", "cap_feats", "adaln", "x_cos", "x_sin", "cap_cos", "cap_sin",
             "x_pad_mask", "cap_pad_mask")

    worst = 1.0
    for s in range(steps):
        latent = load(f"latent_{s}", (1, 16, 1, lat, lat))[0]
        adaln = load(f"adaln_{s}", (1, 256))
        for cap, tag in [(cap_cond, "pos"), (cap_uncond, "neg")][:branches]:
            ins = build_inputs(rm, latent, cap)
            ins["adaln"] = adaln
            r = await fn(inputs={k: nd(ins[k]) for k in order})
            u = torch.as_tensor(r["velocity"].numpy().astype(np.float32))
            vel = unpatchify_velocity(rm, u, ins["x_size"], ins["n_img"])[None]
            ref = load(f"vel_{tag}_{s}", (1, 16, 1, lat, lat))
            c = corr(vel, ref)
            worst = c if np.isnan(c) else min(worst, c)
            print(f"  step {s} {tag}: corr {c:.6f}  max|d| {float((vel-ref).abs().max()):.3e}")
    ok = (not np.isnan(worst)) and worst > 0.997
    print(f"\n[engine-parity] worst corr = {worst:.6f}  {'PASS' if ok else 'CHECK'}")


asyncio.run(main())
