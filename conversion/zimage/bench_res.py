"""Bench one DiT bundle across resolutions. GPU-preferred specialization (the
default path tries ANE, which ANECCompile-fails on the int8 dynamic graph).

  python bench_res.py <bundle.aimodel> [--gpu] 32 64 128     # latent sides
"""
import argparse
import asyncio
import time

import numpy as np
import torch

import coreai.runtime as rt
from zimage_host import build_native_inputs

ORDER = ("img_tokens", "cap_feats", "adaln", "x_cos", "x_sin", "cap_cos", "cap_sin",
         "x_pad_mask", "cap_pad_mask")


def load(n, s):
    return torch.from_numpy(np.fromfile(f"oracle/{n}.f32", "<f4")).reshape(s).float()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("lats", nargs="+", type=int)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    from diffusers import ZImagePipeline
    rm = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu").transformer
    cap = load("cap_cond", (1, 18, 2560))[0]
    adaln = load("adaln_0", (1, 256))

    spec = (rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
            if args.gpu else rt.SpecializationOptions.default())
    t0 = time.time()
    fn = (await rt.AIModel.load(args.bundle, spec)).load_function("main")
    print(f"[load] {time.time()-t0:.1f}s  {args.bundle.split('/')[-1]}", flush=True)

    for lat in args.lats:
        latent = torch.randn(16, 1, lat, lat)
        ins = build_native_inputs(rm, latent, cap)
        ins["adaln"] = adaln
        payload = {k: rt.NDArray(ins[k].detach().to(torch.bfloat16).contiguous()) for k in ORDER}
        try:
            await fn(inputs=payload)                       # warm / specialize
            t = time.time()
            for _ in range(3):
                r = await fn(inputs=payload)
            dt = (time.time() - t) / 3
            o = torch.as_tensor(r["velocity"].numpy().astype(np.float32))
            px = lat * 8
            print(f"  {px:>5}px (n_img={ (lat//2)**2 }): {dt:6.3f} s/fwd  nan={bool(torch.isnan(o).any())}", flush=True)
        except Exception as e:
            print(f"  {lat*8}px: FAIL {type(e).__name__}: {str(e)[:80]}", flush=True)


asyncio.run(main())
