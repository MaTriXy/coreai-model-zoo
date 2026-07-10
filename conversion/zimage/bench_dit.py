"""Benchmark exported DiT bundles (s/forward) on real oracle inputs.

Speed is numerics-independent, so a NaN-producing config is still measurable —
which is how the int8-kernel question gets answered without first fixing overflow.

  python bench_dit.py <bundle.aimodel>[:dtype] ...   # dtype = bf16 (default) | fp16
"""
import asyncio
import json
import os
import sys
import time

import numpy as np
import torch

import coreai.runtime as rt
from zimage_host import build_native_inputs

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")
ORDER = ("img_tokens", "cap_feats", "adaln", "x_cos", "x_sin", "cap_cos", "cap_sin",
         "x_pad_mask", "cap_pad_mask")


def load(n, s):
    return torch.from_numpy(np.fromfile(os.path.join(OUT, f"{n}.f32"), "<f4")).reshape(s).float()


async def main():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    lat, Lc = meta["lat"], meta["cap_cond_L"]
    from diffusers import ZImagePipeline
    rm = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu").transformer
    cap = load("cap_cond", (1, Lc, 2560))[0]
    latent = load("latent_0", (1, 16, 1, lat, lat))[0]
    adaln = load("adaln_0", (1, 256))
    ins = build_native_inputs(rm, latent, cap)
    ins["adaln"] = adaln

    for spec in sys.argv[1:]:
        path, _, dt = spec.partition(":")
        dtype = torch.float16 if dt == "fp16" else torch.bfloat16
        payload = {k: rt.NDArray(ins[k].detach().cpu().to(dtype).contiguous()) for k in ORDER}
        try:
            fn = (await rt.AIModel.load(path, rt.SpecializationOptions.default())).load_function("main")
            await fn(inputs=payload)                       # warm
            t = time.time()
            for _ in range(3):
                r = await fn(inputs=payload)
            dtm = (time.time() - t) / 3
            o = torch.as_tensor(r["velocity"].numpy().astype(np.float32))
            nan = bool(torch.isnan(o).any())
            print(f"{os.path.basename(path):58s} {dtm:6.3f} s/fwd   nan={nan}")
        except Exception as e:
            print(f"{os.path.basename(path):58s} FAIL {type(e).__name__}: {str(e)[:70]}")


asyncio.run(main())
