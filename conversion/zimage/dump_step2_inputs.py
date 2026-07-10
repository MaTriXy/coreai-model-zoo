"""Dump the exact DiT inputs that make the fp16 graph NaN on the Mac, as fp16 raw
bins, so the identical tensors can be replayed on an iPhone.

The fp16 graph is clean at sampler steps 0 and 1 and NaNs at step 2, so the latent
entering step 2 is itself valid: run the loop on the Mac with the same fp16 bundle,
stop at step 2, write the 9 graph inputs, and record the Mac's verdict (NaN).
The device probe replays those bytes. Mac and device then differ only in hardware.

  python dump_step2_inputs.py <fp16_dit.aimodel> --out device_probe
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
ORA = os.path.join(HERE, "oracle")
ORDER = ("img_tokens", "cap_feats", "adaln", "x_cos", "x_sin", "cap_cos", "cap_sin",
         "x_pad_mask", "cap_pad_mask")


def load(n, s):
    return torch.from_numpy(np.fromfile(os.path.join(ORA, f"{n}.f32"), "<f4")).reshape(s).float()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dit")
    ap.add_argument("--out", default="device_probe")
    ap.add_argument("--tag", default="s256")
    args = ap.parse_args()
    out = os.path.join(HERE, args.out)
    os.makedirs(out, exist_ok=True)

    meta = json.load(open(os.path.join(ORA, f"meta_{args.tag}.json")))
    lat, steps, guid = meta["lat"], meta["steps"], meta["guidance"]
    sigmas = load(f"sigmas_{args.tag}", (steps + 1,))
    dsigma = torch.tensor([sigmas[i + 1] - sigmas[i] for i in range(steps)])

    from diffusers import ZImageTransformer2DModel
    print("[dump] loading transformer (host prep) ...", flush=True)
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()

    Lc, Lu = meta_o["cap_cond_L"], meta_o["cap_uncond_L"]
    cap_cond = load("cap_cond", (1, Lc, 2560))[0]
    cap_uncond = load("cap_uncond", (1, Lu, 2560))[0]

    fn = (await rt.AIModel.load(args.dit, rt.SpecializationOptions.default())).load_function("main")

    def nd16(t):
        return rt.NDArray(t.detach().cpu().to(torch.float16).contiguous())

    async def velocity(latent, cap, adaln, dump=None):
        ins = build_native_inputs(rm, latent, cap)
        ins["adaln"] = adaln
        if dump is not None:
            for k in ORDER:
                a = ins[k].detach().cpu().to(torch.float16).contiguous().numpy()
                a.tofile(os.path.join(dump, f"{k}.f16"))
            json.dump({k: list(ins[k].shape) for k in ORDER},
                      open(os.path.join(dump, "shapes.json"), "w"), indent=2)
        r = await fn(inputs={k: nd16(ins[k]) for k in ORDER})
        v = torch.as_tensor(r["velocity"].numpy().astype(np.float32))
        return unpatchify_velocity(rm, v, ins["x_size"], ins["n_img"]), v

    latent = load(f"noise_{args.tag}", (1, 16, lat, lat))
    with torch.no_grad():
        for s in range(3):
            t_norm = float(1.0 - sigmas[s])
            adaln = rm.t_embedder(torch.tensor([t_norm]) * rm.t_scale)
            lat_in = latent[0].unsqueeze(1)                       # [C,1,H,W]
            dump = out if s == 2 else None                        # capture step-2 cond inputs
            pos, raw = await velocity(lat_in, cap_cond, adaln, dump=dump)
            if s == 2:
                nan = bool(torch.isnan(raw).any())
                mx = float(raw[~torch.isnan(raw)].abs().max()) if (~torch.isnan(raw)).any() else 0.0
                print(f"[dump] MAC verdict on step-2 cond inputs: nan={nan} absmax={mx:.3f}", flush=True)
                json.dump({"mac_nan": nan, "mac_absmax": mx, "step": 2, "size": lat * 8},
                          open(os.path.join(out, "mac_verdict.json"), "w"), indent=2)
                break
            neg, _ = await velocity(lat_in, cap_uncond, adaln)
            pred = pos + guid * (pos - neg) if guid > 0 else pos
            latent = latent + dsigma[s] * (-pred).squeeze(1)[None]
            print(f"[dump] step {s} ok, latent|max|={float(latent.abs().max()):.2f}", flush=True)

    tot = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"[dump] wrote {out}/ ({tot/1e6:.1f} MB of fp16 inputs + shapes + mac_verdict)", flush=True)


meta_o = json.load(open(os.path.join(ORA, "meta.json")))
asyncio.run(main())
