"""Teacher-forced parity: NativeZDiT (fp32, the SHIPPED wrapper) vs the captured oracle.

For each step s and branch (cond/uncond), rebuild the DiT graph inputs from
the oracle latent + caption + adaln, run the wrapper, unpatchify, and compare the
velocity to the pipeline's recorded output. Confirms the wrapper +
host prep reproduce the stock diffusers transformer at the target resolution
(before any Core AI export / quantization).
"""
import json
import os

import numpy as np
import torch

from zimage_dit_native import NativeZDiT
from zimage_host import build_native_inputs, unpatchify_velocity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")


def load(name, shape):
    return torch.from_numpy(np.fromfile(os.path.join(OUT, f"{name}.f32"), "<f4")).reshape(shape).float()


def corr(a, b):
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def main():
    meta = json.load(open(os.path.join(OUT, "meta.json")))
    lat, steps, branches = meta["lat"], meta["steps"], meta["branches"]
    Lc, Lu = meta["cap_cond_L"], meta["cap_uncond_L"]
    print(f"[parity] size={meta['size']} lat={lat} steps={steps} branches={branches} "
          f"Lc={Lc} Lu={Lu}")

    from diffusers import ZImageTransformer2DModel
    print("[parity] loading transformer (fp32) ...", flush=True)
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()
    dit = NativeZDiT(rm).eval()

    cap_cond = load("cap_cond", (1, Lc, 2560))[0]
    cap_uncond = load("cap_uncond", (1, Lu, 2560))[0]

    worst = 1.0
    with torch.no_grad():
        for s in range(steps):
            latent = load(f"latent_{s}", (1, 16, 1, lat, lat))[0]     # [C,1,H,W]
            adaln = load(f"adaln_{s}", (1, 256))
            for br, (cap, tag) in enumerate([(cap_cond, "pos"), (cap_uncond, "neg")]):
                if br >= branches:
                    continue
                ins = build_native_inputs(rm, latent, cap)
                u = dit(ins["img_tokens"], ins["cap_feats"], adaln,
                        ins["x_cos"], ins["x_sin"], ins["cap_cos"], ins["cap_sin"],
                        ins["x_pad_mask"], ins["cap_pad_mask"])
                vel = unpatchify_velocity(rm, u, ins["x_size"], ins["n_img"])[None]  # [1,C,1,H,W]
                ref = load(f"vel_{tag}_{s}", (1, 16, 1, lat, lat))
                c = corr(vel, ref)
                md = float((vel - ref).abs().max())
                worst = min(worst, c)
                print(f"  step {s} {tag}: corr {c:.6f}  max|d| {md:.3e}")

    print(f"\n[parity] worst corr = {worst:.6f}  "
          f"{'PASS' if worst > 0.999 else 'CHECK'}")


if __name__ == "__main__":
    main()
