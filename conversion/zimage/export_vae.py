"""Export the Z-Image VAE decoder (16ch AutoencoderKL) to Core AI.

Graph: raw denoised latent z [1,16,h/8,w/8] -> (unscale z/scaling + shift) ->
vae.decode -> image [1,3,h,w]. fp32 (fp16 overflows the VAE activations -> NaN
black frames, per the GLM-Image lesson). Runs ONCE per image, so fp32 cost is
negligible.

Run (coreai-models venv, from conversion/zimage/):
  python export_vae.py --size 512
"""
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from coreai_models.diffusion.components import _patch_nearest_upsample
from coreai_models.export.macos import export_to_coreai

DTYPE = torch.float32
SCALING = 0.3611
SHIFT = 0.1159


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        _patch_nearest_upsample(self.vae.decoder)

    def forward(self, z):
        z = z / SCALING + SHIFT
        return self.vae.decode(z, return_dict=False)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--dyn", action="store_true", help="dynamic latent H/W (one graph for 256/512/1024)")
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()
    name = f"zimage_vae_{'dyn' if args.dyn else args.size}_fp32"

    from diffusers import AutoencoderKL
    print("[vae] loading VAE (fp32) ...", flush=True)
    vae = AutoencoderKL.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="vae", torch_dtype=DTYPE).eval()
    wrap = VAEDecoderWrapper(vae).eval()

    lat = args.size // 8
    ref = {"z": torch.randn(1, 16, lat, lat, dtype=DTYPE)}
    if args.dyn:
        # H and W must share one Dim: the decoder derives W from H (nearest-upsample
        # patch), so independent dims trip a constraint violation. Square only.
        from torch.export import Dim
        side = Dim("lat_side", min=32, max=128)
        dyn = {"z": {2: side, 3: side}}
    else:
        dyn = {"z": None}

    import coreai.runtime as rt
    print("[vae] exporting decoder graph ...", flush=True)
    prog = export_to_coreai(
        wrap, ref, dynamic_shapes=dyn, input_names=("z",), output_names=("image",))
    print("[vae] optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[vae] saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"[vae] bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
