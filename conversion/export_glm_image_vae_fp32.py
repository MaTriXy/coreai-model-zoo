"""Export the GLM-Image VAE decoder (16ch AutoencoderKL) to Core AI.

Graph: raw denoised latent z [1,16,h/8,w/8] -> (unscale z*std+mean) ->
vae.decode -> image [1,3,h,w]. fp16. Completes the on-device T2I output path
(AR + DiT already Core AI; glyph-free text embed is a 1-token host constant).

Run (coreai-models venv): python export_glm_image_vae.py --vae-dir <snap>/vae --size 512
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn

from coreai_models.diffusion.components import _patch_nearest_upsample
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai

DTYPE = torch.float32


class VAEDecoderWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae.to(next(vae.parameters()).dtype)
        _patch_nearest_upsample(self.vae.decoder)
        m = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1)
        s = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1)
        self.register_buffer("lm", m.to(DTYPE))
        self.register_buffer("ls", s.to(DTYPE))

    def forward(self, z):
        z = z * self.ls + self.lm
        return self.vae.decode(z).sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae-dir", required=True)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()
    name = f"glm_image_vae_{args.size}_fp32"

    from diffusers import AutoencoderKL
    print("loading VAE ...", flush=True)
    vae = AutoencoderKL.from_pretrained(args.vae_dir, torch_dtype=DTYPE).eval()
    wrap = VAEDecoderWrapper(vae).eval()

    lat = args.size // 8
    ref = {"z": torch.randn(1, 16, lat, lat, dtype=DTYPE)}
    dyn = {"z": None}

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    import coreai.runtime as rt

    print("exporting VAE decoder graph ...", flush=True)
    prog = export_to_coreai(
        wrap, ref, dynamic_shapes=dyn,
        input_names=("z",), output_names=("image",),
        state_names=None, externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
