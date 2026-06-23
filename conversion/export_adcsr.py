# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "diffusers==0.37.1",
#     "transformers",
#     "numpy",
#     "huggingface_hub",
#     "pytorch-lightning",
# ]
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Export AdcSR (CVPR 2025) — one-step diffusion-GAN ×4 super-resolution — as ONE static Core AI
graph:  lr [1,3,128,128] in [-1,1]  ->  sr [1,3,512,512] in [-1,1].

AdcSR = Adversarial Diffusion Compression of OSEDiff: a pruned Stable Diffusion 2.1 UNet + a
half-size VAE decoder, run in a single forward (no prompt, no noise, no timestep). The exported
graph outputs the RAW SR; AdcSR's per-image color-match is applied HOST-SIDE by CoreAIKit's
SuperResolver after tiling (baking it per-tile blows up uniform tiles → pure-white squares).

Ships fp32 (~1.7 GB). The pruned SD-2.1 attention/group-norm overflow in fp16 (NaN on smooth
tiles; upcast_attention is ignored by the SDPA processor), so fp16 is not usable here.

License: AdcSR code is Apache-2.0; the weights derive from Stable Diffusion 2.1 and carry the
CreativeML Open RAIL++-M license (commercial OK with use restrictions — the license under which
Apple itself ships SD CoreML).

Inputs (auto-downloaded from the Hub):
  - Guaishou74851/AdcSR :: weight/net_params_200.pkl, weight/pretrained/halfDecoder.ckpt
  - flax/stable-diffusion-2-1-base :: unet config only (weights are overwritten by net_params;
    stabilityai/stable-diffusion-2-1-base was removed from the Hub, so the flax mirror is used).

Run:  uv run export_adcsr.py --output-dir out
"""
import argparse
import copy
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from diffusers import UNet2DConditionModel
from diffusers.models.autoencoders.vae import Decoder


def build_model():
    # AdcSR's Net surgically prunes the SD-2.1 UNet (delete time_embedding + cross-attn, ×0.75
    # channels, custom text/time-free forwards) and pairs it with a half VAE decoder. The repo's
    # model.py / forward.py implement this; we clone the repo and import them.
    from huggingface_hub import snapshot_download
    import sys

    repo = snapshot_download("Guaishou74851/AdcSR", allow_patterns=["*.py", "bsr/**", "weight/**"])
    sys.path.insert(0, repo)
    from model import Net  # noqa: E402  (AdcSR repo)

    cfg = UNet2DConditionModel.load_config("flax/stable-diffusion-2-1-base", subfolder="unet")
    unet = UNet2DConditionModel.from_config(cfg)
    half_ckpt = torch.load(
        hf_hub_download("Guaishou74851/AdcSR", "weight/pretrained/halfDecoder.ckpt"),
        map_location="cpu", weights_only=False)
    decoder = Decoder(in_channels=4, out_channels=3,
                      up_block_types=["UpDecoderBlock2D"] * 4,
                      block_out_channels=[64, 128, 256, 256],
                      layers_per_block=2, norm_num_groups=32, act_fn="silu",
                      norm_type="group", mid_block_add_attention=True)
    decoder.load_state_dict(
        {k.replace("decoder.", ""): v for k, v in half_ckpt["state_dict"].items() if "decoder" in k},
        strict=True)
    model = torch.nn.DataParallel(Net(unet, copy.deepcopy(decoder)))
    model.load_state_dict(torch.load(
        hf_hub_download("Guaishou74851/AdcSR", "weight/net_params_200.pkl"),
        map_location="cpu", weights_only=False))
    return torch.nn.Sequential(
        model.module, *decoder.up_blocks,
        decoder.conv_norm_out, decoder.conv_act, decoder.conv_out,
    ).eval()


class AdcSR(nn.Module):
    """Raw one-step SR: lr [1,3,128,128] in [-1,1] -> sr [1,3,512,512] in [-1,1]."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, lr):
        return self.model(lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="out")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    module = AdcSR(build_model()).eval()
    print(f"[PARAM] {sum(p.numel() for p in module.parameters())/1e6:.2f}M")

    example = torch.zeros(1, 3, 128, 128)
    example[:, :, ::4, ::4] = 0.5  # a non-uniform probe so group-norm variance != 0
    with torch.no_grad():
        out = module(example)
    assert out.shape == (1, 3, 512, 512) and torch.isfinite(out).all(), "sanity failed"
    print(f"[SANITY] {list(example.shape)} -> {list(out.shape)}, finite OK")

    exported = torch.export.export(module, args=(example.clone(),)).run_decompositions(get_decomp_table())
    program = TorchConverter().add_exported_program(
        exported_program=exported, input_names=["lr"], output_names=["sr"]).to_coreai()
    program.optimize()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "adcsr_x4_float32.aimodel"
    if path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        shutil.rmtree(path)
    meta = AIModelAssetMetadata()
    meta.author = "Bingchen Li et al. (AdcSR, CVPR 2025)"
    meta.license = "Apache-2.0 (code); weights derived from Stable Diffusion 2.1 (CreativeML OpenRAIL++-M)"
    meta.model_description = (
        "AdcSR one-step diffusion-GAN ×4 super-resolution. lr [1,3,128,128] in [-1,1] -> "
        "sr [1,3,512,512] (raw; host applies per-image color-match). Source: "
        "https://github.com/Guaishou74851/AdcSR (CVPR 2025).")
    meta.creation_date = int(time.time())
    program.save_asset(path, meta)
    print(f"[OK] saved {path}")


if __name__ == "__main__":
    main()
