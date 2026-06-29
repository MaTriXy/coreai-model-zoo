"""Shared helpers for the LTX-Video -> Core AI conversion scripts."""
import sys
import json
import math
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, "/Users/majimadaisuke/Code/coreai")  # coreai_kit

CKPT = "ckpts/ltxv-2b-0.9.6-distilled-04-25.safetensors"

# VAE geometry for ltxv-2b-0.9.6 (patch_size 4, 3 spatio-temporal down blocks).
SPATIAL = 32
TEMPORAL = 8
LATENT_CH = 128
CAPTION_CH = 4096


def cos(a, b):
    a = np.asarray(a).ravel().astype(np.float64)
    b = np.asarray(b).ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def latent_dims(height, width, num_frames):
    lh = height // SPATIAL
    lw = width // SPATIAL
    lf = (num_frames - 1) // TEMPORAL + 1
    return lf, lh, lw


def build_indices_grid(num_frames, height, width, frame_rate=25.0, batch=1):
    """Reproduce pipeline fractional_coords for a SymmetricPatchifier(patch_size=1).

    latent corner coords -> pixel coords (* scale factors, causal temporal fix)
    -> fractional_coords (frame axis * 1/frame_rate). Shape (b, 3, num_tokens).
    """
    lf, lh, lw = latent_dims(height, width, num_frames)
    fcoord, hcoord, wcoord = torch.meshgrid(
        torch.arange(0, lf), torch.arange(0, lh), torch.arange(0, lw), indexing="ij"
    )
    latent_coords = torch.stack([fcoord, hcoord, wcoord], dim=0).reshape(3, -1)  # (3, N)
    latent_coords = latent_coords.unsqueeze(0).repeat(batch, 1, 1).float()  # (b,3,N)
    scale = torch.tensor([TEMPORAL, SPATIAL, SPATIAL], dtype=torch.float32)[None, :, None]
    pixel_coords = latent_coords * scale
    # causal_fix: first temporal frame scale is 1
    pixel_coords[:, 0] = (pixel_coords[:, 0] + 1 - TEMPORAL).clamp(min=0)
    frac = pixel_coords.to(torch.float32)
    frac[:, 0] = frac[:, 0] * (1.0 / frame_rate)
    return frac


def load_dit(dtype=torch.float32):
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    m = Transformer3DModel.from_pretrained(CKPT)
    return m.to(dtype).eval()


def save_video(images, path, fps=24):
    """images: torch/np tensor (1,3,F,H,W) in [0,1]. Writes PNG frames + ffmpeg mp4."""
    import os
    import subprocess
    import tempfile
    from PIL import Image

    if hasattr(images, "detach"):
        images = images.detach().cpu().float().numpy()
    images = np.asarray(images)
    vid = images[0].transpose(1, 2, 3, 0)  # F,H,W,C
    vid = (np.clip(vid, 0, 1) * 255).astype(np.uint8)
    d = tempfile.mkdtemp()
    for i, fr in enumerate(vid):
        Image.fromarray(fr).save(f"{d}/f{i:04d}.png")
    subprocess.run(
        ["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{d}/f%04d.png",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
        check=True, capture_output=True,
    )
    return path, vid.shape


def load_vae(dtype=torch.float32):
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )
    m = CausalVideoAutoencoder.from_pretrained(CKPT)
    return m.to(dtype).eval()


def build_pipeline(device="mps", t5_dir="ckpts/pixart", dtype=torch.bfloat16):
    """Replicate inference.create_ltx_video_pipeline WITHOUT importing
    ltx_video.inference (which hard-imports imageio). Distilled defaults."""
    from transformers import T5EncoderModel, T5Tokenizer
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    from ltx_video.models.autoencoders.causal_video_autoencoder import (
        CausalVideoAutoencoder,
    )
    from ltx_video.schedulers.rf import RectifiedFlowScheduler
    from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline

    with safe_open(CKPT, framework="pt") as f:
        allowed = json.loads(f.metadata()["config"]).get("allowed_inference_steps", None)

    vae = CausalVideoAutoencoder.from_pretrained(CKPT)
    transformer = Transformer3DModel.from_pretrained(CKPT).to(dtype)
    scheduler = RectifiedFlowScheduler.from_pretrained(CKPT)
    text_encoder = T5EncoderModel.from_pretrained(t5_dir, subfolder="text_encoder")
    tokenizer = T5Tokenizer.from_pretrained(t5_dir, subfolder="tokenizer")
    patchifier = SymmetricPatchifier(patch_size=1)

    transformer = transformer.to(device)
    vae = vae.to(device).to(dtype)
    text_encoder = text_encoder.to(device).to(dtype)

    pipe = LTXVideoPipeline(
        transformer=transformer, patchifier=patchifier, text_encoder=text_encoder,
        tokenizer=tokenizer, scheduler=scheduler, vae=vae,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=allowed,
    )
    return pipe.to(device)
