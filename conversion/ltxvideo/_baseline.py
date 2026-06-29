"""Baseline LTX-Video 2B distilled generation with the real torch pipeline.

Confirms the model runs end-to-end + produces a reference video. T5 on host.
Usage: python _baseline.py [H W num_frames seed]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import time
import numpy as np
import torch

import _common as C
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

PROMPT = ("A clear glass of water on a wooden table, slow motion droplet falling "
          "into it creating ripples, cinematic, soft natural light")
NEG = "worst quality, inconsistent motion, blurry, jittery, distorted"


def main():
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    H = int(a[0]) if len(a) > 0 else 256
    W = int(a[1]) if len(a) > 1 else 256
    F = int(a[2]) if len(a) > 2 else 25
    seed = int(a[3]) if len(a) > 3 else 42
    dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
    device = "cpu" if "--cpu" in sys.argv else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[baseline] {H}x{W} x{F}f seed={seed} device={device} dtype={dtype}")

    t0 = time.time()
    pipe = C.build_pipeline(device=device, dtype=dtype)
    print(f"[baseline] pipeline built in {time.time()-t0:.1f}s")

    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    out = pipe(
        prompt=PROMPT, negative_prompt=NEG,
        num_inference_steps=8, guidance_scale=1, stg_scale=0, rescaling_scale=1,
        skip_layer_strategy=SkipLayerStrategy.AttentionValues,
        generator=gen, output_type="pt",
        height=H, width=W, num_frames=F, frame_rate=24,
        decode_timestep=0.05, decode_noise_scale=0.025, stochastic_sampling=True,
        is_video=True, vae_per_channel_normalize=True,
        mixed_precision=False, offload_to_cpu=False,
    ).images
    print(f"[baseline] generated {tuple(out.shape)} in {time.time()-t0:.1f}s")

    np.save("baseline_video.npy", out.detach().cpu().float().numpy())
    path, shape = C.save_video(out, "baseline.mp4", fps=24)
    print(f"[baseline] wrote {path} {shape}; npy saved")


if __name__ == "__main__":
    main()
