"""Convert the LTX-Video causal video VAE *decoder* (latent -> pixels) to Core AI.

The decoder also does the final denoise via timestep conditioning (decode_timestep).
un_normalize (per-channel: latent*std + mean) is baked into the net so the host
just feeds the post-sampler latent + the decode timestep.

Usage: python _conv_vae.py [H W num_frames]
"""
import sys
import numpy as np
import torch
import torch.nn as nn

import _common as C
import coreai_kit as K


class VAEDecWrap(nn.Module):
    """Pure decoder: takes the un-normalized latent + decode timestep -> pixels.

    Un-normalize (latent*std + mean) is left on host so this matches the real
    pipeline's _run_decoder (which un-normalizes before calling decoder.forward).
    """

    def __init__(self, vae, target_shape):
        super().__init__()
        self.vae = vae
        self.target_shape = target_shape

    def forward(self, latent, timestep):
        return self.vae.decoder(
            latent, target_shape=self.target_shape, timestep=timestep
        )


def main():
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 17

    lf, lh, lw = C.latent_dims(H, W, F)
    print(f"[vae] video {H}x{W} x{F}f -> latent {lf}x{lh}x{lw} ({C.LATENT_CH}ch)")

    vae = C.load_vae()
    target_shape = (1, 3, lf * C.TEMPORAL, lh * C.SPATIAL, lw * C.SPATIAL)
    model = VAEDecWrap(vae, target_shape).eval()

    torch.manual_seed(0)
    latent = torch.randn(1, C.LATENT_CH, lf, lh, lw)
    timestep = torch.tensor([0.05])  # decode_timestep

    ex = (latent, timestep)
    names_in = ["latent", "timestep"]
    names_out = ["pixels"]

    with torch.no_grad():
        ref = model(*ex).numpy()
    print(f"[vae] eager out {ref.shape} mean={ref.mean():.4f} std={ref.std():.4f} "
          f"min={ref.min():.3f} max={ref.max():.3f}")

    out = "coreai_out/vae_fp32.aimodel"
    K.convert(model, ex, names_in, names_out, out, optimize=False)
    print("[vae] converted (optimize=False)")

    feed = {n: e.numpy() for n, e in zip(names_in, ex)}
    got = K.run(out, feed, compute="cpu")["pixels"]
    print(f"[vae] coreai out {got.shape} mean={got.mean():.4f} std={got.std():.4f}")
    print(f"[vae] COS = {C.cos(got, ref):.6f}  maxdiff={np.abs(got-ref).max():.3e}")


if __name__ == "__main__":
    main()
