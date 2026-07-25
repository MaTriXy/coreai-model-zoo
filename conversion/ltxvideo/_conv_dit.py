"""Convert the LTX-Video 2B DiT (Transformer3DModel) one denoise step to Core AI.

Per-net gate is converted-vs-eager, so random conditioning at the real shapes is
enough (no T5 / no sampler needed). Apply the TripoSplat converter gotchas.

Usage: python _conv_dit.py [H W num_frames text_seq]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/ — coreai_kit
import numpy as np
import torch
import torch.nn as nn

import _common as C
import coreai_kit as K


class DiTWrap(nn.Module):
    """Tensor-in/tensor-out wrapper: drops return_dict / skip-layer plumbing."""

    def __init__(self, dit):
        super().__init__()
        self.dit = dit

    def forward(self, hidden_states, indices_grid, encoder_hidden_states,
                encoder_attention_mask, timestep):
        return self.dit(
            hidden_states,
            indices_grid=indices_grid,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            return_dict=False,
        )[0]


def main():
    H = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 17
    TS = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    lf, lh, lw = C.latent_dims(H, W, F)
    n_tok = lf * lh * lw
    print(f"[dit] video {H}x{W} x{F}f -> latent {lf}x{lh}x{lw} = {n_tok} tokens, text_seq={TS}")

    torch.manual_seed(0)
    model = DiTWrap(C.load_dit()).eval()

    hidden_states = torch.randn(1, n_tok, C.LATENT_CH)
    indices_grid = C.build_indices_grid(F, H, W)
    enc = torch.randn(1, TS, C.CAPTION_CH)
    enc_mask = torch.ones(1, TS)
    timestep = torch.full((1, 1), 0.7)  # flow-matching t in [0,1]

    ex = (hidden_states, indices_grid, enc, enc_mask, timestep)
    names_in = ["hidden_states", "indices_grid", "encoder_hidden_states",
                "encoder_attention_mask", "timestep"]
    names_out = ["sample"]

    with torch.no_grad():
        ref = model(*ex).numpy()
    print(f"[dit] eager out {ref.shape}  mean={ref.mean():.4f} std={ref.std():.4f}")

    out = "coreai_out/dit_fp32.aimodel"
    K.convert(model, ex, names_in, names_out, out, optimize=False)
    print("[dit] converted (optimize=False)")

    feed = {n: e.numpy() for n, e in zip(names_in, ex)}
    got = K.run(out, feed, compute="cpu")["sample"]
    print(f"[dit] coreai out {got.shape} mean={got.mean():.4f} std={got.std():.4f}")
    print(f"[dit] COS = {C.cos(got, ref):.6f}  maxdiff={np.abs(got-ref).max():.3e}")


if __name__ == "__main__":
    main()
