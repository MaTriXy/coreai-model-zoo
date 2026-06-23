# Community port — NOT an Apple model.
"""VoxCPM AudioVAE decoder (latents[1,64,T] -> 16kHz wav, 640x upsample).

DAC-style causal-conv decoder: depthwise first conv -> 1536ch -> 4 upsample blocks
(Snake + causal transpose-conv + 3 residual units) -> Snake -> conv->1 -> Tanh. ``use_noise_block``
is OFF so it is fully DETERMINISTIC. weight_norm is FOLDED at load (Kokoro lesson #1 — else the
manual convs read random weights). The transpose convs here use ``nn.ConvTranspose1d`` for the torch
gate; the engine export swaps them for zero-insertion + conv1d (Kokoro lesson #5).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RATES = [8, 8, 5, 2]
DEC_DIM = 1536
LATENT = 64


def snake(x, alpha):
    return x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)


class Snake1d(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, ch, 1))

    def forward(self, x):
        return snake(x, self.alpha)


class CausalConv1d(nn.Module):
    """left-pad by 2*pad, conv with padding=0 (causal)."""

    def __init__(self, ci, co, k, stride=1, dilation=1, pad=0, groups=1):
        super().__init__()
        self.pad = pad
        self.conv = nn.Conv1d(ci, co, k, stride=stride, dilation=dilation, padding=0, groups=groups)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad * 2, 0)))


class CausalTransposeConv1d(nn.Module):
    """transpose-conv then trim right by (2*pad - output_pad)."""

    def __init__(self, ci, co, k, stride, pad, output_pad, groups=1):
        super().__init__()
        self.trim = pad * 2 - output_pad
        self.conv = nn.ConvTranspose1d(ci, co, k, stride=stride, padding=0, output_padding=0, groups=groups)

    def forward(self, x):
        y = self.conv(x)
        return y[..., : -self.trim] if self.trim > 0 else y


class ResidualUnit(nn.Module):
    def __init__(self, dim, dilation, groups):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            CausalConv1d(dim, dim, 7, dilation=dilation, pad=pad, groups=groups),
            Snake1d(dim),
            CausalConv1d(dim, dim, 1),
        )

    def forward(self, x):
        return x + self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, ci, co, stride, groups):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(ci),
            CausalTransposeConv1d(ci, co, 2 * stride, stride, math.ceil(stride / 2), stride % 2),
            ResidualUnit(co, 1, groups),
            ResidualUnit(co, 3, groups),
            ResidualUnit(co, 9, groups),
        )

    def forward(self, x):
        return self.block(x)


class AudioVAEDecoder(nn.Module):
    """depthwise=True decoder. forward: latents[B,64,T] -> wav[B,1,640T]."""

    def __init__(self):
        super().__init__()
        layers = [
            CausalConv1d(LATENT, LATENT, 7, pad=3, groups=LATENT),  # depthwise
            CausalConv1d(LATENT, DEC_DIM, 1),
        ]
        ch = DEC_DIM
        for i, stride in enumerate(RATES):
            ci = DEC_DIM // 2 ** i
            co = DEC_DIM // 2 ** (i + 1)
            layers.append(DecoderBlock(ci, co, stride, groups=co))  # depthwise residual units
            ch = co
        layers += [Snake1d(ch), CausalConv1d(ch, 1, 7, pad=3), nn.Tanh()]
        self.model = nn.Sequential(*layers)

    def forward(self, z):
        return self.model(z)


def _fold_wn(g, v):
    """weight_norm dim=0: w = g * v / ||v||_(dims!=0)."""
    norm = v.flatten(1).norm(dim=1).reshape(-1, *([1] * (v.dim() - 1)))
    return g * v / norm


def load_audio_vae(audiovae_sd: dict, dtype=torch.float32) -> AudioVAEDecoder:
    """Load decoder.* from audiovae.pth, folding weight_norm (weight_g/weight_v -> weight)."""
    m = AudioVAEDecoder().to(dtype).eval()
    own = m.state_dict()
    # map checkpoint 'decoder.model.<i>...' to our 'model.<i>...'; our convs nest under '.conv.'
    src = {k[len("decoder."):]: val for k, val in audiovae_sd.items() if k.startswith("decoder.")}
    folded = {}
    # group weight_g/weight_v
    bases = set()
    for k in src:
        if k.endswith(".weight_g") or k.endswith(".weight_v"):
            bases.add(k.rsplit(".", 1)[0])
    for b in bases:
        folded[b + ".weight"] = _fold_wn(src[b + ".weight_g"], src[b + ".weight_v"])
    for k, val in src.items():
        if k.endswith((".weight_g", ".weight_v")):
            continue
        folded[k] = val
    # our module inserts a '.conv' level inside CausalConv1d/CausalTransposeConv1d. Remap conv params:
    # checkpoint key 'model.0.weight' (the conv weight) -> our 'model.0.conv.weight'; Snake 'alpha' &
    # ConvTranspose live the same. Detect by matching against own keys.
    remap = {}
    for k, val in folded.items():
        if k in own:
            remap[k] = val
        else:
            # try inserting '.conv' before the final '.weight'/'.bias'
            parts = k.rsplit(".", 1)
            cand = parts[0] + ".conv." + parts[1]
            if cand in own:
                remap[cand] = val
            else:
                remap[k] = val  # leave; will surface as unexpected
    missing = [k for k in own if k not in remap]
    if missing:
        raise RuntimeError(f"audio_vae: {len(missing)} unloaded, e.g. {missing[:6]}")
    m.load_state_dict({k: v.to(dtype) for k, v in remap.items() if k in own}, strict=True, assign=True)
    return m
