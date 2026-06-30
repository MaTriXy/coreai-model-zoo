# Community port — NOT a Stability model. Clean-rebuilt Oobleck VAE DECODER for Stable Audio.
"""The reference Oobleck decoder uses weight_norm (WNConv1d/WNConvTranspose1d) + alias-free wrappers
that the Core AI exporter's fp16 `_apply` can't swap ("Couldn't swap Conv1d.bias" / weakref). So we
rebuild a PLAIN-torch decoder with the SAME nn.Sequential nesting (so state_dict keys align) and FOLD
weight_norm (`w = g * v / ||v||_(dims!=0)`) at load — the proven zoo recipe (VoxCPM audio_vae.py /
Kokoro: "weight_norm未fold"罠).

latent[1,64,T] -> waveform[1,2,2048*T]. config: channels 128, c_mults [1,2,4,8,16], strides
[2,4,4,8,8], use_snake, final_tanh False. (VAE bottleneck decode is identity, pretransform scale 1.0,
so audio = decoder(latent).)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


def snake_beta(x, alpha, beta):                       # x + (1/(beta+eps)) * sin(x*alpha)^2
    return x + (1.0 / (beta + 1e-9)) * torch.sin(x * alpha).pow(2)


class SnakeBeta(nn.Module):                           # alpha/beta logscale -> exp; params [C]
    def __init__(self, ch):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(ch))
        self.beta = nn.Parameter(torch.zeros(ch))

    def forward(self, x):
        a = self.alpha.unsqueeze(0).unsqueeze(-1).exp()
        b = self.beta.unsqueeze(0).unsqueeze(-1).exp()
        return snake_beta(x, a, b)


class ResidualUnit(nn.Module):                        # layers: Snake, Conv k7 dil, Snake, Conv k1
    def __init__(self, ch, dilation):
        super().__init__()
        pad = (dilation * (7 - 1)) // 2
        self.layers = nn.Sequential(
            SnakeBeta(ch),
            nn.Conv1d(ch, ch, 7, dilation=dilation, padding=pad),
            SnakeBeta(ch),
            nn.Conv1d(ch, ch, 1),
        )

    def forward(self, x):
        return x + self.layers(x)


class DecoderBlock(nn.Module):                         # layers: Snake, ConvT, Res(1), Res(3), Res(9)
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.layers = nn.Sequential(
            SnakeBeta(in_ch),
            nn.ConvTranspose1d(in_ch, out_ch, 2 * stride, stride=stride, padding=math.ceil(stride / 2)),
            ResidualUnit(out_ch, 1),
            ResidualUnit(out_ch, 3),
            ResidualUnit(out_ch, 9),
        )

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self, out_channels=2, channels=128, latent_dim=64,
                 c_mults=(1, 2, 4, 8, 16), strides=(2, 4, 4, 8, 8), final_tanh=False):
        super().__init__()
        c_mults = [1] + list(c_mults)                  # [1,1,2,4,8,16]
        depth = len(c_mults)                           # 6
        layers = [nn.Conv1d(latent_dim, c_mults[-1] * channels, 7, padding=3)]   # layers.0
        for i in range(depth - 1, 0, -1):              # 5..1
            layers.append(DecoderBlock(c_mults[i] * channels, c_mults[i - 1] * channels, strides[i - 1]))
        layers += [
            SnakeBeta(c_mults[0] * channels),
            nn.Conv1d(c_mults[0] * channels, out_channels, 7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity(),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, latent):
        return self.layers(latent)


def _fold(sd: dict, prefix: str) -> dict:
    """Strip `prefix`, fold weight_norm: weight_g[out,1,1] * weight_v / ||weight_v||_(dims 1,2)."""
    out, g_keys = {}, set()
    for k in sd:
        if not k.startswith(prefix):
            continue
        kk = k[len(prefix):]
        if kk.endswith(".weight_g"):
            base = kk[:-len(".weight_g")]
            g = sd[prefix + base + ".weight_g"].float()
            v = sd[prefix + base + ".weight_v"].float()
            norm = v.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
            out[base + ".weight"] = g * v / norm
            g_keys.add(base)
        elif kk.endswith(".weight_v"):
            continue
        else:
            out[kk] = sd[k].float()
    return out


def load_decoder(state_dict: dict, dtype=torch.float32) -> OobleckDecoder:
    m = OobleckDecoder().to(dtype).eval()
    sub = _fold(state_dict, "pretransform.model.decoder.")
    own = m.state_dict()
    sub = {k: v.to(dtype) for k, v in sub.items() if k in own}
    missing = [k for k in own if k not in sub]
    if missing:
        raise RuntimeError(f"oobleck decoder: {len(missing)} unloaded, e.g. {missing[:4]}")
    m.load_state_dict(sub, strict=True)
    return m
