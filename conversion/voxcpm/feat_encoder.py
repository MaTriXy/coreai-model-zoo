# Community port — NOT an Apple model.
"""VoxCPM feat_encoder (LocEnc) + FSQ + enc_to_lm_proj.

LocEnc: per audio patch, prepend a learned special token to the 2 latent steps, run a 4L
bidirectional MiniCPM4 encoder, take the special-token (cls, position 0) output. FSQ: in_proj ->
tanh -> round(h*scale)/scale -> out_proj. enc_to_lm_proj maps LocEnc output into the LM embed space
(this is the ``curr_embed`` fed back into base_lm each decode step).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from feat_decoder import BidirLayer  # noqa: E402
from minicpm4 import RMSNorm, MiniCPM4Cfg, make_rope_tables  # noqa: E402

FEAT = 64
FSQ_SCALE = 9


class LocEnc(nn.Module):
    def __init__(self, cfg: MiniCPM4Cfg, n_layers: int = 4, max_seq: int = 4):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(FEAT, cfg.hidden_size, bias=True)
        self.special_token = nn.Parameter(torch.zeros(1, 1, 1, cfg.hidden_size))
        self.layers = nn.ModuleList([BidirLayer(cfg) for _ in range(n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        cos, sin = make_rope_tables(cfg, max_seq)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(self, x):  # x [B,T,P,64]
        B, T, P, _ = x.shape
        x = self.in_proj(x)                                   # [B,T,P,H]
        sp = self.special_token.expand(B, T, 1, -1)
        x = torch.cat([sp, x], dim=2)                         # [B,T,P+1,H]
        S = P + 1
        x = x.reshape(B * T, S, self.cfg.hidden_size)
        cos = self.cos_table[:S].reshape(1, 1, S, self.cfg.head_dim)
        sin = self.sin_table[:S].reshape(1, 1, S, self.cfg.head_dim)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        cls = x[:, 0, :]
        return cls.reshape(B, T, self.cfg.hidden_size)


class FSQ(nn.Module):
    def __init__(self, hidden=1024, latent=256, scale=FSQ_SCALE):
        super().__init__()
        self.in_proj = nn.Linear(hidden, latent)
        self.out_proj = nn.Linear(latent, hidden)
        self.scale = scale

    def forward(self, h):
        h = torch.tanh(self.in_proj(h))
        h = torch.round(h * self.scale) / self.scale
        return self.out_proj(h)


def load_feat_encoder(sd, n_layers=4, dtype=torch.float32):
    cfg = MiniCPM4Cfg(num_hidden_layers=n_layers, vocab_size=0)
    enc = LocEnc(cfg, n_layers=n_layers).to(dtype).eval()
    own = enc.state_dict()
    pref = "feat_encoder."
    sub = {}
    for k, v in sd.items():
        if k.startswith(pref):
            kk = k[len(pref):].replace("encoder.layers.", "layers.").replace("encoder.norm.", "norm.")
            if kk in own:
                sub[kk] = v.to(dtype)
    missing = [k for k in own if k not in sub and not k.endswith(("cos_table", "sin_table"))]
    if missing:
        raise RuntimeError(f"feat_encoder: unloaded {missing[:6]}")
    enc.load_state_dict(sub, strict=False, assign=True)
    return enc


def load_fsq(sd, dtype=torch.float32):
    m = FSQ().to(dtype).eval()
    sub = {k[len("fsq_layer."):]: v.to(dtype) for k, v in sd.items() if k.startswith("fsq_layer.")}
    m.load_state_dict(sub, strict=True, assign=True)
    return m


def load_linear(sd, prefix, in_f, out_f, dtype=torch.float32):
    m = nn.Linear(in_f, out_f).to(dtype).eval()
    sub = {k[len(prefix):]: v.to(dtype) for k, v in sd.items() if k.startswith(prefix)}
    m.load_state_dict(sub, strict=True, assign=True)
    return m
