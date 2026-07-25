"""LFM2.5-Audio detokenizer — re-authored Core-AI-friendly torch. 8-CB codes @12.5Hz
-> 24kHz wav. FusedEmbedding (8x2048 vocab, dim512, mean over CBs) -> 6x nearest upsample
(->75Hz) -> an 8-layer LFM2 hybrid backbone (5 conv + 3 sliding_attention window=30, run as a
SINGLE masked prefill, no cache) -> Linear(512,1282) -> polar(exp(log_abs), angle) -> custom
"same"-padded iSTFT (n_fft1280/hop320) on the host. Faithful to liquid_audio/detokenizer.py;
the LFM2 attn uses standard HF rotate_half RoPE (theta 1e6), qk-RMSNorm(32); the conv mixer is
the shipped-LFM2 kernel-3 causal depthwise conv.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from depthformer import rms_norm  # shared RMSNorm (float accum, plain weight)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402

DP = hf_snapshot("LiquidAI/LFM2.5-Audio-1.5B", "audio_detokenizer")
CKPT = str(Path(DP) / "model.safetensors")

DIM = 512
N_LAYERS = 8
LAYER_TYPES = ["conv", "conv", "sliding_attention", "conv", "sliding_attention",
               "conv", "sliding_attention", "conv"]
N_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 32
FF = 2304
CONV_K = 3
WINDOW = 30
THETA = 1e6
EPS = 1e-5
CODEBOOKS = 8
CB_VOCAB = 2048           # detok FusedEmbedding vocab (no EOS)
N_FFT, HOP, WIN = 1280, 320, 1280


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(pos, head_dim=HEAD_DIM, theta=THETA):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    f = pos.float()[:, None] * inv[None, :]
    emb = torch.cat([f, f], dim=-1)
    return emb.cos(), emb.sin()


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(DIM, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(DIM, KV_HEADS * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(DIM, DIM, bias=False)
        self.q_ln = nn.Parameter(torch.ones(HEAD_DIM))
        self.k_ln = nn.Parameter(torch.ones(HEAD_DIM))

    def forward(self, x, cos, sin, mask):
        # RAW matmul-softmax attention (not the SDPA composite) + explicit additive
        # sliding-window mask: the SDPA composite crashes MPS lowering over a large
        # prefill query block, while raw matmul lowers (the conformer-encoder recipe).
        b, s, _ = x.shape
        q = rms_norm(self.q_proj(x).view(b, s, N_HEADS, HEAD_DIM), self.q_ln).transpose(1, 2)
        k = rms_norm(self.k_proj(x).view(b, s, KV_HEADS, HEAD_DIM), self.k_ln).transpose(1, 2)
        v = self.v_proj(x).view(b, s, KV_HEADS, HEAD_DIM).transpose(1, 2)
        cs, sn = cos[None, None], sin[None, None]
        q = q * cs + rotate_half(q) * sn
        k = k * cs + rotate_half(k) * sn
        rep = N_HEADS // KV_HEADS                     # GQA: repeat KV heads (16/8=2)
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        scores = (q @ k.transpose(-2, -1)) * (HEAD_DIM ** -0.5) + mask   # [b,H,s,s]
        o = scores.softmax(-1) @ v
        return self.out_proj(o.transpose(1, 2).reshape(b, s, DIM))


class Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(DIM, 3 * DIM, bias=False)
        self.out_proj = nn.Linear(DIM, DIM, bias=False)
        self.conv = nn.Conv1d(DIM, DIM, CONV_K, groups=DIM, bias=False, padding=CONV_K - 1)

    def forward(self, x):
        b, s, _ = x.shape
        bcx = self.in_proj(x).transpose(1, 2)
        gb, gc, xv = bcx.chunk(3, dim=1)
        conv_out = self.conv(gb * xv)[..., :s]
        return self.out_proj((gc * conv_out).transpose(1, 2))


class Layer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        self.operator_norm = nn.Parameter(torch.ones(DIM))
        self.ffn_norm = nn.Parameter(torch.ones(DIM))
        if kind == "conv":
            self.conv = Conv()
        else:
            self.self_attn = Attn()
        self.w1 = nn.Linear(DIM, FF, bias=False)
        self.w3 = nn.Linear(DIM, FF, bias=False)
        self.w2 = nn.Linear(FF, DIM, bias=False)

    def forward(self, x, cos, sin, mask):
        n = rms_norm(x, self.operator_norm)
        r = self.conv(n) if self.kind == "conv" else self.self_attn(n, cos, sin, mask)
        h = x + r
        nf = rms_norm(h, self.ffn_norm)
        return h + self.w2(F.silu(self.w1(nf)) * self.w3(nf))


def sliding_mask(s):
    """Additive causal sliding-window(WINDOW) mask [1,1,s,s]: attend j in (i-W, i]."""
    idx = torch.arange(s)
    d = idx[None, :] - idx[:, None]                  # d[i,j] = j - i
    keep = torch.logical_and(d <= 0, d > -WINDOW)
    m = torch.zeros(s, s)
    m.masked_fill_(~keep, float("-inf"))
    return m[None, None]


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(t) for t in LAYER_TYPES])
        self.embedding_norm = nn.Parameter(torch.ones(DIM))

    def forward(self, x):
        s = x.shape[1]
        cos, sin = rope_cos_sin(torch.arange(s, dtype=torch.int32))
        mask = sliding_mask(s).to(x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return rms_norm(x, self.embedding_norm)


class Detokenizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fused = nn.Embedding(CODEBOOKS * CB_VOCAB, DIM)     # emb.emb
        self.backbone = Backbone()
        self.lin = nn.Linear(DIM, 1282)
        self.register_buffer("window", torch.hann_window(WIN))

    def istft_same(self, spec):
        """Custom 'same'-padded iSTFT (Vocos-style), matches liquid_audio ISTFT."""
        pad = (WIN - HOP) // 2
        B, N, T = spec.shape
        ifft = torch.fft.irfft(spec, N_FFT, dim=1, norm="backward") * self.window[None, :, None]
        out_size = (T - 1) * HOP + WIN
        y = F.fold(ifft, (1, out_size), (1, WIN), stride=(1, HOP))[:, 0, 0, pad:-pad]
        wsq = self.window.square().expand(1, T, -1).transpose(1, 2)
        env = F.fold(wsq, (1, out_size), (1, WIN), stride=(1, HOP)).squeeze()[pad:-pad]
        return y / env

    def forward(self, codes):
        """codes [1,8,T] int -> wav [L]."""
        offsets = torch.arange(CODEBOOKS)[:, None] * CB_VOCAB
        x = self.fused(offsets + codes[0]).mean(0)[None]         # [1,T,512]
        x = F.interpolate(x.mT, 6 * x.shape[1], mode="nearest-exact").mT  # [1,6T,512]
        x = self.backbone(x)
        x = self.lin(x)                                          # [1,6T,1282]
        log_abs, angle = torch.chunk(x.mT.contiguous(), 2, 1)    # [1,641,6T] each
        y = torch.polar(log_abs.exp(), angle)
        return self.istft_same(y)[0]


class DetokSpec(nn.Module):
    """Exportable Core AI graph: upsampled fused embeds [1,S,512] -> spec params
    [1,S,1282] (backbone + lin). FusedEmbedding gather+upsample and polar+iSTFT stay
    on the host (embed gather is a lookup; iSTFT is DSP)."""
    def __init__(self, detok: "Detokenizer"):
        super().__init__()
        self.backbone = detok.backbone
        self.lin = detok.lin

    def forward(self, inputs_embeds):
        return self.lin(self.backbone(inputs_embeds))


def codes_to_embeds(detok, codes):
    """codes [1,8,T] -> upsampled fused embeds [1,6T,512] (host front-end)."""
    offsets = torch.arange(CODEBOOKS)[:, None] * CB_VOCAB
    x = detok.fused(offsets + codes[0]).mean(0)[None]
    return F.interpolate(x.mT, 6 * x.shape[1], mode="nearest-exact").mT


def spec_to_wav(detok, spec):
    """spec [1,S,1282] -> wav [L] (host polar + iSTFT)."""
    log_abs, angle = torch.chunk(spec.mT.contiguous(), 2, 1)
    y = torch.polar(log_abs.exp(), angle)
    return detok.istft_same(y)[0]


def load_detokenizer(dtype=torch.float32):
    from safetensors import safe_open
    m = Detokenizer().to(dtype).eval()
    sd = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        g = lambda k: f.get_tensor(k).to(dtype)  # noqa: E731
        sd["fused.weight"] = g("emb.emb.weight")
        sd["lin.weight"] = g("lin.weight")
        sd["lin.bias"] = g("lin.bias")
        sd["backbone.embedding_norm"] = g("lfm.embedding_norm.weight")
        for i, kind in enumerate(LAYER_TYPES):
            p = f"lfm.layers.{i}."
            b = f"backbone.layers.{i}."
            sd[b + "operator_norm"] = g(p + "operator_norm.weight")
            sd[b + "ffn_norm"] = g(p + "ffn_norm.weight")
            sd[b + "w1.weight"] = g(p + "feed_forward.w1.weight")
            sd[b + "w2.weight"] = g(p + "feed_forward.w2.weight")
            sd[b + "w3.weight"] = g(p + "feed_forward.w3.weight")
            if kind == "conv":
                sd[b + "conv.in_proj.weight"] = g(p + "conv.in_proj.weight")
                sd[b + "conv.out_proj.weight"] = g(p + "conv.out_proj.weight")
                sd[b + "conv.conv.weight"] = g(p + "conv.conv.weight")
            else:
                sd[b + "self_attn.q_proj.weight"] = g(p + "self_attn.q_proj.weight")
                sd[b + "self_attn.k_proj.weight"] = g(p + "self_attn.k_proj.weight")
                sd[b + "self_attn.v_proj.weight"] = g(p + "self_attn.v_proj.weight")
                sd[b + "self_attn.out_proj.weight"] = g(p + "self_attn.out_proj.weight")
                sd[b + "self_attn.q_ln"] = g(p + "self_attn.q_layernorm.weight")
                sd[b + "self_attn.k_ln"] = g(p + "self_attn.k_layernorm.weight")
    sd["window"] = m.window
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:6]}"
    assert not [x for x in missing if x != "window"], f"missing: {missing[:6]}"
    return m
