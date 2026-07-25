"""LFM2.5-Audio Depthformer (RQ audio head) — re-authored in plain, Core-AI-friendly
torch from the audio checkpoint. Per audio frame the LFM hidden [2048] is projected by
depth_linear to 8 codebook inputs [8,1024]; a 6-layer transformer runs an 8-step AR loop
(cur_i = depth_in[i] + embed_raw(prev_code), CB0 cond = 0) with a growing KV cache, and
each step's output is projected by that codebook's head to [2049] logits (greedy argmax).

Faithful to liquid_audio `model/transformer.py` + `lfm2_audio._sample_audio_frame`, EXCEPT
the RoPE: the oracle uses complex `view_as_complex` (adjacent-pair / interleaved) rope —
not Core AI exportable — re-expressed here as real interleaved-pair arithmetic (bit-parity
gated). Depthformer MHA: gqa 32Q/8KV, head_dim 32, qk RMSNorm(32), swiglu GLU ff2816,
RMSNorm eps 1e-5, rope theta 1e6 (MHA default — NOT 1e4).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402

SNAP = hf_snapshot("LiquidAI/LFM2.5-Audio-1.5B")
CKPT = str(Path(SNAP) / "model.safetensors")
CFG_JSON = str(Path(SNAP) / "config.json")

DIM = 1024
N_HEADS = 32
HEAD_DIM = 32          # 1024 / 32
GQA_KV = 8
FF = 2816              # swiglu(4*1024) rounded
CODEBOOKS = 8
AUDIO_VOCAB = 2049     # 2048 codes + EOS
LFM_DIM = 2048
ROPE_THETA = 1e6
EPS = 1e-5


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    out = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (out * weight).type_as(x)


def rope_interleaved(x: torch.Tensor, pos: torch.Tensor, theta: float = ROPE_THETA):
    """x [b,s,heads,head_dim]; pos [s]. Adjacent-pair (interleaved) rope == the oracle's
    complex `view_as_complex(rearrange('(D two) -> D two'))`, in real arithmetic."""
    hd = x.shape[-1]
    half = hd // 2
    freqs = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) * 2.0 / hd))  # [half]
    ang = pos.float()[:, None] * freqs[None, :]                    # [s, half]
    cos = ang.cos()[None, :, None, :]                              # [1,s,1,half]
    sin = ang.sin()[None, :, None, :]
    x1 = x[..., 0::2]                                              # even -> real part
    x2 = x[..., 1::2]                                              # odd  -> imag part
    o1 = x1 * cos - x2 * sin
    o2 = x1 * sin + x2 * cos
    return torch.stack([o1, o2], dim=-1).flatten(-2).type_as(x)


class DepthformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(DIM, DIM + 2 * HEAD_DIM * GQA_KV, bias=False)  # [1536,1024]
        self.out_proj = nn.Linear(DIM, DIM, bias=False)
        self.q_ln = nn.Parameter(torch.ones(HEAD_DIM))
        self.k_ln = nn.Parameter(torch.ones(HEAD_DIM))
        self.operator_norm = nn.Parameter(torch.ones(DIM))
        self.ffn_norm = nn.Parameter(torch.ones(DIM))
        self.w1 = nn.Linear(DIM, FF, bias=False)
        self.w3 = nn.Linear(DIM, FF, bias=False)
        self.w2 = nn.Linear(FF, DIM, bias=False)

    def attn(self, x, kcache=None, vcache=None):
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([DIM, HEAD_DIM * GQA_KV, HEAD_DIM * GQA_KV], dim=-1)
        q = q.view(b, s, N_HEADS, HEAD_DIM)
        k = k.view(b, s, GQA_KV, HEAD_DIM)
        v = v.view(b, s, GQA_KV, HEAD_DIM)
        q = rms_norm(q, self.q_ln)
        k = rms_norm(k, self.k_ln)
        past = 0 if kcache is None else kcache.shape[1]
        pos = torch.arange(past, past + s, dtype=torch.int32)
        q = rope_interleaved(q, pos)
        k = rope_interleaved(k, pos)
        if kcache is not None:
            k = torch.cat([kcache, k], dim=1)
            v = torch.cat([vcache, v], dim=1)
        qh = q.transpose(1, 2)          # [b,H,s,hd]
        kh = k.transpose(1, 2)          # [b,KV,S,hd]
        vh = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=(s > 1), enable_gqa=True)
        out = out.transpose(1, 2).reshape(b, s, DIM)
        return self.out_proj(out), k, v

    def forward(self, x, kcache=None, vcache=None):
        h, k, v = self.attn(rms_norm(x, self.operator_norm), kcache, vcache)
        h = h + x
        g = self.w2(F.silu(self.w1(rms_norm(h, self.ffn_norm))) * self.w3(rms_norm(h, self.ffn_norm)))
        return h + g, k, v


class DepthHead(nn.Module):
    """SharedEmbedding: embed_raw (no norm, conditions next CB) + get_logits (norm->to_logits)."""
    def __init__(self, vocab=AUDIO_VOCAB, dim=DIM):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.embedding_norm = nn.Parameter(torch.ones(dim))
        self.to_logits = nn.Linear(dim, vocab, bias=False)

    def embed(self, tok):
        return self.embedding(tok)

    def logits(self, h):
        return self.to_logits(rms_norm(h, self.embedding_norm))


class Depthformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth_linear = nn.Linear(LFM_DIM, DIM * CODEBOOKS)  # [8192,2048]+bias
        self.layers = nn.ModuleList([DepthformerLayer() for _ in range(6)])
        self.heads = nn.ModuleList([DepthHead() for _ in range(CODEBOOKS)])

    def sample_frame(self, hidden: torch.Tensor) -> list[int]:
        """hidden [2048] -> 8 greedy codebook tokens (matches _sample_audio_frame greedy)."""
        din = self.depth_linear(hidden).view(CODEBOOKS, DIM)      # [8,1024]
        dtok = torch.zeros(DIM, dtype=hidden.dtype)
        kcache = [None] * len(self.layers)
        vcache = [None] * len(self.layers)
        toks = []
        for i in range(CODEBOOKS):
            x = (din[i] + dtok)[None, None, :]                    # [1,1,1024]
            for li, layer in enumerate(self.layers):
                x, kcache[li], vcache[li] = layer(x, kcache[li], vcache[li])
            lg = self.heads[i].logits(x.squeeze(0).squeeze(0))    # [2049]
            tk = int(lg.argmax())
            toks.append(tk)
            dtok = self.heads[i].embed(torch.tensor(tk)).to(hidden.dtype)
        return toks


class AudioEmbedding(nn.Module):
    """Feedback embed: sampled 8-CB frame -> LFM input embed. audio_embedding(tokens+
    offsets).sum(0), a SharedEmbedding table over 8*2049 vocab (to_logits unused in gen)."""
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(CODEBOOKS * AUDIO_VOCAB, LFM_DIM)   # [16392,2048]
        self.register_buffer("offsets", torch.arange(CODEBOOKS) * AUDIO_VOCAB)

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        """toks [8] int -> [2048] feedback embed."""
        return self.embedding(toks + self.offsets).sum(0)


def load_audio_embedding(dtype=torch.float32) -> AudioEmbedding:
    from safetensors import safe_open
    m = AudioEmbedding().to(dtype).eval()
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        m.embedding.weight.data.copy_(f.get_tensor("audio_embedding.embedding.weight").to(dtype))
    return m


def load_depthformer(dtype=torch.float32) -> Depthformer:
    from safetensors import safe_open
    m = Depthformer().to(dtype).eval()
    sd = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        g = lambda k: f.get_tensor(k).to(dtype)  # noqa: E731
        sd["depth_linear.weight"] = g("depth_linear.weight")
        sd["depth_linear.bias"] = g("depth_linear.bias")
        for i in range(6):
            p = f"depthformer.layers.{i}."
            sd[f"layers.{i}.qkv_proj.weight"] = g(p + "operator.qkv_proj.weight")
            sd[f"layers.{i}.out_proj.weight"] = g(p + "operator.out_proj.weight")
            sd[f"layers.{i}.q_ln"] = g(p + "operator.bounded_attention.q_layernorm.weight")
            sd[f"layers.{i}.k_ln"] = g(p + "operator.bounded_attention.k_layernorm.weight")
            sd[f"layers.{i}.operator_norm"] = g(p + "operator_norm.weight")
            sd[f"layers.{i}.ffn_norm"] = g(p + "ffn_norm.weight")
            sd[f"layers.{i}.w1.weight"] = g(p + "feed_forward.w1.weight")
            sd[f"layers.{i}.w2.weight"] = g(p + "feed_forward.w2.weight")
            sd[f"layers.{i}.w3.weight"] = g(p + "feed_forward.w3.weight")
        for i in range(CODEBOOKS):
            p = f"depth_embeddings.{i}."
            sd[f"heads.{i}.embedding.weight"] = g(p + "embedding.weight")
            sd[f"heads.{i}.embedding_norm"] = g(p + "embedding_norm.weight")
            sd[f"heads.{i}.to_logits.weight"] = g(p + "to_logits.weight")
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected: {unexpected[:4]}"
    assert not missing, f"missing: {missing[:4]}"
    return m
