# Community port — NOT an Apple model.
"""Fixed-shape (static-KV) Qwen2.5-1.5B backbone for the dots.tts audio loop.

dots.tts drives the LLM with *continuous* ``inputs_embeds`` (text embeds + the re-encoded
audio patch feedback) and reads HIDDEN states — the lm_head is dead in the loop (the acoustic
path is the DiT flow head + a tiny eos head). So this graph is ``inputs_embeds -> hidden``,
mirroring ``voxcpm/minicpm4.py`` static-decode recipe (itself the qwen3_asr_static pattern):

* write the new K/V into the fixed KV buffer at a DATA-DRIVEN slot, read the WHOLE buffer
  ``[0,buf)``, apply an explicit causal mask ``j <= query_pos``. SDPA is NOT externalized
  (engine SDPA can't take the runtime mask).

Qwen2 deltas vs the MiniCPM4 backbone: **q/k/v HAVE bias**, **no qk_norm** (Qwen3 has it,
Qwen2 doesn't), and a **plain RoPE** (rope_theta 1e6, no LongRoPE short_factor) — bake cos/sin
as constants and apply rotate_half by hand. head_dim 128, GQA 2 kv, 28 layers, hidden 1536.

The plain-torch bodies are the SAME math already gated cos=1.000000 vs the oracle in
``torch_overlays.Qwen2Backbone`` — reformulated for a static KV cache.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA


# --------------------------------------------------------------------------- #
# config (dots.tts llm_config.json — stock Qwen2.5-1.5B)
# --------------------------------------------------------------------------- #
class Qwen2Cfg:
    hidden_size = 1536
    intermediate_size = 8960
    num_hidden_layers = 28
    num_attention_heads = 12
    num_key_value_heads = 2
    head_dim = 1536 // 12  # 128
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    vocab_size = 151672

    def __init__(self, buf: int):
        self.buf = buf


def make_rope_tables(cfg: Qwen2Cfg, buf: int, dtype=torch.float32):
    """Bake cos/sin [buf, head_dim] for plain Qwen2 RoPE (no scaling)."""
    hd = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))  # [hd/2]
    t = torch.arange(buf, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # [buf, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)   # [buf, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q [1,nh,T,hd], k [1,nkv,T,hd]; cos/sin [1,1,T,hd] (indexed by position)."""
    orig = q.dtype
    q = q.float(); k = k.float()
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.to(orig), k.to(orig)


# --------------------------------------------------------------------------- #
# static KV helpers (verbatim from voxcpm/minicpm4.py = qwen3_asr_static)
# --------------------------------------------------------------------------- #
def write_kv_range(cache: torch.Tensor, layer_idx: int, start, x: torch.Tensor) -> None:
    dev = cache.device
    q = x.shape[2]
    li = torch.tensor([layer_idx], dtype=torch.int32, device=dev)
    z = torch.zeros(1, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.int32, device=dev)
    nkv = torch.tensor([cache.shape[2]], dtype=torch.int32, device=dev)
    hd = torch.tensor([cache.shape[4]], dtype=torch.int32, device=dev)
    qn = torch.tensor([q], dtype=torch.int32, device=dev)
    s = (start if torch.is_tensor(start) else torch.tensor([start], dtype=torch.int32, device=dev)).to(torch.int32)
    begin = torch.cat([li, z, z, s, z])
    end = torch.cat([li + 1, one, nkv, s + qn, hd])
    mutable_slice_update(cache, x.unsqueeze(0), begin, end)


def causal_buffer_mask(q_pos: torch.Tensor, buf: int) -> torch.Tensor:
    k_idx = torch.arange(buf, device=q_pos.device)
    return k_idx.reshape(1, 1, 1, buf) <= q_pos.reshape(1, 1, -1, 1)


# --------------------------------------------------------------------------- #
# submodules (plain torch — match upstream `llm.model.*` key names)
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype) * self.weight


class MLP(nn.Module):
    def __init__(self, h, inter):
        super().__init__()
        self.gate_proj = nn.Linear(h, inter, bias=False)
        self.up_proj = nn.Linear(h, inter, bias=False)
        self.down_proj = nn.Linear(inter, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, cfg: Qwen2Cfg):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=True)   # Qwen2: qkv bias
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=True)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)


class Layer(nn.Module):
    def __init__(self, cfg: Qwen2Cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Qwen2Backbone(nn.Module):
    """inputs_embeds -> hidden (final RMSNorm output). Static KV, GQA, baked RoPE."""

    def __init__(self, cfg: Qwen2Cfg, buf: int):
        super().__init__()
        self.cfg = cfg
        self.buf = buf
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.sdpa = SDPA(is_causal=False)  # explicit mask
        cos, sin = make_rope_tables(cfg, buf)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def _attn(self, attn: Attention, x, q_pos, write_start, mask, k_cache, v_cache, li):
        b, q, _ = x.shape
        nh, nkv, hd = attn.nh, attn.nkv, attn.hd
        query = attn.q_proj(x).reshape(b, q, nh, hd).permute(0, 2, 1, 3)
        key = attn.k_proj(x).reshape(b, q, nkv, hd).permute(0, 2, 1, 3)
        value = attn.v_proj(x).reshape(b, q, nkv, hd).permute(0, 2, 1, 3)
        cos = self.cos_table.index_select(0, q_pos.reshape(-1)).reshape(1, 1, q, hd)
        sin = self.sin_table.index_select(0, q_pos.reshape(-1)).reshape(1, 1, q, hd)
        query, key = apply_rope(query, key, cos, sin)
        write_kv_range(k_cache, li, write_start, key)
        write_kv_range(v_cache, li, write_start, value)
        full_k = k_cache.narrow(0, li, 1).squeeze(0)
        full_v = v_cache.narrow(0, li, 1).squeeze(0)
        out = self.sdpa(query, full_k, full_v, attn_mask=mask).permute(0, 2, 1, 3).reshape(b, q, nh * hd)
        return attn.o_proj(out)

    def _run(self, x, q_pos, write_start, mask, k_cache, v_cache):
        for i, layer in enumerate(self.layers):
            h = layer.input_layernorm(x)
            x = x + self._attn(layer.self_attn, h, q_pos, write_start, mask, k_cache, v_cache, i)
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)

    # prefill: q_len = T (baked), positions arange(T), write [0,T).
    def prefill(self, inputs_embeds, k_cache, v_cache):
        b, T, _ = inputs_embeds.shape
        q_pos = torch.arange(T, dtype=torch.int32, device=inputs_embeds.device).unsqueeze(0)
        mask = causal_buffer_mask(q_pos, self.buf)
        return self._run(inputs_embeds, q_pos, 0, mask, k_cache, v_cache)

    # decode: q=1, pos a runtime [1] int32 value.
    def decode(self, inputs_embeds, pos, k_cache, v_cache):
        q_pos = pos.reshape(1, 1)
        mask = causal_buffer_mask(q_pos, self.buf)
        return self._run(inputs_embeds, q_pos, pos, mask, k_cache, v_cache)


def build_kv_state(cfg: Qwen2Cfg, buf: int, dtype=torch.float32):
    n, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    return (torch.zeros(n, 1, nkv, buf, hd, dtype=dtype), torch.zeros(n, 1, nkv, buf, hd, dtype=dtype))


def load_backbone(state_dict: dict, buf: int, dtype=torch.float32) -> Qwen2Backbone:
    """Load `llm.model.*` weights from the upstream dots.tts checkpoint into a static-KV backbone."""
    cfg = Qwen2Cfg(buf)
    m = Qwen2Backbone(cfg, buf).to(dtype).eval()
    own = m.state_dict()
    prefix = "llm.model."
    sub = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            kk = k[len(prefix):]
            if kk in own:
                sub[kk] = v.to(dtype)
    missing = [k for k in own if k not in sub and not k.endswith(("cos_table", "sin_table"))]
    if missing:
        raise RuntimeError(f"{prefix}: {len(missing)} unloaded params, e.g. {missing[:4]}")
    m.load_state_dict(sub, strict=False, assign=True)
    return m
