# Community port — NOT an Apple model.
"""Fixed-shape MiniCPM4 backbone graphs for VoxCPM (base_lm 24L + residual_lm 6L).

VoxCPM drives the two LMs with *continuous* ``inputs_embeds`` (text embeds in prefill, the
re-encoded audio patch ``curr_embed`` in the decode loop) — never token ids mid-loop — and reads
HIDDEN states (there is no lm_head over vocab; the audio path is the diffusion head). So these graphs
are ``inputs_embeds -> hidden`` decoders, mirroring ``conversion/qwen3_asr`` static-decode recipe:

* write the new K/V into the fixed KV buffer at a DATA-DRIVEN slot, read the WHOLE buffer ``[0,buf)``,
  apply an explicit causal mask ``j <= query_pos``. SDPA is NOT externalized (engine SDPA can't take
  the runtime mask).

MiniCPM4 deltas vs the qwen3 overlay (from the VoxCPM 0.5B config): **no qk_norm**, **separate
q/k/v** (bias-free), **use_mup=False** -> plain residual adds + ``scale_emb=1`` (no muP scaling), and
a **fixed LongRoPE** that (because ``short_factor==long_factor`` and ``orig==max==32768``) collapses
to ``scaling_factor=1`` with ``inv_freq /= short_factor`` — so we BAKE cos/sin as constants and apply
rotate_half by hand (avoids the externalized-RoPE-vs-baked-positions trap). head_dim 64, GQA 2 kv.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA


# --------------------------------------------------------------------------- #
# config (VoxCPM-0.5B lm_config; residual_lm differs only in n_layers/vocab)
# --------------------------------------------------------------------------- #
class MiniCPM4Cfg:
    hidden_size = 1024
    intermediate_size = 4096
    num_attention_heads = 16
    num_key_value_heads = 2
    head_dim = 1024 // 16  # 64
    rms_norm_eps = 1e-5
    rope_theta = 10000.0
    # longrope short_factor (== long_factor); 32 entries (head_dim/2). scaling_factor == 1.0.
    short_factor = [
        1.0004360675811768, 1.0668443441390991, 1.1631425619125366, 1.3025742769241333,
        1.5040205717086792, 1.7941505908966064, 2.2101221084594727, 2.802666664123535,
        3.6389970779418945, 4.804192543029785, 6.39855432510376, 8.527148246765137,
        11.277542114257812, 14.684998512268066, 18.69317054748535, 23.13019371032715,
        27.72362518310547, 32.1606559753418, 36.168827056884766, 39.57627868652344,
        42.32667541503906, 44.45526885986328, 46.04962921142578, 47.21482849121094,
        48.05115509033203, 48.64370346069336, 49.05967712402344, 49.34980392456055,
        49.551246643066406, 49.69068145751953, 49.78697967529297, 49.85338592529297,
    ]

    no_rope = False  # residual_lm in VoxCPM2 has no positional embedding

    def __init__(self, num_hidden_layers: int, vocab_size: int, *,
                 hidden_size: int | None = None, intermediate_size: int | None = None,
                 num_attention_heads: int | None = None, num_key_value_heads: int | None = None,
                 head_dim: int | None = None, short_factor: list | None = None,
                 no_rope: bool | None = None):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        # optional overrides (defaults = VoxCPM-0.5B class attrs above; VoxCPM2 passes 2B values)
        if hidden_size is not None:
            self.hidden_size = hidden_size
        if intermediate_size is not None:
            self.intermediate_size = intermediate_size
        if num_attention_heads is not None:
            self.num_attention_heads = num_attention_heads
        if num_key_value_heads is not None:
            self.num_key_value_heads = num_key_value_heads
        if head_dim is not None:
            self.head_dim = head_dim
        if short_factor is not None:
            self.short_factor = short_factor
        if no_rope is not None:
            self.no_rope = no_rope


def make_rope_tables(cfg: MiniCPM4Cfg, buf: int, dtype=torch.float32):
    """Bake cos/sin [buf, head_dim] exactly like VoxCPM MiniCPMLongRoPE (scaling_factor==1).

    inv_freq[i] = 1/theta^(2i/hd); effective = inv_freq[i] / short_factor[i]; freqs = outer(t, eff);
    emb = cat(freqs, freqs); cos/sin = emb.cos()/sin().  (apply_rotary_pos_emb uses rotate_half.)
    """
    hd = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))  # [hd/2]
    ext = torch.tensor(cfg.short_factor, dtype=torch.float32)                                 # [hd/2]
    eff = inv_freq / ext
    t = torch.arange(buf, dtype=torch.float32)
    freqs = torch.outer(t, eff)               # [buf, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)   # [buf, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q [1,nh,T,hd], k [1,nkv,T,hd]; cos/sin [T,hd] (already indexed by position)."""
    orig = q.dtype
    q = q.float(); k = k.float()
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.to(orig), k.to(orig)


# --------------------------------------------------------------------------- #
# static KV helpers (verbatim shape-stable pattern from qwen3_asr_static)
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
# submodules (plain torch — match VoxCPM checkpoint key names exactly)
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
    def __init__(self, cfg: MiniCPM4Cfg):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)


class Layer(nn.Module):
    def __init__(self, cfg: MiniCPM4Cfg):
        super().__init__()
        self.self_attn = Attention(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class MiniCPM4Backbone(nn.Module):
    """inputs_embeds -> hidden. Shared by base_lm (24L, has embed_tokens) and residual_lm (6L)."""

    def __init__(self, cfg: MiniCPM4Cfg, buf: int):
        super().__init__()
        self.cfg = cfg
        self.buf = buf
        if cfg.vocab_size > 0:
            self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.sdpa = SDPA(is_causal=False)  # explicit mask
        self.no_rope = getattr(cfg, "no_rope", False)  # residual_lm (VoxCPM2) has no positional emb
        if self.no_rope:
            self.cos_table = None
            self.sin_table = None
        else:
            cos, sin = make_rope_tables(cfg, buf)
            self.register_buffer("cos_table", cos, persistent=False)
            self.register_buffer("sin_table", sin, persistent=False)

    def _attn(self, attn: Attention, x, q_pos, write_start, mask, k_cache, v_cache, li):
        b, q, _ = x.shape
        nh, nkv, hd = attn.nh, attn.nkv, attn.hd
        query = attn.q_proj(x).reshape(b, q, nh, hd).permute(0, 2, 1, 3)
        key = attn.k_proj(x).reshape(b, q, nkv, hd).permute(0, 2, 1, 3)
        value = attn.v_proj(x).reshape(b, q, nkv, hd).permute(0, 2, 1, 3)
        if not self.no_rope:
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

    # prefill: q_len = T (baked), positions arange(T), write [0,T). returns ALL positions.
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


def build_kv_state(cfg: MiniCPM4Cfg, buf: int, dtype=torch.float32):
    n, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    return (torch.zeros(n, 1, nkv, buf, hd, dtype=dtype), torch.zeros(n, 1, nkv, buf, hd, dtype=dtype))


def load_backbone(state_dict: dict, prefix: str, num_layers: int, vocab_size: int, buf: int,
                  dtype=torch.float32, **cfg_kwargs) -> MiniCPM4Backbone:
    """Load base_lm.* / residual_lm.* weights from the VoxCPM checkpoint into a backbone.

    cfg_kwargs (hidden_size/head_dim/intermediate_size/num_*_heads/short_factor/no_rope) override the
    VoxCPM-0.5B defaults — VoxCPM2 passes the 2B values + no_rope=True for the residual_lm."""
    cfg = MiniCPM4Cfg(num_hidden_layers=num_layers, vocab_size=vocab_size, **cfg_kwargs)
    m = MiniCPM4Backbone(cfg, buf).to(dtype).eval()
    own = m.state_dict()
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
