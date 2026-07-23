# Community port — NOT an Apple model.
"""Static-KV Qwen2 backbone(s) for VibeVoice-Realtime-0.5B: `inputs_embeds -> hidden`.

Both LMs are stock Qwen2 (h896, head_dim 64, 14 heads / 2 KV, qkv bias, no qk_norm, plain RoPE
theta 1e6, eps 1e-6). Adapted verbatim from conversion/dots_tts/backbone.py (same static-KV recipe:
data-driven KV slot write, whole-buffer read, explicit causal mask; SDPA not externalized).

Two configs:
  * MAIN LM  = 4 lower layers, final norm = **Identity** (upstream sets language_model.norm = Identity).
               Host feeds text-token embeds; output hidden is spliced into the TTS LM tail.
  * TTS LM   = 20 upper layers, final norm = real RMSNorm. Host feeds (spliced hidden + type embed).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA


class Qwen2Cfg:
    hidden_size = 896
    intermediate_size = 4864
    num_attention_heads = 14
    num_key_value_heads = 2
    head_dim = 896 // 14  # 64
    rms_norm_eps = 1e-6
    rope_theta = 1000000.0
    vocab_size = 151936

    def __init__(self, num_hidden_layers: int, buf: int, final_norm: bool):
        self.num_hidden_layers = num_hidden_layers
        self.buf = buf
        self.final_norm = final_norm


def make_rope_tables(cfg: Qwen2Cfg, buf: int, dtype=torch.float32):
    hd = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    t = torch.arange(buf, dtype=torch.float32)
    emb = torch.cat((torch.outer(t, inv_freq),) * 2, dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    orig = q.dtype
    q, k = q.float(), k.float()
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.to(orig), k.to(orig)


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
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=True)
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
    """inputs_embeds -> hidden. Static KV, GQA, baked RoPE. final norm optional (Identity for main LM)."""
    def __init__(self, cfg: Qwen2Cfg, buf: int):
        super().__init__()
        self.cfg = cfg
        self.buf = buf
        self.layers = nn.ModuleList([Layer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps) if cfg.final_norm else nn.Identity()
        self.sdpa = SDPA(is_causal=False)
        cos, sin = make_rope_tables(cfg, buf)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def _attn(self, attn, x, q_pos, write_start, mask, k_cache, v_cache, li):
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

    def prefill(self, inputs_embeds, k_cache, v_cache):
        b, T, _ = inputs_embeds.shape
        q_pos = torch.arange(T, dtype=torch.int32, device=inputs_embeds.device).unsqueeze(0)
        mask = causal_buffer_mask(q_pos, self.buf)
        return self._run(inputs_embeds, q_pos, 0, mask, k_cache, v_cache)

    def decode(self, inputs_embeds, pos, k_cache, v_cache):
        q_pos = pos.reshape(1, 1)
        mask = causal_buffer_mask(q_pos, self.buf)
        return self._run(inputs_embeds, q_pos, pos, mask, k_cache, v_cache)

    # arbitrary q_len at an arbitrary start position (a text window appended mid-stream).
    def step(self, inputs_embeds, start, k_cache, v_cache):
        b, T, _ = inputs_embeds.shape
        start_i = start if torch.is_tensor(start) else torch.tensor([start], dtype=torch.int32,
                                                                     device=inputs_embeds.device)
        q_pos = (start_i.reshape(1) + torch.arange(T, dtype=torch.int32, device=inputs_embeds.device)).unsqueeze(0)
        mask = causal_buffer_mask(q_pos, self.buf)
        return self._run(inputs_embeds, q_pos, start_i, mask, k_cache, v_cache)


def build_kv_state(cfg: Qwen2Cfg, buf: int, dtype=torch.float32):
    n, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    return (torch.zeros(n, 1, nkv, buf, hd, dtype=dtype), torch.zeros(n, 1, nkv, buf, hd, dtype=dtype))


def load_backbone(state_dict: dict, prefix: str, num_layers: int, buf: int,
                  final_norm: bool, dtype=torch.float32) -> Qwen2Backbone:
    cfg = Qwen2Cfg(num_layers, buf, final_norm)
    m = Qwen2Backbone(cfg, buf).to(dtype).eval()
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
