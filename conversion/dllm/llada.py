# Community port — NOT an Apple model.
"""Fixed-shape LLaDA-8B backbone overlay for the d3LLM diffusion-LM port.

LLaDA is a masked-diffusion LM: a LLaMA-dense transformer run BIDIRECTIONALLY (no causal
mask) over a fixed-length canvas, predicting the [MASK] positions. The official class name
`LLaDALlamaBlock` is LLaMA-shaped with SEPARATE q/k/v projections (verified from the real
`d3LLM/d3LLM_LLaDA` checkpoint keys `model.transformer.blocks.N.{q_proj,k_proj,v_proj,
attn_out,ff_proj,up_proj,ff_out,attn_norm,ff_norm}.weight`).

Deltas vs the zoo's causal LLaMA-dense overlays (e.g. `voxcpm/minicpm4.py`):
* attention is BIDIRECTIONAL — `F.scaled_dot_product_attention(is_causal=False, attn_mask=None)`,
  exactly the VoxCPM2 LocDiT/feat_decoder BidirAttn pattern (export swaps this one call for the
  coreai SDPA primitive `SDPA(is_causal=False)` and externalize-drops it).
* NO KV cache — vanilla dLLM does a FULL forward over the whole canvas every denoising step;
  export uses a fixed seq bucket (e.g. 256). The d3LLM speedup lives in the host DECODE loop, not
  here.
* no qk-norm, no qkv bias, MHA 32q/32kv head_dim 128 (no GQA).
* RMSNorm = `weight * x` (fp32 variance, eps 1e-5) — NOT Gemma's `(1+w)`.
* RoPE theta = 500000, full-precision (fp32); standard `rotate_half` + `cat(freqs, freqs)` — the
  official `view(...,2,hd/2).unbind(-2)` is identical to `chunk(2, dim=-1)`.
* SwiGLU(llama): `ff_out( silu(ff_proj(x)) * up_proj(x) )`, intermediate 12288 (mlp_hidden_size in
  config; activation_type "silu" so each of ff_proj/up_proj is full-width, NOT halved).
* lm_head = `ff_out` (4096->126464), weight_tying False (independent of wte).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# config (d3LLM/d3LLM_LLaDA == GSAI-ML/LLaDA-8B-Instruct)
# --------------------------------------------------------------------------- #
class LLaDACfg:
    d_model = 4096
    n_layers = 32
    n_heads = 32
    n_kv_heads = 32           # MHA, no GQA
    head_dim = 4096 // 32     # 128
    intermediate = 12288      # mlp_hidden_size (full-width gate & up)
    vocab = 126464
    rms_norm_eps = 1e-5
    rope_theta = 500000.0
    mask_token_id = 126336
    max_seq = 4096

    def __init__(self, num_hidden_layers: int | None = None, max_seq: int | None = None):
        if num_hidden_layers is not None:
            self.n_layers = num_hidden_layers
        if max_seq is not None:
            self.max_seq = max_seq


def make_rope_tables(cfg: LLaDACfg, buf: int, dtype=torch.float32):
    """Bake cos/sin [buf, head_dim] exactly like the official LLaDA RotaryEmbedding.

    inv_freq[i] = 1/theta^(2i/hd); freqs = outer(t, inv_freq); emb = cat(freqs, freqs);
    cos/sin = emb.cos()/sin(). No NTK / longrope scaling (LLaDA uses plain RoPE).
    """
    hd = cfg.head_dim
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))  # [hd/2]
    t = torch.arange(buf, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)            # [buf, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)     # [buf, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """q [b,nh,T,hd], k [b,nkv,T,hd]; cos/sin [1,1,T,hd]. fp32 (rope_full_precision=True)."""
    orig = q.dtype
    q = q.float(); k = k.float()
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q.to(orig), k.to(orig)


# --------------------------------------------------------------------------- #
# submodules (key names match the LLaDA checkpoint exactly)
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        return self.weight * (x.float() * torch.rsqrt(v + self.eps)).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: LLaDACfg):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.nkv * self.hd, bias=False)
        self.attn_out = nn.Linear(self.nh * self.hd, cfg.d_model, bias=False)


class MLP(nn.Module):
    def __init__(self, cfg: LLaDACfg):
        super().__init__()
        self.ff_proj = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)   # gate
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)   # up
        self.ff_out = nn.Linear(cfg.intermediate, cfg.d_model, bias=False)    # down

    def forward(self, x):
        return self.ff_out(F.silu(self.ff_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: LLaDACfg):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ff_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.attn_out = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.ff_proj = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate, bias=False)
        self.ff_out = nn.Linear(cfg.intermediate, cfg.d_model, bias=False)
        self.nh, self.nkv, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim

    def _attn(self, x, cos, sin):
        b, T, _ = x.shape
        q = self.q_proj(x).view(b, T, self.nh, self.hd).transpose(1, 2)    # [b,nh,T,hd]
        k = self.k_proj(x).view(b, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(b, T, self.nkv, self.hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # BIDIRECTIONAL: no causal mask, no padding mask. Export swaps for SDPA(is_causal=False).
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)
        out = out.transpose(1, 2).reshape(b, T, self.nh * self.hd)
        return self.attn_out(out)

    def _mlp(self, x):
        return self.ff_out(F.silu(self.ff_proj(x)) * self.up_proj(x))

    def forward(self, x, cos, sin):
        x = x + self._attn(self.attn_norm(x), cos, sin)
        x = x + self._mlp(self.ff_norm(x))
        return x


class LLaDABackbone(nn.Module):
    """input_ids -> logits[B,T,vocab]. Fixed-shape, bidirectional, no KV (vanilla dLLM forward)."""

    def __init__(self, cfg: LLaDACfg, buf: int):
        super().__init__()
        self.cfg = cfg
        self.buf = buf
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ff_out = nn.Linear(cfg.d_model, cfg.vocab, bias=False)   # lm_head (weight_tying False)
        cos, sin = make_rope_tables(cfg, buf)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(self, input_ids, inputs_embeds=None, return_hidden=False):
        x = self.wte(input_ids) if inputs_embeds is None else inputs_embeds
        b, T, _ = x.shape
        cos = self.cos_table[:T].reshape(1, 1, T, self.cfg.head_dim)
        sin = self.sin_table[:T].reshape(1, 1, T, self.cfg.head_dim)
        hidden = []
        for blk in self.blocks:
            if return_hidden:
                hidden.append(x)                 # input to each block (matches official all_hidden_states)
            x = blk(x, cos, sin)
        x = self.ln_f(x)
        if return_hidden:
            hidden.append(x)                     # post-ln_f (matches official final hidden state)
        logits = self.ff_out(x)
        if return_hidden:
            return logits, hidden
        return logits


def load_backbone(state_dict: dict, num_layers: int, buf: int, dtype=torch.float32) -> LLaDABackbone:
    """Load `model.transformer.*` weights from the LLaDA checkpoint into the overlay.

    Strips the `model.transformer.` prefix; the remaining keys (wte/ln_f/ff_out/blocks.N.*) map
    1:1 onto this module's state_dict. `num_layers` may be < 32 for a fast few-layer parity gate.
    """
    cfg = LLaDACfg(num_hidden_layers=num_layers)
    m = LLaDABackbone(cfg, buf).to(dtype).eval()
    own = m.state_dict()
    prefix = "model.transformer."
    sub = {}
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue
        kk = k[len(prefix):]
        # skip blocks beyond the (possibly truncated) layer count
        if kk.startswith("blocks."):
            li = int(kk.split(".")[1])
            if li >= num_layers:
                continue
        if kk in own:
            sub[kk] = v.to(dtype)
    missing = [k for k in own if k not in sub and not k.endswith(("cos_table", "sin_table"))]
    if missing:
        raise RuntimeError(f"llada overlay: {len(missing)} unloaded params, e.g. {missing[:4]}")
    m.load_state_dict(sub, strict=False, assign=True)
    return m
