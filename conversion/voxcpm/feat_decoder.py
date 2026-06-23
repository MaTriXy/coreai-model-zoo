# Community port — NOT an Apple model.
"""VoxCPM feat_decoder = LocDiT (4L bidirectional MiniCPM4) wrapped in the CFM euler loop.

One audio patch per call: ``(mu=dit_hidden[1,1024], cond[1,64,2], z_noise[1,64,2]) -> pred_feat[1,64,2]``.
The 10-step euler ODE + classifier-free guidance (cfg_value 2.0, cfg-zero-star optimized scale, sway
schedule) is UNROLLED here so the whole thing bakes into one graph; the only stochastic input ``z`` is
host-supplied (no internal randn) so the graph is deterministic — exactly the FLUX/AdcSR diffusion
pattern. Mirrors ``modules/locdit/{unified_cfm,local_dit}.py`` bit-for-bit.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from minicpm4 import MLP, RMSNorm, MiniCPM4Cfg, apply_rope, make_rope_tables

PATCH = 2
FEAT = 64
N_TIMESTEPS = 10
CFG_VALUE = 2.0
SWAY = 1.0


def sinusoidal_pos_emb(x: torch.Tensor, dim: int, scale: float = 1000.0) -> torch.Tensor:
    """SinusoidalPosEmb (local_dit.py): x [N] -> [N, dim]."""
    half = dim // 2
    e = math.log(10000) / (half - 1)
    e = torch.exp(torch.arange(half, dtype=x.dtype, device=x.device) * -e)
    e = scale * x.unsqueeze(1) * e.unsqueeze(0)
    return torch.cat((e.sin(), e.cos()), dim=-1)


class TimeMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear_1 = nn.Linear(dim, dim)
        self.linear_2 = nn.Linear(dim, dim)

    def forward(self, x):
        return self.linear_2(F.silu(self.linear_1(x)))


class BidirAttn(nn.Module):
    """MiniCPM4 attention, full bidirectional, no KV (fresh short sequence). rope over arange(T)."""

    def __init__(self, cfg: MiniCPM4Cfg):
        super().__init__()
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=False)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin):
        b, T, _ = x.shape
        q = self.q_proj(x).reshape(b, T, self.nh, self.hd).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, T, self.nkv, self.hd).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, T, self.nkv, self.hd).permute(0, 2, 1, 3)
        q, k = apply_rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        return self.o_proj(o.permute(0, 2, 1, 3).reshape(b, T, self.nh * self.hd))


class BidirLayer(nn.Module):
    def __init__(self, cfg: MiniCPM4Cfg):
        super().__init__()
        self.self_attn = BidirAttn(cfg)
        self.mlp = MLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


class LocDiT(nn.Module):
    """estimator(x[N,64,2], mu[N,1024], t[N], cond[N,64,2], dt[N]) -> velocity[N,64,2]."""

    def __init__(self, cfg: MiniCPM4Cfg, n_layers: int = 4, max_seq: int = 8):
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(FEAT, cfg.hidden_size)
        self.cond_proj = nn.Linear(FEAT, cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, FEAT)
        self.time_mlp = TimeMLP(cfg.hidden_size)
        self.delta_time_mlp = TimeMLP(cfg.hidden_size)
        self.layers = nn.ModuleList([BidirLayer(cfg) for _ in range(n_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        cos, sin = make_rope_tables(cfg, max_seq)
        self.register_buffer("cos_table", cos, persistent=False)
        self.register_buffer("sin_table", sin, persistent=False)

    def forward(self, x, mu, t, cond, dt):
        H = self.cfg.hidden_size
        x = self.in_proj(x.transpose(1, 2).contiguous())     # [N,2,1024]
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())  # [N,2,1024]
        prefix = cond.size(1)
        te = self.time_mlp(sinusoidal_pos_emb(t, H).to(x.dtype))
        dte = self.delta_time_mlp(sinusoidal_pos_emb(dt, H).to(x.dtype))
        te = te + dte
        seq = torch.cat([(mu + te).unsqueeze(1), cond, x], dim=1)  # [N, 1+2+2, 1024]
        T = seq.size(1)
        cos = self.cos_table[:T].reshape(1, 1, T, self.cfg.head_dim)
        sin = self.sin_table[:T].reshape(1, 1, T, self.cfg.head_dim)
        h = seq
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        h = h[:, prefix + 1:, :]
        return self.out_proj(h).transpose(1, 2).contiguous()  # [N,64,2]


def _optimized_scale(pos_flat, neg_flat):
    dot = (pos_flat * neg_flat).sum(1, keepdim=True)
    sq = (neg_flat ** 2).sum(1, keepdim=True) + 1e-8
    return dot / sq


class CFMDecoder(nn.Module):
    """UnifiedCFM.solve_euler unrolled; z host-supplied. (mu,cond,z) -> pred_feat."""

    def __init__(self, estimator: LocDiT):
        super().__init__()
        self.estimator = estimator

    def forward(self, mu, cond, z):
        # t_span = linspace(1,0,N+1) + sway*(cos(pi/2 * t) - 1 + t)
        ts = torch.linspace(1, 0, N_TIMESTEPS + 1, dtype=mu.dtype, device=mu.device)
        ts = ts + SWAY * (torch.cos(math.pi / 2 * ts) - 1 + ts)
        x = z
        b = x.size(0)
        t = ts[0]
        dt = ts[0] - ts[1]
        zero_init = max(1, int(len(ts) * 0.04))  # =1
        for step in range(1, len(ts)):
            if step <= zero_init:
                dphi = torch.zeros_like(x)
            else:
                x_in = torch.cat([x, x], dim=0)
                mu_in = torch.cat([mu, torch.zeros_like(mu)], dim=0)
                t_in = torch.stack([t, t]).reshape(2 * b)
                dt_in = torch.zeros(2 * b, dtype=x.dtype, device=x.device)  # not mean_mode
                cond_in = torch.cat([cond, cond], dim=0)
                out = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                cond_v, uncond_v = torch.split(out, [b, b], dim=0)
                st = _optimized_scale(cond_v.reshape(b, -1), uncond_v.reshape(b, -1))
                st = st.reshape(b, *([1] * (cond_v.dim() - 1)))
                dphi = uncond_v * st + CFG_VALUE * (cond_v - uncond_v * st)
            x = x - dt * dphi
            t = t - dt
            if step < len(ts) - 1:
                dt = t - ts[step + 1]
        return x


def load_feat_decoder(state_dict: dict, n_layers: int = 4, dtype=torch.float32) -> CFMDecoder:
    cfg = MiniCPM4Cfg(num_hidden_layers=n_layers, vocab_size=0)
    est = LocDiT(cfg, n_layers=n_layers).to(dtype).eval()
    own = est.state_dict()
    pref = "feat_decoder.estimator."
    sub = {}
    for k, v in state_dict.items():
        if k.startswith(pref):
            kk = k[len(pref):].replace("decoder.layers.", "layers.").replace("decoder.norm.", "norm.")
            if kk in own:
                sub[kk] = v.to(dtype)
    missing = [k for k in own if k not in sub and not k.endswith(("cos_table", "sin_table"))]
    if missing:
        raise RuntimeError(f"feat_decoder: {len(missing)} unloaded, e.g. {missing[:6]}")
    est.load_state_dict(sub, strict=False, assign=True)
    return CFMDecoder(est).to(dtype).eval()
