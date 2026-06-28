# Community port — NOT an Apple model.
"""VoxCPM2 (2B) feat_decoder = LocDiT (12L bidirectional MiniCPM4, head_dim 128) in the CFM euler loop.

Same UnifiedCFM.solve_euler unroll as v1 (10-step euler + cfg-zero-star + sway, z host-supplied), so the
CFM wrapper (`feat_decoder.CFMDecoder`) is reused verbatim — only the estimator's input layout changed.

LocDiT layout deltas vs v1 (mirrors `_ref_v2/modules_locdit_local_dit_v2.py`):
* mu is the CONCAT of lm_to_dit_proj(lm_h) + res_to_dit_proj(res_h) => 2048 dims => reshaped to TWO
  hidden-1024 tokens (v1 ADDED them into one token).
* the timestep token is SEPARATE from mu (v1 fused ``mu + t`` into a single token).
* patch_size 4 (v1 was 2): x and cond carry 4 latent steps each.
  sequence = [mu(2), t(1), cond(4), x(4)] = 11 tokens; rope over arange(11); keep the last 4 (=x).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from feat_decoder import CFMDecoder, BidirLayer, TimeMLP, sinusoidal_pos_emb
from minicpm4 import RMSNorm, MiniCPM4Cfg, make_rope_tables

FEAT = 64


class LocDiTV2(nn.Module):
    """estimator(x[N,64,P], mu[N,2*H], t[N], cond[N,64,P], dt[N]) -> velocity[N,64,P]."""

    def __init__(self, cfg: MiniCPM4Cfg, n_layers: int = 12, max_seq: int = 16):
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
        x = self.in_proj(x.transpose(1, 2).contiguous())          # [N, P, H]
        cond = self.cond_proj(cond.transpose(1, 2).contiguous())  # [N, P, H]
        prefix = cond.size(1)
        te = self.time_mlp(sinusoidal_pos_emb(t, H).to(x.dtype))
        dte = self.delta_time_mlp(sinusoidal_pos_emb(dt, H).to(x.dtype))
        te = te + dte                                             # [N, H]
        mu = mu.view(x.size(0), -1, H)                            # [N, n_mu, H]  (n_mu = 2)
        seq = torch.cat([mu, te.unsqueeze(1), cond, x], dim=1)    # [N, n_mu+1+P+P, H]
        T = seq.size(1)
        cos = self.cos_table[:T].reshape(1, 1, T, self.cfg.head_dim)
        sin = self.sin_table[:T].reshape(1, 1, T, self.cfg.head_dim)
        h = seq
        for layer in self.layers:
            h = layer(h, cos, sin)
        h = self.norm(h)
        h = h[:, prefix + mu.size(1) + 1:, :]                     # keep the last P (=x)
        return self.out_proj(h).transpose(1, 2).contiguous()     # [N, 64, P]


def dit_cfg(n_layers: int, short_factor: list, *, hidden=1024, inter=4096,
            heads=16, kv=2, head_dim=128) -> MiniCPM4Cfg:
    return MiniCPM4Cfg(
        num_hidden_layers=n_layers, vocab_size=0,
        hidden_size=hidden, intermediate_size=inter,
        num_attention_heads=heads, num_key_value_heads=kv,
        head_dim=head_dim, short_factor=short_factor, no_rope=False,
    )


def load_feat_decoder_v2(state_dict: dict, short_factor: list, n_layers: int = 12,
                         dtype=torch.float32) -> CFMDecoder:
    cfg = dit_cfg(n_layers, short_factor)
    est = LocDiTV2(cfg, n_layers=n_layers).to(dtype).eval()
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
        raise RuntimeError(f"feat_decoder_v2: {len(missing)} unloaded, e.g. {missing[:6]}")
    est.load_state_dict(sub, strict=False, assign=True)
    return CFMDecoder(est).to(dtype).eval()
