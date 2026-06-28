# Community port — NOT an Apple model.
"""VoxCPM2 (2B) feat_encoder (LocEnc) — 12L bidirectional MiniCPM4, head_dim 128.

Structurally identical to the v1 LocEnc (prepend a learned special token to the P latent steps, run a
bidirectional encoder, take the special-token / cls output); only the config scales (12 layers vs 4,
head_dim 128 vs 64, the 64-entry LongRoPE short_factor) and patch_size is 4 (so the sequence is P+1=5
tokens). Reuses the v1 ``LocEnc`` module with a VoxCPM2 cfg.
"""
from __future__ import annotations

import torch

from feat_encoder import LocEnc
from feat_decoder_v2 import dit_cfg


def load_feat_encoder_v2(state_dict: dict, short_factor: list, n_layers: int = 12,
                         max_seq: int = 8, dtype=torch.float32) -> LocEnc:
    cfg = dit_cfg(n_layers, short_factor)   # encoder shares the dit dims (1024h/4096ffn/16h/2kv/128hd)
    enc = LocEnc(cfg, n_layers=n_layers, max_seq=max_seq).to(dtype).eval()
    own = enc.state_dict()
    pref = "feat_encoder."
    sub = {}
    for k, v in state_dict.items():
        if k.startswith(pref):
            kk = k[len(pref):].replace("encoder.layers.", "layers.").replace("encoder.norm.", "norm.")
            if kk in own:
                sub[kk] = v.to(dtype)
    missing = [k for k in own if k not in sub and not k.endswith(("cos_table", "sin_table"))]
    if missing:
        raise RuntimeError(f"feat_encoder_v2: unloaded {missing[:6]}")
    enc.load_state_dict(sub, strict=False, assign=True)
    return enc
