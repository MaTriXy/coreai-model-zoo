# Community port — NOT an Apple model.
"""Qwen3-ASR AuT audio encoder, reworked to a **fixed shape** for Core AI export.

The HF ``Qwen3ASRAudioEncoder.forward`` is data-dependent (dynamic ``split`` into
``n_window*2 = 100``-frame chunks, ragged ``cu_seqlens`` windowed attention, and a
boolean-mask gather of valid post-CNN frames) and will not ``torch.export`` to one
static graph. This module produces the *same* numbers with a single fixed-shape graph:

  mel ``[1, 128, 100*K]`` (host zero-pads to a whole number of chunks ``K``)
    -> ``[K, 1, 128, 100]``
    -> conv2d1/2/3 (stride 2, 8x down in freq & time) -> ``[K, 480, 16, 13]``
    -> permute+view -> conv_out -> ``[K, 13, 1024]``
    -> + per-chunk sinusoid pos -> flatten -> ``[K*13, 1024]``
    -> 24x windowed self-attention (block = 8 chunks = 104 tokens) via a static
       ``attn_bias [1, S, S]`` (0 in-window&valid / -inf else) -> ``[K*13, 1024]``
    -> ln_post -> proj1 -> gelu -> proj2 -> ``[K*13 = N_max, 2048]``

Host trims ``[N_max, 2048]`` to the clip's real ``N`` token count. ``attn_bias``
encodes both the 104-token windows AND masks the partial last chunk's pad tokens,
which always land at flat indices ``[N, K*13)`` (only the last chunk is partial).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_attn_bias(n_tokens_max: int, n_valid: int, window: int = 104,
                    dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Replicate the HF ``cu_seqlens`` windows as a static additive mask ``[1, S, S]``.

    Valid tokens (0..n_valid-1) attend bidirectionally within their 104-token window
    (``i//window == j//window``); pad tokens (>= n_valid) are masked as keys.
    """
    S = n_tokens_max
    neg = torch.finfo(dtype).min
    idx = torch.arange(S)
    valid = idx < n_valid
    same_window = (idx[:, None] // window) == (idx[None, :] // window)
    allow = same_window & valid[None, :] & valid[:, None]
    bias = torch.where(allow, torch.zeros((), dtype=dtype), torch.full((), neg, dtype=dtype))
    return bias.unsqueeze(0)  # [1, S, S]


class AuTEncoderStatic(nn.Module):
    """Fixed-shape AuT encoder wrapping the HF ``audio_tower`` weights.

    ``n_chunks`` (``K``) is baked at construction: accepts mel of width ``100*K``
    and emits ``K*13`` tokens (host trims to the clip's real ``N``).
    """

    def __init__(self, tower, n_chunks: int) -> None:
        super().__init__()
        self.t = tower
        self.K = int(n_chunks)
        self.chunk_mel = tower.n_window * 2                       # 100 mel frames / chunk
        self.tok_per_chunk = ((((tower.n_window * 2) - 1) // 2 + 1 - 1) // 2 + 1 - 1) // 2 + 1  # 13
        self.H = tower.config.encoder_attention_heads
        self.hd = tower.config.d_model // self.H
        self.scaling = self.hd ** -0.5
        self.window = self.tok_per_chunk * (tower.n_window_infer // self.chunk_mel)  # 104

    def _layer(self, layer, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        residual = x
        h = layer.self_attn_layer_norm(x)
        S, d = h.shape
        sa = layer.self_attn
        q = sa.q_proj(h).view(S, self.H, self.hd).transpose(0, 1)   # [H, S, hd]
        k = sa.k_proj(h).view(S, self.H, self.hd).transpose(0, 1)
        v = sa.v_proj(h).view(S, self.H, self.hd).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # [H, S, S]
        scores = scores + bias                                        # [1, S, S]
        attn = torch.softmax(scores, dim=-1)
        o = torch.matmul(attn, v).transpose(0, 1).reshape(S, d)
        h = sa.out_proj(o)
        x = residual + h
        residual = x
        h = layer.final_layer_norm(x)
        h = layer.fc2(layer.activation_fn(layer.fc1(h)))
        x = residual + h
        if x.dtype == torch.float16:                              # match HF fp16 clamp
            clamp = torch.finfo(x.dtype).max - 1000
            x = torch.clamp(x, min=-clamp, max=clamp)
        return x

    def forward(self, input_features: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        """input_features ``[1, 128, 100*K]`` -> audio_embeds ``[K*13, 2048]``."""
        t = self.t
        nmel = t.num_mel_bins
        # [1, 128, 100*K] -> [K, 1, 128, 100]
        x = input_features.reshape(nmel, self.K, self.chunk_mel).permute(1, 0, 2).contiguous().unsqueeze(1)
        x = F.gelu(t.conv2d1(x))
        x = F.gelu(t.conv2d2(x))
        x = F.gelu(t.conv2d3(x))                                  # [K, 480, 16, 13]
        b, c, f, tt = x.shape
        x = t.conv_out(x.permute(0, 3, 1, 2).contiguous().view(b, tt, c * f))  # [K, 13, 1024]
        pe = t.positional_embedding.positional_embedding[:tt, :].unsqueeze(0).to(x.dtype)
        x = (x + pe).reshape(b * tt, -1)                          # [K*13, 1024]
        for layer in t.layers:
            x = self._layer(layer, x, attn_bias)
        x = t.ln_post(x)
        x = t.proj1(x)
        x = t.act(x)
        x = t.proj2(x)
        return x                                                  # [K*13, 2048]
