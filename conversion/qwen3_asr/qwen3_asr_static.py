# Community port — NOT an Apple model.
"""Fixed-shape Qwen3-ASR decode graphs (prefill q_len=Sp + decode q=1, scalar ``pos``).

The dynamic ship bundle drives the decoder with a *growing* ``position_ids[1, p+1]``: every step is
a new shape, so the engine re-specializes (and probes ANE compile, which fails noisily) each token
-> wedges. This module re-exports the math as TWO fully-static graphs that compile ONCE -> flat
decode (mirrors ``conversion/unlimited_ocr`` and the qwen3-coder-next freeze fix).

Both graphs use ONE authored static attention (the unlimited_ocr recipe): write the new K/V into the
fixed KV buffer at a DATA-DRIVEN slot, read the WHOLE buffer ``[0, buf)``, and apply an explicit
causal mask ``j <= query_pos`` (unwritten slots are ``> pos`` so causality masks them). SDPA is NOT
externalized — the engine-native SDPA op can't take this runtime mask, and (the bug this replaces)
its fused cache read-back returns zeros for a freshly-written static prefill -> garbage logits.

* **prefill** (``Qwen3ASRStaticPrefill``): q_len=Sp (baked). Injects audio via the V+slot trick
  (``index_select`` the AuT ``audio_embeds`` into the embedding stream), writes prompt K/V into slots
  ``[0, Sp)`` with sequential positions ``arange(Sp)``, returns the **last-token** logits (decode step 0).
* **decode** (``Qwen3ASRStaticDecode``): q=1. ``pos`` is a runtime int32 VALUE used to write the new
  K/V at slot ``pos`` and RoPE the single query. Mask ``j <= pos``.

Both reuse the SAME loaded (and quantized, in place) Qwen3 submodules (qkv_proj/qk_norm/o_proj/rope/
mlp/norms/embed/lm_head). The KV cache is shared host-side: the same ``keyCache``/``valueCache``
buffers are passed to the prefill function then the decode loop.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA

from qwen3_asr_decoder import PIPELINED_STATE_NAMES, Qwen3ASRDecoderPipelined

STATE_NAMES = PIPELINED_STATE_NAMES  # ("keyCache", "valueCache")


def write_kv_range(cache: torch.Tensor, layer_idx: int, start, x: torch.Tensor) -> None:
    """Write ``x`` into ``cache[layer_idx, :, :, start:start+q, :]`` at a DATA-DRIVEN slot ``start``.

    cache ``[n_layers,1,n_kv,buf,head_dim]``; x ``[1,n_kv,q,head_dim]``; ``start`` an int (prefill,
    baked) or a ``[1]`` int32 tensor (decode, runtime VALUE). begin/end are built from ``start`` so
    the graph input shapes stay constant -> no per-step Metal recompile.
    """
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
    """[1,1,q,buf] bool, True = attend: key slot j visible to query at absolute pos i iff j <= i."""
    k_idx = torch.arange(buf, device=q_pos.device)
    return k_idx.reshape(1, 1, 1, buf) <= q_pos.reshape(1, 1, -1, 1)


def static_attn(
    attn: nn.Module, sdpa: SDPA, x: torch.Tensor, q_pos: torch.Tensor, write_start,
    mask: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, layer_idx: int,
) -> torch.Tensor:
    """One static-shape Qwen3 attention step: write q new keys at slots [write_start, +q), read the
    whole fixed buffer, masked SDPA. Reuses the loaded attn submodule weights."""
    b, q, _ = x.shape
    nh, nkv, hd = attn.n_heads, attn.n_kv_heads, attn.head_dim
    qkv = attn.qkv_proj(x).reshape(b, q, nh + 2 * nkv, hd).permute(0, 2, 1, 3)
    query_key = qkv.narrow(1, 0, nh + nkv)
    value = qkv.narrow(1, nh + nkv, nkv)                  # [1, nkv, q, hd] — not roped
    query_key = attn.qk_norm(query_key)
    query_key = attn.rope(query_key, position_ids=q_pos)
    query = query_key.narrow(1, 0, nh)                    # [1, nh, q, hd]
    key = query_key.narrow(1, nh, nkv)                    # [1, nkv, q, hd]

    write_kv_range(k_cache, layer_idx, write_start, key)
    write_kv_range(v_cache, layer_idx, write_start, value)
    full_k = k_cache.narrow(0, layer_idx, 1).squeeze(0)   # [1, nkv, buf, hd]
    full_v = v_cache.narrow(0, layer_idx, 1).squeeze(0)

    out = sdpa(query, full_k, full_v, attn_mask=mask).permute(0, 2, 1, 3).reshape(b, q, nh * hd)
    return attn.o_proj(out)


class _StaticBase(nn.Module):
    def __init__(self, base: Qwen3ASRDecoderPipelined) -> None:
        super().__init__()
        self.model = base.model        # Qwen3Model (embed_tokens, layers, norm) — quantized in place
        self.lm_head = base.lm_head
        self.config = base.config
        self.n_audio = base.n_audio_tokens
        self.sdpa = SDPA(is_causal=False)  # explicit mask; default scale 1/sqrt(hd)

    def _run_layers(self, x, q_pos, write_start, mask, k_cache, v_cache):
        m = self.model
        for i, layer in enumerate(m.layers):
            h = layer.input_layernorm(x)
            r = static_attn(layer.self_attn, self.sdpa, h, q_pos, write_start, mask, k_cache, v_cache, i)
            h = x + r
            r = layer.mlp(layer.post_attention_layernorm(h))
            x = h + r
        return m.norm(x)


class Qwen3ASRStaticPrefill(_StaticBase):
    """Static prompt prefill: q_len=Sp (baked), audio injected, writes [0,Sp). Last-token logits."""

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, Sp] int32; audio tokens = V + slot
        audio_embeds: torch.Tensor,  # [N, h] AuT encoder output
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        V = self.config.vocab_size
        N = self.n_audio
        b, Sp = input_ids.shape
        buf = k_cache.size(-2)

        is_aud = input_ids >= V
        slot = (input_ids - V).clamp(0, N - 1)
        e_txt = m.embed_tokens(input_ids.clamp(0, V - 1))
        e_aud = audio_embeds.index_select(0, slot.reshape(-1)).reshape(b, Sp, -1)
        x = torch.where(is_aud.unsqueeze(-1), e_aud.to(e_txt.dtype), e_txt)

        q_pos = torch.arange(Sp, dtype=torch.int32, device=input_ids.device).unsqueeze(0)
        mask = causal_buffer_mask(q_pos, buf)
        x = self._run_layers(x, q_pos, 0, mask, k_cache, v_cache)
        return self.lm_head(x[:, -1:, :])


class Qwen3ASRStaticDecode(_StaticBase):
    """Fully-static single-token decode: input_ids [1,1] (text id < V), pos [1] int32, cache."""

    def forward(
        self,
        input_ids: torch.Tensor,   # [1,1] int32
        pos: torch.Tensor,         # [1] int32 (absolute query position)
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        buf = k_cache.size(-2)
        x = self.model.embed_tokens(input_ids)  # decode tokens < V -> plain embedding
        q_pos = pos.reshape(1, 1)
        mask = causal_buffer_mask(q_pos, buf)
        x = self._run_layers(x, q_pos, pos, mask, k_cache, v_cache)
        return self.lm_head(x)


def build_kv_state(config, cache_len: int, dtype: torch.dtype = torch.float16) -> dict:
    n, nkv, hd = config.num_hidden_layers, config.num_key_value_heads, config.head_dim
    return {
        "k_cache": torch.zeros(n, 1, nkv, cache_len, hd, dtype=dtype),
        "v_cache": torch.zeros(n, 1, nkv, cache_len, hd, dtype=dtype),
    }
