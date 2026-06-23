# Unlimited-OCR (baidu/Unlimited-OCR) DeepseekV2 R-SWA text decoder, authored on the
# Core AI macOS primitives for export. Community port — NOT an Apple model.
#
# This is the export-ready refactor of the verified math in the original plain-torch parity
# (which passed eager parity 9/9 vs the fp32 HF oracle). It mirrors glm4_moe_lite.py
# but with three differences that define this model:
#
#   1. PLAIN MHA (no MLA): use_mla=False, qk_nope/qk_rope_head_dim=0. 10 heads x
#      head_dim 128 (= hidden 1280 / 10), no GQA (kv_heads == q_heads == 10),
#      attention_bias=False, full-128-dim Llama RoPE (rotate_half, theta=10000).
#
#   2. R-SWA attention (SlidingWindowLlamaAttention, sliding_window=128): every token
#      attends the GLOBAL prefix [0, Lm) (the visual tokens + prompt written at
#      prefill) UNION a 128-token sliding tail. Two exportable fetch modes:
#        - full_mask: standard KVCache fetch [0, seq_len) + an additive R-SWA mask
#          `(j<=i) & ((j<Lm) | (j>i-W))` (correctness path; matches the eager parity).
#        - gather: prefix[0,Lm) ++ tail[seq_len-W, seq_len) gathered as two constant
#          slices -> constant-shape SDPA -> FLAT decode latency (the product win).
#
#   3. Greedy softmax MoE (DeepseekV2, topk_method=greedy): scores=softmax(gate),
#      top-6 RAW weights (norm_topk_prob=False, routed_scaling_factor=1.0 -> NO
#      norm / scale / selection bias) + a non-gated shared expert (one MLP at
#      moe_intermediate*2 = 1792). Layer 0 is a dense MLP (first_k_dense_replace=1).
#
# Checkpoint mapping (DeepseekV2 layout, experts stored UNPACKED, one per tensor):
#   model.embed_tokens.weight                                  -> same
#   lm_head.weight (untied, tie_word_embeddings=False)         -> same
#   model.norm.weight                                          -> same
#   model.layers.{L}.{input,post_attention}_layernorm.weight   -> same
#   model.layers.{L}.self_attn.{q,k,v,o}_proj.weight           -> same
#   dense layer 0:  model.layers.0.mlp.{gate,up,down}_proj.weight -> same
#   moe layers 1..11:
#     model.layers.{L}.mlp.gate.weight              [E,d]      -> ...mlp.gate.weight (router)
#     model.layers.{L}.mlp.experts.{e}.{gate,up,down}_proj.weight
#         -> stack_e -> ...mlp.switch_mlp.{gate,up,down}_proj.weight [1,E,out,in]
#     model.layers.{L}.mlp.shared_experts.{g,u,d}_proj.weight -> ...mlp.shared_expert.{...}
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.sdpa import SDPA
from coreai_models.primitives.macos.switch import SwitchGLU


@dataclass
class OcrDecoderConfig:
    """Unlimited-OCR decoder config (values verified from ckpt config.json)."""

    hidden_size: int = 1280
    num_hidden_layers: int = 12
    vocab_size: int = 129280
    intermediate_size: int = 6848  # dense layer-0 FFN width
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = False
    # plain MHA (no MLA)
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    head_dim: int = 128  # 1280 / 10
    attention_bias: bool = False
    rope_theta: float = 10000.0
    sliding_window: int = 128  # R-SWA tail width W
    # greedy-softmax MoE
    moe_intermediate_size: int = 896
    n_routed_experts: int = 64
    n_shared_experts: int = 2  # shared MLP width = moe_intermediate * n_shared = 1792
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 32768

    def is_moe(self, layer_idx: int) -> bool:
        return layer_idx >= self.first_k_dense_replace


# --------------------------------------------------------------------------- #
# Full-128-dim Llama RoPE (rotate_half), theta=10000 — matches the parity script.
# --------------------------------------------------------------------------- #
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q,k: [b,H,s,hd]; cos,sin: [b,s,hd]. Standard Llama rotate_half RoPE."""
    cos = cos.unsqueeze(1).to(q.dtype)  # [b,1,s,hd]
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


# --------------------------------------------------------------------------- #
# R-SWA attention over a FIXED-size KV buffer (StaticKVCache) + a constant-shape
# visibility mask. Standard monotonic write (slot == absolute position); the fetch
# returns the WHOLE fixed buffer (length buf_len, a constant) rather than the
# growing [0,seq_len) slice — so the SDPA key dimension and the mask shape are
# CONSTANT every step. The engine compiles the decode graph ONCE -> flat latency
# (a growing [0,seq_len) fetch instead recompiles the Metal shader per token, the
# Qwen3-Coder-Next freeze). The R-SWA window is expressed purely in the mask:
#   visible(i,j) = (j <= i) & ((j < Lm) | (j > i - W))
# over absolute positions; unwritten slots (j > i) are masked by causality, so the
# zero tail of the buffer contributes nothing. This is exactly the eager-parity
# formula, just evaluated over the full buffer width.
# --------------------------------------------------------------------------- #
def _rswa_bool_mask(
    q_pos: torch.Tensor, key_len: int, prefix_len: int, window: int
) -> torch.Tensor:
    """R-SWA visibility. q_pos [1,q_len] absolute query positions, keys at absolute
    positions [0, key_len). Returns a [1,1,q_len,key_len] bool mask (True = attend):
    causal AND (global prefix OR W-token sliding tail)."""
    device = q_pos.device
    k_idx = torch.arange(key_len, device=device)  # [key_len]
    q_abs = q_pos.reshape(-1, 1)  # [q_len,1]
    causal = k_idx[None, :] <= q_abs
    prefix = k_idx[None, :] < prefix_len
    tail = k_idx[None, :] > (q_abs - window)
    vis = causal & (prefix | tail)  # [q_len, key_len]
    return vis[None, None]


def write_kv_at(cache: torch.Tensor, layer_idx: int, pos: torch.Tensor, x: torch.Tensor) -> None:
    """Write x into cache[layer_idx, :, :, pos:pos+1, :] with a DATA-DRIVEN slot index.

    cache [n_layers,1,n_kv,buf,head_dim]; pos [1] int32 (the absolute slot, a runtime
    VALUE not a shape); x [1,n_kv,1,head_dim]. The begin/end of the slice are built
    from the ``pos`` tensor, so the graph's input shapes stay constant across decode
    steps — no per-step Metal recompile (the recompile that faults on Metal 4)."""
    dev = cache.device
    li = torch.tensor([layer_idx], dtype=torch.int32, device=dev)
    z = torch.zeros(1, dtype=torch.int32, device=dev)
    one = torch.ones(1, dtype=torch.int32, device=dev)
    nkv = torch.tensor([cache.shape[2]], dtype=torch.int32, device=dev)
    hd = torch.tensor([cache.shape[4]], dtype=torch.int32, device=dev)
    p = pos.to(torch.int32)
    begin = torch.cat([li, z, z, p, z])
    end = torch.cat([li + 1, one, nkv, p + 1, hd])
    mutable_slice_update(cache, x.unsqueeze(0), begin, end)


# --------------------------------------------------------------------------- #
# Plain MHA with R-SWA (no MLA).
# --------------------------------------------------------------------------- #
class OcrAttention(nn.Module):
    def __init__(self, config: OcrDecoderConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        proj = self.num_heads * self.head_dim
        bias = config.attention_bias
        self.q_proj = nn.Linear(d, proj, bias=bias)
        self.k_proj = nn.Linear(d, proj, bias=bias)
        self.v_proj = nn.Linear(d, proj, bias=bias)
        self.o_proj = nn.Linear(proj, d, bias=bias)
        self.sdpa = SDPA(scale=self.head_dim**-0.5, is_causal=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
        buf_len: int | None = None,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        H, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(b, s, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, H, hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)

        if kv_cache is None:  # eager parity / stateless
            out = self.sdpa(q, k, v, attn_mask=attn_mask)
        else:  # stateful: monotonic write, fetch the WHOLE fixed buffer (constant shape)
            key, value = kv_cache.update_and_fetch(
                layer_idx, offset, k, v, seq_len=buf_len, query_len=s
            )
            out = self.sdpa(q, key, value, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, H * hd)
        return self.o_proj(out)

    def decode_step(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        pos: torch.Tensor,
        layer_idx: int,
    ) -> torch.Tensor:
        """Fully-static single-token decode: write K/V at slot ``pos`` (data-driven),
        read the whole fixed buffer, masked SDPA. No dynamic shapes anywhere."""
        b, s, _ = x.shape  # b == s == 1
        H, hd = self.num_heads, self.head_dim
        q = self.q_proj(x).view(b, s, H, hd).transpose(1, 2)
        k = self.k_proj(x).view(b, s, H, hd).transpose(1, 2)
        v = self.v_proj(x).view(b, s, H, hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        write_kv_at(k_cache, layer_idx, pos, k)  # k [1,n_kv,1,hd] -> slot pos
        write_kv_at(v_cache, layer_idx, pos, v)
        key = k_cache.narrow(0, layer_idx, 1).squeeze(0)  # [1,n_kv,buf,hd] full buffer
        value = v_cache.narrow(0, layer_idx, 1).squeeze(0)
        out = self.sdpa(q, key, value, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(b, s, H * hd)
        return self.o_proj(out)


# --------------------------------------------------------------------------- #
# Greedy-softmax MoE (DeepseekV2) + non-gated shared expert.
# --------------------------------------------------------------------------- #
class OcrMoE(nn.Module):
    def __init__(self, config: OcrDecoderConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.top_k = config.num_experts_per_tok
        self.gate = nn.Linear(d, config.n_routed_experts, bias=False)
        self.switch_mlp = SwitchGLU(
            d, config.moe_intermediate_size, config.n_routed_experts, bias=False
        )
        self.shared_expert = MLP(d, config.moe_intermediate_size * config.n_shared_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(self.gate(x).float(), dim=-1)  # [b,s,E]
        weights, indices = torch.topk(scores, self.top_k, dim=-1)  # RAW top-k (no norm/scale/bias)
        y = self.switch_mlp(x, indices.to(torch.uint16))  # [b,s,k,d]
        y = (y * weights.unsqueeze(-1)).sum(dim=-2)  # fp32 weighted accumulate
        shared = self.shared_expert(x)  # non-gated, added directly
        return (y + shared.to(torch.float32)).to(x.dtype)


# --------------------------------------------------------------------------- #
class OcrDecoderLayer(nn.Module):
    def __init__(self, config: OcrDecoderConfig, layer_idx: int) -> None:
        super().__init__()
        d = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = OcrAttention(config)
        self.input_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(d, eps=config.rms_norm_eps)
        self.mlp: nn.Module = OcrMoE(config) if config.is_moe(layer_idx) else MLP(d, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor,
        kv_cache: KVCache | None = None,
        offset: int = 0,
        buf_len: int | None = None,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        r = self.self_attn(normed, cos, sin, attn_mask, kv_cache, offset, buf_len, self.layer_idx)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r

    def decode_step(self, x, cos, sin, attn_mask, k_cache, v_cache, pos):
        normed = self.input_layernorm(x)
        r = self.self_attn.decode_step(normed, cos, sin, attn_mask, k_cache, v_cache, pos, self.layer_idx)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


# --------------------------------------------------------------------------- #
class OcrDecoderModel(nn.Module):
    def __init__(self, config: OcrDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [OcrDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        hd = config.head_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, hd, 2).float() / hd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def reset_buffers(self, device: str = "cpu") -> None:
        hd = self.config.head_dim
        self.inv_freq = 1.0 / (
            self.config.rope_theta ** (torch.arange(0, hd, 2, device=device).float() / hd)
        )

    def rope_cos_sin(self, position_ids: torch.Tensor):
        freqs = position_ids[..., None].float() * self.inv_freq  # [b,s,hd/2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [b,s,hd]
        return emb.cos(), emb.sin()

    def forward(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor, prefix_len: int
    ) -> torch.Tensor:
        """Eager parity / stateless teacher-forced forward: full R-SWA mask, no cache."""
        seq_len = inputs_embeds.shape[1]
        cos, sin = self.rope_cos_sin(position_ids)
        mask = _rswa_bool_mask(position_ids, seq_len, prefix_len, self.config.sliding_window)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, cos, sin, mask)
        return self.norm(h)

    def forward_stateful_core(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: KVCache,
        prefix_len: int,
    ) -> torch.Tensor:
        query_len = inputs_embeds.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - query_len
        buf_len = kv_cache._k_cache.size(-2)  # fixed buffer width (constant shape)
        q_pos = position_ids.narrow(1, offset, query_len)
        cos, sin = self.rope_cos_sin(q_pos)
        # R-SWA mask over the WHOLE fixed buffer [0, buf_len) -> constant key dim.
        mask = _rswa_bool_mask(q_pos, buf_len, prefix_len, self.config.sliding_window)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, cos, sin, mask, kv_cache=kv_cache, offset=offset, buf_len=buf_len)
        return self.norm(h)

    def forward_decode(
        self,
        inputs_embeds: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        prefix_len: int,
    ) -> torch.Tensor:
        """Fully-static single-token decode. inputs_embeds [1,1,hidden], pos [1] int32
        (the absolute query position). All shapes constant -> compiles once -> flat."""
        buf_len = k_cache.size(-2)
        q_pos = pos.reshape(1, 1)
        cos, sin = self.rope_cos_sin(q_pos)
        mask = _rswa_bool_mask(q_pos, buf_len, prefix_len, self.config.sliding_window)
        h = inputs_embeds
        for layer in self.layers:
            h = layer.decode_step(h, cos, sin, mask, k_cache, v_cache, pos)
        return self.norm(h)


# State names surfaced via export_to_coreai(state_names=...) and the runtime.
DECODE_STATE_NAMES = ("keyCache", "valueCache")


def build_decode_state(
    config: OcrDecoderConfig, max_seq_len: int, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    n, nkv, hd = config.num_hidden_layers, config.num_key_value_heads, config.head_dim
    return {
        "k_cache": torch.zeros(n, 1, nkv, max_seq_len, hd, dtype=dtype),
        "v_cache": torch.zeros(n, 1, nkv, max_seq_len, hd, dtype=dtype),
    }


class OcrDecoderStatefulForCausalLM(nn.Module):
    """Stateful decoder. forward inputs: inputs_embeds [b, q_len, hidden] (a VLM, so
    the decoder is driven on embeddings, not token ids), position_ids [b, seq_len],
    and the two KV-cache states. ``prefix_len`` (the global prefix length Lm) and
    ``use_gather`` are baked at export as attributes on the module."""

    def __init__(self, config: OcrDecoderConfig) -> None:
        super().__init__()
        self.config = config
        self.model = OcrDecoderModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.last_token_only = False
        self.prefix_len = 0

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        pos: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Static single-token decode graph: inputs_embeds [1,1,hidden], pos [1] int32."""
        h = self.model.forward_decode(inputs_embeds, pos, k_cache, v_cache, self.prefix_len)
        return self.lm_head(h)


class OcrPrefillForCausalLM(nn.Module):
    """Batched prefill graph: writes the global prefix [0,Lm) into the KV cache in ONE
    forward (q_len=Lm) and returns the last-token logits (= decode step 0). Prefill always
    starts at offset 0, so positions are arange(Lm) (baked) and the write region [0,Lm) is
    constant -> static graph (Lm baked) -> compiles once. Uses the plain SwitchGLU/GatherMM
    MoE because q_len>1 (the sym8 metal kernel is decode-only); export this BEFORE metalize.
    Shares the decoder weights with the decode module."""

    def __init__(self, stateful: OcrDecoderStatefulForCausalLM, prefix_len: int) -> None:
        super().__init__()
        self.model = stateful.model
        self.lm_head = stateful.lm_head
        self.prefix_len = prefix_len

    def forward(
        self, inputs_embeds: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
    ) -> torch.Tensor:
        Lm = inputs_embeds.shape[1]
        pos = torch.arange(Lm, dtype=torch.int32, device=inputs_embeds.device).unsqueeze(0)
        kv = KVCache(k_cache, v_cache)
        h = self.model.forward_stateful_core(inputs_embeds, pos, kv, self.prefix_len)
        return self.lm_head(h[:, -1:, :])  # last token -> step 0


# --------------------------------------------------------------------------- #
# Weight loading (safetensors-direct; no transformers / remote code).
# --------------------------------------------------------------------------- #
def ocr_decoder_from_safetensors(
    ckpt_dir: str, target_dtype: torch.dtype = torch.float16
) -> OcrDecoderStatefulForCausalLM:
    """Load the Unlimited-OCR decoder weights from the local checkpoint into the
    authored module. Experts are stored one tensor per expert; stack into the
    [1, E, out, in] SwitchLinear layout. Vision / projector weights are ignored."""
    from safetensors import safe_open

    cfg_path = os.path.join(ckpt_dir, "config.json")
    raw = json.load(open(cfg_path)).get("language_config", {})
    config = OcrDecoderConfig(
        hidden_size=raw.get("hidden_size", 1280),
        num_hidden_layers=raw.get("num_hidden_layers", 12),
        vocab_size=raw.get("vocab_size", 129280),
        intermediate_size=raw.get("intermediate_size", 6848),
        moe_intermediate_size=raw.get("moe_intermediate_size", 896),
        n_routed_experts=raw.get("n_routed_experts", 64),
        n_shared_experts=raw.get("n_shared_experts", 2),
        num_experts_per_tok=raw.get("num_experts_per_tok", 6),
        first_k_dense_replace=raw.get("first_k_dense_replace", 1),
        num_attention_heads=raw.get("num_attention_heads", 10),
        num_key_value_heads=raw.get("num_key_value_heads", 10),
        sliding_window=raw.get("sliding_window", 128),
        max_position_embeddings=raw.get("max_position_embeddings", 32768),
    )

    with torch.device("meta"):
        model = OcrDecoderStatefulForCausalLM(config)
    model.to(dtype=target_dtype)

    st = glob.glob(os.path.join(ckpt_dir, "model-*.safetensors"))[0]
    E = config.n_routed_experts
    expert_stack: dict[tuple[int, str], dict[int, torch.Tensor]] = {}
    sd: dict[str, torch.Tensor] = {}
    with safe_open(st, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            if key.startswith(("model.vision_model.", "model.sam_model.", "model.projector.")):
                continue  # vision tower — handled by the DeepEncoder export
            # routed experts: model.layers.{L}.mlp.experts.{e}.{proj}_proj.weight
            if ".mlp.experts." in key:
                parts = key.split(".")
                li = int(parts[2])
                e = int(parts[5])
                proj = parts[6]  # gate_proj / up_proj / down_proj
                t = f.get_tensor(key)
                if t.dtype != target_dtype:
                    t = t.to(target_dtype)
                expert_stack.setdefault((li, proj), {})[e] = t
                continue
            local = key.replace(".mlp.shared_experts.", ".mlp.shared_expert.")
            t = f.get_tensor(key)
            if t.dtype != target_dtype and t.is_floating_point():
                t = t.to(target_dtype)
            sd[local] = t

    for (li, proj), per_expert in list(expert_stack.items()):
        if len(per_expert) != E:
            raise RuntimeError(f"layer {li} {proj}: got {len(per_expert)} experts, expected {E}")
        stacked = torch.stack([per_expert[e] for e in range(E)], dim=0)  # [E,out,in]
        sd[f"model.layers.{li}.mlp.switch_mlp.{proj}.weight"] = stacked.unsqueeze(0).contiguous()

    model.load_state_dict(sd, assign=True, strict=False)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()

    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unloaded params: {meta[:8]} ... ({len(meta)} total)")
    model.eval()
    return model
