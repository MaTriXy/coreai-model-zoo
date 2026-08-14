# North-Micro-Vision / Cohere `cohere_compass` for the Core AI authoring path.
#
# Community port — NOT an Apple model.
#
# The vision half is NOT here on purpose: this checkpoint's tower is a Qwen3-VL
# visual encoder in every structural respect — `patch_embed.proj [d,3,2,16,16]`,
# fused `blocks.N.attn.qkv`, `merger` + `deepstack_merger_list`, a `pos_embed`
# table interpolated to the grid — only with SigLIP2-SO400M's dimensions
# (hidden 1152 / MLP 4304 / 27 blocks / 16 heads, deepstack at [8,16,24]). The
# zoo's `Qwen3VLVisionEncoder` loads `model.visual.*` with ZERO missing or
# unexpected keys (verified), so `vision_encoder_from_hf` below is a config shim
# over it rather than a second implementation.
#
# What IS new is the decoder, and it is not a Llama with different numbers:
#
#   * **Parallel block.** One norm per layer (`input_layernorm`, and the weight
#     dump has no `post_attention_layernorm` at all): attention and MLP both read
#     it and their outputs are summed into the residual —
#     `h = x + attn(ln(x)) + mlp(ln(x))`. Wiring it as a serial pre-norm block
#     runs and produces plausible text; it is simply a different model.
#   * **Cohere LayerNorm**, not RMSNorm: the mean IS subtracted, and there is no
#     bias. Using RMSNorm because the weight shape matches is the same silent
#     class of error.
#   * **Half the layers have no positional encoding at all.** `layer_types` runs
#     SSSF x 7: the 21 `sliding_attention` layers carry interleaved M-RoPE
#     (sections [24,20,20], theta 5e4) over a 4096 window, and the 7
#     `full_attention` layers have `rope_parameters: null` — the reference
#     passes them `position_embeddings=None` and they attend without rotary.
#   * **logit_scale 0.25** multiplies the logits, and the 262 144-entry
#     embedding is tied to the head. At fp16 that table alone is 1.07 GB, which
#     is what decides whether this model reaches a phone.
#
# The multimodal contract is the zoo's usual one (cf. qwen3_vl_pipelined):
# image tokens arrive as EXTENSION ids `V + slot` and the deepstack rows are
# added at image positions in the first three layers.
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from coreai_models.models.macos.qwen3_vl_pipelined import (
    Qwen3VLVisionEncoder,
    mrope_masks,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA

__all__ = [
    "CohereCompassConfig",
    "CohereCompassPipelinedForCausalLM",
    "CohereCompassTextOnlyForCausalLM",
    "cohere_compass_config_from_dict",
    "vision_encoder_from_hf",
    "PIPELINED_STATE_NAMES",
]

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


@dataclass
class CohereCompassConfig:
    hidden_size: int = 2048
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 6144
    vocab_size: int = 262_144
    layer_norm_eps: float = 1e-5
    logit_scale: float = 0.25
    sliding_window: int = 4096
    tie_word_embeddings: bool = True
    # per-layer-type: "sliding_attention" | "full_attention"
    layer_types: list[str] = field(default_factory=list)
    sliding_rope_theta: float = 50_000.0
    mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    @property
    def n_layers_with_rope(self) -> int:
        return sum(t == "sliding_attention" for t in self.layer_types)


def cohere_compass_config_from_dict(raw: dict) -> CohereCompassConfig:
    """Authoring config from a raw config.json (the whole file or its text_config)."""
    t = raw.get("text_config", raw)
    rope = t.get("rope_parameters") or {}
    sliding = rope.get("sliding_attention") or {}
    return CohereCompassConfig(
        hidden_size=t["hidden_size"],
        num_hidden_layers=t["num_hidden_layers"],
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        head_dim=t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"],
        intermediate_size=t["intermediate_size"],
        vocab_size=t["vocab_size"],
        layer_norm_eps=t.get("layer_norm_eps", 1e-5),
        logit_scale=t.get("logit_scale") or 1.0,
        sliding_window=t.get("sliding_window") or 0,
        tie_word_embeddings=bool(t.get("tie_word_embeddings", True)),
        layer_types=list(t["layer_types"]),
        sliding_rope_theta=float(sliding.get("rope_theta", 50_000.0)),
        mrope_section=list(sliding.get("mrope_section", [24, 20, 20])),
    )


class _Attention(nn.Module):
    """GQA with fused qkv. Rotary and the sliding window are per layer TYPE:
    `sliding_attention` gets interleaved M-RoPE inside a 4096 window,
    `full_attention` gets neither (the reference hands it no position
    embeddings at all)."""

    def __init__(self, config: CohereCompassConfig, layer_idx: int, kv_idx: int) -> None:
        super().__init__()
        self.layer_idx = kv_idx
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.sliding = config.layer_types[layer_idx] == "sliding_attention"

        dim = config.hidden_size
        self.qkv_proj = nn.Linear(
            dim, (self.n_heads + 2 * self.n_kv_heads) * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.sdpa = SDPA(
            is_causal=True,
            window_size=config.sliding_window if self.sliding else 0,
        )
        if self.sliding:
            self.rope = initialize_rope(base=config.sliding_rope_theta)
            masks = mrope_masks(self.head_dim, config.mrope_section)
            self.register_buffer("mask_t", masks[0].view(1, 1, 1, -1), persistent=False)
            self.register_buffer("mask_h", masks[1].view(1, 1, 1, -1), persistent=False)
            self.register_buffer("mask_w", masks[2].view(1, 1, 1, -1), persistent=False)

    def _mrope(self, x, pos_t, pos_h, pos_w):
        return (
            self.rope(x, position_ids=pos_t) * self.mask_t
            + self.rope(x, position_ids=pos_h) * self.mask_h
            + self.rope(x, position_ids=pos_w) * self.mask_w
        )

    def forward(self, x, position_ids, pos_t, pos_h, pos_w, cache: KVCache):
        b, q, _ = x.shape
        n_heads, n_kv = self.n_heads, self.n_kv_heads
        qkv = (
            self.qkv_proj(x)
            .reshape(b, q, n_heads + 2 * n_kv, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        query_key = qkv.narrow(1, 0, n_heads + n_kv)
        value = qkv.narrow(1, n_heads + n_kv, n_kv)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(q)
        torch._check_is_size(seq_len)
        offset = seq_len - q
        torch._check_is_size(offset)

        if self.sliding:
            query_key = self._mrope(query_key, pos_t, pos_h, pos_w)
        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv)

        key, value = cache.update_and_fetch(
            self.layer_idx, offset, key, value, seq_len=seq_len, query_len=q
        )
        out = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(b, q, n_heads * self.head_dim)
        )
        return self.o_proj(out)


class _Block(nn.Module):
    """Cohere PARALLEL block: one norm, two branches, one residual."""

    def __init__(self, config: CohereCompassConfig, layer_idx: int, kv_idx: int) -> None:
        super().__init__()
        self.self_attn = _Attention(config, layer_idx, kv_idx)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        # Cohere's norm subtracts the mean and has no bias.
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps, bias=False
        )

    def forward(self, x, position_ids, pos_t, pos_h, pos_w, cache):
        normed = self.input_layernorm(x)
        return (
            x
            + self.self_attn(normed, position_ids, pos_t, pos_h, pos_w, cache)
            + self.mlp(normed)
        )


class CohereCompassPipelinedForCausalLM(nn.Module):
    """Engine-shaped `cohere_compass` decoder with the VLM static inputs.

    ``(input_ids [1,s] dyn-or-static, position_ids [1,total] dyn,
       image_embeds [N,h], deepstack_embeds [3N,h],
       rope_shift_start [1], rope_shift_amount [1],
       keyCache/valueCache) -> logits``

    Same contract as the Qwen3-VL rider — image tokens are extension ids
    ``V + slot`` and the first three layers add their deepstack rows at image
    positions — because this checkpoint's tower is that tower.
    """

    coreai_externalize_specs: tuple = ()

    def __init__(self, config: CohereCompassConfig, grid_h: int = 16, grid_w: int = 16) -> None:
        super().__init__()
        self.config = config
        self.grid_h, self.grid_w = grid_h, grid_w
        self.n_image_tokens = grid_h * grid_w
        self.n_deepstack = 3

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # Only the sliding layers hold KV that the engine's single pair indexes;
        # every layer needs a slot, so the KV index is the plain layer index.
        self.layers = nn.ModuleList(
            [_Block(config, i, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps, bias=False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
        rope_shift_start: torch.Tensor,
        rope_shift_amount: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        V = cfg.vocab_size
        N = self.n_image_tokens
        b, s = input_ids.shape

        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        p = position_ids.narrow(-1, offset, s)

        is_img = input_ids >= V
        slot = (input_ids - V).clamp(0, N - 1)
        flat_slot = slot.reshape(-1)

        e_txt = self.embed_tokens(input_ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, flat_slot).reshape(b, s, -1)
        x = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)

        shift = torch.where(p >= rope_shift_start, rope_shift_amount, torch.zeros_like(p))
        p_text = p - shift
        s0 = p - slot
        row = torch.div(slot, self.grid_w, rounding_mode="floor")
        col = slot - row * self.grid_w
        pos_t = torch.where(is_img, s0, p_text)
        pos_h = torch.where(is_img, s0 + row, p_text)
        pos_w = torch.where(is_img, s0 + col, p_text)
        img_f = is_img.unsqueeze(-1).to(x.dtype)

        cache = KVCache(k_cache, v_cache)
        for i, layer in enumerate(self.layers):
            x = layer(x, position_ids, pos_t, pos_h, pos_w, cache)
            if i < self.n_deepstack:
                ds = deepstack_embeds.index_select(0, flat_slot + i * N)
                x = x + ds.reshape(b, s, -1).to(x.dtype) * img_f

        return self.lm_head(self.norm(x)) * cfg.logit_scale

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id_or_dir: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> "CohereCompassPipelinedForCausalLM":
        raw, sd = _load_state_dict(hf_id_or_dir, "model.language_model.", target_dtype)
        config = cohere_compass_config_from_dict(raw)
        model = cls(config, grid_h=grid_h, grid_w=grid_w).to(dtype=target_dtype)

        out: dict[str, torch.Tensor] = {}
        for i in range(config.num_hidden_layers):
            pre = f"layers.{i}.self_attn."
            out[pre + "qkv_proj.weight"] = torch.cat(
                [sd.pop(pre + n + "_proj.weight") for n in ("q", "k", "v")], dim=0
            )
        for k, v in sd.items():
            out[k] = v
        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [
            k for k in missing
            if not k.endswith(("mask_t", "mask_h", "mask_w", "lm_head.weight"))
        ]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        model.eval()
        return model

    # -- export -------------------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_query: int = 1,
        trace_past: int = 64,
        static_ids: bool | None = None,
    ) -> dict:
        cfg = self.config
        N, h = self.n_image_tokens, cfg.hidden_size
        input_ids = torch.randint(1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
        position_ids = torch.arange(trace_past + trace_query, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(
            cfg.num_hidden_layers, 1, cfg.num_key_value_heads,
            trace_kv_len, cfg.head_dim, dtype=target_dtype,
        )
        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "image_embeds": torch.zeros(N, h, dtype=target_dtype),
            "deepstack_embeds": torch.zeros(3 * N, h, dtype=target_dtype),
            "rope_shift_start": torch.tensor([1 << 30], dtype=torch.int32),
            "rope_shift_amount": torch.tensor([0], dtype=torch.int32),
            "k_cache": k_cache,
            "v_cache": torch.zeros_like(k_cache),
        }
        if static_ids is None:
            static_ids = trace_query == 1
        pos_min = max(2, trace_query) if static_ids else 2
        seq_pos = torch.export.Dim("seq_pos", min=pos_min, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        ids_shape = (
            None if static_ids
            else {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)}
        )
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": {
                "input_ids": ids_shape,
                "position_ids": {1: seq_pos},
                "image_embeds": None,
                "deepstack_embeds": None,
                "rope_shift_start": None,
                "rope_shift_amount": None,
                "k_cache": {KVCache.seq_len_dim(): k_seq},
                "v_cache": {KVCache.seq_len_dim(): v_seq},
            },
            "input_names": (
                "input_ids", "position_ids", "image_embeds", "deepstack_embeds",
                "rope_shift_start", "rope_shift_amount",
            ),
            "output_names": ("logits",),
            "state_names": PIPELINED_STATE_NAMES,
        }


class CohereCompassTextOnlyForCausalLM(CohereCompassPipelinedForCausalLM):
    """The same decoder with NO image inputs — a plain 2B Cohere text model.

    It exists for the same reason the LFM2.5-VL port needed one: `llm-runner`
    cannot bind `image_embeds`/`deepstack_embeds`, so neither `llm-benchmark`
    nor `coreai_gate.py` can drive the VLM bundle, and the Mac speed number has
    to come from the same weights in a shape they can run. With no extension
    ids every M-RoPE axis collapses to the text position, which is exactly what
    this forward does.
    """

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.config
        s = input_ids.shape[1]
        seq_len = position_ids.shape[-1]
        torch._check_is_size(s)
        torch._check_is_size(seq_len)
        offset = seq_len - s
        torch._check_is_size(offset)
        p = position_ids.narrow(-1, offset, s)

        x = self.embed_tokens(input_ids.clamp(0, cfg.vocab_size - 1))
        cache = KVCache(k_cache, v_cache)
        for layer in self.layers:
            x = layer(x, position_ids, p, p, p, cache)
        return self.lm_head(self.norm(x)) * cfg.logit_scale

    def build_export_spec(  # type: ignore[override]
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_query: int = 1,
        trace_past: int = 64,
        static_ids: bool | None = None,
    ) -> dict:
        spec = super().build_export_spec(
            target_dtype, max_context_length, trace_kv_len, trace_query, trace_past, static_ids
        )
        for key in ("image_embeds", "deepstack_embeds", "rope_shift_start", "rope_shift_amount"):
            spec["reference_inputs"].pop(key)
            spec["dynamic_shapes"].pop(key)
        spec["input_names"] = ("input_ids", "position_ids")
        return spec


# --------------------------------------------------------------------------- #
# Vision: a shim over the Qwen3-VL tower, not a second implementation
# --------------------------------------------------------------------------- #
class _VisionShim:
    """The attribute surface `Qwen3VLVisionEncoder` reads off a vision config."""

    def __init__(self, raw_vision: dict) -> None:
        self.hidden_size = raw_vision["hidden_size"]
        self.intermediate_size = raw_vision["intermediate_size"]
        self.depth = raw_vision["depth"]
        self.num_heads = raw_vision["num_heads"]
        self.in_channels = raw_vision.get("in_channels", 3)
        self.patch_size = raw_vision["patch_size"]
        self.temporal_patch_size = raw_vision["temporal_patch_size"]
        self.spatial_merge_size = raw_vision["spatial_merge_size"]
        self.out_hidden_size = raw_vision["out_hidden_size"]
        self.num_position_embeddings = raw_vision["num_position_embeddings"]
        self.deepstack_visual_indexes = list(raw_vision["deepstack_visual_indexes"])


def vision_encoder_from_hf(
    hf_id_or_dir: str,
    target_dtype: torch.dtype = torch.float16,
    grid_h: int = 16,
    grid_w: int = 16,
) -> Qwen3VLVisionEncoder:
    """Load `model.visual.*` into the zoo's Qwen3-VL tower.

    Verified to load with zero missing/unexpected keys: this checkpoint's tower
    IS that architecture, at SigLIP2-SO400M's dimensions.
    """
    raw, sd = _load_state_dict(hf_id_or_dir, "model.visual.", torch.float32)
    vcfg = _VisionShim(raw["vision_config"])
    model = Qwen3VLVisionEncoder(vcfg, grid_h=grid_h, grid_w=grid_w).float()

    conv = sd.pop("patch_embed.proj.weight")
    sd["patch_proj.weight"] = conv.reshape(conv.shape[0], -1)
    sd["patch_proj.bias"] = sd.pop("patch_embed.proj.bias")
    pos_table = sd.pop("pos_embed.weight")
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [
        k for k in missing
        if not k.endswith(("pos_embed_const", "cos_const", "sin_const"))
    ]
    if missing or unexpected:
        raise RuntimeError(f"vision load mismatch: missing={missing} unexpected={unexpected}")
    model._init_positional_constants(pos_table)
    return model.to(dtype=target_dtype).eval()


def _load_state_dict(hf_id_or_dir: str, prefix: str, dtype: torch.dtype):
    """(raw config.json, {key_without_prefix: tensor})."""
    import glob

    from safetensors import safe_open

    model_dir = hf_id_or_dir
    if not os.path.isdir(model_dir):
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            hf_id_or_dir,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    sd: dict[str, torch.Tensor] = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(prefix):
                    sd[key.removeprefix(prefix)] = f.get_tensor(key).to(dtype)
    if not sd:
        raise RuntimeError(f"no keys under prefix {prefix!r} in {model_dir}")
    return raw, sd
