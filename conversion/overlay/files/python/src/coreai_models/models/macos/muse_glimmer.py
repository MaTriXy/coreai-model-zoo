# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Muse-Glimmer text decoder (meta-models/Muse-Glimmer-30B), macOS GPU path.

Re-authored from `transformers` `MuseGlimmerTextModel` (5.15). Gemma-3 is the
closest shipped port; the differences that matter here are:

  * Layer pattern is `sliding × 3 + full × 1`, but the full layers are **NoPE**.
    `layer_rope_theta[i] == 0` marks them, mirroring the HF model handing those
    layers `position_embeddings=None`. Only the sliding layers get rotary.
  * Weight-less RMSNorm (no learned scale) on the embedding output and on Q/K.
  * A sigmoid output gate per attention layer:
    `attn_out * sigmoid(gate_proj(x))`, where `gate_proj` reads the same
    pre-attention hidden states as Q/K/V — so it is folded into a single fused
    Q/K/V/G projection here (4 GEMMs -> 1).
  * Sandwich norms use two epsilons: `rms_norm_eps` on the pre-norms,
    `post_norm_eps` (1e-8) on the post-norms.
  * Logits are pre-scaled by `output_multiplier` before a Gemma-style tanh
    softcap: `T * tanh(logits * mult / T)`.

`qk_scale_factor` (3.87) is folded into the SDPA scale instead of scaling Q:
attention is `softmax((aQ)·K^T / sqrt(d))` with `a` a scalar, and rotary is a
rotation, so `a / sqrt(d)` as the SDPA scale is algebraically identical and
saves a full-width multiply per layer.
"""

import coreai_torch.composite_ops
import torch
import torch.nn as nn
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm, RMSNormPlusOne
from coreai_models.primitives.macos.rope import RoPE
from coreai_models.primitives.macos.sdpa import SDPA


class WeightlessRMSNorm(nn.Module):
    """RMSNorm with no learned scale (HF `MuseGlimmerRMSNorm(with_scale=False)`).

    The unit scale is kept as an explicit fp32 tensor rather than dropped so the
    op still lowers as the RMSNorm composite (which takes the scale as a graph
    input), and so `RMSNormImpl`'s fp32-scale branch is taken — that branch
    keeps the normalize-then-scale product in fp32 before the down-cast, which
    is what HF's `.float()` / `.type_as()` pair does.

    Held as a plain attribute (not a buffer/parameter) so it stays out of the
    state dict and off the meta device during `_init_model`.
    """

    def __init__(self: Self, dim: int, eps: float) -> None:
        super().__init__()
        self._impl = coreai_torch.composite_ops.RMSNormImpl(eps=eps)
        with torch.device("cpu"):
            self._scale = torch.ones(dim, dtype=torch.float32)

    def forward(self: Self, x: torch.Tensor) -> torch.Tensor:
        return self._impl(x, self._scale)


class NormedEmbedding(nn.Embedding):
    """Token embedding followed by a weight-less RMSNorm.

    HF keeps the norm outside the embedding matrix (it cannot be folded in)
    because the DFlash drafter needs to embed without it.
    """

    def __init__(
        self: Self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None,
        norm_eps: float,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.embed_norm = WeightlessRMSNorm(embedding_dim, eps=norm_eps)

    def forward(self: Self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(super().forward(input_ids))


class Attention(nn.Module):
    def __init__(self: Self, config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", None) or dim // n_heads

        is_local = config.layer_types[layer_idx] == "sliding_attention"
        # Fold `qk_scale_factor` into the SDPA scale (see module docstring).
        self.sdpa = SDPA(
            scale=float(config.qk_scale_factor) * head_dim**-0.5,
            window_size=config.sliding_window if is_local else 0,
            is_causal=True,
        )

        # NoPE layers are marked by a zero entry in `layer_rope_theta`; they run
        # without any rotary at all rather than with theta=0.
        layer_theta = float(config.layer_rope_theta[layer_idx])
        self.rope = RoPE(base=layer_theta, scale=1.0) if layer_theta else None

        # Q / K / V / gate all read the same pre-attention hidden states, so
        # they are one projection. Layout: [Q heads | K heads | V heads | G heads].
        self.qkvg_proj = nn.Linear(dim, 2 * (n_heads + n_kv_heads) * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.qk_norm = WeightlessRMSNorm(head_dim, eps=config.rms_norm_eps)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        qkvg = (
            self.qkvg_proj(x)
            .reshape(batch_size, query_len, 2 * (n_heads + n_kv_heads), self.head_dim)
            .permute(0, 2, 1, 3)
        )

        query_key = qkvg.narrow(1, 0, n_heads + n_kv_heads)
        value = qkvg.narrow(1, n_heads + n_kv_heads, n_kv_heads)
        gate = qkvg.narrow(1, n_heads + 2 * n_kv_heads, n_heads)

        query_key = self.qk_norm(query_key)

        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)

        if self.rope is not None:
            rope_positions = position_ids.narrow(-1, offset, query_len)
            query_key = self.rope(query_key, position_ids=rope_positions)

        query = query_key.narrow(1, 0, n_heads)
        key = query_key.narrow(1, n_heads, n_kv_heads)

        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        attn_dim = n_heads * self.head_dim
        output = (
            self.sdpa(query=query, key=key, value=value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, attn_dim)
        )
        # `gate` carries the same head-major layout as `output`, so the
        # element-wise product matches HF's `attn_out * sigmoid(gate_proj(x))`.
        gate = gate.permute(0, 2, 1, 3).reshape(batch_size, query_len, attn_dim)
        return self.o_proj(output * torch.sigmoid(gate))


class TransformerBlock(nn.Module):
    def __init__(self: Self, config, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size)

        pre_eps = config.rms_norm_eps
        post_eps = getattr(config, "post_norm_eps", pre_eps)
        self.input_layernorm = RMSNormPlusOne(hidden_size, eps=pre_eps)
        self.post_attention_layernorm = RMSNormPlusOne(hidden_size, eps=post_eps)
        self.pre_feedforward_layernorm = RMSNormPlusOne(hidden_size, eps=pre_eps)
        self.post_feedforward_layernorm = RMSNormPlusOne(hidden_size, eps=post_eps)

    def forward(
        self: Self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + self.post_attention_layernorm(r)
        r = self.mlp(self.pre_feedforward_layernorm(h))
        return h + self.post_feedforward_layernorm(r)


class MuseGlimmerModel(nn.Module):
    def __init__(self: Self, config) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = NormedEmbedding(
            config.vocab_size,
            hidden_size,
            getattr(config, "pad_token_id", None),
            norm_eps=config.rms_norm_eps,
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # Final norm keeps a learned scale and is *not* the +1 variant.
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self: Self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class MuseGlimmerForCausalLM(BaseForCausalLM):
    """Engine-compatible Muse-Glimmer text decoder."""

    _HF_MODEL_CLASS = None  # no HF modeling class in the pinned transformers

    # The checkpoint keys the text tower as `model.language_model.layers.N.*`.
    # `from_hf_memory_efficient` streams one layer at a time only for keys that
    # read `model.layers.N.*` *after* the prefix is stripped, and it applies
    # `_mutate_state_dict` (where the Q/K/V/G fusion happens) only to those
    # per-layer slices — anything else lands in one shared dict that is loaded
    # unmutated. Stripping `model.language_` rather than the whole
    # `model.language_model.` is what puts the layers on the streaming path;
    # it also leaves `model.embed_tokens` / `model.norm` on their real module
    # paths and skips the vision tower without loading it.
    _HF_TEXT_PREFIX = "model.language_"

    # `lm_head.weight` is untied (`tie_word_embeddings: false`, 2.7 GB in fp16)
    # and sits at the checkpoint root, outside the text prefix, so the prefix
    # filter drops it. Declaring it here makes the loader fill it explicitly.
    _HF_EXTRA_ROOT_KEYS = ("lm_head.weight",)

    @classmethod
    def from_hf_memory_efficient(  # type: ignore[override]
        cls,
        huggingface_model_id: str,
        max_context_length: int | None = None,
        target_dtype: torch.dtype = torch.float16,
        mmap_path: str | None = None,
        num_layers: int | None = None,
        hf_config_attr: str | None = "text_config",
        hf_state_dict_prefix: str = "",
    ) -> "MuseGlimmerForCausalLM":
        """Force the text prefix regardless of what the caller passes.

        The prefix is a correctness requirement, not a preference: any other
        spelling silently routes the layers off the streaming path, so the Q/K/V/G
        fusion never runs (see `_HF_TEXT_PREFIX`).
        """
        return super().from_hf_memory_efficient(
            huggingface_model_id,
            max_context_length=max_context_length,
            target_dtype=target_dtype,
            mmap_path=mmap_path,
            num_layers=num_layers,
            hf_config_attr=hf_config_attr,
            hf_state_dict_prefix=cls._HF_TEXT_PREFIX,
        )

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        text_config = getattr(hf_config, "text_config", None) or hf_config
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
            text_config.layer_types = list(text_config.layer_types)[:num_layers]
            text_config.layer_rope_theta = list(text_config.layer_rope_theta)[:num_layers]
        # The checkpoint's `tie_word_embeddings` is False (lm_head is a separate
        # top-level tensor); read it off the text config, falling back to the
        # top-level flag for text-only re-uploads of the same family.
        text_config.tie_word_embeddings = getattr(
            text_config, "tie_word_embeddings", getattr(hf_config, "tie_word_embeddings", False)
        )
        return text_config

    @override
    def _init_model(self: Self, config) -> None:
        self.model = MuseGlimmerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self: Self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        logits = self.lm_head(out)
        # `T * tanh(logits * mult / T)` — the pre-scale and the softcap are one
        # step in HF; keep both so the sampled distribution matches.
        softcap = float(self.config.final_logit_softcapping)
        multiplier = float(self.config.output_multiplier)
        return softcap * torch.tanh(logits * (multiplier / softcap))

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Keys arrive in one of two forms depending on the loading path:
        # (a) Raw checkpoint keys:   "model.language_model.layers.0.self_attn.q_proj.weight"
        # (b) Already-stripped keys: "layers.0.self_attn.q_proj.weight"
        #     (when from_hf_memory_efficient strips "model.language_model.")
        # Normalize both to "model.layers.N.*" / "model.embed_tokens.*" / "lm_head.*".
        for key in list(state_dict.keys()):
            if key.startswith(("model.vision_tower.", "model.vision_adapter.", "model.vision_projection.")):
                del state_dict[key]
            elif key.startswith("model.language_model."):
                state_dict["model." + key[len("model.language_model.") :]] = state_dict.pop(key)
            elif key.startswith("layers.") or key.startswith("norm.") or key == "embed_tokens.weight":
                state_dict["model." + key] = state_dict.pop(key)
            # "lm_head.weight" and already-"model.*" keys pass through unchanged.

        max_layer = -1
        for k in state_dict:
            if k.startswith("model.layers."):
                max_layer = max(max_layer, int(k.split(".")[2]))

        for i in range(max_layer + 1):
            # Fuse Q/K/V/gate into one projection. Order must match the narrow
            # layout in `Attention.forward`.
            combined = []
            for proj in ("q_proj", "k_proj", "v_proj", "gate_proj"):
                weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                if weight_key not in state_dict:
                    combined = []
                    break
                combined.append(state_dict.pop(weight_key))
            if combined:
                state_dict[f"model.layers.{i}.self_attn.qkvg_proj.weight"] = torch.concat(
                    combined, axis=0
                )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight
        return result
