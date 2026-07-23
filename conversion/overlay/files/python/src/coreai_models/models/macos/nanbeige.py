"""Nanbeige4.2 recurrent Llama decoder for the Core AI authoring path."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.llama.configuration_llama import LlamaConfig
from typing_extensions import override

from coreai_models.models.macos.llama import LlamaForCausalLM, LlamaModel
from coreai_models.primitives.macos.cache import KVCache


class NanbeigeConfig(LlamaConfig):
    model_type = "nanbeige"

    def __init__(
        self,
        num_loops: int = 1,
        loop_loss_weights: list[float] | None = None,
        skip_loop_final_norm: bool = False,
        **kwargs,
    ) -> None:
        self.num_loops = num_loops
        self.loop_loss_weights = [] if loop_loss_weights is None else loop_loss_weights
        self.skip_loop_final_norm = skip_loop_final_norm
        super().__init__(**kwargs)


try:
    AutoConfig.register("nanbeige", NanbeigeConfig)
except ValueError:
    pass


def validate_nanbeige_config(config) -> None:
    """Accept only the released Nanbeige4.2-3B recurrent-Llama architecture."""
    if config.model_type != "nanbeige":
        raise ValueError(f"model_type must be 'nanbeige', got {config.model_type!r}")
    if config.num_hidden_layers != 22:
        raise ValueError(f"num_hidden_layers must be 22, got {config.num_hidden_layers}")
    if getattr(config, "num_loops", 1) != 2:
        raise ValueError(f"num_loops must be 2, got {getattr(config, 'num_loops', 1)}")
    if getattr(config, "skip_loop_final_norm", False):
        raise ValueError("skip_loop_final_norm must be false")
    if getattr(config, "loop_loss_weights", []):
        raise ValueError("loop_loss_weights must be empty")

    layer_types = getattr(config, "layer_types", None)
    if layer_types and any(layer_type != "full_attention" for layer_type in layer_types):
        raise ValueError("layer_types must contain only full_attention")
    if getattr(config, "sliding_window", None) is not None:
        raise ValueError("sliding_window is not supported; all layers must use full attention")

    unsupported = {
        "attention_bias": getattr(config, "attention_bias", False),
        "mlp_bias": getattr(config, "mlp_bias", False),
        "qk_layernorm": getattr(config, "qk_layernorm", False),
        "emb_neighbor_num": getattr(config, "emb_neighbor_num", None) is not None,
        "emb_split_num": getattr(config, "emb_split_num", None) is not None,
        "ngram_vocab_size_ratio": getattr(config, "ngram_vocab_size_ratio", None) is not None,
        "ngram_mod_force_prime": getattr(config, "ngram_mod_force_prime", False),
        "ngram_embedding_hidden_size": getattr(config, "ngram_embedding_hidden_size", None)
        is not None,
        "ngram_fused_mode": getattr(config, "ngram_fused_mode", "average") != "average",
        "emb_tp_num": getattr(config, "emb_tp_num", None) is not None,
        "ngram_compressed_tokenizer": getattr(config, "ngram_compressed_tokenizer", False),
        "skip_ngram_for_input": getattr(config, "skip_ngram_for_input", False),
        "insert_ngram_layer_idx": bool(getattr(config, "insert_ngram_layer_idx", [])),
        "ngram_insert_all_layers": getattr(config, "ngram_insert_all_layers", False),
        "ngram_layer_downproject_size": getattr(config, "ngram_layer_downproject_size", None)
        is not None,
        "enable_double_loop_split": getattr(config, "enable_double_loop_split", False),
        "loop_middle_layers": getattr(config, "loop_middle_layers", None) is not None,
        "loop_share_kv": getattr(config, "loop_share_kv", False),
        "mhc_diff_for_loop": getattr(config, "mhc_diff_for_loop", False),
        "mhc_double_stream_position_for_loop": getattr(
            config, "mhc_double_stream_position_for_loop", None
        )
        is not None,
        "enable_hyper_connection": getattr(config, "enable_hyper_connection", False),
        "enable_mhc": getattr(config, "enable_mhc", False),
        "enable_h_res_identity": getattr(config, "enable_h_res_identity", False),
        "mhc_identity_nohresparam": getattr(config, "mhc_identity_nohresparam", False),
        "num_residual_streams": getattr(config, "num_residual_streams", 4) != 4,
        "mhc_sinkhorn_iterations": getattr(config, "mhc_sinkhorn_iterations", 20) != 20,
        "mhc_init_gating_factor": getattr(config, "mhc_init_gating_factor", 0.01) != 0.01,
        "enable_depth_attention": getattr(config, "enable_depth_attention", False),
        "depth_attention_stride": getattr(config, "depth_attention_stride", None) is not None,
        "depth_attention_recent_window": bool(
            getattr(config, "depth_attention_recent_window", 0)
        ),
        "depth_attention_static_anchor_once": not getattr(
            config, "depth_attention_static_anchor_once", True
        ),
    }
    enabled = [name for name, value in unsupported.items() if value]
    if enabled:
        raise ValueError("unsupported Nanbeige options: " + ", ".join(enabled))

    if getattr(config, "pretraining_tp", 1) != 1:
        raise ValueError("pretraining_tp must be 1")
    if getattr(config, "hidden_act", "silu") != "silu":
        raise ValueError("hidden_act must be 'silu'")
    if getattr(config, "attention_dropout", 0.0) != 0.0:
        raise ValueError("attention_dropout must be 0")


class OffsetKVCache:
    """Route one physical pass to its disjoint logical cache-layer range."""

    def __init__(self, cache: KVCache, layer_offset: int) -> None:
        self.cache = cache
        self.layer_offset = layer_offset

    def update_and_fetch(self, layer_idx: int, *args, **kwargs):
        return self.cache.update_and_fetch(layer_idx + self.layer_offset, *args, **kwargs)


def create_cache_tensors(config, dtype: torch.dtype = torch.float32):
    logical_config = copy.copy(config)
    logical_config.num_hidden_layers *= config.num_loops
    return KVCache.create_cache_tensors(logical_config, dtype=dtype)


class NanbeigeModel(LlamaModel):
    def __init__(self, config: NanbeigeConfig) -> None:
        if not getattr(config, "_nanbeige_released_config_validated", False):
            validate_nanbeige_config(config)
        super().__init__(config)
        self.num_loops = config.num_loops

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        num_layers = len(self.layers)
        if cache is not None:
            expected = self.num_loops * num_layers
            actual = cache._k_cache.size(0)
            if actual != expected or cache._v_cache.size(0) != expected:
                raise ValueError(f"Nanbeige requires {expected} cache layers, got {actual}")

        for loop_idx in range(self.num_loops):
            loop_cache = OffsetKVCache(cache, loop_idx * num_layers) if cache is not None else None
            for layer in self.layers:
                h = layer(h, position_ids, loop_cache)
            h = self.norm(h)
        return h


class NanbeigeForCausalLM(LlamaForCausalLM):
    _HF_MODEL_CLASS = None

    @classmethod
    def _get_reauthored_config(cls, hf_config, max_context_length=None, num_layers=None):
        validate_nanbeige_config(hf_config)
        if num_layers is not None and not 1 <= num_layers <= hf_config.num_hidden_layers:
            raise ValueError(
                f"num_layers must be between 1 and {hf_config.num_hidden_layers}, got {num_layers}"
            )
        hf_config._nanbeige_released_config_validated = True
        return super()._get_reauthored_config(hf_config, max_context_length, num_layers)

    @override
    def _init_model(self, config: NanbeigeConfig) -> None:
        self.model = NanbeigeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
