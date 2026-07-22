#!/usr/bin/env python3
"""Synthetic checks for the Core AI Nanbeige4.2 recurrent-Llama overlay."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

REPOSITORY = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY))
MODELS = REPOSITORY / "conversion/overlay/files/python/src/coreai_models/models/macos"


def load_overlay_module(name: str):
    spec = importlib.util.spec_from_file_location(name, MODELS / f"{name.rsplit('.', 1)[-1]}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_overlay_module("coreai_models.models.macos.llama")
nanbeige = load_overlay_module("coreai_models.models.macos.nanbeige")


def config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 22,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "max_position_embeddings": 8,
        "rms_norm_eps": 1e-5,
        "num_loops": 2,
        "loop_loss_weights": [],
        "skip_loop_final_norm": False,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return nanbeige.NanbeigeConfig(**values)


class Nanbeige42Test(unittest.TestCase):
    def test_validation(self):
        nanbeige.validate_nanbeige_config(config())
        invalid = (
            ("num_hidden_layers", 21),
            ("num_loops", 1),
            ("skip_loop_final_norm", True),
            ("loop_loss_weights", [1.0]),
            ("attention_bias", True),
            ("mlp_bias", True),
            ("qk_layernorm", True),
            ("layer_types", ["full_attention"] * 21 + ["sliding_attention"]),
            ("sliding_window", 4),
            ("emb_neighbor_num", 2),
            ("emb_split_num", 2),
            ("ngram_vocab_size_ratio", 0.25),
            ("ngram_mod_force_prime", True),
            ("ngram_embedding_hidden_size", 16),
            ("ngram_fused_mode", "concat"),
            ("emb_tp_num", 2),
            ("ngram_compressed_tokenizer", True),
            ("skip_ngram_for_input", True),
            ("insert_ngram_layer_idx", [0]),
            ("ngram_insert_all_layers", True),
            ("ngram_layer_downproject_size", 16),
            ("enable_double_loop_split", True),
            ("loop_middle_layers", 11),
            ("loop_share_kv", True),
            ("mhc_diff_for_loop", True),
            ("mhc_double_stream_position_for_loop", 1),
            ("enable_hyper_connection", True),
            ("enable_mhc", True),
            ("enable_h_res_identity", True),
            ("mhc_identity_nohresparam", True),
            ("num_residual_streams", 3),
            ("mhc_sinkhorn_iterations", 19),
            ("mhc_init_gating_factor", 0.02),
            ("enable_depth_attention", True),
            ("depth_attention_stride", 11),
            ("depth_attention_recent_window", 1),
            ("depth_attention_static_anchor_once", False),
            ("pretraining_tp", 2),
            ("hidden_act", "relu"),
            ("attention_dropout", 0.1),
        )
        for option, value in invalid:
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, option):
                nanbeige.validate_nanbeige_config(config(**{option: value}))

    def test_shared_layers_cache_count_and_cached_parity(self):
        torch.manual_seed(7)
        model = nanbeige.NanbeigeForCausalLM(config()).float().eval()
        self.assertEqual(len(model.model.layers), 22)
        self.assertEqual(len({id(layer) for layer in model.model.layers}), 22)
        self.assertEqual(sum(isinstance(module, nn.Linear) for module in model.modules()), 111)

        k_cache, v_cache = nanbeige.create_cache_tensors(model.config)
        self.assertEqual(k_cache.shape, (44, 1, 1, 8, 8))
        self.assertEqual(v_cache.shape, k_cache.shape)

        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        with torch.no_grad():
            full = model(
                input_ids,
                torch.arange(3, dtype=torch.int32).unsqueeze(0),
                torch.zeros_like(k_cache),
                torch.zeros_like(v_cache),
            )
            cached = torch.cat(
                [
                    model(
                        input_ids[:, index : index + 1],
                        torch.arange(index + 1, dtype=torch.int32).unsqueeze(0),
                        k_cache,
                        v_cache,
                    )
                    for index in range(input_ids.shape[1])
                ],
                dim=1,
            )
        torch.testing.assert_close(full, cached, rtol=1e-4, atol=1e-4)

    def test_pass_order_and_truncated_export(self):
        events = []

        class Layer(nn.Module):
            def __init__(self, index):
                super().__init__()
                self.index = index

            def forward(self, hidden, position_ids, cache):
                events.append(("layer", self.index, cache.layer_offset if cache else None))
                return hidden

        class Norm(nn.Module):
            def forward(self, hidden):
                events.append(("norm",))
                return hidden

        full = nanbeige.NanbeigeForCausalLM(config())
        full.model.layers = nn.ModuleList([Layer(index) for index in range(22)])
        full.model.norm = Norm()
        full.model(
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
        )
        one_pass = [("layer", index, None) for index in range(22)] + [("norm",)]
        self.assertEqual(events, one_pass * 2)
        events.clear()

        source = config()
        truncated = nanbeige.NanbeigeForCausalLM._get_reauthored_config(
            source, max_context_length=8, num_layers=1
        )
        model = nanbeige.NanbeigeForCausalLM(truncated)
        model.model.layers = nn.ModuleList([Layer(0)])
        model.model.norm = Norm()
        k_cache, v_cache = nanbeige.create_cache_tensors(model.config)
        model(
            torch.tensor([[1]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.int32),
            k_cache,
            v_cache,
        )
        self.assertEqual(k_cache.shape[0], 2)
        self.assertEqual(events, [("layer", 0, 0), ("norm",), ("layer", 0, 1), ("norm",)])

    def test_int8hu_quantizes_only_physical_modules(self):
        from coreai_models.export.compression import quantize_pytorch_model

        from conversion.export_nanbeige41_decode_pipelined import (
            head_quant_spec,
            linear_quant_config,
        )

        cfg = config(hidden_size=32, intermediate_size=64, num_attention_heads=4)
        model = nanbeige.NanbeigeForCausalLM(cfg).eval()
        k_cache, v_cache = nanbeige.create_cache_tensors(cfg)
        inputs = (
            torch.ones((1, 1), dtype=torch.int32),
            torch.ones((1, 1), dtype=torch.int32),
            k_cache,
            v_cache,
        )
        quant_config = linear_quant_config()
        quant_config["module_name_configs"][r".*lm_head$"] = head_quant_spec("block32", True)
        model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        quantized = quantize_pytorch_model(
            model,
            inputs,
            {name: None for name in ("input_ids", "position_ids", "k_cache", "v_cache")},
            quant_config,
        )
        self.assertEqual(
            sum(type(module).__name__ == "ParametrizedLinear" for module in quantized.modules()),
            111,
        )


if __name__ == "__main__":
    unittest.main()
