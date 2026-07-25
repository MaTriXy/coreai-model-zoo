"""Export the MiniCPM-V-4.6 TEXT decode core (qwen3_5 hybrid backbone) for the Core AI
pipelined engine. The text backbone IS qwen3_5_text (== qwen3.6 arch, 0.8B/24L), so this
reuses `Qwen3_5ForCausalLMStateful` verbatim; only the weight loading differs (MiniCPM nests
text weights under `model.language_model.*` and we build the config WITHOUT AutoConfig — the
coreai-models venv's transformers has no `minicpmv4_6`).

This is the LLM half of the VLM. The vision encoder (.aimodel) + the host-side vision-feature
splice are separate (export_minicpmv46_vision.py / app wiring). Text-only this bundle is a
complete qwen3_5 decode core and is gated as such (engine greedy == python greedy).

Run (grab _GPU_LOCK for the GPU steps):
    coreai-models/.venv/bin/python conversion/export_minicpmv46_decode_pipelined.py [int8lin|fp16]
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from _paths import hf_snapshot

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5Config,
    Qwen3_5ForCausalLMStateful,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16
TEXT_PREFIX = "model.language_model."


def snapshot_dir(hf_id: str = "openbmb/MiniCPM-V-4.6") -> str:
    return hf_snapshot(hf_id)


def build_config() -> Qwen3_5Config:
    lt = (["linear_attention"] * 3 + ["full_attention"]) * 6  # 24L, every 4th full
    return Qwen3_5Config(
        hidden_size=1024, num_hidden_layers=24, vocab_size=248094, intermediate_size=3584,
        rms_norm_eps=1e-6, tie_word_embeddings=True, head_dim=256,
        num_attention_heads=8, num_key_value_heads=2, attn_output_gate=True,
        partial_rotary_factor=0.25, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        full_attention_interval=4, layer_types=lt,
    )


def load_text_weights(model: Qwen3_5ForCausalLMStateful, snap: str) -> None:
    ckpts = sorted(glob.glob(snap + "/model*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"no safetensors in {snap}")
    sd = {}
    for ckpt in ckpts:
        with safe_open(ckpt, framework="pt", device="cpu") as f:
            for k in f.keys():  # noqa: SIM118
                if k.startswith(TEXT_PREFIX):
                    sd["model." + k[len(TEXT_PREFIX):]] = f.get_tensor(k).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unloaded params: {meta[:6]}")
    print(f"[load] {len(sd)} text tensors (prefix-remapped, tied head)")


def linear_quant_config(dtype: str = "int8") -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": dtype, "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def mixed48_palett_config() -> dict:
    """Mixed 4/8-bit k-means PALETTIZATION (coreai-opt). Bulk weights 4-bit; the SENSITIVE ones 8-bit:
    the big-vocab lm_head + the GDN linear-attn projections (the SSM recurrence is precision-sensitive)
    + the full-attention q/k/v/o. Robust MLP (gate/up/down) stays 4-bit = where the size shrinks most.
    conv1d (short SSM conv) + embeddings stay fp16. NOTE: on the GPU, LUT (palettized) dequant has been
    SLOWER than int8-linear (gemma4: int4km 31/41 vs int8lin 57/72) — this is primarily a SIZE play; the
    decode-speed verdict is empirical (does coreai-opt 0.2.0's LUT path beat int8 here?)."""
    g = {"type": "per_grouped_channel", "axis": 0, "group_size": 32}
    spec4 = {"n_bits": 4, "granularity": g, "enable_per_channel_scale": False}
    spec8 = {"n_bits": 8, "granularity": g, "enable_per_channel_scale": False}
    eight = {"op_state_spec": {"weight": spec8}}
    return {
        "global_config": {"op_state_spec": {"weight": spec4}},   # bulk (MLP) = 4-bit
        "module_name_configs": {
            r".*lm_head$": eight,            # big-vocab head -> 8-bit
            r".*\.linear_attn\..*proj.*$": eight,  # GDN in/out projections -> 8-bit (SSM sensitive)
            r".*self_attn\..*proj$": eight,  # full-attn q/k/v/o -> 8-bit
            r".*embed_tokens$": None,        # exclude (fp16 gather)
            r".*conv1d$": None,              # exclude (short SSM conv)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "mixed48"])
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--hf-id", default="openbmb/MiniCPM-V-4.6",
                    help="HF repo to load weights/tokenizer from (e.g. openbmb/MiniCPM-V-4.6-Thinking)")
    ap.add_argument("--tag", default="",
                    help="bundle-name suffix to avoid overwriting base (e.g. _thinking)")
    args = ap.parse_args()

    snap = snapshot_dir(args.hf_id)
    name = f"minicpmv46{args.tag}_text_decode_{args.mode}"
    cfg = build_config()
    print(f"[cfg] {cfg.num_hidden_layers}L full {cfg.num_full_layers} linear {cfg.num_linear_layers} "
          f"vocab {cfg.vocab_size}")

    model = Qwen3_5ForCausalLMStateful(cfg).eval()
    load_text_weights(model, snap)

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"[loopfree] enabled on {n_lin} linear layers")

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
        "conv_state": state["conv_state"], "rec_state": state["rec_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None, "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq}, "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None, "rec_state": None,
    }

    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("[quant] linear int8 per-block-32 ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, linear_quant_config("int8"))
    elif args.mode == "mixed48":
        from coreai_models.export.compression import palettize_pytorch_model
        # Untie the head so the palettizer quantizes it (it silently skips shared params).
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print("[palett] mixed 4/8 k-means g32 (MLP 4-bit; head+attn+GDN 8-bit; conv/embed fp16) ...")
        model = palettize_pytorch_model(
            model, tuple(reference_inputs.values()), mixed48_palett_config())

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("[export] -> Core AI dialect ...")
    prog = export_to_coreai(
        model, reference_inputs, dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"), output_names=("logits",),
        state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[save] {aimodel}")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    meta = {
        "metadata_version": "0.2", "kind": "llm", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {"tokenizer": args.hf_id, "vocab_size": cfg.vocab_size,
                     "max_context_length": args.max_ctx, "embedded_tokenizer": True,
                     "function_map": {"main": ["main"]}},
        "source": {"model_definition": "torch", "hf_model_id": args.hf_id},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    # tokenizer: copy files directly (transformers 4.57.6 may not know the custom class)
    tdir = out_dir / "tokenizer"
    tdir.mkdir()
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json"):
        src = Path(snap) / fn
        if src.exists():
            shutil.copy(src, tdir / fn)
    print(f"[done] bundle: {out_dir}")
    print(f"  gate: llm-runner --model {out_dir} --inference-engine-variant coreai-sequential ...")


if __name__ == "__main__":
    main()
