"""Export the MiniCPM-V-4.6 head-split DECODE CORE (inputs_embeds -> hidden) for the VLM path.

The VLM deployment split: the giant tied embed/lm_head table (248094×1024 ≈ ⅓ of the model)
stays on the CPU/ANE FRONT-END (embed gather + vision-feature splice + final lm_head matmul);
the GPU decode core holds ONLY the qwen3_5 hybrid transformer and takes `inputs_embeds`. This
is exactly `Qwen3_5DecodeCore` (already in the overlay). The vision features are spliced into
inputs_embeds on the host, so NO image-token / static-buffer wiring is needed in this graph.

Run (GPU; _GPU_LOCK held):
    coreai-models/.venv/bin/python ../coreai-models-community/conversion/export_minicpmv46_core_decode_pipelined.py [int8lin|fp16]
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
from torch import nn
from _paths import exports_dir, hf_snapshot

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import Qwen3_5Config, Qwen3_5DecodeCore

DTYPE = torch.float16
TEXT_PREFIX = "model.language_model."
SNAP = hf_snapshot("openbmb/MiniCPM-V-4.6")


def build_config() -> Qwen3_5Config:
    lt = (["linear_attention"] * 3 + ["full_attention"]) * 6
    return Qwen3_5Config(
        hidden_size=1024, num_hidden_layers=24, vocab_size=248094, intermediate_size=3584,
        rms_norm_eps=1e-6, tie_word_embeddings=True, head_dim=256,
        num_attention_heads=8, num_key_value_heads=2, attn_output_gate=True,
        partial_rotary_factor=0.25, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        full_attention_interval=4, layer_types=lt)


def linear_quant_config() -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": "int8", "qscheme": "symmetric_with_clipping",
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
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "int8lin"])
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()
    name = f"minicpmv46_core_decode_{args.mode}"
    cfg = build_config()

    core = Qwen3_5DecodeCore(cfg).eval()
    # Load ONLY the transformer (layers + norm); the embed table lives on the front-end.
    sd = {}
    with safe_open(glob.glob(SNAP + "/model.safetensors")[0], framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if not k.startswith(TEXT_PREFIX):
                continue
            local = "model." + k[len(TEXT_PREFIX):]
            if local.startswith("model.embed_tokens"):
                continue  # front-end holds it
            sd[local] = f.get_tensor(k).to(DTYPE)
    # Replace the giant unused embed with a dummy so it is NOT baked into the core graph.
    core.model.embed_tokens = nn.Embedding(2, cfg.hidden_size).to(DTYPE)
    core.load_state_dict(sd, strict=False, assign=True)
    core.model.reset_buffers()
    meta = [n for n, p in core.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unloaded: {meta[:6]}")
    print(f"[load] {len(sd)} transformer tensors (embed on front-end, dummy in core)")

    n_lin = 0
    for layer in core.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"[loopfree] {n_lin} linear layers")

    spec = core.build_macos_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        query_len=1, offset=64, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
    # Decode-only: static [1,1,hidden] query (S=1), like the LLM core. position_ids stays dynamic.
    spec["dynamic_shapes"]["inputs_embeds"] = None
    ref = spec["reference_inputs"]

    if args.mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("[quant] int8 per-block-32 ...")
        core = quantize_pytorch_model(core, tuple(ref.values()), spec["dynamic_shapes"],
                                      linear_quant_config())

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("[export] head-split core -> Core AI dialect ...")
    prog = export_to_coreai(
        core, ref, dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"], output_names=spec["output_names"],
        state_names=spec["state_names"], externalize_modules=specs)
    prog.optimize()

    out_dir = exports_dir() / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[save] {aimodel}")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    (out_dir / "metadata.json").write_text(json.dumps({
        "metadata_version": "0.2", "kind": "llm-decode-core", "name": name,
        "io": {"input": "inputs_embeds[1,S,1024]+position_ids", "output": "hidden[1,S,1024]",
               "states": list(spec["state_names"])},
        "note": "head-split: embed gather + vision splice + tied lm_head are on the front-end",
        "source": {"hf_model_id": "openbmb/MiniCPM-V-4.6"},
        "compilation": {"date": datetime.now(timezone.utc).isoformat()},
    }, indent=2))
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
