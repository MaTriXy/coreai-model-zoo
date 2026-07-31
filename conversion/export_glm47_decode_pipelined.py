"""Export a DECODE-ONLY (S=1, loop-free) GLM-4.7-Flash (glm4_moe_lite) bundle
for the Core AI pipelined engine.

GLM-4.7-Flash = MLA attention (DeepSeek-V3-style, authored in the naive
materialized form) on all 47 layers + a 64-expert top-4 sparse-MoE FFN with one
non-gated shared expert (layer 0 is a dense MLP, first_k_dense_replace=1). The
attention rides standard KV-cache states (NO conv / SSM state — simpler than the
qwen3.6 hybrid); the MoE block lowers through the SwitchGLU/GatherMM composite
with in-graph sigmoid routing + selection-only bias + topk.

~3B active params -> per-token int8 weight read ~3 GB: Mac-class decode from a
30B-quality local-coding model (the 64 GB Studio is the target; KV cache is
materialized full-MHA, ~8 GB at 8192 ctx, so this is firmly Mac-only).

Quantization (int8lin / int8hu):
  * global: linear int8 per-block-32 `symmetric_with_clipping` (the ship recipe)
    on every Linear except norms/embedding — includes the MLA down/up
    projections (q_a/q_b/kv_a/kv_b/o_proj);
  * MoE expert weights are 4D [1, E, out, in] SwitchLinear params -> per-module
    override with multi-dim blocks [1, 1, 1, 32] (Apple's MoE recipe shape);
  * the router (`mlp.gate`) stays fp16 — quantizing it can flip discrete expert
    selection for ~0.1% of total bytes. (GLM's shared expert is NOT gated, so
    there is no shared_expert_gate to skip.)
  * lm_head is untied (vocab 154880): int8hu quantizes it with plain `symmetric`
    (absmax) per-block-32 — the big-vocab-head rule (clipping crushes outlier
    rows); int8lin leaves it fp16.

RAM/disk: the fp16 model is ~60 GB in RAM; the quantizer runs with `mmap_dir` so
the finalized int8 tensors are disk-backed (~30 GB temp, removed after
save_asset). Keep >= 70 GB disk free before running.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_glm47_decode_pipelined.py \
          int8hu --head-sym
Modes: fp16 (control; ~60 GB bundle — debug/truncated only), int8lin,
       int8hu (ship: + int8 absmax lm_head).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.glm4_moe_lite import (
    DECODE_STATE_NAMES,
    Glm4MoeLiteStatefulForCausalLM,
    build_decode_state,
    glm4_moe_lite_from_hf,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Ship recipe: per-block-32 linear int8 + the 4D SwitchLinear override."""
    block_2d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
    }
    block_4d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {
            "type": "per_block",
            "block_size": [1, 1, 1, 32],
            "axis": None,
        },
    }
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": block_2d},
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
            "coreai_models.primitives.macos.switch.SwitchLinear": {
                "module_state_spec": {"weight": block_4d},
                "op_input_spec": None,
                "op_output_spec": None,
            },
        },
        # router stays fp16 (discrete-selection risk, ~0.1% of bytes);
        # lm_head handled per-mode below.
        "module_name_configs": {
            r".*mlp\.gate$": None,
            r".*lm_head$": None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8hu",
                    choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--hf-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None,
                    help="debug: truncated-layer export (early MLA bisection)")
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"])
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu: plain symmetric (absmax) head — SHIP shape")
    ap.add_argument("--no-quant-mmap", action="store_true",
                    help="keep quantized tensors in RAM (debug/truncated only)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode == "int8hu" and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 ...", flush=True)
    model = glm4_moe_lite_from_hf(args.hf_id, target_dtype=DTYPE)
    if args.num_layers is not None:
        # truncated bisection build: keep the first N layers, drop the rest
        model.model.layers = model.model.layers[: args.num_layers]
        model.config.num_hidden_layers = args.num_layers
    model.eval()
    cfg = model.config
    print(f"model ready | {cfg.num_hidden_layers} layers, "
          f"heads={cfg.num_attention_heads}, qk_head_dim={cfg.qk_head_dim}, "
          f"E={cfg.n_routed_experts}/top{cfg.num_experts_per_tok}", flush=True)

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # static [1, 1]
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
    }

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    quant_mmap = None

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(
                args.head_quant, args.head_sym)
        if not args.no_quant_mmap:
            quant_mmap = out_dir / "_quant_mmap"
            quant_mmap.mkdir()
        print(f"quantizing (linear int8 per-block-32, mode={args.mode}, "
              f"mmap={quant_mmap}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
            mmap_dir=str(quant_mmap) if quant_mmap else None)

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("exporting decode-only graph to Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=DECODE_STATE_NAMES,
        externalize_modules=specs,
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx)
    save_tokenizer(args.hf_id, out_dir)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
