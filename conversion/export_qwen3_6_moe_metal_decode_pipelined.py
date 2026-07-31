"""Export a DECODE-ONLY Qwen3.6-35B-A3B bundle whose 256-expert top-8 MoE runs the
``gather_qmm`` custom Metal kernel — reading ONLY the 8 routed experts (a 32× over-read removed).

The metal-MoE sibling of ``export_qwen3_6_decode_pipelined.py``. Everything outside the routed
experts is identical to the shipped int8 recipe (int8 per-block-32 linears incl. the shared
expert, fp16 router/shared-gate, absmax int8 head). The ONLY change is the routed-expert FFN:

  * shipped: ``SwitchGLU``/``GatherMM`` reads ALL 256 experts' weights per token (over-read-bound;
    30.9 tok/s reading the full ~35 GB/token).
  * here: ``MetalSwitchGLU`` (models/macos/moe_metal.py) with the ``sym8`` scheme — a custom matvec
    that takes the routed indices as a kernel input and reads only the 8 routed experts (8/256),
    quantized with the CLEAN symmetric-linear int8 (per-K-block-32, the shipped int8lin recipe;
    fp32-oracle margin-gate-clean on LFM2.5-8B-A1B). Decode runs at active-param (~3B) bandwidth.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_qwen3_6_moe_metal_decode_pipelined.py \
          sym8 --hf-id Qwen/Qwen3.6-35B-A3B
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.moe_metal import metalize_moe
from coreai_models.models.macos.qwen3_5_moe import (
    DECODE_STATE_NAMES,
    Qwen3_5MoeStatefulForCausalLM,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """int8 per-block-32 linear for the NON-routed-expert weights (incl. the shared expert).
    Routed experts (SwitchLinear) -> None (kept fp16) because the gather_qmm kernel owns them."""
    block_2d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
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
            # routed experts kept fp16 -> metalize_moe sym8-quantizes them into the kernel buffers
            "coreai_models.primitives.macos.switch.SwitchLinear": None,
        },
        "module_name_configs": {
            r".*mlp\.gate$": None,
            r".*shared_expert_gate$": None,
            r".*lm_head$": None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="sym8", choices=["sym8"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument("--head-quant", default="block32")
    ap.add_argument("--head-sym", action="store_true")
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}_gather"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 ...", flush=True)
    model = Qwen3_5MoeStatefulForCausalLM.from_hf_memory_efficient(
        args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE, num_layers=args.num_layers)
    model.eval()
    cfg = model.config

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"loop-free single-step enabled on {n_lin} linear layers", flush=True)
    print(f"E={cfg.num_experts}/top{cfg.num_experts_per_tok}, moe_inter={cfg.moe_intermediate_size}, "
          f"hidden={cfg.hidden_size}, vocab={cfg.vocab_size}", flush=True)

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

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) int8 the NON-routed-expert weights (routed experts excluded -> stay fp16 for metalize)
    from coreai_models.export.compression import quantize_pytorch_model
    cfg_q = linear_quant_config("int8")
    cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(args.head_quant, args.head_sym)
    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing non-routed-expert linears (int8 per-block-32; routed experts kept fp16) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) metalize the routed MoE: SwitchGLU -> MetalSwitchGLU (gather_qmm sym8; reads only 8/256)
    print(f"metalizing routed MoE -> gather_qmm sym8 (reads only top-{cfg.num_experts_per_tok}/"
          f"{cfg.num_experts}) ...", flush=True)
    kernel = metalize_moe(model, scheme="sym8")

    # 3) export with the kernel; drop gather_mm (gone) + gated_delta_update (inlined, as shipped)
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print("exporting decode-only graph (custom kernel embedded) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs=reference_inputs, custom_kernels=[kernel],
        dynamic_shapes=dynamic_shapes, input_names=("input_ids", "position_ids"),
        output_names=("logits",), state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx)
    save_tokenizer(args.hf_id, out_dir, via_transformers=False)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
