"""Export a DECODE-ONLY LFM2.5-MoE bundle whose experts run the ``gather_qmm``
custom Metal kernel — the fix for the GatherMM dense-overread.

This is the metal-MoE sibling of ``export_lfm2_moe_decode_pipelined.py``. Everything
OUTSIDE the experts is treated identically to the shipped ``int8lin`` recipe (int8
per-block-32 linears, fp32 attention q/k/v/out, fp16 router, tied fp16 head) so the
ONLY change vs that 39 tok/s baseline is how the 32-expert top-4 FFN is computed:

  * baseline: ``SwitchGLU`` -> ``GatherMM`` composite, which gathers the routed experts
    then runs a DENSE matmul that READS ALL 32 experts' rows (over-read-bound; LFM2.5
    int8 decode is BW-saturated reading its full 8.8 GB/token).
  * here: ``MetalSwitchGLU`` (models/macos/moe_metal.py) — a custom matvec that takes the
    routed indices as a kernel input and reads ONLY the 4 routed experts' weight slabs
    (``QP[w, n, e]`` with ``e = IDX[slot]``). Decode runs at ACTIVE-param bandwidth.

The kernel owns its weight format (k-means palettized in-file), so the experts are
EXCLUDED from Core AI's quantizer (kept fp16) and palettized by ``metalize_moe`` after.

Modes:
  int8km — 256-entry codebook experts (the quality floor; matches int8 MoE numerics).
  int4km — 16-entry codebook, 8 nibbles/uint32 experts (half the bytes; the size/speed lever).

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_lfm2_moe_metal_decode_pipelined.py \
          int8km --hf-id LiquidAI/LFM2.5-8B-A1B
Bench: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model <out> -p 128 -g 256 -n 3
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.lfm2_moe import (
    DECODE_STATE_NAMES,
    build_decode_state,
    lfm2_moe_from_hf,
)
from coreai_models.models.macos.moe_metal import metalize_moe
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8", block: int = 32) -> dict:
    """int8 per-block linear quant for the NON-expert weights — the int8lin recipe.

    Experts (``SwitchLinear``) are set to None (kept fp16) because the gather_qmm kernel
    owns their layout and palettizes them itself. Attention projections + router + lm_head
    excluded by name; norms/conv/embedding excluded by type."""
    block_2d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {"type": "per_block", "block_size": block, "axis": 1},
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
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
            # experts stay fp16 -> metalize_moe palettizes them into the kernel's buffers
            "coreai_models.primitives.macos.switch.SwitchLinear": None,
        },
        "module_name_configs": {
            r".*lm_head$": None,                                       # tied fp16 head (int8lin)
            r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$": None,    # fp32-kept attention
            r".*feed_forward\.gate$": None,                            # router stays fp16
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="sym8", choices=["sym8", "int8km", "int4km"])
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-8B-A1B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    # sym8 = clean symmetric-linear int8 experts (matches the shipped int8lin quality, fp32-oracle
    # margin gate); km8/km4 = k-means (km4 = the iPhone-jetsam-safe compact bundle).
    scheme = {"sym8": "sym8", "int8km": "km8", "int4km": "km4"}[args.mode]
    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}_gather"

    print(f"loading {args.hf_id} fp16 ...", flush=True)
    model = lfm2_moe_from_hf(args.hf_id, target_dtype=DTYPE)
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_full_layers} full / "
          f"{cfg.num_conv_layers} conv), dense={cfg.num_dense_layers}, "
          f"E={cfg.num_experts}/top{cfg.num_experts_per_tok}, vocab={cfg.vocab_size}", flush=True)

    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,
    }

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) quantize the NON-expert weights (experts excluded -> stay fp16 for metalize)
    from coreai_models.export.compression import quantize_pytorch_model

    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing non-expert linears (int8 per-block-32; experts kept fp16) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, linear_quant_config("int8"),
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) metalize the MoE: SwitchGLU -> MetalSwitchGLU (gather_qmm; reads only routed experts)
    print(f"metalizing MoE experts -> gather_qmm {args.mode} (reads only top-{cfg.num_experts_per_tok}/"
          f"{cfg.num_experts}) ...", flush=True)
    kernel = metalize_moe(model, scheme=scheme)

    # 3) export with the custom kernel; drop gather_mm + gated_delta_update externalize specs
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print(f"exporting decode-only graph (custom kernel embedded) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model,
        reference_inputs=reference_inputs,
        custom_kernels=[kernel],
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
