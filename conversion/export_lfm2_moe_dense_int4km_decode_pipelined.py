"""Kernel-program lever #2 (dense-path coverage) on LFM2.5-8B-A1B: the gather_qmm MoE-experts bundle
(``export_lfm2_moe_metal_decode_pipelined.py``) PLUS the proven fused int4km matvec wired into the
UN-kernelized dense path that is still fp16 in that recipe — lm_head (tied fp16, the single biggest
per-token read) and the 6 full-attention layers' q_proj / out_proj.

Why: LFM2.5 decode is weight-bandwidth-bound. The experts are already active-param-bound via gather_qmm,
but lm_head (2048x128000 fp16 = 524 MB/token) and attn q/o stay fp16 in the shipped recipe. int4km reads
4-bit weights -> ~4x fewer bytes than fp16 for exactly those matvecs. Per-op de-risk already measured:
int4km lm_head (N=128000-class) beats MPSGraph fp16 matmul 2.77x (ondevice/_dense_int4km_microbench.py).
This bundle is the END-TO-END A/B partner for the baseline int8km bundle.

GATE before any claim: decode tok/s up AND greedy token-match vs the fp16-head baseline on multi-token
reasoning (int4km lm_head changes the final logits — the Nanbeige lesson: single-token survives, reasoning
can crater). k/v stay fp16 (N=512 matvecs never pay — the Mac lesson). DROP-IN: experts unchanged.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_lfm2_moe_dense_int4km_decode_pipelined.py \
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
from coreai_models.models.macos.gemma4_metal_attn_int4km import MetalInt4KMLinear
from coreai_models.models.macos.gemma4_metal_mlp import (
    build_fused_int4km_kernel,
    export_to_coreai_with_kernels,
)
from coreai_models.models.macos.lfm2_moe import (
    DECODE_STATE_NAMES,
    build_decode_state,
    lfm2_moe_from_hf,
)
from coreai_models.models.macos.moe_metal import metalize_moe
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8", block: int = 32) -> dict:
    """int8 per-block linear quant for the NON-expert weights — identical to the baseline recipe.
    lm_head + attn q/k/v/out + router are excluded here (stay fp16); #2 then int4km's lm_head + attn q/o."""
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
            "coreai_models.primitives.macos.switch.SwitchLinear": None,
        },
        "module_name_configs": {
            r".*lm_head$": None,                                       # int4km'd below
            r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$": None,    # q/o int4km'd below; k/v stay fp16
            r".*feed_forward\.gate$": None,                            # router stays fp16
        },
    }


def metalize_dense_int4km(model, kernel) -> int:
    """Swap the still-fp16 dense matvecs that dominate the per-token read for the fused int4km kernel:
    lm_head (biggest single read) + full-attention layers' q_proj / out_proj. k/v_proj stay fp16
    (N=512 — small-N matvecs never pay). Shares ONE kernel object (gemma pattern). Returns #swapped."""
    n = 0
    model.lm_head = MetalInt4KMLinear(model.lm_head, kernel)
    n += 1
    for layer in model.model.layers:
        attn = getattr(layer, "self_attn", None)
        if attn is not None and hasattr(attn, "q_proj"):
            attn.q_proj = MetalInt4KMLinear(attn.q_proj, kernel)
            attn.out_proj = MetalInt4KMLinear(attn.out_proj, kernel)
            n += 2
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8km", choices=["sym8", "int8km", "int4km"])
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-8B-A1B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    scheme = {"sym8": "sym8", "int8km": "km8", "int4km": "km4"}[args.mode]
    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}_gather_dense_int4km"

    print(f"loading {args.hf_id} fp16 ...", flush=True)
    model = lfm2_moe_from_hf(args.hf_id, target_dtype=DTYPE)
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_full_layers} full / "
          f"{cfg.num_conv_layers} conv), E={cfg.num_experts}/top{cfg.num_experts_per_tok}, "
          f"vocab={cfg.vocab_size}", flush=True)

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

    # 1) quantize the NON-expert weights (experts + lm_head + attn q/o + router excluded -> fp16)
    from coreai_models.export.compression import quantize_pytorch_model

    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing non-expert linears (int8 per-block-32) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, linear_quant_config("int8"),
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) metalize the MoE experts -> gather_qmm (reads only the routed top-k)
    print(f"metalizing MoE experts -> gather_qmm {args.mode} ...", flush=True)
    moe_kernel = metalize_moe(model, scheme=scheme)

    # 3) #2: int4km the still-fp16 dense path (lm_head + full-attn q/o); share ONE int4km kernel
    dense_kernel = build_fused_int4km_kernel(name="lfm_dense_int4km")
    n_dense = metalize_dense_int4km(model, dense_kernel)
    print(f"metalized dense int4km: {n_dense} matvecs (lm_head + {cfg.num_full_layers} full layers' q/o)",
          flush=True)

    # 4) export with BOTH custom kernels embedded
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print("exporting decode-only graph (gather_qmm + dense int4km embedded) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model,
        reference_inputs=reference_inputs,
        custom_kernels=[moe_kernel, dense_kernel],
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
