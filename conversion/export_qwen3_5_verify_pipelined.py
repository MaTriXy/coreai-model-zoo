"""Export a STATIC-S VERIFY bundle (S=K+1 tokens per forward) for Qwen3.5/3.6 hybrids.

Spec-decode (Stream C) needs a verify-forward: feed the last committed token plus K
draft tokens, get per-position logits [1, S, vocab] back in ONE forward, so K tokens
are verified per weight read. On the qwen3_5 GDN hybrid the dynamic-query graph is
blocked twice over (the GatedDeltaUpdate while_loop doesn't lower on the GPU
delegate; JIT dynamic-query dies ANECompile->MTL4 on OS27), so this export keeps
`input_ids` STATIC at [1, S] — the same fixed-query class as the shipped decode-only
bundles — and routes every linear-attention layer through the loop-free CHUNKED scan
(`use_loopfree_chunk`, `_gated_delta_chunk`). The known fp16 chunked-inverse NaN is a
chunk>=64 problem; verify chunks are S<=17. Numerics gated in
`_smoke/test_qwen35_verify_chunk_parity.py` (chunk==step at S in {2,9,17}; 0.8B
model-level argmax 9/9 + end-state parity).

The graph doubles as a chunked PREFILL graph (feed S prompt tokens per call), and the
spec-decode host loop keeps states honest via snapshot + exact-S re-anchor (states
advance S tokens per call; only commit when every fed token is accepted/committed).

position_ids and the KV seq dim stay dynamic exactly like the decode export, so the
same growing-KV runtime contract holds. Quantization recipes mirror the decode
script so a verify bundle is weight-identical to its shipped decode sibling.

Run:  python export_qwen3_5_verify_pipelined.py [fp16|int8lin|int8hu|int4lin] \
          [--hf-id Qwen/Qwen3.5-0.8B] [--s 9] [--out-dir exports]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5StatefulForCausalLM,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear per-block-32 — identical to the decode export (ship shape)."""
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
                }
            },
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
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int8hu", "int4lin"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--s", type=int, default=9,
                    help="static verify query length S=K+1 (keep <=17: fp16 chunk-inverse "
                         "NaN is a chunk>=64 problem, gated at S<=17)")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"])
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu only: plain symmetric (absmax) head — 27B ship shape")
    ap.add_argument("--metal", action="store_true",
                    help="swap the in-graph fp16 chunk-inverse GDN scan for the fp32 sequential "
                         "Metal kernel (qwen3_5_gdn_metal) — decode-exact recurrence, no matrix "
                         "inverse; closes the verify-vs-decode per-forward gap (C1 profile)")
    ap.add_argument("--num-layers", type=int, default=None,
                    help="debug: truncated-layer export")
    ap.add_argument("--doublings", type=int, default=None,
                    help="chunk scan only: doubling rounds for the triangular inverse "
                         "(exact iff 2**d >= S; default keeps the model's 12)")
    ap.add_argument("--emit-hidden", default=None, choices=["pre", "post"],
                    help="add a second `hidden` output [1,S,hidden] (pre- or "
                         "post-final-norm) for the MTP drafter; logits unchanged")
    ap.add_argument("--scan", default="chunk", choices=["chunk", "unroll", "while"],
                    help="GDN scan formulation: chunk = loop-free doubling-inverse (ship), "
                         "unroll = static-S step recurrence unrolled in-graph, "
                         "while = native GatedDeltaUpdate while_loop composite "
                         "(Apple 177354777 retest)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_verify_s{args.s}_{args.mode}"
    if args.mode == "int8hu" and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.metal:
        name += "_metal"
    if args.scan != "chunk":
        name += f"_{args.scan}"
    if args.doublings is not None:
        name += f"_d{args.doublings}"
    if args.emit_hidden is not None:
        name += f"_h{args.emit_hidden}"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 ...")
    model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
        args.hf_id, max_context_length=args.max_ctx, target_dtype=DTYPE,
        hf_config_attr="text_config", num_layers=args.num_layers)
    model.eval()
    model.emit_hidden = args.emit_hidden
    cfg = model.config
    output_names = ("logits",) if args.emit_hidden is None else ("logits", "hidden")

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            if args.scan == "chunk":
                layer.linear_attn.use_loopfree_chunk = True
                if args.doublings is not None:
                    layer.linear_attn.chunk_doublings = args.doublings
            elif args.scan == "unroll":
                layer.linear_attn.use_loopfree_unroll = True
            # "while": leave the GatedDeltaUpdate composite (native while_loop) in place
            n_lin += 1
    print(f"GDN scan={args.scan} on {n_lin} linear layers (S={args.s})")

    # Verify trace: S static query, dynamic full-length positions, dynamic KV seq.
    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, args.s), dtype=torch.int32)
    position_ids = torch.arange(trace_past + args.s, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
        "rec_state": state["rec_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=args.s, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,  # STATIC [1, S] — fixed-query class, no while_loop
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,
        "rec_state": None,
    }

    if args.mode in ("int8lin", "int8hu", "int4lin"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int4" if args.mode == "int4lin" else "int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {
                r".*lm_head$": head_quant_spec(args.head_quant, args.head_sym)}
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone())
        print(f"quantizing (linear per-block-32, mode={args.mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    # The chunk path never calls the GatedDeltaUpdate composite. SDPA must NOT be
    # externalized here: its lower-right causal mask emits a `k_len >= S` guard that
    # the externalize pipeline's auto-Dim (min=2) violates at static S>1. Decomposed
    # in-graph SDPA keeps the identical mask (plain arange/compare ops) — at S<=17
    # the unfused attention cost is noise next to the per-forward weight read.
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update",
                                            "scaled_dot_product_attention")]

    # --metal: replace the in-graph fp16 chunk-inverse recurrence with the fp32 sequential
    # Metal kernel (set AFTER quantization — it only touches the cheap GDN recurrence; the
    # weight-heavy in_proj/out_proj stay quantized in-graph). Registered with the converter.
    custom_kernels: list = []
    if args.metal:
        from coreai_models.models.macos.qwen3_5_gdn_metal import metalize_gdn_chunk
        custom_kernels = [metalize_gdn_chunk(model)]
        print(f"[metal] GDN fp32 sequential-scan kernel on {n_lin} linear layers (S={args.s})")

    print(f"exporting static S={args.s} verify graph to Core AI dialect ...")
    if args.metal:
        from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
        prog = export_to_coreai_with_kernels(
            model,
            reference_inputs,
            custom_kernels=custom_kernels,
            dynamic_shapes=dynamic_shapes,
            input_names=("input_ids", "position_ids"),
            output_names=output_names,
            state_names=DECODE_STATE_NAMES,
            externalize_modules=specs,
        )
    else:
        prog = export_to_coreai(
            model,
            reference_inputs,
            dynamic_shapes=dynamic_shapes,
            input_names=("input_ids", "position_ids"),
            output_names=output_names,
            state_names=DECODE_STATE_NAMES,
            externalize_modules=specs,
        )
    print("optimizing ...")
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    extra = {"verify_query_len": args.s}
    if args.emit_hidden is not None:
        extra["emit_hidden"] = args.emit_hidden
    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
                          extra=extra)
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
