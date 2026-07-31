"""Export a DECODE-ONLY (S=1, loop-free) LFM2.5-MoE bundle for the Core AI
pipelined engine.

LFM2.5-8B-A1B (HF model_type `lfm2_moe`) = the LFM2.5 conv + full-attention
hybrid (18 short-conv + 6 GQA attention layers; reused verbatim from the lfm2
overlay — NO recurrent scan, so the decode graph is loop-free) with a 32-expert
top-4 sparse-MoE FFN on every layer past the first two (which stay dense). The
MoE block routes with SIGMOID + a selection-only expert bias and has NO shared
expert (the LFM2.5-MoE recipe, distinct from Qwen3.5-MoE); experts lower through
the SwitchGLU/GatherMM composite.

1.5B active params -> int4 expert read ~0.9 GB/token: iPhone-class decode from
an 8.3B-quality model. State = growing KV (6 attn layers) + ONE fixed-shape conv
state [18, 1, hidden, kernel-1], within the pipelined extra-states budget.

input_ids is STATIC [1,1]; position_ids + KV seq stay dynamic -> pipelined
engine. Prefill runs as pipelined S=1 steps: set COREAI_CHUNK_THRESHOLD=1.

Quantization (mirrors the qwen3.6-MoE ship recipe):
  * global: linear per-block-32 `symmetric_with_clipping` on every Linear
    except norms/conv1d/embedding AND the attention q/k/v/out projections
    (kept fp32 — LFM2.5's large q/k-norm gains amplify fp16/quant matmul noise,
    the lfm2 module-header lesson);
  * MoE expert weights are 4D [1, E, out, in] SwitchLinear params -> per-type
    override with multi-dim blocks [1, 1, 1, 32];
  * the router (`feed_forward.gate`) stays fp16 — quantizing it can flip
    discrete expert selection for ~0.1% of bytes;
  * lm_head is tied to the embedding (stays fp16) except int8hu, which unties +
    quantizes it with plain `symmetric` (absmax) per-block-32 (big-vocab-head
    rule: clipping crushes outlier rows).

Numerics gate BEFORE believing any mode: _smoke/test_lfm2moe_parity.py (torch
fp32 ladder) then the on-device/Mac oracle gate on the exported bundle.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_lfm2_moe_decode_pipelined.py \
          int4lin --hf-id LiquidAI/LFM2.5-8B-A1B
Modes: fp16 (control), int8lin, int8hu (+ absmax lm_head), int4lin (size lever).
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.lfm2_moe import (
    DECODE_STATE_NAMES,
    build_decode_state,
    lfm2_moe_from_hf,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8", block: int = 32) -> dict:
    """Per-block linear quant + the 4D SwitchLinear (MoE expert) override.
    Attention projections + router + lm_head excluded by name; norms/conv/
    embedding excluded by type."""
    block_2d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {"type": "per_block", "block_size": block, "axis": 1},
    }
    block_4d = {
        "dtype": dtype,
        "qscheme": "symmetric_with_clipping",
        "granularity": {"type": "per_block", "block_size": [1, 1, 1, block], "axis": None},
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
            "coreai_models.primitives.macos.switch.SwitchLinear": {
                "module_state_spec": {"weight": block_4d},
                "op_input_spec": None,
                "op_output_spec": None,
            },
        },
        "module_name_configs": {
            r".*lm_head$": None,
            r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$": None,
            r".*feed_forward\.gate$": None,  # router stays fp16 (selection risk)
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int4lin",
                    choices=["fp16", "int8lin", "int8hu", "int4lin"])
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-8B-A1B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"])
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu: plain symmetric (absmax) head — SHIP shape")
    ap.add_argument("--int4-block", type=int, default=32)
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode == "int8hu" and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")

    print(f"loading {args.hf_id} fp16 ...", flush=True)
    model = lfm2_moe_from_hf(args.hf_id, target_dtype=DTYPE)
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers ({cfg.num_full_layers} full / "
          f"{cfg.num_conv_layers} conv), dense={cfg.num_dense_layers}, "
          f"E={cfg.num_experts}/top{cfg.num_experts_per_tok}, vocab={cfg.vocab_size}",
          flush=True)

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
    quant_mmap = None

    if args.mode in ("int8lin", "int8hu", "int4lin"):
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int8" if args.mode in ("int8lin", "int8hu") else "int4"
        block = args.int4_block if dtype == "int4" else 32
        cfg_q = linear_quant_config(dtype, block=block)
        if args.mode == "int8hu":
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(
                args.head_quant, args.head_sym)
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone())
        if not args.no_quant_mmap:
            quant_mmap = out_dir / "_quant_mmap"
            quant_mmap.mkdir()
        print(f"quantizing (linear {dtype} per-block-{block}, mode={args.mode}, "
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
