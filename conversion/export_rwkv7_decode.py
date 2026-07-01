"""Export a DECODE-ONLY (S=1) RWKV-7 "Goose" bundle for Core AI.

RWKV-7 is a pure-recurrent / linear-attention LLM — NO attention, NO KV cache.
Every layer is a WKV7 delta-rule matrix-state time-mix + sqrelu channel-mix, so
the decode graph is loop-free at S=1 and carries only TWO fixed-shape states:
  * recState  [num_layers, 1, num_heads, head_dim, head_dim]  (the WKV7 matrix S)
  * shiftState[num_layers, 1, 2, hidden]                      (token-shift prev hidden)
This is the O(1)-per-token win: constant memory, unbounded context, no growing KV.

EVERYTHING is static (input_ids [1,1], no dynamic seq), so the exported graph is a
fixed-shape decode step. The per-token recurrence lowers to STANDARD Core AI ops —
no custom composite (verified: granite4h's pure-torch Mamba2 step exports the same
way), so no GatedDeltaUpdate-style externalization is needed.

Run (coreai-models venv):
  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_rwkv7_decode.py [fp32|fp16] \
      [--hf-id RWKV/RWKV7-Goose-World3-1.5B-HF] [--out-dir exports]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.rwkv7 import (
    DECODE_STATE_NAMES,
    build_decode_state,
    rwkv7_from_hf,
)


def quant_config(dtype: str = "int8", block: int = 32, keep_proj: bool = False) -> dict:
    """Weight-only linear per-block scale-multiply dequant (no LUT — the qwen3.5/lfm2
    ship recipe). Embedding + LayerNorm + GroupNorm excluded by type. With
    ``keep_proj`` the recurrence-critical projections (r/k/v/o_proj + all LoRA
    linears) stay fp16 — they feed the fp32 WKV7 delta-rule; only the FFN
    (the weight bulk) + lm_head quantize."""
    def spec(d: str, b: int) -> dict:
        return {
            "op_state_spec": {"weight": {
                "dtype": d, "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": b, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        }
    name_configs: dict = {}
    if keep_proj:
        name_configs[r".*\.(r_proj|k_proj|v_proj|o_proj)$"] = None
        name_configs[r".*_lora\.lora\.\d+$"] = None
    return {
        "execution_mode": "eager",
        "global_config": spec(dtype, block),
        "module_type_configs": {
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.normalization.LayerNorm": None,
            "torch.nn.modules.normalization.GroupNorm": None,
        },
        "module_name_configs": name_configs,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="fp32",
                    choices=["fp32", "fp16", "int8", "int8keepproj", "int4", "int4keepproj"])
    ap.add_argument("--hf-id", default="RWKV/RWKV7-Goose-World3-1.5B-HF")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--block", type=int, default=32, help="per-block granularity for int modes")
    ap.add_argument("--last-token-only", action="store_true",
                    help="return only the last token's logits (S=1 -> no-op, kept for symmetry)")
    args = ap.parse_args()

    is_quant = args.mode.startswith("int")
    dtype = torch.float16 if is_quant else (torch.float32 if args.mode == "fp32" else torch.float16)
    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"

    print(f"loading {args.hf_id} ({args.mode}) ...", flush=True)
    model = rwkv7_from_hf(args.hf_id, target_dtype=dtype)
    model.last_token_only = args.last_token_only
    cfg = model.config
    print(f"{cfg.num_hidden_layers} layers, hidden={cfg.hidden_size}, heads={cfg.num_heads}, "
          f"vocab={cfg.vocab_size}", flush=True)

    # Decode trace: S=1 static query; RWKV-7 is positionless so position_ids is a
    # static [1,1] placeholder. Both states are fixed-shape.
    input_ids = torch.zeros(1, 1, dtype=torch.int32)
    position_ids = torch.zeros(1, 1, dtype=torch.int32)
    state = build_decode_state(cfg, dtype=dtype)
    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "rec_state": state["rec_state"],
        "shift_state": state["shift_state"],
    }
    # Fully static graph — no dynamic dims anywhere.
    dynamic_shapes = {k: None for k in reference_inputs}

    if is_quant:
        from coreai_models.export.compression import quantize_pytorch_model
        qdtype = "int8" if args.mode.startswith("int8") else "int4"
        keep_proj = "keepproj" in args.mode
        print(f"quantizing (linear {qdtype} per-block-{args.block}"
              f"{', proj/lora kept fp16' if keep_proj else ', all linears'}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes,
            quant_config(qdtype, args.block, keep_proj))

    print("exporting decode-only graph to Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=DECODE_STATE_NAMES,
        externalize_modules=[],   # no composite ops used (pure standard-op recurrence)
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
