"""Export a DECODE-ONLY ZAYA1-8B bundle whose 16-expert top-1 MoE runs the
``gather_qmm`` custom Metal kernel (reads only the 1 routed expert; the MoD
"skip" expert is the identity path handled in-graph).

zoo's first Compressed Convolutional Attention (CCA) + EDA router export. The
CCA conv-state + prev_hs decode states ride alongside the KV pair (2 extra
fixed-shape states, within the pipelined ≤2 budget), the lfm2.py loop-free step.

Two-stage de-risk:
  * default (mode=sym8): experts -> MetalSwitchGLU sym8; everything else fp16,
    CCA weights fp32 (QK-norm amplifies fp16 projection error — lfm2 lesson).
    Validates the GRAPH + gather. ~9GB.
  * --int8-rest: additionally int8 per-block-32 the router/o_proj-free non-expert
    linears + int8 head -> the ship size (~8.5GB).

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
        ../coreai-models-community/conversion/export_zaya_moe_metal_decode_pipelined.py \
        sym8      # --ckpt defaults to <work root>/_zaya_ckpt, see conversion/_paths.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from _paths import work_path

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.moe_metal import metalize_moe
from coreai_models.models.macos.zaya import (
    DECODE_STATE_NAMES,
    build_decode_state,
    zaya_from_hf,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16
HF_ID = "Zyphra/ZAYA1-8B"


def linear_quant_config() -> dict:
    """int8 per-block-32 for the SAFE non-expert linears only. Excludes the CCA
    self_attn block (QK-norm + temp amplify fp16/int8 projection error) and the
    router (int8 would perturb the top-1 argmax). Routed experts -> None (the
    gather_qmm kernel owns them)."""
    block_2d = {"dtype": "int8", "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}
    return {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": block_2d},
                          "op_input_spec": None, "op_output_spec": None},
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
            "coreai_models.primitives.macos.switch.SwitchLinear": None,  # experts -> metal
        },
        "module_name_configs": {
            r".*\.self_attn\..*": None,   # CCA stays fp32
            r".*\.router\..*": None,      # router stays fp16
            r".*lm_head$": None,          # head set explicitly below
        },
    }


def head_quant_spec() -> dict:
    return {"op_state_spec": {"weight": {"dtype": "int8",
            "qscheme": "symmetric_with_clipping",
            "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None}


def write_bundle_metadata(out_dir: Path, name: str, cfg, max_ctx: int) -> None:
    meta = {"metadata_version": "0.2", "kind": "llm", "name": name,
            "assets": {"main": f"{name}.aimodel"},
            "language": {"tokenizer": HF_ID, "vocab_size": cfg.vocab_size,
                         "max_context_length": max_ctx, "embedded_tokenizer": True,
                         "function_map": {"main": ["main"]}},
            "source": {"model_definition": "torch", "hf_model_id": HF_ID},
            "compression": None,
            "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def save_tokenizer(out_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(HF_ID, allow_patterns=[
        "tokenizer*", "*.txt", "chat_template*", "*.jinja", "special_tokens*",
        "vocab*", "merges*"]))
    (out_dir / "tokenizer").mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and (f.name.startswith("tokenizer") or f.name in (
                "vocab.json", "merges.txt", "chat_template.jinja",
                "special_tokens_map.json")):
            shutil.copy2(f, out_dir / "tokenizer" / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="sym8",
                    choices=["sym8", "int8km", "int4km"])
    ap.add_argument("--ckpt", default=str(work_path("_zaya_ckpt")))
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--int8-rest", action="store_true",
                    help="also int8 the safe non-expert linears + head (ship size)")
    ap.add_argument("--fp32-cca", action="store_true",
                    help="keep CCA projections fp32 (needs dtype-casting in ZayaCCA)")
    ap.add_argument("--fp16-experts", action="store_true",
                    help="NO expert quantization: plain fp16 SwitchGLU / dense GatherMM "
                         "(Mac max-quality; reads all 16 experts/token = slower but lossless)")
    args = ap.parse_args()

    scheme = {"sym8": "sym8", "int8km": "km8", "int4km": "km4"}[args.mode]
    name = "zaya1_8b_decode_fp16experts" if args.fp16_experts else f"zaya1_8b_decode_{args.mode}_gather"
    if args.int8_rest:
        name += "_int8rest"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading ZAYA1-8B fp16 (CCA fp32) from {args.ckpt} ...", flush=True)
    model = zaya_from_hf(args.ckpt, target_dtype=DTYPE, stateful=True, fp32_attn=args.fp32_cca)
    if args.num_layers is not None:
        model.model.layers = model.model.layers[: args.num_layers]
        model.config.num_hidden_layers = args.num_layers
    model.eval()
    cfg = model.config
    print(f"model ready | {cfg.num_hidden_layers} layers ({cfg.num_att_layers} att/"
          f"{cfg.num_moe_layers} moe), E={cfg.num_experts} top{cfg.moe_router_topk} "
          f"(+MoD skip), moe_inter={cfg.moe_intermediate_size}, hidden={cfg.hidden_size}, "
          f"vocab={cfg.vocab_size}", flush=True)

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
        "conv_state": state["conv_state"], "prev_hs_state": state["prev_hs_state"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None, "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None, "prev_hs_state": None,
    }

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if args.int8_rest:
        from coreai_models.export.compression import quantize_pytorch_model
        cfg_q = linear_quant_config()
        cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec()
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
        print("int8 per-block-32 the safe non-expert linears + head ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
            mmap_dir=str(quant_mmap))
    else:
        quant_mmap = None

    if args.fp16_experts:
        # NO metalize: experts stay fp16 SwitchGLU -> dense GatherMM (reads all E).
        # Keep gather_mm in the externalize specs (it is USED, not replaced).
        kernels = []
        specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
        print("exporting decode-only graph (fp16 experts, dense GatherMM, NO quant) ...", flush=True)
    else:
        print(f"metalizing routed MoE -> gather_qmm {scheme} (reads only top-1/{cfg.num_experts}) ...",
              flush=True)
        kernels = [metalize_moe(model, scheme=scheme)]
        specs = [s for s in _EXTERNALIZE_SPECS
                 if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
        print("exporting decode-only graph (custom kernel embedded) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs=reference_inputs, custom_kernels=kernels,
        dynamic_shapes=dynamic_shapes, input_names=("input_ids", "position_ids"),
        output_names=("logits",), state_names=DECODE_STATE_NAMES,
        externalize_modules=specs)
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    write_bundle_metadata(out_dir, name, cfg, args.max_ctx)
    save_tokenizer(out_dir)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
