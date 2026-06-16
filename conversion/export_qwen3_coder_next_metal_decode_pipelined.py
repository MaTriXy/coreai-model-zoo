"""Export a DECODE-ONLY Qwen3-Coder-Next 80B-A3B bundle whose 512-expert top-10 MoE runs the
``gather_qmm`` custom Metal kernel — reading ONLY the 10 routed experts (a 51× over-read removed,
the largest over-read of any MoE that fits on a Mac).

The metal-MoE sibling of ``export_qwen3_6_moe_metal_decode_pipelined.py``, with one structural
difference forced by scale: the 80B model is ~159 GB in fp16, which does NOT fit the 128 GB host
as a full state-dict. So instead of ``from_hf_memory_efficient`` (full fp16) + ``metalize_moe``,
this uses ``from_hf_streaming_metal_moe`` (models/macos/moe_metal_streaming.py): the routed experts
are read one MoE layer at a time and immediately sym8-quantized into the ``MetalSwitchGLU`` kernel
buffers, the fp16 freed before the next layer (peak RAM ≈ one layer's experts + the accumulating
int8 ≈ 85 GB, well under the fp16 footprint). Everything outside the routed experts is the shipped
int8 recipe (int8 per-block-32 linears incl. the shared expert, fp16 router/shared-gate, absmax
int8 head — pass ``--head-sym`` for the big 151936-vocab head per the big-vocab-head lesson).

Arch == Qwen3.6's ``qwen3_5_moe`` (hybrid GatedDeltaNet/full-attn 3:1, GVA 32v/16k, partial-RoPE,
sigmoid-gated shared expert); only the loader/config differ (flat config, ``model.`` prefix,
per-expert keys, layer_types from full_attention_interval) — see models/macos/qwen3_next.py.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_qwen3_coder_next_metal_decode_pipelined.py \
          sym8 --hf-id Qwen/Qwen3-Coder-Next --head-sym
      # smoke first:  ... --num-layers 4 --head-sym   (4-layer integration before the full ~hour run)
GPU exclusivity: grab the community-repo _GPU_LOCK and confirm no diffgemma/llm-runner is using the
GPU (RAM: 85 GB streaming + another GPU job will OOM the 128 GB host).
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.moe_metal_streaming import from_hf_streaming_metal_moe
from coreai_models.models.macos.qwen3_next import (
    DECODE_STATE_NAMES,
    QWEN3_NEXT_PREFIX,
    Qwen3NextStatefulForCausalLM,
    build_decode_state,
    qwen3_next_config_from_json,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """int8 per-block-32 linear for the NON-routed-expert weights (incl. the shared expert).
    Routed experts are already MetalSwitchGLU (streamed sym8) -> not in any quant spec, untouched."""
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
            "coreai_models.primitives.macos.switch.SwitchLinear": None,
        },
        "module_name_configs": {
            r".*mlp\.gate$": None,
            r".*shared_expert_gate$": None,
            r".*lm_head$": None,
        },
    }


def head_quant_spec(gran: str, sym: bool) -> dict:
    if gran == "perchan":
        g: dict = {"type": "per_channel", "axis": 0}
    else:
        g = {"type": "per_block", "block_size": int(gran[len("block"):]), "axis": 1}
    return {
        "op_state_spec": {"weight": {"dtype": "int8",
                                     "qscheme": "symmetric" if sym else "symmetric_with_clipping",
                                     "granularity": g}},
        "op_input_spec": None,
        "op_output_spec": None,
    }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, cfg, max_ctx: int) -> None:
    meta = {
        "metadata_version": "0.2", "kind": "llm", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {"tokenizer": hf_id, "vocab_size": cfg.vocab_size,
                     "max_context_length": max_ctx, "embedded_tokenizer": True,
                     "function_map": {"main": ["main"]}},
        "source": {"model_definition": "torch", "hf_model_id": hf_id},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def save_tokenizer(hf_id: str, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(hf_id, allow_patterns=[
        "tokenizer*", "*.txt", "chat_template*", "vocab*", "merges*", "special_tokens*"]))
    (out_dir / "tokenizer").mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and (f.name.startswith("tokenizer") or f.name in (
                "vocab.json", "merges.txt", "chat_template.jinja", "special_tokens_map.json")):
            shutil.copy2(f, out_dir / "tokenizer" / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="sym8", choices=["sym8"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3-Coder-Next")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument("--head-quant", default="block32")
    ap.add_argument("--head-sym", action="store_true",
                    help="absmax symmetric head (recommended: the 151936-vocab head gates clean only absmax)")
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}_gather"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    # Build config from the FLAT qwen3_next config.json (cached from the bg download).
    from huggingface_hub import snapshot_download
    cfg_dir = Path(snapshot_download(args.hf_id, allow_patterns=["config.json"]))
    flat_cfg = json.load(open(cfg_dir / "config.json"))
    cfg = qwen3_next_config_from_json(flat_cfg, num_layers=args.num_layers)
    print(f"E={cfg.num_experts}/top{cfg.num_experts_per_tok}, moe_inter={cfg.moe_intermediate_size}, "
          f"hidden={cfg.hidden_size}, layers={cfg.num_hidden_layers}, vocab={cfg.vocab_size}", flush=True)

    # Stream the 80B: routed experts -> sym8 MetalSwitchGLU one layer at a time (fp16 never whole).
    print(f"streaming {args.hf_id} -> {args.mode} gather (per-expert, fp16 never fully materialized) ...",
          flush=True)
    model, kernel = from_hf_streaming_metal_moe(
        Qwen3NextStatefulForCausalLM, cfg, args.hf_id, scheme=args.mode,
        prefix=QWEN3_NEXT_PREFIX, num_layers=args.num_layers, target_dtype=DTYPE)
    model.eval()

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"loop-free single-step enabled on {n_lin} linear layers", flush=True)

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

    # 1) int8 the NON-routed-expert weights (routed experts are already MetalSwitchGLU -> untouched).
    from coreai_models.export.compression import quantize_pytorch_model
    cfg_q = linear_quant_config("int8")
    cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(args.head_quant, args.head_sym)
    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing non-routed-expert linears (int8 per-block-32; routed experts already sym8) ...",
          flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) export with the streamed kernel; drop gather_mm (replaced) + gated_delta_update (inlined).
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print("exporting decode-only graph (custom gather_qmm kernel embedded) ...", flush=True)
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
    write_bundle_metadata(out_dir, name, args.hf_id, cfg, args.max_ctx)
    save_tokenizer(args.hf_id, out_dir)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}")
    print(f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
