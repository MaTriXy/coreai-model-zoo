"""Export a DECODE-ONLY Youtu-LLM-2B bundle with ABSORBED-MLA attention.

Youtu-LLM-2B is a DENSE DeepSeek-MLA decoder (Tencent) — the same MLA as
GLM-4.7-Flash but with a plain gated MLP on every layer and a tied lm_head. The
absorbed-MLA decode caches only the [512] kv-latent + [64] rope key (vs the naive
materialized per-head K/V) and folds the kv_b up-projection into the q-lift /
latent-value readout. Its K(576/192)!=V(512/128) shape is exactly what the
``mla_metal_sdpa`` flash-decode kernel was built for, so ``--metal-sdpa`` (default)
swaps the eager einsum core for the fused, online-softmax staged kernel.

Everything is the shipped int8 recipe: int8 per-block-32 linears (the MLA
projections + the dense MLP gate/up/down), absmax/int8 head. W_UK/W_UV (the small
absorbed lifts) are buffers -> stay fp16.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_youtu_decode_pipelined.py \
          --hf-id tencent/Youtu-LLM-2B [--eager-graph] [--split-g 8] [--num-layers N]
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
from coreai_models.models.macos.youtu_absorbed import (
    DECODE_STATE_NAMES_ABSORBED,
    YoutuAbsorbedStatefulForCausalLM,
    build_absorbed_decode_state,
    youtu_absorbed_from_hf,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """int8 per-block-32 for every dense linear (MLA q_a/q_b/kv_a/o_proj + the MLP
    gate/up/down). SDPA/RoPE/RMSNorm/Embedding -> None. W_UK/W_UV are buffers (not
    Linear) -> untouched (stay fp16). lm_head is set separately by the caller."""
    block_2d = {"dtype": dtype, "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}
    return {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": block_2d}, "op_input_spec": None, "op_output_spec": None},
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


def head_quant_spec(gran: str, sym: bool) -> dict:
    if gran == "perchan":
        g: dict = {"type": "per_channel", "axis": 0}
    else:
        g = {"type": "per_block", "block_size": int(gran[len("block"):]), "axis": 1}
    return {"op_state_spec": {"weight": {"dtype": "int8",
            "qscheme": "symmetric" if sym else "symmetric_with_clipping", "granularity": g}},
            "op_input_spec": None, "op_output_spec": None}


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, cfg, max_ctx: int) -> None:
    meta = {"metadata_version": "0.2", "kind": "llm", "name": name,
            "assets": {"main": f"{name}.aimodel"},
            "language": {"tokenizer": hf_id, "vocab_size": cfg.vocab_size,
                         "max_context_length": max_ctx, "embedded_tokenizer": True,
                         "function_map": {"main": ["main"]}},
            "source": {"model_definition": "torch", "hf_model_id": hf_id},
            "compression": None,
            "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def save_tokenizer(hf_id: str, out_dir: Path) -> None:
    from huggingface_hub import snapshot_download
    src = Path(snapshot_download(hf_id, allow_patterns=[
        "tokenizer*", "*.txt", "chat_template*", "*.jinja", "special_tokens*", "vocab*", "merges*"]))
    (out_dir / "tokenizer").mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and (f.name.startswith("tokenizer") or f.name in (
                "vocab.json", "merges.txt", "chat_template.jinja", "special_tokens_map.json")):
            shutil.copy2(f, out_dir / "tokenizer" / f.name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hf-id", default="tencent/Youtu-LLM-2B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--linear-dtype", default="int8", choices=["int8", "int4"],
                    help="quant for the dense linears (int4 keeps an int8 head to dodge the int4 cliff)")
    ap.add_argument("--eager-graph", action="store_true",
                    help="keep the absorbed eager einsum core (probe) instead of the flash-decode kernel")
    ap.add_argument("--split-g", type=int, default=32,
                    help="staged MLA kernel sequence-split factor (# threadgroups for occupancy)")
    ap.add_argument("--per-head", action="store_true",
                    help="use the non-staging per-head baseline kernel (A/B; default = staged)")
    ap.add_argument("--head-quant", default="block32")
    ap.add_argument("--head-sym", action="store_true")
    ap.add_argument("--no-quant-mmap", action="store_true")
    args = ap.parse_args()

    metal = not args.eager_graph
    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_absorbed_{args.linear_dtype}" + ("_msdpa" if metal else "_graph")
    if metal and not args.per_head and args.split_g != 8:
        name += f"_g{args.split_g}"  # keep g8 unsuffixed (the ship bundle); tag other split factors
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 (absorbed) ...", flush=True)
    causal = youtu_absorbed_from_hf(args.hf_id, target_dtype=DTYPE)
    if args.num_layers is not None:
        causal.model.layers = causal.model.layers[: args.num_layers]
        causal.config.num_hidden_layers = args.num_layers
    model = YoutuAbsorbedStatefulForCausalLM.from_causal_lm(causal)
    model.eval()
    cfg = causal.config
    print(f"model ready | {cfg.num_hidden_layers} layers, kv_lora={cfg.kv_lora_rank}, "
          f"qk_head_dim={cfg.qk_head_dim}, v_head_dim={cfg.v_head_dim}, vocab={cfg.vocab_size}", flush=True)

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_absorbed_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids,
        "kv_a": state["kv_a"], "kv_b": state["kv_b"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    a_seq = torch.export.Dim("a_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    b_seq = torch.export.Dim("b_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None, "position_ids": {1: seq_pos},
        "kv_a": {KVCache.seq_len_dim(): a_seq},
        "kv_b": {KVCache.seq_len_dim(): b_seq},
    }

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) quantize the dense linears (MLA projections + the dense MLP).
    from coreai_models.export.compression import quantize_pytorch_model
    cfg_q = linear_quant_config(args.linear_dtype)
    cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(args.head_quant, args.head_sym)
    quant_mmap = None
    if not args.no_quant_mmap:
        quant_mmap = out_dir / "_quant_mmap"
        quant_mmap.mkdir()
    print("quantizing dense linears (int8 per-block-32) ...", flush=True)
    model = quantize_pytorch_model(
        model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q,
        mmap_dir=str(quant_mmap) if quant_mmap else None)

    # 2) optionally swap the MLA core for the absorbed flash-decode kernel.
    kernels: list = []
    if metal:
        from coreai_models.models.macos.mla_metal_sdpa import metalize_mla
        staged = not args.per_head
        kind = f"staged (split_g={args.split_g})" if staged else "per-head baseline"
        print(f"metalizing MLA -> absorbed flash-decode kernel: {kind} ...", flush=True)
        kernels.extend(metalize_mla(causal, scale=cfg.qk_head_dim ** -0.5,
                                    staged=staged, split_g=args.split_g))
    else:
        print("MLA kept as eager einsum graph (probe mode) ...", flush=True)

    # 3) export the decode-only graph.
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update", "gather_mm")]
    print("exporting decode-only graph ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs=reference_inputs, custom_kernels=kernels,
        dynamic_shapes=dynamic_shapes, input_names=("input_ids", "position_ids"),
        output_names=("logits",), state_names=DECODE_STATE_NAMES_ABSORBED, externalize_modules=specs)
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
