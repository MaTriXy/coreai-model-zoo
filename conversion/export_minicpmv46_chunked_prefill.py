"""Export the MiniCPM-V-4.6 TEXT decode core as a DYNAMIC-query graph that does CHUNKED
PREFILL — the TTFT experiment.

The shipped bundle (`export_minicpmv46_decode_pipelined.py`) freezes input_ids to [1,1] +
the loop-free single-step GDN, so the engine processes the prompt ONE token at a time
(prefill tok/s == decode tok/s == TTFT bottleneck). This variant instead:

  * sets `use_loopfree_chunk = True` on every GDN layer  → the loop-free chunked scan
    (`_gated_delta_chunk`, parity-gated in _smoke/test_gdn_chunked_parity.py) processes a
    whole query block in parallel and still lowers (no while_loop);
  * exports with a DYNAMIC input_ids query length (`build_macos_export_spec`) + `last_token_only`,
    so the default (pipelined) engine feeds the WHOLE prompt as one block (chunked prefill)
    and feeds 1 token per step for decode — both served by this single graph.

Run (grab _GPU_LOCK for the GPU steps):
    coreai-models/.venv/bin/python conversion/export_minicpmv46_chunked_prefill.py [int8lin|fp16]
Then benchmark on Mac (default engine — no crash now that the graph accepts a multi-token block):
    .build/out/Products/Release/llm-benchmark --model exports/minicpmv46_text_chunked_int8lin -p 128 -g 64 -n 3
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from _bundle import head_quant_spec
from _paths import hf_snapshot

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5Config,
    Qwen3_5ForCausalLMStateful,
)

DTYPE = torch.float16
TEXT_PREFIX = "model.language_model."


def snapshot_dir() -> str:
    return hf_snapshot("openbmb/MiniCPM-V-4.6")


def build_config() -> Qwen3_5Config:
    lt = (["linear_attention"] * 3 + ["full_attention"]) * 6
    return Qwen3_5Config(
        hidden_size=1024, num_hidden_layers=24, vocab_size=248094, intermediate_size=3584,
        rms_norm_eps=1e-6, tie_word_embeddings=True, head_dim=256,
        num_attention_heads=8, num_key_value_heads=2, attn_output_gate=True,
        partial_rotary_factor=0.25, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=16,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_conv_kernel_dim=4,
        full_attention_interval=4, layer_types=lt,
    )


def load_text_weights(model: Qwen3_5ForCausalLMStateful) -> None:
    ckpt = glob.glob(snapshot_dir() + "/model.safetensors")[0]
    sd = {}
    with safe_open(ckpt, framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith(TEXT_PREFIX):
                sd["model." + k[len(TEXT_PREFIX):]] = f.get_tensor(k).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unloaded params: {meta[:6]}")
    print(f"[load] {len(sd)} text tensors (prefix-remapped, tied head)")


def linear_quant_config(dtype: str = "int8") -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": dtype, "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
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
    ap = argparse.ArgumentParser()
    # int8hu = int8 body + UNTIED int8 lm_head (head fp16 in int8lin = 508MB read/token = ~half the
    # per-token bandwidth; quantizing it is the proven ~+40% decode lever on the 0.8B class).
    # int8hu_metal = int8hu + the fp32 GDN chunked-scan Metal kernel (numerically STABLE at chunk=32/64,
    # where the in-graph fp16 doubling-inverse NaNs) → unlocks the big-chunk prefill win (~32-45x).
    ap.add_argument("mode", nargs="?", default="int8hu_metal",
                    choices=["fp16", "int8lin", "int8hu", "int8hu_metal"])
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--chunk-cap", type=int, default=63,
                    help="metal mode: max prefill chunk (engine COREAI_CHUNK_THRESHOLD must be <= this). "
                         "63 not 64: torch.export auto-guards S != 64 (a downstream numel special-case).")
    args = ap.parse_args()
    is_metal = args.mode == "int8hu_metal"
    quant_mode = "int8hu" if is_metal else args.mode

    name = f"minicpmv46_text_chunked_{args.mode}"
    cfg = build_config()
    model = Qwen3_5ForCausalLMStateful(cfg).eval()
    load_text_weights(model)

    n_lin = 0
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_chunk = True   # chunked prefill + S==1 decode
            n_lin += 1
    # last_token_only=True (head computed once → the big amortization win; output [1,1,vocab]).
    # NOTE: the STOCK pipelined engine binds query=1 from this output shape, so chunked prefill via the
    # engine needs full-query output — BUT the engine's S>1 compute is buggy for this 4-state GDN bundle
    # (wrong tokens on device). So chunked prefill is driven by a custom HOST LOOP (low-level run with
    # [1,chunk] + manual 4-state threading + provided output buffer), which keeps last_token_only.
    model.last_token_only = True
    print(f"[chunked] use_loopfree_chunk on {n_lin} linear layers; last_token_only=True")

    # Dynamic query-length graph (one graph: chunked prefill at S>1, decode at S==1).
    spec = model.build_macos_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        query_len=8, offset=0, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
    reference_inputs = spec["reference_inputs"]
    dynamic_shapes = spec["dynamic_shapes"]

    if quant_mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model
        cfg_q = linear_quant_config("int8")
        if quant_mode == "int8hu":
            # Untie the head (the eager quantizer silently skips shared params) then quantize it.
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec("block32", True)}
            print("[quant] linear int8 per-block-32 + UNTIED int8 head (block32 symmetric) ...")
        else:
            print("[quant] linear int8 per-block-32 (fp16 tied head) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    custom_kernels: list = []
    if is_metal:
        # Swap the GDN scan for the fp32 Metal kernel (takes precedence over use_loopfree_chunk in
        # the forward). Cap the input_ids query dim to chunk_cap (< the kernel's chunk_max buffer).
        from coreai_models.models.macos.qwen3_5_gdn_metal import metalize_gdn_chunk
        custom_kernels = [metalize_gdn_chunk(model)]
        seq_pos = torch.export.Dim("seq_pos", min=1, max=args.max_ctx - 1)
        dynamic_shapes = dict(dynamic_shapes)
        dynamic_shapes["input_ids"] = {1: torch.export.Dim("q_ids", min=1, max=args.chunk_cap)}
        dynamic_shapes["position_ids"] = {1: seq_pos}
        print(f"[metal] GDN fp32 chunked-scan kernel on {n_lin} layers; query cap {args.chunk_cap}")

    print("[export] -> Core AI dialect (dynamic query) ...")
    if is_metal:
        from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
        prog = export_to_coreai_with_kernels(
            model, reference_inputs, custom_kernels=custom_kernels, dynamic_shapes=dynamic_shapes,
            input_names=spec["input_names"], output_names=spec["output_names"],
            state_names=spec["state_names"], externalize_modules=specs)
    else:
        prog = export_to_coreai(
            model, reference_inputs, dynamic_shapes=dynamic_shapes,
            input_names=spec["input_names"], output_names=spec["output_names"],
            state_names=spec["state_names"], externalize_modules=specs)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[save] {aimodel}")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    meta = {
        "metadata_version": "0.2", "kind": "llm", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {"tokenizer": "openbmb/MiniCPM-V-4.6", "vocab_size": cfg.vocab_size,
                     "max_context_length": args.max_ctx, "embedded_tokenizer": True,
                     "function_map": {"main": ["main"]}},
        "source": {"model_definition": "torch", "hf_model_id": "openbmb/MiniCPM-V-4.6"},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    tdir = out_dir / "tokenizer"
    tdir.mkdir()
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
               "generation_config.json"):
        src = Path(snapshot_dir()) / fn
        if src.exists():
            shutil.copy(src, tdir / fn)
    print(f"[done] bundle: {out_dir}")


if __name__ == "__main__":
    main()
