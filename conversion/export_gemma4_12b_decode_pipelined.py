"""Export a DECODE-ONLY (S=1, loop-free) Gemma 4 *dense* (12B) bundle for the
Core AI pipelined engine.

``google/gemma-4-12B-it`` (HF ``model_type: gemma4_unified``) is a clean dense
interleaved-attention Gemma decoder — NO PLE / AltUp / Laurel / MoE / KV-sharing /
double-wide-MLP (those belong to the on-device E2B/E4B siblings). Its only
wrinkles vs gemma3 are the dual head_dim (sliding 256 / full 512) and the
``attention_k_eq_v`` full layers (1 KV head, value == raw k_proj). Both ride a
single growing KV pair via the unified-cache decode core (``gemma4_dense_pipelined``),
so the bundle loads on the STOCK pipelined engine — exactly 2 states, no patch.

Engine contract: ``(input_ids [1,1] static, position_ids [1,S] dynamic) ->
logits [1,1,262144]`` with ONE growing KV pair (keyCache/valueCache
``[48,1,8,S,512]``; sliding layers padded 256->512, full layers padded 1->8 KV
heads). ``embed_tokens`` + the tied ``lm_head`` + final softcap live IN-GRAPH.

Modes:
  fp16    - no quantization (graph-correctness baseline; ~24 GB bundle)
  int8lin - linear int8 per-block-32 incl. an UNTIED lm_head (anchor; ~12 GB)
  int4lin - linear int4 per-block-32 incl. an UNTIED lm_head — THE ship config
            (~7 GB; from the QAT checkpoint = "int4 ~= bf16 by design")

--lin-sym uses plain absmax `symmetric` (no clipping) for ALL linears — the q4_0
grid the QAT checkpoints were TRAINED for (llama.cpp Q4_0 = per-block-32 absmax).
Default is `symmetric_with_clipping` (the qwen/gemma ship recipe).

--hf-id selects the checkpoint (default the QAT-unquantized 12B). Gate AFTER export
with _smoke/test_gemma4_12b_decode_oracle_gate.py against the matching fp32 oracle.

Run (from a coreai-models checkout with the gemma4_dense overlay):
  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_gemma4_12b_decode_pipelined.py \
    int4lin --lin-sym
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from huggingface_hub import snapshot_download

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.gemma4_dense_metal_sdpa import metalize_full_sdpa
from coreai_models.models.macos.gemma4_dense_pipelined import (
    PIPELINED_STATE_NAMES,
    Gemma4DensePipelinedForCausalLM,
)
from coreai_models.models.macos.gemma4_dense_text import Gemma4DenseForCausalLM
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels

DEFAULT_HF_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized"
DTYPE = torch.float16


def bundle_basename(hf_id: str) -> str:
    """``gemma4_<size>`` derived from the hf-id (so the 31B export isn't misnamed 12b)."""
    import re

    m = re.search(r"gemma-?4-(\w+?)-it", hf_id.lower())
    size = m.group(1) if m else "12b"
    return f"gemma4_{size}" + ("_qat" if "qat" in hf_id.lower() else "")


def q4_0_presnap_linears(model, block: int = 32) -> int:
    """Snap every nn.Linear weight to the exact ggml q4_0 grid (block-32 along in-features,
    d = signed_absmax/-8, q in [-8,7]) BEFORE the int4 store, so the QAT-q4_0-trained weights
    sit on their designed operating point. This is the closest q4_0 reachable through
    coreai-opt's positive-scale int4 (-8,7): the +absmax element of blocks whose signed absmax
    is positive is within 1/8, every other element is exact. See notes in the bench repo."""
    import torch as _t
    n = 0
    for m in model.modules():
        if isinstance(m, _t.nn.Linear) and m.weight.shape[-1] % block == 0:
            w = m.weight.data
            x = w.reshape(-1, block).float()
            vmax = _t.gather(x, 1, x.abs().argmax(1, keepdim=True))
            d = vmax / -8.0
            d = _t.where(d == 0, _t.ones_like(d), d)
            q = _t.clamp(_t.round(x / d) + 8, 0, 15)
            m.weight.data = ((q - 8) * d).reshape(w.shape).to(w.dtype)
            n += 1
    return n


def linear_quant_config(dtype: str = "int4", qscheme: str = "symmetric_with_clipping") -> dict:
    """Weight-only linear per-block-32 (scale-multiply dequant, no LUT) incl. the head.

    Embeddings / norms / SDPA / RoPE carry no quantized weight; the in-graph
    ``embed_tokens`` table stays fp16 so the gather is exact (the head is untied
    and quantized separately by ``main``).
    """
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
                    "qscheme": qscheme,
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
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*embed_tokens$": None},
    }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, cfg, max_ctx: int) -> None:
    meta = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {
            "tokenizer": hf_id,
            "vocab_size": cfg.vocab_size,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "source": {"model_definition": "torch", "hf_model_id": hf_id},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def save_tokenizer(model_dir: str, out_dir: Path) -> None:
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja"):
        src = Path(model_dir) / f
        if src.exists():
            shutil.copy(src, tok_dir / f)
    # Chat-EOS fix: gemma4's base tokenizer_config declares eos_token = "<eos>" (the
    # document-end token, id 1), but the instruct model ENDS A CHAT TURN with "<turn|>"
    # (id 106) — generation_config lists eos_token_id = [1, 106, 50]. A generic chat app
    # derives its stop sequence from tokenizer.eos_token alone, so with "<eos>" it never
    # stops on the turn terminator and runs to max_tokens. Set eos_token to the turn
    # terminator (the gemma convention — gemma2/3 use "<end_of_turn>") so any app stops
    # cleanly. Render-safe: the chat template emits literal "<turn|>", not {{ eos_token }}.
    cfg_path = tok_dir / "tokenizer_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        if cfg.get("eos_token") == "<eos>":
            cfg["eos_token"] = "<turn|>"
            cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int4lin",
                    choices=["fp16", "int8lin", "int4lin"])
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID)
    ap.add_argument("--lin-sym", action="store_true",
                    help="plain absmax symmetric (no clipping) — the q4_0-grid variant")
    ap.add_argument("--q4-0", dest="q4_0", action="store_true",
                    help="pre-snap linear weights to the exact ggml q4_0 grid before the int4 "
                         "store (implies the symmetric -8,7 grid) so the QAT-q4_0 weights hit "
                         "their designed operating point — the quality-preserving int4 path")
    ap.add_argument("--metal-sdpa", action="store_true",
                    help="replace the FULL layers' SDPA with a custom flash-decode Metal kernel "
                         "(the scratch-heap-crash bypass; see gemma4_dense_metal_sdpa)")
    ap.add_argument("--split-g", type=int, default=8,
                    help="with --metal-sdpa: SIMD-groups per head for the higher-occupancy G-way "
                         "sequence-split kernel (default 8, faster at long context); 0 = the simple "
                         "1-SIMD-group kernel")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--num-layers", type=int, default=None,
                    help="debug: truncated-layer export")
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()

    split_g = args.split_g or None  # 0 -> simple 1-SIMD-group kernel
    name = f"{bundle_basename(args.hf_id)}_decode_{args.mode}" + ("sym" if args.lin_sym else "") + ("_q40" if args.q4_0 else "")
    if args.metal_sdpa:
        name += "_msdpa" + (f"_g{split_g}" if split_g else "")
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    model_dir = snapshot_download(
        args.hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json",
                                    "tokenizer*", "chat_template*"])
    print(f"loading {args.hf_id} (fp16) ...", flush=True)
    causal = Gemma4DenseForCausalLM.from_local(
        model_dir, dtype=DTYPE, num_layers=args.num_layers).eval()
    cfg = causal.config

    model = Gemma4DensePipelinedForCausalLM(causal).eval()
    spec = model.build_export_spec(
        target_dtype=DTYPE, max_context_length=args.max_ctx,
        trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode in ("int8lin", "int4lin"):
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int4" if args.mode == "int4lin" else "int8"
        # q4_0 pre-snap wants the -8,7 grid so the snapped values round-trip; force symmetric.
        qscheme = "symmetric" if (args.lin_sym or args.q4_0) else "symmetric_with_clipping"
        # Untie the head so the quantizer actually quantizes it (it skips shared params).
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        if args.q4_0:
            nq = q4_0_presnap_linears(model)
            print(f"pre-snapped {nq} linear weights to the exact q4_0 grid", flush=True)
        print(f"quantizing (linear {dtype} per-block-32, {qscheme}, incl. untied head) ...",
              flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], linear_quant_config(dtype, qscheme))

    # Metalize AFTER quant (like the MoE ports) so the custom op isn't in the quant trace; the
    # full-layer SDPA carries no quantizable weight, so order is otherwise irrelevant.
    custom_kernels: list = []
    if args.metal_sdpa:
        variant = f"G-way split-{split_g}" if split_g else "1-SIMD-group"
        print(f"metalizing FULL-layer SDPA -> custom flash-decode kernel ({variant}, "
              "scratch-heap bypass) ...", flush=True)
        custom_kernels = [metalize_full_sdpa(model, split_g=split_g)]

    print("exporting decode-only engine graph to Core AI dialect ...", flush=True)
    if custom_kernels:
        prog = export_to_coreai_with_kernels(
            model,
            reference_inputs=spec["reference_inputs"],
            custom_kernels=custom_kernels,
            dynamic_shapes=spec["dynamic_shapes"],
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            state_names=spec["state_names"],
            externalize_modules=[],
        )
    else:
        prog = export_to_coreai(
            model,
            reference_inputs=spec["reference_inputs"],
            dynamic_shapes=spec["dynamic_shapes"],
            input_names=spec["input_names"],
            output_names=spec["output_names"],
            state_names=spec["state_names"],
            externalize_modules=[],
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

    write_bundle_metadata(out_dir, name, args.hf_id, cfg, args.max_ctx)
    save_tokenizer(model_dir, out_dir)

    import subprocess
    sz = subprocess.run(["du", "-sh", str(out_dir)], capture_output=True, text=True).stdout.split()[0]
    print(f"bundle ready: {out_dir} ({sz})", flush=True)
    print("gate next: .venv/bin/python ../_smoke/test_gemma4_12b_decode_oracle_gate.py "
          f"exports/{name}")
    print(f"bench: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3")


if __name__ == "__main__":
    main()
