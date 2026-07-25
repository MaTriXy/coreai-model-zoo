"""Export a DECODE-ONLY (S=1) Gemma 4 E2B bundle for the Core AI pipelined engine.

The graph rides `CoreAIPipelinedEngine` with the per-token-inputs patch:
inputs (input_ids [1,1] static, position_ids [1,S] dynamic, ple_tokens
[1,1,L,ld] fp16 per-token host input) + ONE unified growing KV pair
(keyCache/valueCache [15,1,1,S,512], sliding layers padded 256->512) ->
logits [1,1,262144]. embed_tokens + lm_head + the PLE projection live in-graph;
only the giant per-layer-embedding table stays on the host (mmap gather by the
engine's PerTokenInputProvider, see ondevice/artifacts/gemma4_gather_raw/).

Modes:
  fp16    - no quantization (graph-correctness baseline; big)
  int4lin - linear int4 per-block-32 incl. an UNTIED lm_head - THE ship config
            (~1.15 GB/token, oracle 8/8; scale-multiply dequant, no LUT)
  int8lin - linear int8 per-block-32 incl. an UNTIED lm_head (anchor config,
            ~2.3 GB/token, BW-bound)
  int4km  - k-means int4 group-32 on FFN+attn+UNTIED head - numerics 8/8 but
            2.25x SLOWER than int4lin at the same bytes: eager-palettized LUT
            dequant is the slow class on this delegate (kept for evidence)

--tbl (combine with a mode, e.g. `int4lin --tbl`): the PLE table itself becomes a
STATIC graph INPUT (ple_table [V,L*ld] int8 + ple_scale [V] f32, gathered in-graph
by input_ids) instead of a per-token ple_tokens input. No host provider, no decode
wait on the sampled token -> the engine's full pipeline survives in decode
(decode ~= prefill). Needs apps/coreai-pipelined-static-inputs.patch and an app
that binds the two dump files via EngineOptions.staticInputBuffers (OWNED
storageModeShared buffers preferred — see knowledge/pipelined-engine.md for the
buffer-mode traps).

Numerics gate AFTER export: teacher-forced oracle 8/8 on the GPU delegate with the
int8 mmap PLE gather (probe pattern in the source repo: _smoke/probe_gemma4_decode_parity.py).
Measured M4 Max (release, chunk=1, p128 g256): int4lin 70.9 decode / 85.3 prefill;
int4lin --tbl 77.0 / 87.1; int8lin 57.2 / 71.9; int4km 31.5 / 41.0 (k-means LUT
dequant is the slow class).

--hf-id picks the checkpoint (default google/gemma-4-E2B-it). The QAT releases
(`google/gemma-4-{E2B,E4B}-it-qat-q4_0-unquantized`, bf16 weights TRAINED for q4_0
rounding) ride the same path — the bundle name gains a `_qat` suffix and E4B is
derived from the config (42 layers, hidden 2560, 2 KV heads, 24 KV slots). The PLE
dump dir (--raw-dir) and the oracle MUST be regenerated from the same checkpoint.

Run (from a coreai-models checkout with the model overlay):
  python export_gemma4_decode_pipelined.py [fp16|int8lin|int4lin|int4km] [--tbl] [--max-ctx 4096]
Requires the gemma4_text + gemma4_pipelined model overlay (see conversion/README.md) and,
to RUN the bundle, the full Swift patch stack (provider bundles: through
apps/coreai-pipelined-per-token-inputs.patch; --tbl bundles: also
apps/coreai-pipelined-static-inputs.patch) + COREAI_CHUNK_THRESHOLD=1.
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
from coreai_models.models.macos.gemma4_pipelined import (
    PIPELINED_STATE_NAMES,
    Gemma4PipelinedForCausalLM,
    Gemma4PipelinedTblForCausalLM,
)
from coreai_models.models.macos.gemma4_text import Gemma4ForCausalLM

DEFAULT_HF_ID = "google/gemma-4-E2B-it"
DTYPE = torch.float16


def bundle_basename(hf_id: str) -> str:
    """gemma4_{e2b|e4b}[_qat] from the checkpoint id."""
    low = hf_id.lower()
    size = "e4b" if "e4b" in low else "e2b"
    return f"gemma4_{size}" + ("_qat" if "qat" in low else "")


def int4km_config() -> dict:
    """k-means int4 g32 — the gemma4-verified recipe class (FFN+head 8/8, attn 8/8).

    Embeddings and the PLE per-layer gate/projection stay fp16 (small, and the
    embed gather must stay exact); norms/SDPA/RoPE carry no quantizable weight.
    """
    spec = {
        "n_bits": 4,
        "granularity": {"type": "per_grouped_channel", "axis": 0, "group_size": 32},
        "enable_per_channel_scale": False,
    }
    return {
        "global_config": {"op_state_spec": {"weight": spec}},
        "module_name_configs": {
            r".*embed_tokens$": None,
            r".*per_layer_input_gate$": None,
            r".*per_layer_projection$": None,
            r".*per_layer_model_projection$": None,
        },
    }


def linear_quant_config(dtype: str = "int8", qscheme: str = "symmetric_with_clipping") -> dict:
    """Weight-only linear per-block-32 (scale-multiply dequant, no LUT) incl. the head.

    dtype "int8" = the qwen ship recipe; "int4" = the int4-bytes-without-LUT candidate
    (the int4 K-MEANS variant measured ~2x slower than int8lin on this delegate — the
    LUT gather, not bandwidth, dominates it). qscheme "symmetric" (plain absmax, no
    clipping) matches the q4_0 grid the QAT checkpoints were TRAINED for (llama.cpp
    Q4_0 = per-block-32 absmax symmetric) — the --lin-sym probe.
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
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int4km",
                    choices=["fp16", "int8lin", "int4lin", "int4km"])
    ap.add_argument("--tbl", action="store_true",
                    help="table-as-INPUT variant: ple_table/ple_scale ride as STATIC "
                         "graph inputs (no per-token provider, full decode "
                         "pipelining). Needs the engine static-inputs patch + "
                         "--raw-dir.")
    ap.add_argument("--raw-dir", default="gemma4_gather_raw",
                    help="gather-table dump dir (embed_per_layer.i8/.scale.f32 + "
                         "meta.json)")
    ap.add_argument("--hf-id", default=DEFAULT_HF_ID,
                    help="checkpoint id (E2B/E4B, base or QAT-unquantized)")
    ap.add_argument("--lin-sym", action="store_true",
                    help="linear modes: plain absmax symmetric (no clipping) — the "
                         "QAT-grid-aligned variant (q4_0 = per-block-32 absmax)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()
    name = (f"{bundle_basename(args.hf_id)}_decode_{args.mode}"
            + ("sym" if args.lin_sym else "") + ("_tbl" if args.tbl else ""))

    model_dir = snapshot_download(
        args.hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"])
    print(f"loading {args.hf_id} (fp16) ...")
    causal = Gemma4ForCausalLM.from_local(model_dir, dtype=DTYPE).eval()
    cfg = causal.config

    # The giant PLE table must NOT be traced into the graph (it arrives per token
    # as the ple_tokens input / per-row-int8 as the ple_table input) — drop the
    # module so nothing can reference it and ~4.7 GB leaves RAM before tracing.
    del causal.model.embed_tokens_per_layer

    if args.tbl:
        import numpy as np

        raw = Path(args.raw_dir)
        meta = json.loads((raw / "meta.json").read_text())
        v, pld = meta["V"], meta["PLD"]
        ple_q = torch.from_numpy(np.array(
            np.memmap(raw / "embed_per_layer.i8", np.int8, "r", shape=(v, pld))))
        ple_s = torch.from_numpy(
            np.fromfile(raw / "embed_per_layer.scale.f32", np.float32))
        model = Gemma4PipelinedTblForCausalLM(causal).eval()
        spec = model.build_export_spec(
            target_dtype=DTYPE, max_context_length=args.max_ctx,
            trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
            ple_table=ple_q, ple_scale=ple_s)
    else:
        model = Gemma4PipelinedForCausalLM(causal).eval()
        spec = model.build_export_spec(
            target_dtype=DTYPE, max_context_length=args.max_ctx,
            trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode == "int4km":
        from coreai_models.export.compression import palettize_pytorch_model

        # Untie the head so the palettizer actually quantizes it (it silently
        # skips shared parameters); the in-graph embed stays fp16 for exact gathers.
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print("palettizing (int4 k-means g32; FFN+attn+head; embeddings/PLE-proj excluded) ...")
        model = palettize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), int4km_config())
    elif args.mode in ("int8lin", "int4lin"):
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int4" if args.mode == "int4lin" else "int8"
        qscheme = "symmetric" if args.lin_sym else "symmetric_with_clipping"
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing (linear {dtype} per-block-32, {qscheme}, incl. untied head) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], linear_quant_config(dtype, qscheme))

    print("exporting decode-only engine graph to Core AI dialect ...")
    prog = export_to_coreai(
        model,
        reference_inputs=spec["reference_inputs"],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=spec["state_names"],
        externalize_modules=[],  # gemma4 opts out (orphan PLE front-end norms)
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

    write_bundle_metadata(out_dir, name, args.hf_id, cfg, args.max_ctx)
    # Copy tokenizer files straight from the HF snapshot — the venv's transformers
    # predates the gemma4 tokenizer's extra_special_tokens format and can't
    # round-trip it through AutoTokenizer.
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir()
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
              "chat_template.jinja", "generation_config.json"):
        src = Path(model_dir) / f
        if src.exists():
            shutil.copy(src, tok_dir / f)
    import subprocess

    sz = subprocess.run(["du", "-sh", str(out_dir)], capture_output=True, text=True).stdout.split()[0]
    print(f"bundle ready: {out_dir} ({sz})")
    print("gate next: .venv/bin/python ../_smoke/probe_gemma4_decode_parity.py "
          f"exports/{name}/{name}.aimodel")


if __name__ == "__main__":
    main()
