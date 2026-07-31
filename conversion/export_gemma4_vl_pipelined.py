"""Export Gemma 4 E2B VISION (VL) for the Core AI pipelined engine.

The Qwen3-VL rider recipe on the gemma4 decode-only port — TWO artifacts:

* ``<name>/`` — the TEXT DECODER bundle on the existing gemma4 pipelined
  contract (S=1 static ids, dynamic position ramp, ONE padded KV pair,
  in-graph embed/head/PLE-projection) PLUS ONE new static input
  ``image_embeds [n_slots, 1536] f16``. The host rewrites the prompt's
  ``<image_soft_token>`` ids to EXTENSION ids ``V + slot``; in-graph
  ``x = ids < V ? embed_tokens[ids] : image_embeds[ids - V]`` and the PLE
  gather reads the PAD row (id 0) for extension ids — byte-identical PLE
  tables to the text ship. Image span attention is CAUSAL (E2B-it
  ``use_bidirectional_attention=None``, verified vs the fp32 HF mask dump),
  positions are the standard ramp: NO rope changes, NO extra states. With no
  extension ids the graph degenerates to the text decoder.

* ``<name>_vision/`` — the fixed-grid VISION ENCODER ``.aimodel``:
  ``patches [2304, 768] -> image_embeds [256, 1536]`` fp16, run once per
  image (48x48-patch square = 768x768 px, what the processor emits for any
  square image; soft tokens 0..255 of the 280-slot static buffer).

Modes mirror export_gemma4_decode_pipelined.py: fp16 | int8lin | int4lin
(+ --lin-sym), --tbl for the table-as-static-input variant (ship). The QAT
checkpoint (google/gemma-4-E2B-it-qat-q4_0-unquantized) contains the FULL
multimodal model incl. vision tower + clip buffers; pair --tbl with the
MATCHING dump (--raw-dir ondevice/artifacts/gemma4_qat_gather_raw).

Numerics gating: torch ladder _smoke/test_gemma4vl_torch_ladder.py (vision
cos 1.000, prefill 4/4, decode 16/16 vs the fp32 HF oracle) BEFORE this
script; the .aimodel gate is _smoke/check_gemma4vl_aimodel_gate.py.

Run (cwd coreai-models-community/conversion, coreai-models/.venv python):
  python export_gemma4_vl_pipelined.py int4lin --tbl \
      --hf-id google/gemma-4-E2B-it-qat-q4_0-unquantized \
      --raw-dir ../../ondevice/artifacts/gemma4_qat_gather_raw
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from _bundle import write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.gemma4_pipelined import (
    Gemma4VLPipelinedForCausalLM,
    Gemma4VLPipelinedTblForCausalLM,
)
from coreai_models.models.macos.gemma4_text import Gemma4ForCausalLM
from coreai_models.models.macos.gemma4_vision import Gemma4VisionEncoder

DTYPE = torch.float16
N_IMAGE_SLOTS = 280  # max soft-token budget; the 48x48 square grid fills 256


def bundle_basename(hf_id: str) -> str:
    low = hf_id.lower()
    size = "e4b" if "e4b" in low else "e2b"
    return f"gemma4_{size}" + ("_qat" if "qat" in low else "") + "_vl"


def linear_quant_config(dtype: str = "int8", qscheme: str = "symmetric_with_clipping") -> dict:
    """Weight-only linear per-block-32 incl. the untied head (the gemma4 recipe)."""
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


def export_vision(args, base: str) -> None:
    out_dir = Path(args.out_dir) / f"{base}_vision"
    print(f"loading vision tower ({args.hf_id}) fp16, grid {args.grid}x{args.grid} ...")
    vis = Gemma4VisionEncoder.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid, grid_w=args.grid)
    patches = torch.zeros(vis.n_patches, 3 * vis.vcfg["patch_size"] ** 2, dtype=DTYPE)

    print("exporting vision graph ...")
    prog = export_to_coreai(
        vis,
        {"patches": patches},
        dynamic_shapes={"patches": None},
        input_names=("patches",),
        output_names=("image_embeds",),
        state_names=(),
        externalize_modules=[
            s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"
        ],
    )
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    prog.save_asset(out_dir / f"{base}_vision.aimodel", rt.AIModelAssetMetadata())
    import subprocess

    sz = subprocess.run(["du", "-sh", str(out_dir)], capture_output=True, text=True).stdout.split()[0]
    print(f"vision ready: {out_dir} ({sz})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int4lin",
                    choices=["fp16", "int8lin", "int4lin"])
    ap.add_argument("--tbl", action="store_true",
                    help="PLE table as STATIC graph input (ship variant; needs "
                         "--raw-dir matching the checkpoint)")
    ap.add_argument("--raw-dir", default="../../ondevice/artifacts/gemma4_qat_gather_raw",
                    help="PLE gather-table dump dir (embed_per_layer.i8/.scale.f32 + meta.json)")
    ap.add_argument("--hf-id", default="google/gemma-4-E2B-it-qat-q4_0-unquantized")
    ap.add_argument("--lin-sym", action="store_true",
                    help="plain absmax symmetric (QAT q4_0-grid-aligned probe)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--grid", type=int, default=48,
                    help="vision patch grid side (48 = 768x768 square, 256 soft tokens)")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    base = bundle_basename(args.hf_id)
    name = (f"{base}_decode_{args.mode}"
            + ("sym" if args.lin_sym else "") + ("_tbl" if args.tbl else ""))

    if not args.skip_vision:
        export_vision(args, base)
    if args.skip_decoder:
        return

    model_dir = snapshot_download(
        args.hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "*.json"])
    print(f"loading {args.hf_id} text decoder (fp16) ...")
    causal = Gemma4ForCausalLM.from_local(model_dir, dtype=DTYPE).eval()
    cfg = causal.config

    # Giant PLE table must NOT be traced (rides as ple_tokens / ple_table input).
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
        model = Gemma4VLPipelinedTblForCausalLM(
            causal, n_image_slots=N_IMAGE_SLOTS).eval()
        spec = model.build_export_spec(
            target_dtype=DTYPE, max_context_length=args.max_ctx,
            trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
            ple_table=ple_q, ple_scale=ple_s)
    else:
        model = Gemma4VLPipelinedForCausalLM(
            causal, n_image_slots=N_IMAGE_SLOTS).eval()
        spec = model.build_export_spec(
            target_dtype=DTYPE, max_context_length=args.max_ctx,
            trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode in ("int8lin", "int4lin"):
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int4" if args.mode == "int4lin" else "int8"
        qscheme = "symmetric" if args.lin_sym else "symmetric_with_clipping"
        model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing (linear {dtype} per-block-32, {qscheme}, incl. untied head) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], linear_quant_config(dtype, qscheme))

    print("exporting decode-only VL engine graph to Core AI dialect ...")
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

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx)
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


if __name__ == "__main__":
    main()
