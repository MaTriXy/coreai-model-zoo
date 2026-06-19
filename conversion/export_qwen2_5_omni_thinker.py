"""Export the Qwen2.5-Omni *Thinker* (audio-understanding decoder) to Core AI.

Follows the qwen3_vl recipe (conversion/export_qwen3_vl_pipelined.py):
  - int8lin quant via `quantize_pytorch_model` (symmetric_with_clipping,
    per-block-32 axis 1; Embedding/SDPA/RoPE/RMSNorm/lm_head kept full).
  - `externalize_modules` = the standard specs minus gated_delta_update.
  - Two graphs from the same quantized weights:
      <name>      dynamic-query  -> engine ship bundle (chunked prefill+decode)
      <name>_s1   static S=1     -> the PYTHON oracle-gate bundle (the python
                  runtime cannot execute dynamic-shaped outputs).
Bundle contract: input_ids, position_ids, audio_embeds + 2 KV states -> logits.
Audio placeholder ids = V+slot index the static audio_embeds.

Run: coreai-models/.venv/bin/python ondevice/export_qwen2_5_omni_thinker.py [--dynamic]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch

import coreai.runtime as rt
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.compression import quantize_pytorch_model
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen2_5_omni_thinker import (
    PIPELINED_STATE_NAMES,
    Qwen2_5OmniThinkerForCausalLM,
)

ROOT = Path(__file__).resolve().parents[1]
HF_ID = "Qwen/Qwen2.5-Omni-3B"
REF = ROOT / "_smoke/qwen2_5omni_ref.pt"
ART = Path(__file__).resolve().parent / "artifacts"
DTYPE = torch.float16
MAX_CTX = 4096


def linear_quant_config() -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": "int8",
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
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dynamic", action="store_true", help="also export the dynamic engine bundle")
    ap.add_argument("--n-audio", type=int, default=0,
                    help="audio-embed input width N; 0 = read from the oracle ref (=100, the 4 s "
                         "clip). Use the MAX (e.g. 750 = K*50 for the K=15 / 30 s encoder) for the "
                         "variable-length shippable bundle; the host zero-pads audio_embeds to N "
                         "and unreferenced rows are harmless.")
    args = ap.parse_args()

    ref = torch.load(REF, weights_only=False)
    n_audio = args.n_audio if args.n_audio > 0 else int(ref["n_audio_tokens"])
    # Tag the bundle name when N is decoupled from the oracle so the N=100 bundle
    # (the proven 4 s pair + python gate) is never clobbered.
    suffix = f"_n{n_audio}" if args.n_audio > 0 else ""

    print(f"loading {HF_ID} thinker (fp16, n_audio={n_audio}) ...")
    model = Qwen2_5OmniThinkerForCausalLM.from_hf(
        HF_ID, n_audio_tokens=n_audio, target_dtype=DTYPE, max_context_length=MAX_CTX
    ).eval()

    spec = model.build_export_spec(DTYPE, MAX_CTX, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
    print("quantizing (int8lin: symmetric per-block-32) ...")
    model = quantize_pytorch_model(
        model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"],
        linear_quant_config(),
    )

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    variants = [(f"qwen2_5_omni_3b_thinker_int8lin{suffix}_s1",
                 model.build_export_spec(DTYPE, MAX_CTX, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
                                         trace_query=1))]
    if args.dynamic:
        variants.append((f"qwen2_5_omni_3b_thinker_int8lin{suffix}", spec))

    for vname, vspec in variants:
        print(f"exporting decoder graph ({vname}) ...")
        prog = export_to_coreai(
            model,
            vspec["reference_inputs"],
            dynamic_shapes=vspec["dynamic_shapes"],
            input_names=vspec["input_names"],
            output_names=vspec["output_names"],
            state_names=PIPELINED_STATE_NAMES,
            externalize_modules=specs,
        )
        prog.optimize()
        out = ART / f"{vname}.aimodel"
        ART.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out, ignore_errors=True)
        prog.save_asset(out, rt.AIModelAssetMetadata())
        sz = subprocess.run(["du", "-sh", str(out)], capture_output=True, text=True).stdout.split()[0]
        print(f"SAVED {out} ({sz})")


if __name__ == "__main__":
    main()
