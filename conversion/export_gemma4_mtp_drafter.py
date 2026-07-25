# Community port — NOT an Apple model.
"""Export the Gemma-4 E2B MTP drafter (Section-11 transplant) as a Core AI bundle.

The drafter cross-attends the MAIN mixed-bit bundle's unified KV pair, so its
state contract is IDENTICAL (keyCache/valueCache [15,1,1,seq,512]) — the spec-decode
host loop binds the SAME state NDArrays to the main decode/verify functions and to
this graph (read-only here). See gemma4_mtp_drafter.py for the wiring notes and
coreai-models-community/knowledge/gemma4-mixedbit-qat-transplant.md (P5b/P5c).

Composition:
  * 4 tiny layers + pre/post proj, weights dequantized fp16 (13 MB)
  * lm_head [262144, 256] int4 -> the shipped affine int4 Metal kernel (33 MB)
  * embed = PackedInt2Embedding sharing the main table bytes (packed, in-graph)
  * static int8 activation fake-quant at the 18 QAT boundaries (part of the
    transplant: the drafter was trained behind these; fp16-only drafting measurably
    drops acceptance — the parity gate is tok/step vs the real tflite)

Gate first: _smoke/check_gemma4_mtp_drafter_parity_real.py (PASS 2026-07-03:
torch 2.043 tok/step vs tflite 2.000 on the sky prompt).

Run (coreai-models checkout):
  .venv/bin/python ../coreai-models-community/conversion/export_gemma4_mtp_drafter.py
Then AOT:
  xcrun coreai-build compile exports/gemma4_e2b_mtp_drafter/gemma4_e2b_mtp_drafter.aimodel \
    --output exports/gemma4_e2b_mtp_drafter --platform iOS --architecture h18p --expect-frequent-reshapes
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from _paths import code_path

from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.gemma4_metal_mlp_int2 import (
    MetalInt4AffLinear,
    build_fused_int4_kernel,
)
from coreai_models.models.macos.gemma4_mixedbit_pipelined import PackedInt2Embedding
from coreai_models.models.macos.gemma4_mtp_drafter import (
    HIDDEN,
    VOCAB,
    Gemma4MtpDrafter,
    load_transplant,
)

DTYPE = torch.float16
N_SLOTS = 15
HD_MAX = 512
DEFAULT_EXTRACT = str(code_path("litertlm-convert", "out", "gemma4e2b_extract"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", default=DEFAULT_EXTRACT)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--max-seq", type=int, default=1024,
                    help="STATIC cross-attention length (host mask handles validity)")
    args = ap.parse_args()

    name = "gemma4_e2b_mtp_drafter"
    model = Gemma4MtpDrafter()
    load_transplant(model, args.extract, dtype=DTYPE, fp_head=False)

    fw = safe_open(str(Path(args.extract) / "gemma4e2b_mixedbit_weights.safetensors"),
                   framework="pt")
    model.embed_tokens = PackedInt2Embedding(
        fw.get_tensor("embed.composite"), fw.get_tensor("embed.composite.scale"),
        HIDDEN, embed_scale=HIDDEN ** 0.5)
    k4 = build_fused_int4_kernel("gemma4_mtp_head_int4aff")
    model.lm_head = MetalInt4AffLinear(
        fw.get_tensor("drafter.lm_head"), fw.get_tensor("drafter.lm_head.scale"),
        VOCAB, 256, k4)
    model = model.to(DTYPE)
    for layer in model.layers:
        layer.inv_freq = layer.inv_freq.float()  # composite RoPE requires fp32 freqs
    model.eval()

    # FULLY STATIC graph: fixed-length slot inputs + explicit 0/1 masks + scalar pos.
    # (Dynamic narrows on inputs crash the iOS MPSGraph ViewOp path; a static graph
    # also specializes exactly once — no per-length respecialization in the loop.)
    max_seq = args.max_seq
    slot_shape = (1, 1, max_seq, HD_MAX)
    reference_inputs = {
        "input_ids": torch.randint(1, VOCAB, (1, 1), dtype=torch.int32),
        "hidden": torch.zeros(1, 1, HIDDEN, dtype=DTYPE),
        "pos": torch.tensor([[64]], dtype=torch.int32),
        "mask_sliding": torch.zeros(1, 1, 1, max_seq, dtype=DTYPE),
        "mask_full": torch.zeros(1, 1, 1, max_seq, dtype=DTYPE),
        "k11": torch.zeros(*slot_shape, dtype=DTYPE),
        "v11": torch.zeros(*slot_shape, dtype=DTYPE),
        "k14": torch.zeros(*slot_shape, dtype=DTYPE),
        "v14": torch.zeros(*slot_shape, dtype=DTYPE),
    }

    print("exporting drafter graph (fully static) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model,
        reference_inputs=reference_inputs,
        custom_kernels=[k4],
        dynamic_shapes=None,
        input_names=("input_ids", "hidden", "pos", "mask_sliding", "mask_full",
                     "k11", "v11", "k14", "v14"),
        output_names=("logits", "proj_hidden", "amax"),
        state_names=(),
        externalize_modules=(),
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())

    meta = {
        "metadata_version": "0.2",
        "kind": "llm-drafter",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "source": {
            "model_definition": "torch",
            "weights": "litert-community/gemma-4-E2B-it-litert-lm Section 11 (MTP drafter transplant)",
        },
        "drafter": {
            "draft_steps": 3,
            "max_seq": 1024,
            "shared_state_contract": "gemma4_e2b_mixedbit keyCache/valueCache [15,1,1,seq,512]",
        },
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
