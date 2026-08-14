"""Export LFM2.5-VL (450M / 3B) for the Core AI pipelined engine — two graphs.

* ``<name>_vision/`` — the fixed-grid VISION ENCODER ``.aimodel``:
  ``patches [1024, 768] -> image_embeds [256, text_hidden]``. SigLIP2 NaFlex:
  the host patchifies a 512x512 tile into 16x16x3 flattened patches (see
  ``_smoke/lfm25vl_preprocess.py``), the tower's patch embedding is a LINEAR,
  and the 16x16 position table is bilinearly resized to the 32x32 grid at load
  time, so nothing positional survives into the graph. 32x32 patches -> 2x
  pixel-unshuffle -> 256 tokens = the checkpoint's own ``max_image_tokens``.

* ``<name>/`` — the TEXT DECODER LanguageBundle on the pipelined-engine
  contract, which is the SHIPPED LFM2 decoder plus one static input:

  ``(input_ids [1,1] static, position_ids [1,total] dyn, image_embeds [256,h],
     keyCache/valueCache/convState) -> logits``

  The host rewrites the prompt's ``<image>`` ids (id 396) to EXTENSION ids
  ``V + slot``, slot 0..255 in the encoder's row-major token order; in-graph
  ``embedding = ids < V ? embed_tokens[ids] : image_embeds[ids - V]``. With
  zero image embeds and no extension ids the graph IS the LFM2 text decoder,
  so ``llm-benchmark`` runs it unchanged.

  Prefill runs as pipelined S=1 steps (``COREAI_CHUNK_THRESHOLD=1``), the same
  contract the shipped LFM2.5-1.2B / 2.6B decode bundles use.

Numerics: gate the torch re-authoring FIRST —
``_smoke/test_lfm25vl_torch_ladder.py --ref _smoke/lfm2_5_vl_450m_ref_512x512.npz``
(vision ladder cos 1.000000 at every seam + 48/48 token-exact greedy against
the fp32 oracle). The ``.aimodel`` gate is ``_smoke/test_lfm25vl_aimodel_gate.py``.

Requires the lfm2 + lfm2_vl model overlay on ``coreai-models`` and the
pipelined-engine extra-states patch to RUN the decoder bundle (the conv state
rides as a fixed-shape extra state).

Run:  python export_lfm25vl_pipelined.py [fp16|int8lin|int8hu|int4lin] \
          [--hf-id LiquidAI/LFM2.5-VL-450M] [--vision-mode fp16|int8lin]

Modes are the LFM2 ship recipe (see export_lfm2_decode_pipelined.py): the
attention projections and the embedding stay high precision, the MLP and
conv-mixer linears carry the quantization. ``--vision-mode int8lin`` halves the
tower's weight bandwidth, which is the dominant term in a VLM's time-to-first-
token; the tower's own numerics are gated separately.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata
from _paths import exports_dir
from export_lfm2_decode_pipelined import linear_quant_config, save_tokenizer

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.lfm2 import DECODE_STATE_NAMES
from coreai_models.models.macos.lfm2_vl import (
    Lfm2VlPipelinedForCausalLM,
    Lfm2VlVisionEncoder,
    lfm2_text_core_from_hf,
)

DTYPE = torch.float16
GRID = 32  # patch grid: one 512x512 tile at patch 16 -> 32x32 -> 256 tokens


def vision_quant_config(block: int) -> dict:
    """int8 per-block on the tower + projector Linears; norms stay fp16.

    ``block`` is 32 where every Linear's in_features divides it and 16
    otherwise — the 3B tower's 4304-wide MLP intermediate is the case that
    forces 16 (the same rule the MiniCPM-V SigLIP export ran into).
    """
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": "int8",
                    "qscheme": "symmetric_with_clipping",
                    "granularity": {"type": "per_block", "block_size": block, "axis": 1},
                }
            },
            "op_input_spec": None,
            "op_output_spec": None,
        },
        "module_type_configs": {
            "torch.nn.modules.normalization.LayerNorm": None,
            "coreai_models.primitives.macos.sdpa.SDPA": None,
        },
    }


def vision_block_size(vis: Lfm2VlVisionEncoder) -> int:
    dims = {
        vis.vcfg.patch_dim,
        vis.vcfg.hidden_size,
        vis.vcfg.intermediate_size,
        vis.vcfg.hidden_size * vis.vcfg.downsample_factor**2,
        vis.vcfg.projector_hidden_size,
    }
    return 32 if all(d % 32 == 0 for d in dims) else 16


def export_vision(args, short: str, out_root: Path) -> None:
    name = f"{short}_vision_{args.vision_mode}"
    out_dir = out_root / name
    print(f"loading vision tower ({args.hf_id}) ...")
    vis = Lfm2VlVisionEncoder.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=GRID, grid_w=GRID
    )
    patches = torch.zeros(vis.n_patches, vis.vcfg.patch_dim, dtype=DTYPE)
    print(
        f"  grid {GRID}x{GRID} -> {vis.n_patches} patches -> {vis.n_tokens} tokens, "
        f"hidden {vis.vcfg.hidden_size} x {vis.vcfg.num_hidden_layers}L"
    )

    if args.vision_mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model

        block = vision_block_size(vis)
        print(f"quantizing vision (linear int8 per-block-{block}) ...")
        vis = quantize_pytorch_model(
            vis, (patches,), {"patches": None}, vision_quant_config(block)
        )

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("exporting vision graph ...")
    prog = export_to_coreai(
        vis,
        {"patches": patches},
        dynamic_shapes={"patches": None},
        input_names=("patches",),
        output_names=("image_embeds",),
        state_names=(),
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"vision ready: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "mode", nargs="?", default="int8lin",
        choices=["fp16", "int8lin", "int8hu", "int4lin"]
    )
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-VL-450M")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--vision-mode", default="fp16", choices=["fp16", "int8lin"])
    ap.add_argument("--int4-block", type=int, default=32)
    ap.add_argument("--int8-block", type=int, default=32,
                    help="per-block granularity for int8 (16 halves the per-weight "
                         "range error at ~+6%% size; the 450M decoder is small enough "
                         "that block-32 int8 measurably moves the logits)")
    ap.add_argument("--head-sym", action="store_true",
                    help="int8hu only: absmax (no clipping) for the untied head")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument(
        "--text-core",
        action="store_true",
        help="export the decoder WITHOUT the image_embeds input (suffix _textcore): "
        "the same weights as a plain LFM2 text bundle. llm-benchmark cannot bind "
        "the VLM bundle's image buffer, so this is the Mac speed proxy — the same "
        "one the MiniCPM-V-4.6 card documents — and a usable text model besides.",
    )
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    out_root = Path(args.out_dir) if args.out_dir else exports_dir()

    if not args.skip_vision:
        export_vision(args, short, out_root)
    if args.skip_decoder:
        return

    suffix = "_textcore" if args.text_core else ""
    name = f"{short}_decode_{args.mode}{suffix}{args.tag}"
    n_tokens = (GRID // 2) ** 2

    print(f"loading {args.hf_id} text decoder fp16 ...")
    if args.text_core:
        model = lfm2_text_core_from_hf(args.hf_id, target_dtype=DTYPE)
        spec = model.build_macos_export_spec(
            DTYPE, args.max_ctx, query_len=1, offset=64,
            trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
        )
        spec["dynamic_shapes"]["input_ids"] = None  # static [1,1] decode query
    else:
        model = Lfm2VlPipelinedForCausalLM.from_hf(
            args.hf_id, target_dtype=DTYPE, n_image_tokens=n_tokens
        )
        spec = model.build_export_spec(
            DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1
        )
    cfg = model.config
    print(
        f"{cfg.num_hidden_layers} layers ({cfg.num_full_layers} full / "
        f"{cfg.num_conv_layers} conv), ff_dim={cfg.ff_dim}, vocab={cfg.vocab_size}"
        + ("" if args.text_core else f", image tokens={n_tokens}")
    )

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int4" if args.mode == "int4lin" else "int8"
        block = args.int4_block if dtype == "int4" else 32
        cfg_q = linear_quant_config(dtype, block=block)
        if args.mode == "int8hu":
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(
                "block32", args.head_sym
            )
            model.lm_head.weight = torch.nn.Parameter(
                model.lm_head.weight.detach().clone()
            )
        print(f"quantizing (linear {dtype} per-block-{block}, mode={args.mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q
        )

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print(f"exporting decoder graph ({name}) ...")
    prog = export_to_coreai(
        model,
        spec["reference_inputs"],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=DECODE_STATE_NAMES,
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(
        out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
        mode=None if args.mode == "fp16" else args.mode,
        language_extra=None if args.text_core else {
            "image_tokens": n_tokens, "image_patch_grid": [GRID, GRID]
        },
    )
    save_tokenizer(args.hf_id, out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")
    print(
        f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3"
    )


if __name__ == "__main__":
    main()
