"""Export North-Micro-Vision (Cohere `cohere_compass`, 2.4B) for the pipelined engine.

Two artifacts, the same shape the Qwen3-VL rider ships — because this
checkpoint's vision tower IS the Qwen3-VL tower at SigLIP2-SO400M dimensions
(verified: the zoo's encoder loads `model.visual.*` with zero missing keys and
reproduces every seam at cos 1.000000):

* ``<name>_vision/`` — the fixed-grid VISION ENCODER ``.aimodel``:
  ``patches [1024, 1536] -> (image_embeds [256, 2048], deepstack_embeds [768, 2048])``.
  A 512x512 canvas at patch 16 gives 32x32 patches, and the 2x2 merge makes 256
  tokens. The upstream processor is native-resolution (it kept the 640x480
  fixture's own 30x40 grid); baking a square grid is this export's choice and it
  stretches non-square images.

* ``<name>/`` — the TEXT DECODER LanguageBundle on the pipelined-engine contract
  with the multimodal state on the static-input hook:

  ``(input_ids [1,1] static, position_ids [1,total] dyn, image_embeds [256,h],
     deepstack_embeds [768,h], rope_shift_start [1], rope_shift_amount [1],
     keyCache/valueCache) -> logits``

  Image tokens are EXTENSION ids ``V + slot``; the first three layers add their
  deepstack rows at image positions. Positions follow the Qwen3-VL rope-shift
  contract (an image consumes only max(H,W) rope positions), which this
  checkpoint shares.

What is NOT Qwen3-VL is the decoder: a parallel Cohere block (one LayerNorm,
attention and MLP summed into the residual), mean-subtracting LayerNorm without
bias, `SSSF x 7` layer types where the 21 sliding layers carry interleaved
M-RoPE inside a 4096 window and the 7 full-attention layers have no positional
encoding at all, `logit_scale 0.25`, and a 262 144-entry embedding tied to the
head — 1.07 GB of fp16 that no linear quantization touches.

Numerics: gate the torch re-authoring FIRST —
``_smoke/test_northmv_torch_ladder.py --ref _smoke/north_micro_vision_instruct_ref_512x512.npz``
(vision cos 1.000000 at every seam, full chain cos 1.000000 + token-exact).
The oracle needs transformers git main; 5.15.0 does not know `cohere_compass`.

Run:  python export_northmv_pipelined.py [fp16|int8lin|int4lin] \
          [--hf-id CohereLabs/North-Micro-Vision-Instruct] [--grid 16]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import write_bundle_metadata
from _paths import exports_dir

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.cohere_compass import (
    PIPELINED_STATE_NAMES,
    CohereCompassPipelinedForCausalLM,
    CohereCompassTextOnlyForCausalLM,
    vision_encoder_from_hf,
)

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8", block: int = 32) -> dict:
    """Weight-only linear per-block, the zoo's ship recipe.

    Norms and the embedding stay high precision by type; `lm_head` by name
    because it is TIED — the eager quantizer skips shared parameters anyway, so
    naming it keeps the intent visible rather than relying on that.
    """
    spec = {
        "op_state_spec": {
            "weight": {
                "dtype": dtype,
                "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": block, "axis": 1},
            }
        },
        "op_input_spec": None,
        "op_output_spec": None,
    }
    return {
        "execution_mode": "eager",
        "global_config": spec,
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "torch.nn.modules.normalization.LayerNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def export_vision(args, short: str, out_root: Path) -> None:
    name = f"{short}_vision_fp16"
    out_dir = out_root / name
    print(f"loading vision tower ({args.hf_id}) ...")
    vis = vision_encoder_from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid, grid_w=args.grid
    )
    vcfg = vis.vcfg
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size**2
    patches = torch.zeros(vis.n_patches, patch_dim, dtype=DTYPE)
    print(f"  merged grid {args.grid}x{args.grid} -> {vis.n_patches} patches "
          f"-> {args.grid ** 2} tokens, hidden {vcfg.hidden_size} x {vcfg.depth}L")

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("exporting vision graph ...")
    prog = export_to_coreai(
        vis,
        {"patches": patches},
        dynamic_shapes={"patches": None},
        input_names=("patches",),
        output_names=("image_embeds", "deepstack_embeds"),
        state_names=(),
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
    print(f"vision ready: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int4lin"])
    ap.add_argument("--hf-id", default="CohereLabs/North-Micro-Vision-Instruct")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--grid", type=int, default=16,
                    help="merged vision grid side (16 = a 512x512 canvas, 256 tokens)")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--text-core", action="store_true",
                    help="export the decoder WITHOUT the image inputs (suffix _textcore): "
                         "the Mac speed proxy, because llm-runner cannot bind an image buffer")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    out_root = Path(args.out_dir) if args.out_dir else exports_dir()

    if not args.skip_vision:
        export_vision(args, short, out_root)
    if args.skip_decoder:
        return

    name = f"{short}_decode_{args.mode}{'_textcore' if args.text_core else ''}"
    cls = CohereCompassTextOnlyForCausalLM if args.text_core else CohereCompassPipelinedForCausalLM
    print(f"loading {args.hf_id} text decoder fp16 ...")
    model = cls.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid, grid_w=args.grid
    )
    cfg = model.config
    n_sliding = cfg.n_layers_with_rope
    print(f"{cfg.num_hidden_layers} layers ({n_sliding} sliding / "
          f"{cfg.num_hidden_layers - n_sliding} full-attention NoPE), "
          f"ff={cfg.intermediate_size}, vocab={cfg.vocab_size}, "
          f"image tokens={model.n_image_tokens}")

    spec = model.build_export_spec(
        DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1
    )

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        dtype = "int4" if args.mode == "int4lin" else "int8"
        print(f"quantizing (linear {dtype} per-block-{args.block}) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"],
            linear_quant_config(dtype, args.block),
        )

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print(f"exporting decoder graph ({name}) ...")
    prog = export_to_coreai(
        model,
        spec["reference_inputs"],
        dynamic_shapes=spec["dynamic_shapes"],
        input_names=spec["input_names"],
        output_names=spec["output_names"],
        state_names=PIPELINED_STATE_NAMES,
        externalize_modules=specs,
    )
    print("optimizing ...")
    prog.optimize()

    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt
    from transformers import AutoTokenizer

    prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
    write_bundle_metadata(
        out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
        mode=None if args.mode == "fp16" else args.mode,
        language_extra=None if args.text_core else {
            "image_tokens": model.n_image_tokens,
            "image_patch_grid": [args.grid * 2, args.grid * 2],
        },
    )
    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
