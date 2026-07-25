"""Export MinerU2.5-Pro (opendatalab/MinerU2.5-Pro-2605-1.2B, Apache-2.0) for the
Core AI pipelined engine — the zoo's whole-page doc-OCR (2-stage host pipeline).

Two artifacts per run (mirrors export_qwen3_vl_pipelined.py, stock Qwen2-VL, minus deepstack):

* ``<name>/`` — the TEXT DECODER LanguageBundle on the pipelined-engine
  contract: dynamic-query ids/positions, ONE KV pair, logits out. Multimodal
  state rides the static-input hook: ``image_embeds [N,h]``,
  ``rope_shift_start [1]``, ``rope_shift_amount [1]``. Zero embeds +
  shift_start=1<<30 -> a plain Qwen2 text decoder.

* ``<name>_vision/`` — the fixed-grid VISION ENCODER ``.aimodel``:
  ``patches [n_patch, C*T*P*P] -> image_embeds [N, out_h]``, fp16.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_mineru_pipelined.py \
          [fp16|int8lin|int8hu] [--hf-id DIR] [--grid-h 22 --grid-w 31]

Modes follow the qwen3.5 recipe: int8lin = per-block-32 linear int8 body;
int8hu = + untied int8 head (absmax symmetric). Vision stays fp16 in all modes.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from _paths import work_path

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.mineru import (
    PIPELINED_STATE_NAMES,
    MinerUPipelinedForCausalLM,
    MinerUVisionEncoder,
)

DTYPE = torch.float16
DEFAULT_HF = str(work_path("_mineru_dl"))


def linear_quant_config(dtype: str = "int8") -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {
                "weight": {
                    "dtype": dtype,
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


def head_quant_spec() -> dict:
    return {
        "op_state_spec": {
            "weight": {
                "dtype": "int8",
                "qscheme": "symmetric",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1},
            }
        },
        "op_input_spec": None,
        "op_output_spec": None,
    }


def write_bundle_metadata(out_dir: Path, name: str, hf_id: str, vocab: int, max_ctx: int,
                          functions: tuple[str, ...] = ("main",)) -> None:
    meta = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {
            "tokenizer": hf_id,
            "vocab_size": vocab,
            "max_context_length": max_ctx,
            "embedded_tokenizer": True,
            "function_map": {"main": list(functions)},
        },
        "source": {"model_definition": "torch", "hf_model_id": hf_id},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _specs():
    return [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]


def export_vision(args, name: str) -> None:
    out_dir = Path(args.out_dir) / f"{name}_vision"
    print(f"loading vision tower ({args.hf_id}) ...")
    vis = MinerUVisionEncoder.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid_h, grid_w=args.grid_w)
    vcfg = vis.vcfg
    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size ** 2
    patches = torch.zeros(vis.n_patches, patch_dim, dtype=DTYPE)

    print("exporting vision graph ...")
    prog = export_to_coreai(
        vis,
        {"patches": patches},
        dynamic_shapes={"patches": None},
        input_names=("patches",),
        output_names=("image_embeds",),
        state_names=(),
        externalize_modules=_specs(),
    )
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt

    prog.save_asset(out_dir / f"{name}_vision.aimodel", rt.AIModelAssetMetadata())
    print(f"vision ready: {out_dir}  (N={vis.grid_h * vis.grid_w} tokens)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8lin", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--hf-id", default=DEFAULT_HF)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--grid-h", type=int, default=22, help="merged vision grid height")
    ap.add_argument("--grid-w", type=int, default=31, help="merged vision grid width")
    ap.add_argument("--skip-vision", action="store_true")
    ap.add_argument("--skip-decoder", action="store_true")
    ap.add_argument("--skip-dynamic", action="store_true")
    ap.add_argument("--skip-s1", action="store_true")
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="also export a multifunction bundle <name>_pf<N>: "
                         "'main' = static S=1 decode + 'prefill' = static S=N chunk, weights "
                         "shared. The engine feeds the prompt in S=N chunks (chunked prefill) "
                         "then 1 token/step for decode — ~4.7x prefill speedup, no token cut.")
    args = ap.parse_args()

    name = f"mineru_decode_{args.mode}"

    if not args.skip_vision:
        export_vision(args, "mineru")
    if args.skip_decoder:
        return

    print(f"loading {args.hf_id} text decoder fp16 ...")
    model = MinerUPipelinedForCausalLM.from_hf(
        args.hf_id, target_dtype=DTYPE, grid_h=args.grid_h, grid_w=args.grid_w)
    cfg = model.config

    spec = model.build_export_spec(DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec()}
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing ({args.mode}) ...")
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()),
            spec["dynamic_shapes"], cfg_q)

    import coreai.runtime as rt
    from transformers import AutoTokenizer

    variants = []
    if not args.skip_dynamic:
        variants.append((name, spec))
    if not args.skip_s1:
        variants.append((f"{name}_s1", model.build_export_spec(
            DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)))

    for vname, vspec in variants:
        print(f"exporting decoder graph ({vname}) ...")
        prog = export_to_coreai(
            model,
            vspec["reference_inputs"],
            dynamic_shapes=vspec["dynamic_shapes"],
            input_names=vspec["input_names"],
            output_names=vspec["output_names"],
            state_names=PIPELINED_STATE_NAMES,
            externalize_modules=_specs(),
        )
        print("optimizing ...")
        prog.optimize()

        out_dir = Path(args.out_dir) / vname
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        aimodel = out_dir / f"{vname}.aimodel"
        print(f"saving {aimodel} ...")
        prog.save_asset(aimodel, rt.AIModelAssetMetadata())
        write_bundle_metadata(out_dir, vname, args.hf_id, cfg.vocab_size, args.max_ctx)
        try:
            AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
        except Exception as e:  # noqa: BLE001
            print(f"  (tokenizer save skipped: {e})")
        print(f"bundle ready: {out_dir}")

    # -- chunked-prefill multifunction bundle (main S=1 + prefill S=N) ------------------
    if args.prefill_chunk:
        from coreai_models.export.macos import export_to_coreai_multifunction

        # coreai-torch bug workaround (from export_qwen3_vl_pipelined.py): a static S=N causal
        # SDPA emits a `key_seq >= N` guard inside the externalized submodule whose fallback
        # Dim(min=1) then violates it, and the body must stay dynamic. Retry the export with the
        # min/max bounds torch itself suggests in the ConstraintViolation message.
        import re as _re

        import coreai_torch.converter as _ct_conv
        from torch.export import Dim as _Dim

        _orig_export_module = _ct_conv._torch_export_module

        def _export_module_with_retry(prep):
            for _attempt in range(3):
                try:
                    return _orig_export_module(prep)
                except Exception as e:  # noqa: BLE001 — retry only on suggested fixes
                    fixes = {
                        name: (int(mn), int(mx) if mx else None)
                        for name, mn, mx in _re.findall(
                            r"(\w+) = Dim\('\w+', min=(\d+)(?:, max=(\d+))?\)", str(e))
                    }
                    if not fixes:
                        raise
                    print(f"[externalize-retry] {prep.name}: {fixes}")
                    rebuilt: dict[str, object] = {}

                    def _remap(dims):
                        if dims is None:
                            return None
                        out = {}
                        for j, d in dims.items():
                            name = getattr(d, "__name__", None)
                            if name not in fixes:
                                out[j] = d
                                continue
                            if name not in rebuilt:
                                mn, mx = fixes[name]
                                kwargs = {"min": mn}
                                if mx is not None:
                                    kwargs["max"] = mx
                                rebuilt[name] = _Dim(name, **kwargs)
                            out[j] = rebuilt[name]
                        return out

                    prep.dynamic_shapes = tuple(
                        _remap(dims) for dims in prep.dynamic_shapes)
            return _orig_export_module(prep)

        _ct_conv._torch_export_module = _export_module_with_retry

        pf = args.prefill_chunk
        vname = f"{name}_pf{pf}"
        print(f"exporting multifunction decoder (main S=1 + prefill S={pf}) ...")
        entries = [
            ("main", model.build_export_spec(
                DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)),
            ("prefill", model.build_export_spec(
                DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN,
                trace_query=pf, static_ids=True)),
        ]
        prog = export_to_coreai_multifunction(model, entries, externalize_modules=_specs())
        print("optimizing ...")
        prog.optimize()

        out_dir = Path(args.out_dir) / vname
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        aimodel = out_dir / f"{vname}.aimodel"
        print(f"saving {aimodel} ...")
        prog.save_asset(aimodel, rt.AIModelAssetMetadata())
        write_bundle_metadata(out_dir, vname, args.hf_id, cfg.vocab_size, args.max_ctx,
                              functions=("main", "prefill"))
        try:
            AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
        except Exception as e:  # noqa: BLE001
            print(f"  (tokenizer save skipped: {e})")
        print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
