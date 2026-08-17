"""Export the Qwen3.8 MTP-head drafter (S-token stateful) for spec-decode.

The Qwen3.8 checkpoints ship a trained 1-layer DeepSeek-V3-style drafter under
`mtp.*` (transformers 5.x ignores it). Graph per step:
    x = fc(concat(RMSNorm_emb(embed(token)), RMSNorm_hid(hidden_in)))
    x = decoder_layer(x)          # gated full attention + SwiGLU (target family)
    logits = lm_head(norm(x))     # head weight shared with the target
    outputs: logits, hidden_out (recurrent hidden for the next draft step)
Inputs: input_ids [1,S] static, hidden_in [1,S,hidden] static (the target
verify bundle's `hidden` output rows — export the target with --emit-hidden),
position_ids [1,seq] dynamic; state: 1-layer KV (keyCache/valueCache), dynamic
seq dim like the decode exports. One graph serves the fresh-KV v1 loop AND the
committed-context v2 loop (persistent KV + host replay of committed rows).

Conventions (concat order, recurrent hidden, which target hidden) were gated
empirically in ondevice/_mtp_alpha_probe.py — pass the winners; they are
recorded in the bundle metadata for the host.

Run:  python export_qwen3_8_mtp.py [fp16|int8hu] --hf-id Qwen/Qwen3.8-27B \
          [--hidden-source post] [--concat-order eh] [--recurrent-hidden prenorm]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import head_quant_spec, write_bundle_metadata

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_8_mtp import (
    MTP_STATE_NAMES,
    Qwen3_8MTPDrafter,
)

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear per-block-32 — identical to the target ship recipe."""
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
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", nargs="?", default="int8hu", choices=["fp16", "int8hu"])
    ap.add_argument("--hf-id", default="Qwen/Qwen3.8-27B")
    ap.add_argument("--s", type=int, default=1,
                    help="static query length (1 = draft/replay step; >1 batches "
                         "committed-row replay)")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--head-quant", default="block32",
                    choices=["block32", "block16", "block8", "perchan"])
    ap.add_argument("--head-sym", action="store_true", default=True,
                    help="plain symmetric (absmax) head — 27B ship shape (default)")
    ap.add_argument("--concat-order", default="eh", choices=["eh", "he"])
    ap.add_argument("--recurrent-hidden", default="postnorm",
                    choices=["prenorm", "postnorm"])
    ap.add_argument("--hidden-source", default="post", choices=["pre", "post"],
                    help="which target hidden the probe gated this drafter with "
                         "(metadata only — the host must feed a matching verify "
                         "bundle exported with --emit-hidden <this>)")
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_mtp_s{args.s}_{args.mode}"
    if args.mode == "int8hu":
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")

    print(f"loading MTP head from {args.hf_id} ...")
    model = Qwen3_8MTPDrafter.from_checkpoint(args.hf_id, target_dtype=DTYPE)
    model.eval()
    model.concat_order = args.concat_order
    model.recurrent_hidden = args.recurrent_hidden
    cfg = model.config

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, args.s), dtype=torch.int32)
    hidden_in = torch.zeros(1, args.s, cfg.hidden_size, dtype=DTYPE)
    position_ids = torch.arange(trace_past + args.s, dtype=torch.int32).unsqueeze(0)
    state = model.build_kv_state(TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)

    reference_inputs = {
        "input_ids": input_ids,
        "hidden_in": hidden_in,
        "position_ids": position_ids,
        "k_cache": state["k_cache"],
        "v_cache": state["v_cache"],
    }
    seq_pos = torch.export.Dim("seq_pos", min=args.s, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None,
        "hidden_in": None,
        "position_ids": {1: seq_pos},
        "k_cache": {3: k_seq},
        "v_cache": {3: v_seq},
    }

    if args.mode == "int8hu":
        from coreai_models.export.compression import quantize_pytorch_model

        cfg_q = linear_quant_config("int8")
        cfg_q["module_name_configs"] = {
            r".*lm_head$": head_quant_spec(args.head_quant, args.head_sym)}
        model.lm_head.weight = torch.nn.Parameter(
            model.lm_head.weight.detach().clone())
        print("quantizing (linear per-block-32 int8, sym int8 head) ...")
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes, cfg_q)

    # Decomposed in-graph SDPA (the externalized composite's lower-right mask
    # emits a k_len >= S guard that breaks at static S — same as the verify
    # export); no GDN in this graph.
    specs = [s for s in _EXTERNALIZE_SPECS
             if s.composite_op_name not in ("gated_delta_update",
                                            "scaled_dot_product_attention")]

    print(f"exporting static S={args.s} MTP drafter graph ...")
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "hidden_in", "position_ids"),
        output_names=("logits", "hidden"),
        state_names=MTP_STATE_NAMES,
        externalize_modules=specs,
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

    write_bundle_metadata(
        out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx,
        extra={"mtp_query_len": args.s, "concat_order": args.concat_order,
               "recurrent_hidden": args.recurrent_hidden,
               "hidden_source": args.hidden_source})
    from transformers import AutoTokenizer

    AutoTokenizer.from_pretrained(args.hf_id).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}")


if __name__ == "__main__":
    main()
