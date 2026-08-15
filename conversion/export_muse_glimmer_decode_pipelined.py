"""Export a decode-pipelined Muse-Glimmer-30B text bundle for the Core AI engine.

Muse-Glimmer-30B (`meta-models/Muse-Glimmer-30B`, HF `model_type: muse_glimmer`)
is a 30B image-text-to-text VLM; this exports the TEXT decoder only (the vision
tower is dropped in `_mutate_state_dict`). The text tower is dense plain-KV
attention — no MoE, no SSM — so it rides the same KV-only decode path as the
Nanbeige/Llama archetype this is trimmed from. What is *not* textbook:

  * `sliding(2048) x 3 + full x 1` layer pattern where the FULL layers are NoPE
    (`layer_rope_theta[i] == 0`); only the sliding layers carry rotary.
  * A sigmoid output gate per attention layer, fused with Q/K/V into one
    projection in the re-authored module (`qkvg_proj`).
  * Weight-less RMSNorm on the embedding output and on Q/K, and two epsilons
    across the sandwich norms (1e-5 pre, 1e-8 post).
  * `qk_scale_factor` folded into the SDPA scale; logits pre-scaled by
    `output_multiplier` then tanh-softcapped at 20.

Big untied head: vocab 202048 x 6656 = 2.7 GB in fp16, `tie_word_embeddings:
false`. The `*hu` modes quantize it with plain `symmetric` (absmax) via
`--head-sym` — big-vocab heads are fat-tailed and `symmetric_with_clipping`
craters the outlier rows.

Size: the text tower is ~26.4 B params, so fp16 is ~53 GB in RAM and int4
lands near 15 GB of weights. Mac-only by construction. The quantizer runs with
`mmap_dir` so the finalized tensors are disk-backed; keep >= 80 GB free.

Run:  cd ~/code/coreai/coreai-models && .venv/bin/python \
          ../coreai-models-community/conversion/export_muse_glimmer_decode_pipelined.py \
          int4hu --head-sym --static-ids
      # smoke first:  ... int8hu --head-sym --num-layers 4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from _bundle import head_quant_spec, save_tokenizer, write_bundle_metadata

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.muse_glimmer import MuseGlimmerForCausalLM
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16


def linear_quant_config(dtype: str = "int8") -> dict:
    """Weight-only linear int8/int4 per-block-32 (scale-multiply dequant, no LUT).

    Norms/embedding/SDPA/RoPE excluded; `lm_head` excluded by name (stays fp16
    unless a `*hu` mode adds an explicit head spec). `RMSNormPlusOne` is the
    centered norm this model uses for all four sandwich norms, and
    `WeightlessRMSNorm` carries no parameters at all — both are listed so the
    exclusion is readable rather than incidental.
    """
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
            "coreai_models.models.macos.muse_glimmer.WeightlessRMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*lm_head$": None},
    }


def build_kv_reference(cfg, max_ctx: int, static_ids: bool = False):
    """KV-only reference inputs + dynamic shapes.

    static_ids=True fixes `input_ids` at [1, 1] (the qwen3_5 loop-free device
    pattern). With chunkThreshold=1 every step is S=1 anyway, so nothing is lost
    at decode, and it removes the per-step input_ids re-specialization that is
    pathological on a bundle this size. `position_ids` and the KV stay dynamic.
    """
    if static_ids:
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)  # trace_past 64 + 1
        ids_dyn = None
        pos_dyn = {1: torch.export.Dim("seq_pos", min=2, max=max_ctx - 1)}
    else:
        input_ids = torch.randint(1, cfg.vocab_size, (1, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
        position_ids = torch.arange(
            QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32
        ).unsqueeze(0)
        ids_dyn = {1: torch.export.Dim("seq_ids", max=max_ctx - 2)}
        pos_dyn = {1: torch.export.Dim("seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_ctx - 1)}

    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved

    reference_inputs = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }
    dynamic_shapes = {
        "input_ids": ids_dyn,
        "position_ids": pos_dyn,
        "k_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx
            )
        },
        "v_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx
            )
        },
    }
    return reference_inputs, dynamic_shapes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "mode", nargs="?", default="int4hu",
        choices=["fp16", "int8lin", "int8hu", "int4lin", "int4hu"],
    )
    ap.add_argument("--hf-id", default="meta-models/Muse-Glimmer-30B")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=8192)
    ap.add_argument(
        "--head-quant", default="block32", choices=["block32", "block16", "block8", "perchan"]
    )
    ap.add_argument(
        "--head-sym", action="store_true",
        help="*hu modes: plain symmetric (absmax, no clipping) for the 202048-vocab head",
    )
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer export")
    ap.add_argument(
        "--static-ids", action="store_true",
        help="fix input_ids at [1, 1] — the device decode pattern (see build_kv_reference)",
    )
    ap.add_argument(
        "--no-quant-mmap", action="store_true",
        help="keep quantized tensors in RAM (debug/truncated only)",
    )
    args = ap.parse_args()

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_decode_{args.mode}"
    if args.mode in ("int8hu", "int4hu") and (args.head_quant != "block32" or args.head_sym):
        name += f"_{args.head_quant}" + ("_sym" if args.head_sym else "")
    if args.static_ids:
        name += "_s1"
    if args.num_layers is not None:
        name += f"_l{args.num_layers}"

    print(f"loading {args.hf_id} fp16 (memory-efficient) ...", flush=True)
    model = MuseGlimmerForCausalLM.from_hf_memory_efficient(
        args.hf_id,
        max_context_length=args.max_ctx,
        target_dtype=DTYPE,
        num_layers=args.num_layers,
        hf_config_attr="text_config",
    )
    model.eval()
    cfg = model.config
    print(
        f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} "
        f"q/kv={cfg.num_attention_heads}/{cfg.num_key_value_heads} vocab={cfg.vocab_size} "
        f"tied={cfg.tie_word_embeddings} window={cfg.sliding_window}",
        flush=True,
    )
    nope = sum(1 for t in cfg.layer_rope_theta if not t)
    print(f"NoPE layers: {nope}/{cfg.num_hidden_layers}", flush=True)

    reference_inputs, dynamic_shapes = build_kv_reference(
        cfg, args.max_ctx, static_ids=args.static_ids
    )

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    quant_mmap = None

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        base = "int4" if "int4" in args.mode else "int8"
        cfg_q = linear_quant_config(base)
        if args.mode in ("int8hu", "int4hu"):
            cfg_q["module_name_configs"][r".*lm_head$"] = head_quant_spec(
                args.head_quant, args.head_sym
            )
            # The eager quantizer skips shared params; this head is untied, but
            # clone anyway so it is quantized in place regardless of load path.
            model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
        if not args.no_quant_mmap:
            quant_mmap = out_dir / "_quant_mmap"
            quant_mmap.mkdir()
        print(
            f"quantizing (linear {base} per-block-32, mode={args.mode}, mmap={quant_mmap}) ...",
            flush=True,
        )
        model = quantize_pytorch_model(
            model,
            tuple(reference_inputs.values()),
            dynamic_shapes,
            cfg_q,
            mmap_dir=str(quant_mmap) if quant_mmap else None,
        )

    print("exporting decode graph to Core AI dialect ...", flush=True)
    prog = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"),
        output_names=("logits",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt

    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())

    write_bundle_metadata(out_dir, name, args.hf_id, cfg.vocab_size, args.max_ctx, mode=args.mode)
    # `via_transformers=False`: the checkpoint declares `tokenizer_class:
    # TokenizersBackend`, a transformers-5 concept the pinned 4.57 export venv
    # cannot instantiate — the raw-file copy is the only path that runs, and it
    # also keeps `chat_template.jinja` verbatim.
    save_tokenizer(args.hf_id, out_dir, via_transformers=False)
    if quant_mmap is not None:
        shutil.rmtree(quant_mmap, ignore_errors=True)
    print(f"bundle ready: {out_dir}", flush=True)
    print(
        f"run: COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model {out_dir} -p 128 -g 256 -n 3",
        flush=True,
    )


if __name__ == "__main__":
    main()
