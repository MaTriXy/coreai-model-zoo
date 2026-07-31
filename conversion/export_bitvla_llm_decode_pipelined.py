"""Export a decode-pipelined BitVLA language model (BitNet b1.58 2B4T, 1.58-bit ternary) bundle.

The autoregressive action-token generator. The 7 per-layer linears run the generalized fused
2-bit ternary matvec (``bitnet_ternary_metal``); lm_head stays fp16. Input is **inputs_embeds**
[1,1,2560] (host-side embed lookup + spliced 256 vision embeds), so the graph injects image
features. Static-ids S=1 (M=1 ternary kernel is decode-only; prefill = loop one position).

  # smoke export (2 layers, GPU — grab _GPU_LOCK):
  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_bitvla_llm_decode_pipelined.py --num-layers 2
  # full export:
  ... export_bitvla_llm_decode_pipelined.py
GPU exclusivity: grab the community-repo _GPU_LOCK before the export path.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from _bundle import write_bundle_metadata
from _paths import work_path

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.models.macos.bitvla_llm import load_bitvla_llm
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.primitives.macos.cache import KVCache

CK = str(work_path("_bitvla_ckpt", "bitvla_bf16", "model.safetensors"))
DTYPE = torch.float16
HIDDEN = 2560


def build_reference(cfg, max_ctx: int):
    """inputs_embeds [1,1,H] static (S=1); position_ids + KV cache dynamic over context."""
    inputs_embeds = torch.randn(1, 1, HIDDEN, dtype=DTYPE)
    position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)

    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved

    reference_inputs = {"inputs_embeds": inputs_embeds, "position_ids": position_ids,
                        "k_cache": k_cache, "v_cache": v_cache}
    dynamic_shapes = {
        "inputs_embeds": None,                                    # fixed [1,1,H]
        "position_ids": {1: torch.export.Dim("seq_pos", min=2, max=max_ctx - 1)},
        "k_cache": {KVCache.seq_len_dim(): torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        "v_cache": {KVCache.seq_len_dim(): torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
    }
    return reference_inputs, dynamic_shapes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--num-layers", type=int, default=None)
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--action-head", action="store_true",
                    help="slice lm_head to the 256 action rows (656MB->1.3MB; device decode)")
    args = ap.parse_args()

    print(f"loading BitVLA LLM (layers={args.num_layers or 'all'}, action_head={args.action_head}) ...", flush=True)
    model, kernel, _ = load_bitvla_llm(CK, num_layers=args.num_layers, dtype=DTYPE,
                                       action_head=args.action_head)
    cfg = model.config
    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} q/kv="
          f"{cfg.num_attention_heads}/{cfg.num_key_value_heads} vocab={cfg.vocab_size}", flush=True)

    name = "bitvla_llm_decode_ternary_s1" + ("_act" if args.action_head else "") + \
           (f"_l{args.num_layers}" if args.num_layers else "")
    reference_inputs, dynamic_shapes = build_reference(cfg, args.max_ctx)

    print("exporting ternary decode graph to Core AI (custom kernel) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs, custom_kernels=[kernel], dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"), output_names=("logits",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME),
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
    write_bundle_metadata(out_dir, name, "lxsy/bitvla-bf16", cfg.vocab_size, args.max_ctx,
                          embedded_tokenizer=False,
                          language_extra={"input": "inputs_embeds[1,1,2560]"})
    print(f"bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
