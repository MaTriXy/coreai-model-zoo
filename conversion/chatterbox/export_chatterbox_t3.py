"""Export the Chatterbox T3 (AR speech-token model) as an EMBEDS-IN Core AI decode graph.

T3 is a plain-Llama backbone driven by custom host-assembled input embeddings, so the
graph takes `inputs_embeds [b,q,1024]` (not input_ids) + position_ids + KV cache -> speech
logits [b,q,8194]. Adapted from the nanbeige plain-Llama export; the host does the
cond-prefix / text / speech embed assembly + CFG + sampling (like Stable Audio / VoxCPM,
NOT the ChatSession engine). fp32-parity of the overlay is proven (cosine 0.999999 vs the
chatterbox T3). This produces the int8 body bundle; gate with the python coreai runtime.

Run: cd ~/code/coreai/coreai-models && .venv/bin/python \
       ../coreai-models-community/conversion/chatterbox/export_chatterbox_t3.py [--num-layers N]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.export.macos import export_to_coreai
from coreai_models.models.macos.chatterbox_t3 import chatterbox_t3_from_pretrained
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16
HID = 1024


def linear_quant_config(dtype: str = "int8") -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {"dtype": dtype, "qscheme": "symmetric_with_clipping",
                              "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
        },
        "module_name_configs": {r".*speech_head$": None},  # head stays fp16
    }


def build_reference(cfg, max_ctx: int):
    inputs_embeds = torch.randn(1, QUANT_TRACE_QUERY_LEN, HID, dtype=DTYPE)
    position_ids = torch.arange(QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32).unsqueeze(0)
    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved
    reference_inputs = {"inputs_embeds": inputs_embeds, "position_ids": position_ids,
                        "k_cache": k_cache, "v_cache": v_cache}
    dynamic_shapes = {
        "inputs_embeds": {1: torch.export.Dim("seq_ids", max=max_ctx - 2)},
        "position_ids": {1: torch.export.Dim("seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_ctx - 1)},
        "k_cache": {KVCache.seq_len_dim(): torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        "v_cache": {KVCache.seq_len_dim(): torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
    }
    return reference_inputs, dynamic_shapes


def main() -> None:
    from huggingface_hub import snapshot_download
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8", choices=["fp16", "int8"])
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--num-layers", type=int, default=None)
    args = ap.parse_args()

    snap = snapshot_download("ResembleAI/chatterbox")
    name = f"chatterbox_t3_decode_{args.mode}" + (f"_l{args.num_layers}" if args.num_layers else "")
    print(f"loading T3 fp16 from {snap} ...", flush=True)
    model = chatterbox_t3_from_pretrained(snap, target_dtype=DTYPE)
    if args.num_layers is not None:
        model.model.layers = model.model.layers[: args.num_layers]
        model.config.num_hidden_layers = args.num_layers
    model.eval()
    cfg = model.config
    print(f"T3 | {cfg.num_hidden_layers}L hidden={cfg.hidden_size} heads={cfg.num_attention_heads} "
          f"speech_vocab={cfg.speech_vocab_size}", flush=True)

    reference_inputs, dynamic_shapes = build_reference(cfg, args.max_ctx)

    if args.mode == "int8":
        from coreai_models.export.compression import quantize_pytorch_model
        print("quantizing body int8 per-block-32 (speech_head fp16) ...", flush=True)
        model = quantize_pytorch_model(model, tuple(reference_inputs.values()),
                                       dynamic_shapes, linear_quant_config("int8"))

    print("exporting embeds-in decode graph ...", flush=True)
    prog = export_to_coreai(
        model, reference_inputs, dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"), output_names=("logits",),
        state_names=(KEY_CACHE_NAME, VALUE_CACHE_NAME))
    prog.optimize()

    out_dir = Path(args.out_dir) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    print(f"bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
