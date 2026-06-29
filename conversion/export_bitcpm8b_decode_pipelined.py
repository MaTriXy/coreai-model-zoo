"""Export a decode-pipelined BitCPM-CANN-8B (MiniCPM4-8B, 1.58-bit ternary) bundle.

zoo's first ternary model + first sub-int8 packed-GEMM Metal kernel. The 7 per-layer linears run
the fused 2-bit ternary matvec (``bitcpm_ternary_metal``); embed (Q4_K) + untied head (Q6_K) stay
fp16. Weights come from the TQ2_0 gguf (no self-quantization — the ternary IS the ship weight).

  # CPU parity (no GPU): generate greedily, eyeball coherence
  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/export_bitcpm8b_decode_pipelined.py --check --num-layers 32

  # smoke export (2 layers, GPU — grab _GPU_LOCK):
  ... export_bitcpm8b_decode_pipelined.py --num-layers 2
  # full export:
  ... export_bitcpm8b_decode_pipelined.py
GPU exclusivity: grab the community-repo _GPU_LOCK before the export path.
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from coreai_models.export._constants import (
    KEY_CACHE_NAME,
    QUANT_TRACE_OFFSET,
    QUANT_TRACE_QUERY_LEN,
    TRACE_KV_CACHE_SEQ_LEN,
    VALUE_CACHE_NAME,
)
from coreai_models.models.macos.bitcpm import load_bitcpm8b_from_gguf
from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.primitives.macos.cache import KVCache

GGUF = "/Users/majimadaisuke/code/coreai/_bitcpm_ckpt/bitcpm4-8b-tq2_0.gguf"
HF = "/Users/majimadaisuke/code/coreai/_bitcpm_ckpt/hf"
DTYPE = torch.float16


def build_kv_reference(cfg, max_ctx: int, static_ids: bool = False):
    """KV-only decode reference + dynamic shapes (mirrors nanbeige).

    static_ids=False (DEFAULT, Mac gate): dynamic input_ids -> the engine prefills the prompt in one
      call. FAST on Mac; on iPhone it respecializes per S=1 step (slow) -> ship `_s1` there later.
    static_ids=True: input_ids fixed [1,1] (iPhone-fast, chunkThreshold=1)."""
    if static_ids:
        input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
        position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)
        ids_dyn = None
        pos_dyn = {1: torch.export.Dim("seq_pos", min=2, max=max_ctx - 1)}
    else:
        input_ids = torch.randint(1, cfg.vocab_size, (1, QUANT_TRACE_QUERY_LEN), dtype=torch.int32)
        position_ids = torch.arange(
            QUANT_TRACE_QUERY_LEN + QUANT_TRACE_OFFSET, dtype=torch.int32).unsqueeze(0)
        ids_dyn = {1: torch.export.Dim("seq_ids", max=max_ctx - 2)}
        pos_dyn = {1: torch.export.Dim("seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_ctx - 1)}

    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved
    reference_inputs = {"input_ids": input_ids, "position_ids": position_ids,
                        "k_cache": k_cache, "v_cache": v_cache}
    dynamic_shapes = {
        "input_ids": ids_dyn,
        "position_ids": pos_dyn,
        "k_cache": {KVCache.seq_len_dim(): torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
        "v_cache": {KVCache.seq_len_dim(): torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)},
    }
    return reference_inputs, dynamic_shapes


@torch.no_grad()
def cpu_generate(model, cfg, prompt: str, new: int = 8):
    """Prefill + greedy decode via KVCache on CPU — eyeball arch+mup+ternary-kernel(torch_defn)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt").int()
    buf = max(ids.shape[1] + new + 8, TRACE_KV_CACHE_SEQ_LEN + 1)
    saved = cfg.max_position_embeddings
    cfg.max_position_embeddings = buf
    k_cache, v_cache = KVCache.create_cache_tensors(cfg, dtype=DTYPE)
    cfg.max_position_embeddings = saved
    out = []
    cur = ids
    pos0 = 0
    for step in range(new):
        T = cur.shape[1]
        position_ids = torch.arange(pos0 + T, dtype=torch.int32).unsqueeze(0)
        logits = model(cur, position_ids, k_cache, v_cache)
        nxt = int(logits[0, -1].float().argmax())
        out.append(nxt)
        print("  ->", nxt, repr(tok.decode([nxt])), flush=True)
        pos0 = pos0 + T
        cur = torch.tensor([[nxt]], dtype=torch.int32)
    print("GENERATION:", repr(tok.decode(out)), flush=True)


def write_bundle_metadata(out_dir: Path, name: str, cfg, max_ctx: int):
    meta = {"metadata_version": "0.2", "kind": "llm", "name": name,
            "assets": {"main": f"{name}.aimodel"},
            "language": {"tokenizer": "openbmb/BitCPM-CANN-8B", "vocab_size": cfg.vocab_size,
                         "max_context_length": max_ctx, "embedded_tokenizer": True,
                         "function_map": {"main": ["main"]}},
            "source": {"model_definition": "torch", "hf_model_id": "openbmb/BitCPM-CANN-8B"},
            "compression": None,
            "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []}}
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="CPU greedy parity (no export, no GPU)")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--num-layers", type=int, default=None, help="debug: truncated-layer build")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--static-ids", action="store_true",
                    help="fix input_ids [1,1] (iPhone-fast _s1); default dynamic-ids (Mac gate)")
    args = ap.parse_args()

    print(f"loading BitCPM-8B from gguf (layers={args.num_layers or 'all'}) ...", flush=True)
    model, kernel = load_bitcpm8b_from_gguf(GGUF, num_layers=args.num_layers, dtype=DTYPE)
    cfg = model.config
    print(f"hidden={cfg.hidden_size} layers={cfg.num_hidden_layers} q/kv="
          f"{cfg.num_attention_heads}/{cfg.num_key_value_heads} vocab={cfg.vocab_size} "
          f"scale_emb={cfg.scale_emb} resid={cfg.residual_scale:.4f} logit_div={cfg.logit_div}", flush=True)

    if args.check:
        cpu_generate(model, cfg, args.prompt)
        return

    name = "bitcpm_8b_decode_ternary" + ("_s1" if args.static_ids else "") + \
           (f"_l{args.num_layers}" if args.num_layers else "")
    reference_inputs, dynamic_shapes = build_kv_reference(cfg, args.max_ctx, static_ids=args.static_ids)

    print("exporting ternary decode graph to Core AI (custom kernel) ...", flush=True)
    prog = export_to_coreai_with_kernels(
        model, reference_inputs, custom_kernels=[kernel], dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids"), output_names=("logits",),
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
    write_bundle_metadata(out_dir, name, cfg, args.max_ctx)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(HF, trust_remote_code=True).save_pretrained(out_dir / "tokenizer")
    print(f"bundle ready: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
