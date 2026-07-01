"""P3.5: export ONE unified Qwen3-ASR bundle with two entrypoints — ``prefill`` (q_len=Sp) and
``decode`` (q=1, scalar ``pos``) — sharing a SINGLE copy of the int8hu weights and one KV state.

Two separate static bundles are 2×2.3 GB (> iPhone jetsam). This merges them to ~2.3 GB (mirror
``unlimited_ocr/export_decoder.py::export_unified``): one ``coreai_torch.TorchConverter`` + two
``add_pytorch_module`` calls. Quant is in-place so both wrappers share the quantized submodules.

Same externalization fix as the static bundles: DROP ``rope`` (engine-native RoPE mishandles the
baked-constant position_ids -> garbage) and ``scaled_dot_product_attention`` (explicit mask).

Run:  python export_unified.py [--mode int8hu] [--cache-len 256]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import coreai_torch
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS
from coreai_models.export.mlir_ops import register_custom_torch_lowering, remove_functionalization
from qwen3_asr_decoder import Qwen3ASRDecoderPipelined
from qwen3_asr_static import (
    STATE_NAMES,
    Qwen3ASRStaticDecode,
    Qwen3ASRStaticPrefill,
    build_kv_state,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_qwen3_vl_pipelined import head_quant_spec, linear_quant_config  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
DTYPE = torch.float16
OUTDIR = Path(__file__).resolve().parent
ART = OUTDIR / "artifacts"
_DROP = {"scaled_dot_product_attention", "rope"}


def mk_export_fn(ref: dict, dyn: dict | None = None):
    def export_fn(m):
        with torch.no_grad():
            ep = torch.export.export(m, args=(), kwargs=ref, dynamic_shapes=dyn)
        ep = ep.run_decompositions(coreai_torch.get_decomp_table())
        remove_functionalization(ep)
        return ep
    return export_fn


def write_unified_metadata(out_dir: Path, name: str, vocab: int, max_ctx: int) -> None:
    meta = {
        "metadata_version": "0.2", "kind": "llm", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {"tokenizer": MODEL, "vocab_size": vocab, "max_context_length": max_ctx,
                     "embedded_tokenizer": True, "function_map": {"prefill": ["prefill"], "decode": ["decode"]}},
        "source": {"model_definition": "torch", "hf_model_id": MODEL},
        "compression": None,
        "compilation": {"date": datetime.now(timezone.utc).isoformat(), "targets": []},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--cache-len", type=int, default=1024, help="KV buffer; 1024 fits 30s audio + transcript")
    ap.add_argument("--max-sp", type=int, default=448, help="max prompt length (≤30s: ~390 audio + template)")
    ap.add_argument("--max-n", type=int, default=400, help="max audio token count (K=30 -> 390)")
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()
    CL = args.cache_len

    d = np.load(OUTDIR / "oracle_tokens.npz")
    Sp = int(d["input_ids"].shape[1]); N = int(d["encoder_out"].shape[0])
    print(f"Sp={Sp} N={N} cache_len={CL} mode={args.mode}", flush=True)

    print(f"loading decoder from {MODEL} (fp16, N={N}) ...", flush=True)
    base = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=N, target_dtype=DTYPE)
    cfg = base.config
    h = cfg.hidden_size

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model
        spec = base.build_export_spec(DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec()}
            base.lm_head.weight = torch.nn.Parameter(base.lm_head.weight.detach().clone())
        print(f"quantizing base ({args.mode}, in place) ...", flush=True)
        base = quantize_pytorch_model(
            base, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)

    prefill = Qwen3ASRStaticPrefill(base).eval()
    decode = Qwen3ASRStaticDecode(base).eval()
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

    st = build_kv_state(cfg, CL, DTYPE)
    pref_ref = {"input_ids": torch.zeros(1, Sp, dtype=torch.int32),
                "audio_embeds": torch.zeros(N, h, dtype=DTYPE),
                "k_cache": st["k_cache"].clone(), "v_cache": st["v_cache"].clone()}
    dec_ref = {"input_ids": torch.zeros(1, 1, dtype=torch.int32),
               "pos": torch.tensor([Sp], dtype=torch.int32),
               "k_cache": st["k_cache"].clone(), "v_cache": st["v_cache"].clone()}

    # PREFILL is DYNAMIC in prompt length Sp and audio count N -> ONE bundle handles any clip ≤30s
    # (the engine specializes once per length, cached on disk; decode stays fully static).
    sp_dim = torch.export.Dim("sp", min=16, max=args.max_sp)
    n_dim = torch.export.Dim("n", min=1, max=args.max_n)
    pref_dyn = {"input_ids": {1: sp_dim}, "audio_embeds": {0: n_dim}, "k_cache": None, "v_cache": None}

    print("building unified bundle (prefill[dynamic] + decode[static], shared weights+state) ...", flush=True)
    conv = coreai_torch.TorchConverter()
    conv.add_pytorch_module(
        prefill, export_fn=mk_export_fn(pref_ref, pref_dyn), externalize_modules=specs,
        input_names=("input_ids", "audio_embeds"), output_names=("logits",),
        state_names=STATE_NAMES, entrypoint_name="prefill")
    conv.add_pytorch_module(
        decode, export_fn=mk_export_fn(dec_ref), externalize_modules=specs,
        input_names=("input_ids", "pos"), output_names=("logits",),
        state_names=STATE_NAMES, entrypoint_name="decode")
    register_custom_torch_lowering(conv)
    prog = conv.to_coreai()
    print("optimizing ...", flush=True)
    prog.optimize()

    import coreai.runtime as rt
    from transformers import AutoTokenizer
    name = f"qwen3_asr_1.7b_{args.mode}_unified_cl{CL}"  # prefill dynamic in Sp/N (variable length)
    out_dir = ART / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    write_unified_metadata(out_dir, name, cfg.vocab_size, args.max_ctx)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(out_dir / "tokenizer")
    sz = subprocess.run(["du", "-sh", str(aimodel)], capture_output=True, text=True).stdout.split()[0]
    print(f"unified bundle ready: {out_dir} ({sz})", flush=True)


if __name__ == "__main__":
    main()
