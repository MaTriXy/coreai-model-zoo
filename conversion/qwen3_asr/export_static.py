"""P3: export the Qwen3-ASR decoder as TWO fully-static Core AI bundles — prefill (q_len=Sp) and
decode (q=1, scalar ``pos``) — so the engine compiles each ONCE -> flat decode (no per-step
re-specialization / ANE-probe wedge that the dynamic ``position_ids[1,p+1]`` bundle hit).

Quantizes the base decoder int8hu ONCE (in place); the prefill + decode wrappers share the
quantized Qwen3 submodules. The KV cache (``keyCache``/``valueCache`` state) is shared HOST-SIDE:
the same NDArray buffers are passed to the prefill function then the decode loop (see gate_static.py).

Run (community venv):
    python export_static.py [--mode int8hu] [--cache-len 256]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from qwen3_asr_decoder import Qwen3ASRDecoderPipelined
from qwen3_asr_static import (
    STATE_NAMES,
    Qwen3ASRStaticDecode,
    Qwen3ASRStaticPrefill,
    build_kv_state,
)

# reuse the qwen3.5 ship quant recipes verbatim
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_qwen3_vl_pipelined import head_quant_spec, linear_quant_config, write_bundle_metadata  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
DTYPE = torch.float16
AUDIO_TOKEN_ID = 151676
V = 151936
OUTDIR = Path(__file__).resolve().parent
ART = OUTDIR / "artifacts"


def _du(p: Path) -> str:
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    print(f"saving {aimodel} ...", flush=True)
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    return aimodel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--cache-len", type=int, default=256, help="baked KV buffer length")
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()
    CACHE_LEN = args.cache_len

    # prompt length Sp + audio token count N from the oracle (ja1)
    d = np.load(OUTDIR / "oracle_tokens.npz")
    Sp = int(d["input_ids"].shape[1])           # 70
    N = int(d["encoder_out"].shape[0])          # 55
    print(f"Sp={Sp} N={N} cache_len={CACHE_LEN} mode={args.mode}", flush=True)

    print(f"loading decoder from {MODEL} (fp16, N={N}) ...", flush=True)
    base = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=N, target_dtype=DTYPE)
    cfg = base.config

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model
        spec = base.build_export_spec(DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)
        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec()}
            base.lm_head.weight = torch.nn.Parameter(base.lm_head.weight.detach().clone())  # untie head
        print(f"quantizing base ({args.mode}, in place) ...", flush=True)
        base = quantize_pytorch_model(
            base, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)

    prefill = Qwen3ASRStaticPrefill(base).eval()
    decode = Qwen3ASRStaticDecode(base).eval()

    # Externalization drops (verified via debug_attn isolation, layer-0 attn cos):
    #  - scaled_dot_product_attention: the engine-native SDPA op can't take our explicit causal mask.
    #  - rope: the engine-native RoPE op mishandles a BAKED-CONSTANT position_ids (arange(Sp)) in this
    #    fully-static graph -> wrong rotation that grows with position (cos 1.0 decomposed vs 0.69
    #    externalized). Decomposed RoPE is bit-exact. RMSNorm/GatherMM stay externalized (fast + correct).
    _DROP = {"scaled_dot_product_attention", "rope"}
    static_specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

    h = cfg.hidden_size
    NL, NKV, HD = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
    st = build_kv_state(cfg, CACHE_LEN, DTYPE)

    ART.mkdir(parents=True, exist_ok=True)
    base_name = f"qwen3_asr_1.7b_{args.mode}"

    # --- prefill bundle (q_len=Sp, explicit causal mask; SDPA externalization dropped) ---
    print(f"exporting static PREFILL graph (q_len={Sp}) ...", flush=True)
    pref_ref = {
        "input_ids": torch.zeros(1, Sp, dtype=torch.int32),
        "audio_embeds": torch.zeros(N, h, dtype=DTYPE),
        "k_cache": st["k_cache"].clone(),
        "v_cache": st["v_cache"].clone(),
    }
    prog_p = export_to_coreai(
        prefill, pref_ref, dynamic_shapes=None,
        input_names=("input_ids", "audio_embeds"), output_names=("logits",),
        state_names=STATE_NAMES, externalize_modules=static_specs)
    pdir = ART / f"{base_name}_prefill_sp{Sp}"
    paim = _save(prog_p, pdir)
    write_bundle_metadata(pdir, pdir.name, MODEL, cfg.vocab_size, args.max_ctx)
    print(f"prefill bundle: {pdir} ({_du(paim)})", flush=True)

    # --- decode bundle (q=1, scalar pos; DROP SDPA externalization for the explicit mask) ---
    print("exporting static DECODE graph (q=1, scalar pos) ...", flush=True)
    dec_ref = {
        "input_ids": torch.zeros(1, 1, dtype=torch.int32),
        "pos": torch.tensor([Sp], dtype=torch.int32),
        "k_cache": st["k_cache"].clone(),
        "v_cache": st["v_cache"].clone(),
    }
    prog_d = export_to_coreai(
        decode, dec_ref, dynamic_shapes=None,
        input_names=("input_ids", "pos"), output_names=("logits",),
        state_names=STATE_NAMES, externalize_modules=static_specs)
    ddir = ART / f"{base_name}_decode_cl{CACHE_LEN}"
    daim = _save(prog_d, ddir)
    write_bundle_metadata(ddir, ddir.name, MODEL, cfg.vocab_size, args.max_ctx)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(ddir / "tokenizer")
    print(f"decode bundle: {ddir} ({_du(daim)})", flush=True)
    print("\nDONE. prefill + decode static bundles ready.", flush=True)


if __name__ == "__main__":
    main()
