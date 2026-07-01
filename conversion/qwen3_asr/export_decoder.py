"""P2: export the Qwen3-ASR decoder to Core AI pipelined bundles (int8hu) + eager-int8 sanity.

Mirrors export_qwen3_vl_pipelined.py (decoder half) but for Qwen3ASRDecoderPipelined: inputs
input_ids/position_ids/audio_embeds + KV state. Two graphs from the same quantized weights:
  - dynamic query  -> ship bundle (chunked prefill + decode)
  - static S=1     -> `_s1` gate bundle (python runtime can't run dynamic-shaped outputs)
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from qwen3_asr_decoder import PIPELINED_STATE_NAMES, Qwen3ASRDecoderPipelined

# reuse the qwen3.5 ship quant recipes verbatim
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_qwen3_vl_pipelined import head_quant_spec, linear_quant_config, write_bundle_metadata  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
DTYPE = torch.float16
AUDIO_TOKEN_ID = 151676
V = 151936
OUTDIR = Path(__file__).resolve().parent
ART = OUTDIR / "artifacts"


@torch.no_grad()
def eager_int8_sanity(model, n_audio: int) -> None:
    """Confirm the quantized model still argmaxes the golden first token on ja1 (eager)."""
    d = np.load(OUTDIR / "oracle_tokens.npz")
    ids = torch.from_numpy(d["input_ids"]).long()
    audio = torch.from_numpy(d["encoder_out"]).to(DTYPE)
    golden_first = int(d["gen_ids"][0])
    aud_pos = (ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    ids[0, aud_pos] = V + torch.arange(len(aud_pos))
    s = ids.shape[1]
    pos = torch.arange(s, dtype=torch.int32).unsqueeze(0)
    cfg = model.config
    kc = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 128, cfg.head_dim, dtype=DTYPE)
    vc = torch.zeros_like(kc)
    logits = model(ids, pos, audio, kc, vc)[0, -1].float()
    am = int(logits.argmax())
    print(f"[eager int8] first-token argmax={am} golden={golden_first} "
          f"{'OK' if am == golden_first else 'MISMATCH'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--n-audio", type=int, default=55, help="N audio-embed slots baked into the bundle")
    ap.add_argument("--max-ctx", type=int, default=4096)
    ap.add_argument("--skip-dynamic", action="store_true")
    args = ap.parse_args()

    print(f"loading decoder from {MODEL} (fp16, N={args.n_audio}) ...", flush=True)
    model = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=args.n_audio, target_dtype=DTYPE)
    cfg = model.config
    spec = model.build_export_spec(DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN)

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model
        cfg_q = linear_quant_config("int8")
        if args.mode == "int8hu":
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec()}
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())
        print(f"quantizing ({args.mode}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(spec["reference_inputs"].values()), spec["dynamic_shapes"], cfg_q)
        try:
            eager_int8_sanity(model, args.n_audio)
        except Exception as e:  # noqa: BLE001
            print(f"[eager int8] skipped ({e})", flush=True)

    specs = list(_EXTERNALIZE_SPECS)
    import coreai.runtime as rt
    from transformers import AutoTokenizer

    name = f"qwen3_asr_1.7b_decode_{args.mode}_n{args.n_audio}"
    variants = []
    if not args.skip_dynamic:
        variants.append((name, spec))
    variants.append((f"{name}_s1", model.build_export_spec(
        DTYPE, args.max_ctx, trace_kv_len=TRACE_KV_CACHE_SEQ_LEN, trace_query=1)))

    ART.mkdir(parents=True, exist_ok=True)
    for vname, vspec in variants:
        print(f"exporting decoder graph ({vname}) ...", flush=True)
        prog = export_to_coreai(
            model, vspec["reference_inputs"], dynamic_shapes=vspec["dynamic_shapes"],
            input_names=vspec["input_names"], output_names=vspec["output_names"],
            state_names=PIPELINED_STATE_NAMES, externalize_modules=specs)
        prog.optimize()
        out_dir = ART / vname
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        aimodel = out_dir / f"{vname}.aimodel"
        print(f"saving {aimodel} ...", flush=True)
        prog.save_asset(aimodel, rt.AIModelAssetMetadata())
        write_bundle_metadata(out_dir, vname, MODEL, cfg.vocab_size, args.max_ctx)
        AutoTokenizer.from_pretrained(MODEL).save_pretrained(out_dir / "tokenizer")
        import subprocess
        sz = subprocess.run(["du", "-sh", str(aimodel)], capture_output=True, text=True).stdout.split()[0]
        print(f"bundle ready: {out_dir} ({sz})", flush=True)


if __name__ == "__main__":
    main()
