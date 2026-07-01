# Community port — NOT an Apple model.
"""Linchpin test for variable-length: does a DYNAMIC-q prefill (dynamic prompt length Sp + dynamic
audio count N) compile CLEANLY on the engine (one specialization per length, no wedge) and stay
correct? Few-layer fp16 for speed. Also times a 2nd same-length call (specialization cache).
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from qwen3_asr_decoder import Qwen3ASRDecoderPipelined
from qwen3_asr_static import STATE_NAMES, Qwen3ASRStaticPrefill, build_kv_state

MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_TOKEN_ID = 151676
V = 151936
OUTDIR = Path(__file__).resolve().parent
CL = 256
_DROP = {"scaled_dot_product_attention", "rope"}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    a = ap.parse_args()
    DTYPE = torch.float16

    import coreai.runtime as rt
    d = np.load(OUTDIR / "oracle_tokens.npz")
    ids = torch.from_numpy(d["input_ids"]).long()[0]
    audio = torch.from_numpy(d["encoder_out"]).to(DTYPE)
    N = int(audio.shape[0]); Sp = ids.shape[0]
    prompt = ids.clone()
    aud_pos = (ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    prompt[aud_pos] = V + torch.arange(N)
    prompt_i32 = prompt.unsqueeze(0).to(torch.int32)

    base = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=N, target_dtype=DTYPE)
    base.model.layers = base.model.layers[: a.layers]
    base.config.num_hidden_layers = a.layers
    prefill = Qwen3ASRStaticPrefill(base).eval()
    st = build_kv_state(base.config, CL, DTYPE)
    with torch.no_grad():
        eager = prefill(prompt_i32, audio, st["k_cache"].clone(), st["v_cache"].clone())[0, -1].float().numpy()

    # DYNAMIC prefill: prompt length Sp and audio count N both dynamic
    h = base.config.hidden_size
    sp_dim = torch.export.Dim("sp", min=16, max=512)
    n_dim = torch.export.Dim("n", min=1, max=400)
    ref = {"input_ids": torch.zeros(1, Sp, dtype=torch.int32),
           "audio_embeds": torch.zeros(N, h, dtype=DTYPE),
           "k_cache": st["k_cache"].clone(), "v_cache": st["v_cache"].clone()}
    dyn = {"input_ids": {1: sp_dim}, "audio_embeds": {0: n_dim}, "k_cache": None, "v_cache": None}
    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
    prog = export_to_coreai(prefill, ref, dynamic_shapes=dyn,
                            input_names=("input_ids", "audio_embeds"), output_names=("logits",),
                            state_names=STATE_NAMES, externalize_modules=specs)
    prog.optimize()
    out = OUTDIR / "artifacts" / "_dyn_prefill.aimodel"
    if out.exists():
        shutil.rmtree(out)
    prog.save_asset(out, rt.AIModelAssetMetadata())

    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    mdl = await rt.AIModel.load(str(out), gpu)
    fn = mdl.load_function("main")

    def run():
        state = {"keyCache": rt.NDArray(np.zeros((a.layers, 1, 8, CL, 128), dtype=np.float16)),
                 "valueCache": rt.NDArray(np.zeros((a.layers, 1, 8, CL, 128), dtype=np.float16))}
        return fn(inputs={"input_ids": rt.NDArray(np.ascontiguousarray(prompt_i32.numpy())),
                          "audio_embeds": rt.NDArray(np.ascontiguousarray(audio.numpy()))}, state=state)

    t0 = time.time(); r1 = await asyncio.wait_for(run(), timeout=300); t1 = (time.time() - t0) * 1000
    eng = r1["logits"].numpy().astype(np.float32).reshape(-1)
    t0 = time.time(); await asyncio.wait_for(run(), timeout=300); t2 = (time.time() - t0) * 1000

    cos = float(np.dot(eng, eager) / (np.linalg.norm(eng) * np.linalg.norm(eager) + 1e-9))
    print(f"layers={a.layers}  DYNAMIC-q prefill (Sp={Sp},N={N})")
    print(f"  1st call (compile+run) = {t1:.0f} ms | 2nd same-length = {t2:.0f} ms")
    print(f"  logits cos(engine,eager) = {cos:.5f}  (eager argmax={int(eager.argmax())} engine={int(eng.argmax())})")


if __name__ == "__main__":
    asyncio.run(main())
