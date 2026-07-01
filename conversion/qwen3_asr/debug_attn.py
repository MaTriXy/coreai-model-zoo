# Community port — NOT an Apple model.
"""Isolate ONE static attention block (layer 0) on the engine: input = injected embeddings,
output = attention residual-add output. Pinpoints whether qk_norm/rope/masked-SDPA is the bug.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, "/tmp/qwen3-asr-official")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from qwen3_asr_decoder import Qwen3ASRDecoderPipelined
from qwen3_asr_static import STATE_NAMES, build_kv_state, causal_buffer_mask, static_attn
from coreai_models.primitives.macos.sdpa import SDPA

MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_TOKEN_ID = 151676
V = 151936
OUTDIR = Path(__file__).resolve().parent
CL = 256


class OneAttn(nn.Module):
    def __init__(self, base):
        super().__init__()
        layer = base.model.layers[0]
        self.input_layernorm = layer.input_layernorm   # hold ONLY used submodules
        self.self_attn = layer.self_attn
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x, k_cache, v_cache):
        b, Sp, _ = x.shape
        buf = k_cache.size(-2)
        q_pos = torch.arange(Sp, dtype=torch.int32, device=x.device).unsqueeze(0)
        mask = causal_buffer_mask(q_pos, buf)
        h = self.input_layernorm(x)
        r = static_attn(self.self_attn, self.sdpa, h, q_pos, 0, mask, k_cache, v_cache, 0)
        return x + r


async def main() -> None:
    import coreai.runtime as rt
    d = np.load(OUTDIR / "oracle_tokens.npz")
    ids = torch.from_numpy(d["input_ids"]).long()[0]
    audio = torch.from_numpy(d["encoder_out"]).to(torch.float16)
    N = int(audio.shape[0]); Sp = ids.shape[0]
    prompt = ids.clone()
    aud_pos = (ids == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    prompt[aud_pos] = V + torch.arange(N)
    prompt_i32 = prompt.unsqueeze(0).to(torch.int32)

    base = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=N, target_dtype=torch.float16)
    # injected embeddings (precomputed, proven correct on engine)
    m_cfg = base.config
    is_aud = prompt_i32 >= V
    slot = (prompt_i32 - V).clamp(0, N - 1)
    e_txt = base.model.embed_tokens(prompt_i32.clamp(0, V - 1))
    e_aud = audio.index_select(0, slot.reshape(-1)).reshape(1, Sp, -1)
    x = torch.where(is_aud.unsqueeze(-1), e_aud.to(e_txt.dtype), e_txt).detach()

    m = OneAttn(base).eval()
    st = build_kv_state(m_cfg, CL, torch.float16)
    with torch.no_grad():
        eager = m(x, st["k_cache"].clone(), st["v_cache"].clone())[0].float().numpy()  # [Sp,h]

    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="none", choices=["none", "all_but_sdpa", "rope", "rmsnorm"])
    a = ap.parse_args()
    if a.ext == "none":
        specs = []
    elif a.ext == "all_but_sdpa":
        specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "scaled_dot_product_attention"]
    elif a.ext == "rope":
        specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name == "rope"]
    else:  # rmsnorm
        specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name == "rms_norm"]
    print(f"externalize = {a.ext} -> {[s.composite_op_name for s in specs]}")

    ref = {"x": x, "k_cache": st["k_cache"].clone(), "v_cache": st["v_cache"].clone()}
    prog = export_to_coreai(m, ref, dynamic_shapes=None, input_names=("x",), output_names=("y",),
                            state_names=STATE_NAMES, externalize_modules=specs)
    prog.optimize()
    out = OUTDIR / "artifacts" / "_dbg_attn.aimodel"
    if out.exists():
        shutil.rmtree(out)
    prog.save_asset(out, rt.AIModelAssetMetadata())

    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    mdl = await rt.AIModel.load(str(out), gpu)
    fn = mdl.load_function("main")
    state = {"keyCache": rt.NDArray(np.zeros((m_cfg.num_hidden_layers, 1, 8, CL, 128), dtype=np.float16)),
             "valueCache": rt.NDArray(np.zeros((m_cfg.num_hidden_layers, 1, 8, CL, 128), dtype=np.float16))}
    res = await fn(inputs={"x": rt.NDArray(np.ascontiguousarray(x.numpy()))}, state=state)
    eng = res["y"].numpy().astype(np.float32)[0]  # [Sp,h]

    cosp = [float(np.dot(eager[i], eng[i]) / (np.linalg.norm(eager[i]) * np.linalg.norm(eng[i]) + 1e-9)) for i in range(Sp)]
    print(f"layer-0 attn-out per-token cos: min={min(cosp):.4f} mean={np.mean(cosp):.4f} last={cosp[-1]:.4f}")
    print(f"worst tokens: {[ (i, round(c,3)) for i,c in sorted(enumerate(cosp), key=lambda t:t[1])[:5] ]}")


if __name__ == "__main__":
    asyncio.run(main())
