# Community port — NOT an Apple model.
"""Export the VoxCPM2 (2B) bundles to Core AI .aimodel + engine-gate each vs my torch overlay (which is
already bit-exact vs the official model in the gate_v2_*_torch.py suite — so the overlay IS the oracle).

Bundles (mirror v1, scaled to 2B / 48kHz):
  base / res   : MiniCPM4 decode (q=1, static-KV) [+ optional q=32 prefill]   (base 28L, res 8L no-rope)
  feat_decoder : CFMDecoder(LocDiT 12L) unrolled euler : (mu[1,2048],cond[1,64,4],z[1,64,4]) -> [1,64,4]
  feat_encoder : LocEnc 12L + enc_to_lm_proj : pred_feat[1,1,4,64] -> curr_embed[1,1,2048]
  vocoder      : AudioVAE v2 decoder (depthwise, SR-cond baked) : latents[1,64,Tf] -> wav[1,1,1920Tf]

  python export_v2.py --which {base,res,feat_decoder,feat_encoder,vocoder} [--mode fp16] [--cache-len 512]
                      [--prefill-len 32] [--vae-frames 8]
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.export.compression import quantize_pytorch_model  # noqa: E402
from minicpm4 import build_kv_state, load_backbone  # noqa: E402
from feat_decoder_v2 import load_feat_decoder_v2  # noqa: E402
from feat_encoder_v2 import load_feat_encoder_v2  # noqa: E402
from audio_vae_v2 import load_audio_vae_v2  # noqa: E402

_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
ART = Path(__file__).resolve().parent / "artifacts"
FEAT, PATCH = 64, 4

# Weight-only symmetric per-channel int8 (coreai-opt PT2E, no activation calibration). LM-only
# (base/res); DiT + VAE stay fp16 (MLX precedent + continuous-feedback quant-sensitivity, v1 lesson).
INT8_CFG = {"execution_mode": "eager", "global_config": {
    "op_state_spec": {"weight": {"dtype": "int8", "qscheme": "symmetric",
                                 "granularity": {"type": "per_channel", "axis": 0}}},
    "op_input_spec": None, "op_output_spec": None}}


def snap():
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]


def load_v2():
    from safetensors.torch import load_file
    lm = json.load(open(snap() + "/config.json"))["lm_config"]
    return load_file(snap() + "/model.safetensors"), lm


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    print(f"  saving {aim.name} ...", flush=True)
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


class PrefillWrap(nn.Module):
    def __init__(self, bb):
        super().__init__(); self.bb = bb

    def forward(self, inputs_embeds, k_cache, v_cache):
        return self.bb.prefill(inputs_embeds, k_cache, v_cache)


class DecodeWrap(nn.Module):
    def __init__(self, bb):
        super().__init__(); self.bb = bb

    def forward(self, inputs_embeds, pos, k_cache, v_cache):
        return self.bb.decode(inputs_embeds, pos, k_cache, v_cache)


class CFMWrap(nn.Module):
    def __init__(self, cfm):
        super().__init__(); self.cfm = cfm

    def forward(self, mu, cond, z):
        return self.cfm(mu, cond, z)


class FeatEncWrap(nn.Module):
    def __init__(self, enc, enc2lm):
        super().__init__(); self.enc = enc; self.enc2lm = enc2lm

    def forward(self, pred_feat):
        return self.enc2lm(self.enc(pred_feat))


class VocoderWrap(nn.Module):
    def __init__(self, dec):
        super().__init__(); self.dec = dec

    def forward(self, latents):
        return self.dec(latents)


def _linear(sd, prefix, dtype):
    w = sd[prefix + ".weight"].to(dtype)
    m = nn.Linear(w.shape[1], w.shape[0], bias=(prefix + ".bias") in sd).to(dtype).eval()
    m.weight = nn.Parameter(w, requires_grad=False)
    if m.bias is not None:
        m.bias = nn.Parameter(sd[prefix + ".bias"].to(dtype), requires_grad=False)
    return m


async def gate_backbone(which, DT, sd, lm, cache_len, prefill_len, int8=False):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    tag = "int8" if int8 else "fp16"
    H = lm["hidden_size"]
    ov = dict(hidden_size=H, intermediate_size=lm["intermediate_size"],
              num_attention_heads=lm["num_attention_heads"], num_key_value_heads=lm["num_key_value_heads"],
              head_dim=lm["kv_channels"], short_factor=lm["rope_scaling"]["short_factor"])
    nl, vocab, no_rope = (28, 73448, False) if which == "base" else (8, 0, True)
    bb = load_backbone(sd, f"{'base_lm' if which=='base' else 'residual_lm'}.", nl, vocab,
                       cache_len, DT, no_rope=no_rope, **ov).to(DT).eval()
    kc, vc = build_kv_state(bb.cfg, cache_len, DT)
    NKV, HD = bb.cfg.num_key_value_heads, bb.cfg.head_dim

    # ---- decode bundle ----
    dec_ref = {"inputs_embeds": torch.zeros(1, 1, H, dtype=DT), "pos": torch.tensor([0], dtype=torch.int32),
               "k_cache": kc.clone(), "v_cache": vc.clone()}
    dec_model = DecodeWrap(bb).eval()
    if int8:
        print(f"  [{which}] int8 weight-only quantize (decode) ...", flush=True)
        dec_model = quantize_pytorch_model(
            dec_model, (dec_ref["inputs_embeds"], dec_ref["pos"], dec_ref["k_cache"], dec_ref["v_cache"]),
            {"inputs_embeds": None, "pos": None, "k_cache": None, "v_cache": None}, dict(INT8_CFG))
    prog = export_to_coreai(dec_model, dec_ref, dynamic_shapes=None,
                            input_names=("inputs_embeds", "pos"), output_names=("hidden",),
                            state_names=("keyCache", "valueCache"))
    ddir = ART / f"voxcpm2_{which}_{tag}_decode_cl{cache_len}"
    aim = _save(prog, ddir)
    print(f"  -> {ddir.name} ({_du(aim)})", flush=True)

    # gate: thread K decode steps, torch-buffers vs engine-state
    K = 6
    torch.manual_seed(0)
    embs = [torch.randn(1, 1, H, dtype=DT) for _ in range(K)]
    tk, tv = build_kv_state(bb.cfg, cache_len, DT)
    t_hid = []
    with torch.inference_mode():
        for i, e in enumerate(embs):
            t_hid.append(bb.decode(e, torch.tensor([i], dtype=torch.int32), tk, tv).reshape(-1).float())
    dfn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    state = {"keyCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16)),
             "valueCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16))}
    cs = []
    for i, e in enumerate(embs):
        r = await dfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(e.numpy())),
                              "pos": rt.NDArray(np.ascontiguousarray(np.array([i], np.int32)))}, state=state)
        cs.append(cos(r["hidden"].numpy(), t_hid[i]))
    print(f"  decode engine-vs-torch min cos={min(cs):.6f}  {['%.4f'%c for c in cs]}")

    # ---- prefill bundle (q=prefill_len) ----
    if prefill_len:
        pref_ref = {"inputs_embeds": torch.zeros(1, prefill_len, H, dtype=DT),
                    "k_cache": kc.clone(), "v_cache": vc.clone()}
        pre_model = PrefillWrap(bb).eval()
        if int8:
            print(f"  [{which}] int8 weight-only quantize (prefill q={prefill_len}) ...", flush=True)
            pre_model = quantize_pytorch_model(
                pre_model, (pref_ref["inputs_embeds"], pref_ref["k_cache"], pref_ref["v_cache"]),
                {"inputs_embeds": None, "k_cache": None, "v_cache": None}, dict(INT8_CFG))
        progp = export_to_coreai(pre_model, pref_ref, dynamic_shapes=None,
                                 input_names=("inputs_embeds",), output_names=("hidden",),
                                 state_names=("keyCache", "valueCache"))
        pdir = ART / f"voxcpm2_{which}_{tag}_prefill_t{prefill_len}"
        aimp = _save(progp, pdir)
        print(f"  -> {pdir.name} ({_du(aimp)})", flush=True)
        torch.manual_seed(1)
        pe = torch.randn(1, prefill_len, H, dtype=DT)
        pk, pv = build_kv_state(bb.cfg, cache_len, DT)
        with torch.inference_mode():
            t_p = bb.prefill(pe, pk, pv).reshape(-1).float()
        pfn = (await rt.AIModel.load(str(aimp), gpu)).load_function("main")
        pstate = {"keyCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16)),
                  "valueCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16))}
        rp = await pfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(pe.numpy()))}, state=pstate)
        cs.append(cos(rp["hidden"].numpy(), t_p))
        print(f"  prefill q={prefill_len} engine-vs-torch cos={cs[-1]:.6f}")
    return min(cs)


async def gate_ff(which, DT, sd, lm, vae_frames):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    sf = lm["rope_scaling"]["short_factor"]
    torch.manual_seed(0)

    if which == "feat_decoder":
        cfm = load_feat_decoder_v2(sd, sf, dtype=DT)
        ref = {"mu": torch.randn(1, lm["hidden_size"], dtype=DT),
               "cond": torch.randn(1, FEAT, PATCH, dtype=DT), "z": torch.randn(1, FEAT, PATCH, dtype=DT)}
        with torch.inference_mode():
            t_out = cfm(ref["mu"], ref["cond"], ref["z"]).reshape(-1).float()
        prog = export_to_coreai(CFMWrap(cfm).eval(), ref, dynamic_shapes=None,
                                input_names=("mu", "cond", "z"), output_names=("pred_feat",), state_names=None)
        ddir = ART / "voxcpm2_feat_decoder_fp16"
        aim = _save(prog, ddir); print(f"  -> {ddir.name} ({_du(aim)})", flush=True)
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        r = await fn(inputs={k: rt.NDArray(np.ascontiguousarray(v.numpy())) for k, v in ref.items()})
        return cos(r["pred_feat"].numpy(), t_out)

    if which == "feat_encoder":
        enc = load_feat_encoder_v2(sd, sf, dtype=DT)
        enc2lm = _linear(sd, "enc_to_lm_proj", DT)
        m = FeatEncWrap(enc, enc2lm).eval()
        ref = {"pred_feat": torch.randn(1, 1, PATCH, FEAT, dtype=DT)}
        with torch.inference_mode():
            t_out = m(ref["pred_feat"]).reshape(-1).float()
        prog = export_to_coreai(m, ref, dynamic_shapes=None,
                                input_names=("pred_feat",), output_names=("curr_embed",), state_names=None)
        ddir = ART / "voxcpm2_feat_encoder_fp16"
        aim = _save(prog, ddir); print(f"  -> {ddir.name} ({_du(aim)})", flush=True)
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        r = await fn(inputs={"pred_feat": rt.NDArray(np.ascontiguousarray(ref["pred_feat"].numpy()))})
        return cos(r["curr_embed"].numpy(), t_out)

    # vocoder
    vae_ck = torch.load(snap() + "/audiovae.pth", map_location="cpu", weights_only=True)
    dec = load_audio_vae_v2(vae_ck.get("state_dict", vae_ck), DT)
    m = VocoderWrap(dec).eval()
    ref = {"latents": torch.randn(1, FEAT, vae_frames, dtype=DT)}
    with torch.inference_mode():
        t_out = m(ref["latents"]).reshape(-1).float()
    prog = export_to_coreai(m, ref, dynamic_shapes=None,
                            input_names=("latents",), output_names=("wav",), state_names=None)
    ddir = ART / f"voxcpm2_vocoder_fp16_t{vae_frames}"
    aim = _save(prog, ddir); print(f"  -> {ddir.name} ({_du(aim)})", flush=True)
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(ref["latents"].numpy()))})
    wav = r["wav"].numpy()
    print(f"  wav shape={wav.shape} peak={np.abs(wav).max():.3f} (peak~0 => ConvTranspose-zeros trap)")
    return cos(wav, t_out)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["base", "res", "feat_decoder", "feat_encoder", "vocoder"])
    ap.add_argument("--mode", default="fp16", choices=["fp16"])
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--prefill-len", type=int, default=0, help="also export a q=N prefill bundle")
    ap.add_argument("--vae-frames", type=int, default=8)
    ap.add_argument("--int8", action="store_true", help="weight-only int8 (base/res only)")
    a = ap.parse_args()
    DT = torch.float16
    ART.mkdir(parents=True, exist_ok=True)
    sd, lm = load_v2()

    if a.which in ("base", "res"):
        lo = await gate_backbone(a.which, DT, sd, lm, a.cache_len, a.prefill_len, int8=a.int8)
    else:
        lo = await gate_ff(a.which, DT, sd, lm, a.vae_frames)
    print(f"\n>>> {a.which} export+engine: cos={lo:.6f} -> {'GATE PASS' if lo >= 0.99 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
