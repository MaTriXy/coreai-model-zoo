# Community port — NOT an Apple model.
"""FULL-utterance E2E host loop driving ALL exported Core AI bundles through the real dots.tts
generation loop — the Swift blueprint, gated vs the golden. Bootstraps the schedule/prefill-embeds
from the upstream (oracle --e2e) so no tokenizer reimpl is needed; the AUDIO is produced by the
engine feedback loop (solver -> patch_encoder -> backbone -> next), not replayed.

Per patch k (soar, no-prompt "Hello from Core A I.", 15 patches):
  1. solver: DiT bundle over fm_sequence[:fm_seq_len] (padded to S=164) + coordinate_proj(noise_k)
  2. denormalize (latent_stats) -> gate vs golden e2e.latent{k}
  3. append latent_proj(denorm) to fm_sequence (+4)
  4. patch_encoder bundle (threaded static KV) -> LLM embed
  5. backbone bundle decode step (LLM embed @ pos 15+k) -> new hidden
  6. append hidden_proj(hidden) to fm_sequence (+1) if next span is audio
Then concat 15 denorm patches -> vocoder bundle -> wav, gate vs golden wav.

  PYTHONPATH=. <coreai-venv>/bin/python e2e_full.py --src <weights/dots.tts-soar>
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coreai.runtime as rt  # noqa: E402
from backbone import build_kv_state as bb_kv, load_backbone  # noqa: E402
from patch_encoder import build_kv_state as pe_kv, load_patch_encoder  # noqa: E402
from export_dit import build_fm_attn_mask, build_fm_pos_ids  # noqa: E402
from host_solver import solve_soar, DiTEngine, cos  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
DT = torch.float16
CAP = 164          # DiT bucket-32 total_len (serves fm_seq_len <= 160)
LATENT_START = CAP - 4  # 160


def _linear(sd, prefix):
    w = sd[prefix + ".weight"].to(DT)
    m = nn.Linear(w.shape[1], w.shape[0]).to(DT).eval()
    with torch.no_grad():
        m.weight.copy_(w); m.bias.copy_(sd[prefix + ".bias"].to(DT))
    return m


async def _load_fn(bundle):
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    aim = ART / bundle / f"{bundle}.aimodel"
    return (await rt.AIModel.load(str(aim), gpu)).load_function("main")


def _nd(x):
    return rt.NDArray(np.ascontiguousarray(x.detach().numpy().astype(np.float16)))


async def solve_mf_grow(dit, coord, cond, g_cond, attn_mask, pos_ids, noise, nfe, DT):
    """MeanFlow solver over the growing fm_sequence: batch-1, no CFG, +duration. cond [1,S-4,1024]."""
    latent_start = cond.shape[1]
    z = noise.clone()
    times = torch.linspace(0.0, 1.0, nfe + 1, dtype=DT)
    for step in range(nfe):
        t = times[step].reshape(1)
        dt = (times[step + 1] - times[step]).reshape(1)
        zp = coord(z)
        z_c = torch.cat([cond, zp], 1)               # [1,S,1024]
        vt = await dit(x=z_c, timesteps=t, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g_cond, duration=dt)
        z = z + vt[:, latent_start:] * dt.view(-1, 1, 1)
    return z


async def main_run(src, mode="soar", nfe=4, bb_prec="fp16"):
    torch.set_grad_enabled(False)
    src = Path(src)
    sd = load_file(str(src / "model.safetensors"))
    cfg_json = json.loads((src / "config.json").read_text())
    z = np.load(ART / "oracle_ref_e2e_soar.npz")

    # ---- host glue projections ----
    hidden_proj = _linear(sd, "hidden_proj")     # 1536->1024
    latent_proj = _linear(sd, "latent_proj")     # 128->1024
    coordinate_proj = _linear(sd, "coordinate_proj")  # 128->1024
    st = torch.load(str(src / "latent_stats.pt"), weights_only=False)
    mean = torch.as_tensor(st["mean"]).to(DT); std = torch.sqrt(torch.as_tensor(st["var"]).to(DT))

    # ---- engine bundles (mf reuses soar backbone/patchenc/vocoder/glue — only the DiT differs) ----
    bb_fn = await _load_fn(f"dots_backbone_{bb_prec}_decode_cl512")
    pe_fn = await _load_fn("dots_patchenc_fp16_buf1000")
    voc_fn = await _load_fn("dots_vocoder_fp16_t60")
    if mode == "mf":
        dit_fn = await _load_fn("dots_dit_mf_fp16_s164")
        dit_names = ("x", "timesteps", "attn_mask", "pos_ids", "g_cond", "duration")
    else:
        dit_fn = await _load_fn("dots_dit_soar_fp16_s164")
        dit_names = ("x", "timesteps", "attn_mask", "pos_ids", "g_cond")
    dit_engine = DiTEngine(dit_fn, dit_names)

    # ---- seed ----
    prefill_embeds = torch.from_numpy(z["e2e.prefill_embeds"]).to(DT)  # [1,15,1536]
    P = prefill_embeds.shape[1]
    g_cond = torch.from_numpy(z["e2e.fm_null_g_cond"]).to(DT)          # [1,1024]
    n_patches = int(z["e2e.num_latents"][0])
    CACHE = 512

    # ---- backbone state (static KV via decode-via-decode) ----
    bb_cfg = load_backbone(sd, CACHE, DT).cfg
    nl, nkv, hd = bb_cfg.num_hidden_layers, bb_cfg.num_key_value_heads, bb_cfg.head_dim
    bb_state = {"keyCache": rt.NDArray(np.zeros((nl, 1, nkv, CACHE, hd), np.float16)),
                "valueCache": rt.NDArray(np.zeros((nl, 1, nkv, CACHE, hd), np.float16))}

    async def bb_decode(emb, pos):
        r = await bb_fn(inputs={"inputs_embeds": _nd(emb),
                                "pos": rt.NDArray(np.ascontiguousarray(np.array([pos], np.int32)))},
                        state=bb_state)
        return torch.from_numpy(np.asarray(r["hidden"].numpy())).to(DT)

    # prefill-via-decode: 15 steps @ pos 0..14
    llm_hidden = None
    for i in range(P):
        llm_hidden = await bb_decode(prefill_embeds[:, i:i + 1], i)   # [1,1,1536]

    # ---- fm_sequence buffers (CAP x 1024) ----
    fm = torch.zeros(1, CAP, 1024, dtype=DT)
    fm_cfg = torch.zeros(1, CAP, 1024, dtype=DT)
    fm_len = 0
    zero_hidden = torch.zeros(1, 1, 1536, dtype=DT)

    def append_hidden(h):
        nonlocal fm_len
        fm[:, fm_len:fm_len + 1] = hidden_proj(h[:, -1:])
        fm_cfg[:, fm_len:fm_len + 1] = hidden_proj(zero_hidden)
        fm_len += 1

    def append_history(lat):
        nonlocal fm_len
        p = latent_proj(lat)  # [1,4,1024]
        fm[:, fm_len:fm_len + 4] = p
        fm_cfg[:, fm_len:fm_len + 4] = p
        fm_len += 4

    append_hidden(llm_hidden)  # fm_len -> 1

    # ---- patch_encoder state ----
    _, pe_cfg = load_patch_encoder(sd, cfg_json, 1000, DT)
    pnl, pnh, phd = pe_cfg.n_layers, pe_cfg.n_heads, pe_cfg.head_dim
    pe_state = {"keyCache": rt.NDArray(np.zeros((pnl, 1, pnh, 1000, phd), np.float16)),
                "valueCache": rt.NDArray(np.zeros((pnl, 1, pnh, 1000, phd), np.float16))}
    conv_tail = torch.zeros(1, pe_cfg.input_dim, 1, dtype=DT)
    pe_seq = 0

    patch_cos = []
    denorm_patches = []
    for k in range(n_patches):
        # ---- solver over padded fm_sequence ----
        cond = fm[:, :LATENT_START].clone()       # [1,160,1024] (0:fm_len filled, rest 0)
        uncond = fm_cfg[:, :LATENT_START].clone()
        attn_mask = build_fm_attn_mask(fm_len, CAP).to(DT)
        pos_ids = build_fm_pos_ids(fm_len, CAP)
        noise = torch.from_numpy(z[f"randn{k}"]).to(DT)
        if mode == "mf":
            patch = await solve_mf_grow(dit_engine, coordinate_proj, cond, g_cond,
                                        attn_mask, pos_ids, noise, nfe, DT)      # [1,4,128] normalized
        else:
            patch = await solve_soar(dit_engine, coordinate_proj, cond, uncond, g_cond,
                                     attn_mask, pos_ids, noise, DT, engine=True)

        denorm = patch.to(DT) * std + mean
        denorm_patches.append(denorm)
        c = cos(denorm.numpy(), z[f"e2e.latent{k}"])
        patch_cos.append(c)

        # ---- feedback (upstream _consume_audio_patch): history gets the NORMALIZED patch
        # (latent_proj input), patch_encoder gets the DENORMALIZED patch ----
        append_history(patch)
        r = await pe_fn(inputs={"latent_patch": _nd(denorm), "conv_tail": _nd(conv_tail),
                                "pos": rt.NDArray(np.ascontiguousarray(np.array([pe_seq], np.int32)))},
                        state=pe_state)
        emb = torch.from_numpy(np.asarray(r["embedding"].numpy())).to(DT)     # [1,1,1536]
        # new_conv_tail = the last latent frame of the (denormalized) input — host-derived
        conv_tail = denorm[:, -1:, :].transpose(1, 2).contiguous()           # [1,128,1]
        pe_seq += pe_cfg.out_ds_rate
        llm_hidden = await bb_decode(emb, P + k)
        if k < n_patches - 1:
            append_hidden(llm_hidden)

    print(f"  per-patch denorm-latent cos vs golden: min={min(patch_cos):.6f}  "
          f"mean={sum(patch_cos)/len(patch_cos):.6f}")
    print(f"    {['%.4f' % c for c in patch_cos]}")

    # ---- vocode: concat patches [1,60,128] -> [1,128,60] -> wav ----
    latents = torch.cat(denorm_patches, dim=1).transpose(1, 2)  # [1,128,60]
    rv = await voc_fn(inputs={"latents": _nd(latents)})
    wav = np.asarray(rv["wav"].numpy()).reshape(-1)
    cw = cos(wav, z["wav"])
    af = np.abs(np.fft.rfft(wav)); ff = np.fft.rfftfreq(len(wav), 1 / 48000)
    sb = af[(ff >= 200) & (ff <= 4000)].sum() / af.sum()
    peak = float(np.abs(wav).max())
    if mode == "mf":
        # mf DiT + soar-oracle noise => valid speech, NOT a cos-match to the soar golden
        ok = 0.05 < peak <= 1.0 and sb > 0.6 and not np.isnan(wav).any()
        print(f"\n>>> mf E2E ({nfe} NFE, {bb_prec} bb): wav {wav.shape} peak={peak:.3f} "
              f"speechband={sb:.2f} cos-vs-soar={cw:.3f} -> {'PASS (valid speech)' if ok else 'FAIL'}")
    else:
        ok = cw >= 0.99
        print(f"\n>>> soar E2E ({bb_prec} bb): wav cos={cw:.6f} "
              f"(per-patch mean {sum(patch_cos)/len(patch_cos):.4f}, min {min(patch_cos):.4f}) "
              f"-> {'PASS' if ok else 'FAIL'}")

    import soundfile as sf
    sf.write(str(ART / f"e2e_engine_{mode}.wav"), wav.astype(np.float32), 48000)
    sys.exit(0 if ok else 1)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", default="soar", choices=["soar", "mf"])
    ap.add_argument("--nfe", type=int, default=4)
    ap.add_argument("--bb-prec", default="fp16", choices=["fp16", "int8", "int4"])
    a = ap.parse_args()
    await main_run(a.src, mode=a.mode, nfe=a.nfe, bb_prec=a.bb_prec)


if __name__ == "__main__":
    asyncio.run(main())
