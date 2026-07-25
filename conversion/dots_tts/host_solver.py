# Community port — NOT an Apple model.
"""Host-side DiT SOLVER glue driving the exported Core AI DiT bundle end-to-end — the Swift
blueprint for one audio patch. Mirrors core.py:

  soar (flow_matching): euler, num_steps=10, CFG. z0=noise; for i in 10: t=i/10;
      z_c[:,latent_start:]=coordinate_proj(z); z_cfg likewise; DiT(cat([z_c,z_cfg]), t.rep2,
      g_cond=[g,0]) -> vt[:,latent_start:]; v = vt_c + gs*(vt_c - vt_u); z += 0.1*v.
  mf (meanflow): nfe=4, no CFG. times=linspace(0,1,5); for step: t,dt; z_c[:,latent_start:]=
      coordinate_proj(z); DiT(z_c, t, duration=dt, g_cond) -> vt[:,latent_start:]; z += vt*dt.

Gate: run the SAME solver twice — once calling the ENGINE DiT bundle, once the torch overlay,
with the SAME replayed noise — and compare the final patch. cos~1.0 proves the host solver + the
engine DiT bundle compose to the same denoised patch (the numeric core of the per-patch loop).

  PYTHONPATH=. <coreai-venv>/bin/python host_solver.py --mode soar --src <..-soar>
  PYTHONPATH=. <coreai-venv>/bin/python host_solver.py --mode mf   --src <..-mf>
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
from torch_overlays import DiTOverlay  # noqa: E402

ART = Path(__file__).resolve().parent / "artifacts"
GS = 1.2        # guidance_scale (soar) = model.py generate default (oracle used it)
NUM_STEPS = 10  # soar euler steps (oracle --num-steps 10)
NFE = 10        # mf steps (oracle used num_steps=10 for both decoders)


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _linear(sd, prefix, DT):
    w = sd[prefix + ".weight"].to(DT)
    m = nn.Linear(w.shape[1], w.shape[0]).to(DT).eval()
    with torch.no_grad():
        m.weight.copy_(w); m.bias.copy_(sd[prefix + ".bias"].to(DT))
    return m


class DiTEngine:
    """Async wrapper: call the exported DiT bundle like the torch overlay."""

    def __init__(self, fn, names):
        self.fn = fn; self.names = names

    async def __call__(self, **kw):
        import coreai.runtime as rt
        inp = {n: rt.NDArray(np.ascontiguousarray(kw[n].numpy())) for n in self.names}
        r = await self.fn(inputs=inp)
        return torch.from_numpy(np.asarray(r["velocity"].numpy()))


async def solve_soar(dit, coord, cond, uncond, g_cond, attn_mask, pos_ids, noise, DT, engine):
    """dit: callable(x,timesteps,attn_mask,pos_ids,g_cond)->velocity. cond/uncond [1,1,1024]."""
    latent_start = cond.shape[1] + 0  # S-4; here cond has 1 pos, +4 noise -> latent_start=1
    S = cond.shape[1] + 4
    z = noise.clone()  # [1,4,128]
    g2 = torch.cat([g_cond, torch.zeros_like(g_cond)], 0)  # [2,1024]
    for i in range(NUM_STEPS):
        t = torch.tensor([i / NUM_STEPS], dtype=DT)
        zp = coord(z)  # [1,4,1024]
        z_c = torch.cat([cond, zp], 1)      # [1,S,1024]
        z_cfg = torch.cat([uncond, zp], 1)  # [1,S,1024]
        x = torch.cat([z_c, z_cfg], 0)      # [2,S,1024]
        ts = t.repeat(2)
        kw = dict(x=x, timesteps=ts, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g2)
        vt = (await dit(**kw)) if engine else dit(x, ts, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g2)
        vt = vt[:, latent_start:]           # [2,4,128]
        v = vt[:1] + GS * (vt[:1] - vt[1:2])
        z = z + (1.0 / NUM_STEPS) * v
    return z


async def solve_mf(dit, coord, cond, g_cond, attn_mask, pos_ids, noise, DT, engine):
    latent_start = cond.shape[1]
    z = noise.clone()
    times = torch.linspace(0.0, 1.0, NFE + 1, dtype=DT)
    for step in range(NFE):
        t = times[step].reshape(1)
        dt = (times[step + 1] - times[step]).reshape(1)
        zp = coord(z)
        z_c = torch.cat([cond, zp], 1)
        kw = dict(x=z_c, timesteps=t, attn_mask=attn_mask, pos_ids=pos_ids, g_cond=g_cond, duration=dt)
        vt = (await dit(**kw)) if engine else dit(z_c, t, attn_mask=attn_mask, pos_ids=pos_ids,
                                                  g_cond=g_cond, duration=dt)
        z = z + vt[:, latent_start:] * dt.view(-1, 1, 1)
    return z


async def run(mode, src, DT):
    torch.set_grad_enabled(False)
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    src = Path(src)
    sd = load_file(str(src / "model.safetensors"))
    coord = _linear(sd, "coordinate_proj", DT)
    cfg = json.loads((src / "config.json").read_text())["DiT"]
    dit_mode = "meanflow" if mode == "mf" else "flow_matching"
    ov = DiTOverlay(cfg, mode=dit_mode).to(DT).eval()
    ov.load_upstream({k: v for k, v in sd.items() if k.startswith("velocity_field_predictor.")})

    z = np.load(ART / ("oracle_ref_mf.npz" if mode == "mf" else "oracle_ref.npz"))
    kx = torch.from_numpy(z["dit.kw_x"]).to(DT)          # [B,5,1024] scattered
    cond = kx[0:1, :1]                                    # [1,1,1024] cond hidden
    attn_mask = torch.from_numpy(z["dit.kw_attn_mask"]).to(DT)
    pos_ids = torch.from_numpy(z["dit.kw_pos_ids"]).to(torch.float32)
    g_all = torch.from_numpy(z["dit.kw_g_cond"]).to(DT)
    g_cond = g_all[0:1]
    noise = torch.from_numpy(z["randn0"]).to(DT)          # replayed first-patch noise

    # engine DiT bundle (S=5)
    names = ("x", "timesteps", "attn_mask", "pos_ids", "g_cond") + (("duration",) if mode == "mf" else ())
    aim = ART / f"dots_dit_{mode}_fp16_s5" / f"dots_dit_{mode}_fp16_s5.aimodel"
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    eng = DiTEngine(fn, names)

    if mode == "mf":
        p_t = await solve_mf(ov, coord, cond, g_cond, attn_mask, pos_ids, noise, DT, engine=False)
        p_e = await solve_mf(eng, coord, cond, g_cond, attn_mask, pos_ids, noise, DT, engine=True)
    else:
        uncond = kx[1:2, :1]
        p_t = await solve_soar(ov, coord, cond, uncond, g_cond, attn_mask, pos_ids, noise, DT, engine=False)
        p_e = await solve_soar(eng, coord, cond, uncond, g_cond, attn_mask, pos_ids, noise, DT, engine=True)

    c = cos(p_e.numpy(), p_t.numpy())
    print(f"  {mode} solver: engine-driven vs torch-driven final patch {tuple(p_e.shape)}  cos={c:.6f}")

    # ---- GOLDEN gate: denormalize(solved patch) vs the oracle's patch_encoder.in_latent_patch ----
    ls_path = src / "latent_stats.pt"
    if ls_path.exists() and "patch_encoder.in_latent_patch" in z.files:
        st = torch.load(str(ls_path), weights_only=False)
        mean = torch.as_tensor(st["mean"]).to(DT); std = torch.sqrt(torch.as_tensor(st["var"]).to(DT))
        golden = torch.from_numpy(z["patch_encoder.in_latent_patch"]).to(torch.float32)
        # solver emits [1,4,128] normalized; denormalize per-channel (last dim=128)
        den_e = (p_e.to(DT) * std + mean).to(torch.float32)
        den_t = (p_t.to(DT) * std + mean).to(torch.float32)
        cg_e = cos(den_e.numpy(), golden.numpy()); cg_t = cos(den_t.numpy(), golden.numpy())
        print(f"  GOLDEN: denorm(engine-patch) vs oracle in_latent_patch cos={cg_e:.6f}  (torch {cg_t:.6f})")
        return min(c, cg_e)
    return c


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["soar", "mf"])
    ap.add_argument("--src", required=True)
    a = ap.parse_args()
    c = await run(a.mode, a.src, torch.float16)
    print(f"\n>>> host_solver/{a.mode}: cos={c:.6f} -> {'PASS' if c >= 0.99 else 'FAIL'}")
    sys.exit(0 if c >= 0.99 else 1)


if __name__ == "__main__":
    asyncio.run(main())
