# Community port — end-to-end ON-ENGINE pipeline: DiT bundle (8-step rf-euler) -> VAE bundle -> audio.
"""Drives the exported DiT + VAE Core AI bundles with the host sampler loop, reproducing the
reference generation. Conditioning + noise are taken from the captured oracle (the conditioner
export is a separate step). Gate = audio cos vs ref_oracle + a listenable wav.
"""
import json, os, sys, asyncio, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
import coreai.runtime as rt

DIT = os.path.join(HERE, "artifacts", "sa_dit_fp16_probe", "sa_dit_fp16_probe.aimodel")
VAE = os.path.join(HERE, "artifacts", "sa_vae_fp16_s256", "sa_vae_fp16_s256.aimodel")

io = torch.load(os.path.join(HERE, "dit_io.pt"))
cac = io["cross_attn_cond"].float(); ge = io["global_embed"].float(); mask = io["cross_attn_cond_mask"].float()
samp = torch.load(os.path.join(HERE, "sampler_traj.pt"))
noise = samp["traj"][0]["x"].float()                      # x at step 0 = the seed noise
tsched = samp["t_schedule"] + [0.0]                       # 8 steps -> append 0
ref_audio = torch.load(os.path.join(HERE, "ref_oracle.pt"))["audio"].float()
SR = json.load(open(os.path.join(HERE, "model_config.json")))["sample_rate"]


def nd(a): return rt.NDArray(np.ascontiguousarray(a.detach().numpy().astype(np.float32)))


async def main():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    dit = (await rt.AIModel.load(DIT, gpu)).load_function("main")
    vae = (await rt.AIModel.load(VAE, gpu)).load_function("main")

    x = noise.clone()
    for i in range(8):
        t = torch.tensor([tsched[i]], dtype=torch.float32)
        r = await dit(inputs={"x": nd(x), "t": nd(t), "cross_attn_cond": nd(cac),
                              "global_embed": nd(ge), "cross_attn_cond_mask": nd(mask)})
        v = torch.as_tensor(r["v"].numpy().astype(np.float32))
        dt = tsched[i + 1] - tsched[i]
        x = x + dt * v
        print(f"  step {i}: t={tsched[i]:.4f} dt={dt:+.4f}  x std={x.std():.4f}", flush=True)

    # final latent vs reference latent
    ref_lat = samp["final_latent"].float()
    lc = torch.nn.functional.cosine_similarity(x.reshape(-1), ref_lat.reshape(-1), dim=0).item()
    print(f"[e2e] engine latent vs reference latent cos={lc:.6f}")

    r = await vae(inputs={"latent": nd(x)})
    audio = torch.as_tensor(r["audio"].numpy().astype(np.float32))
    ac = torch.nn.functional.cosine_similarity(audio.reshape(-1), ref_audio.reshape(-1), dim=0).item()
    print(f"[e2e] engine AUDIO vs reference cos={ac:.6f}  {'PASS' if ac >= 0.99 else 'CHECK'}")

    import soundfile as sf
    sf.write(os.path.join(HERE, "engine_sample.wav"), audio[0].clamp(-1, 1).numpy().T, SR)
    print("[e2e] saved engine_sample.wav")


asyncio.run(main())
