# Community port — FULLY self-contained on-engine generation: tokens -> cond bundle -> DiT x8 -> VAE.
import json, os, sys, asyncio, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))
import coreai.runtime as rt
from transformers import AutoTokenizer

A = os.path.join(HERE, "artifacts")
COND = os.path.join(A, "sa_cond_fp16", "sa_cond_fp16.aimodel")
DIT = os.path.join(A, "sa_dit_fp16_probe", "sa_dit_fp16_probe.aimodel")
VAE = os.path.join(A, "sa_vae_fp16_s256", "sa_vae_fp16_s256.aimodel")
SR = json.load(open(os.path.join(HERE, "model_config.json")))["sample_rate"]
PROMPT, SECS = "128 BPM tech house drum loop", 11.0

tok = AutoTokenizer.from_pretrained("t5-base")
enc = tok([PROMPT], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
input_ids = enc["input_ids"].int(); attn = enc["attention_mask"].float()
secs_norm = torch.tensor([SECS / 256.0], dtype=torch.float32)

samp = torch.load(os.path.join(HERE, "sampler_traj.pt"))
noise = samp["traj"][0]["x"].float()
tsched = samp["t_schedule"] + [0.0]
ref_audio = torch.load(os.path.join(HERE, "ref_oracle.pt"))["audio"].float()


def nd(a): return rt.NDArray(np.ascontiguousarray(a.detach().numpy().astype(np.float32)))
def ndi(a): return rt.NDArray(np.ascontiguousarray(a.detach().numpy().astype(np.int32)))


async def main():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    cond = (await rt.AIModel.load(COND, gpu)).load_function("main")
    dit = (await rt.AIModel.load(DIT, gpu)).load_function("main")
    vae = (await rt.AIModel.load(VAE, gpu)).load_function("main")

    r = await cond(inputs={"input_ids": ndi(input_ids), "attention_mask": nd(attn), "seconds_norm": nd(secs_norm)})
    cac = torch.as_tensor(r["cross_attn_cond"].numpy().astype(np.float32))
    ge = torch.as_tensor(r["global_embed"].numpy().astype(np.float32))
    mask = torch.as_tensor(r["cond_mask"].numpy().astype(np.float32))
    print(f"[full] conditioner bundle -> cross{tuple(cac.shape)} global{tuple(ge.shape)} mask{tuple(mask.shape)}", flush=True)

    x = noise.clone()
    for i in range(8):
        t = torch.tensor([tsched[i]], dtype=torch.float32)
        rr = await dit(inputs={"x": nd(x), "t": nd(t), "cross_attn_cond": nd(cac),
                               "global_embed": nd(ge), "cross_attn_cond_mask": nd(mask)})
        v = torch.as_tensor(rr["v"].numpy().astype(np.float32))
        x = x + (tsched[i + 1] - tsched[i]) * v
    rr = await vae(inputs={"latent": nd(x)})
    audio = torch.as_tensor(rr["audio"].numpy().astype(np.float32))

    ac = torch.nn.functional.cosine_similarity(audio.reshape(-1), ref_audio.reshape(-1), dim=0).item()
    print(f"[full] FULL on-engine (tokens->cond->DiT x8->VAE) AUDIO vs reference cos={ac:.6f}  "
          f"{'PASS' if ac >= 0.99 else 'CHECK'}")
    import soundfile as sf
    sf.write(os.path.join(HERE, "engine_full_sample.wav"), audio[0].clamp(-1, 1).numpy().T, SR)
    print("[full] saved engine_full_sample.wav")


asyncio.run(main())
