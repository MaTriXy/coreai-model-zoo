# Community port — generate several DIFFERENT prompts via the fp16 engine bundles (show it's not just drums).
import json, os, sys, asyncio, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
import coreai.runtime as rt
from transformers import AutoTokenizer

A = os.path.join(HERE, "artifacts")
COND = os.path.join(A, "sa_cond_fp16b", "sa_cond_fp16b.aimodel")
DIT = os.path.join(A, "sa_dit_fp16", "sa_dit_fp16.aimodel")
VAE = os.path.join(A, "sa_vae_fp16", "sa_vae_fp16.aimodel")
SR = 44100
tSched = [1.0, 0.9943756, 0.9844802, 0.9579123, 0.8909032, 0.7455466, 0.5124974, 0.27388501, 0.0]
PROMPTS = [
    ("ambient", "ambient pad, slow and dreamy, ethereal, no drums"),
    ("orchestral", "epic orchestral trailer, soaring strings and brass"),
    ("piano", "gentle solo piano melody, emotional"),
    ("bass", "funky disco bassline with synth stabs, 120 BPM"),
]
tok = AutoTokenizer.from_pretrained("t5-base")
def ndf(a): return rt.NDArray(np.ascontiguousarray(a.astype(np.float16)))
def ndi(a): return rt.NDArray(np.ascontiguousarray(a.astype(np.int32)))

async def main():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    cond = (await rt.AIModel.load(COND, gpu)).load_function("main")
    dit = (await rt.AIModel.load(DIT, gpu)).load_function("main")
    vae = (await rt.AIModel.load(VAE, gpu)).load_function("main")
    import soundfile as sf
    rng = np.random.default_rng(0)
    for tag, prompt in PROMPTS:
        enc = tok([prompt], truncation=True, max_length=64, padding="max_length", return_tensors="np")
        ids = enc["input_ids"].astype(np.int32); attn = enc["attention_mask"].astype(np.float16)
        r = await cond(inputs={"input_ids": ndi(ids), "attention_mask": ndf(attn),
                               "seconds_norm": ndf(np.array([11/256.0]))})
        cac = r["cross_attn_cond"].numpy(); ge = r["global_embed"].numpy(); mask = r["cond_mask"].numpy()
        x = rng.standard_normal((1, 64, 256)).astype(np.float32)
        for i in range(8):
            rr = await dit(inputs={"x": ndf(x), "t": ndf(np.array([tSched[i]])),
                                   "cross_attn_cond": ndf(cac), "global_embed": ndf(ge), "cross_attn_cond_mask": ndf(mask)})
            v = rr["v"].numpy().astype(np.float32)
            x = x + (tSched[i+1]-tSched[i]) * v
        rr = await vae(inputs={"latent": ndf(x)})
        audio = rr["audio"].numpy().astype(np.float32)[0]   # [2,N]
        out = os.path.join(HERE, f"demo_{tag}.wav")
        sf.write(out, np.clip(audio.T, -1, 1), SR)
        print(f"[var] {tag}: peak={np.abs(audio).max():.2f} rms={np.sqrt((audio**2).mean()):.3f} -> {out}", flush=True)
asyncio.run(main())
