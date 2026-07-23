# Community port — NOT an Apple model.
"""VibeVoice-Realtime-0.5B multi-speaker dialogue = host-level turn switching (the zoo differentiator).

The 0.5B streaming model ships single-speaker voice presets (one .pt per speaker). A dialogue is
produced by generating each "Speaker N:" turn independently with that speaker's voice seed and
concatenating — entirely on Core AI engines. Pairs with the shipped Sortformer diarization
(generate a conversation -> diarize who-said-what, all on-device).

Each turn: seed 3 KV streams from the turn's .pt, run the generate() loop with fresh random noise,
whole-sequence-decode, append audio. Reuses EngineBackbone + the sampler from host_e2e.

  PYTHONPATH=. <coreai-venv>/bin/python host_multispeaker.py \
      --turns seed_turn0.npz seed_turn1.npz --out artifacts/conversation.wav
"""
from __future__ import annotations
import argparse, asyncio, glob, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dpm_solver import DPMSolverMultistepScheduler
from host_e2e import EngineBackbone, ART, SNAP, TW, SW, VAE_DIM

DEC_T = 64  # per-turn decode buffer (~8.5 s); turns are short


async def main():
    import coreai.runtime as rt
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", nargs="+", default=["seed_turn0.npz", "seed_turn1.npz"])
    ap.add_argument("--out", default=str(ART / "conversation.wav"))
    ap.add_argument("--ddpm-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--gap-ms", type=int, default=250)
    a = ap.parse_args()
    buf = a.cache_len
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    def load(name):
        return rt.AIModel.load(glob.glob(str(ART / name / f"{name}.aimodel"))[0], gpu)

    main_fn = (await load(f"vibevoice_mainlm_fp16_decode_cl{buf}")).load_function("main")
    tts_fn = (await load(f"vibevoice_ttslm_fp16_decode_cl{buf}")).load_function("main")
    neg_fn = (await load(f"vibevoice_ttslm_fp16_decode_cl{buf}")).load_function("main")
    head_fn = (await load("vibevoice_diffusion_head_fp16")).load_function("main")
    conn_fn = (await load("vibevoice_connector_fp16")).load_function("main")
    dec_fn = (await load(f"vibevoice_decoder_fp16_t{DEC_T}")).load_function("main")

    sd = load_file(SNAP)
    z = np.load(ART / "oracle_ref.npz")
    embed_tokens = sd["model.language_model.embed_tokens.weight"].float().numpy()
    type_emb = sd["model.tts_input_types.weight"].float().numpy()
    eos_w1 = sd["tts_eos_classifier.fc1.weight"].float(); eos_b1 = sd["tts_eos_classifier.fc1.bias"].float()
    eos_w2 = sd["tts_eos_classifier.fc2.weight"].float(); eos_b2 = sd["tts_eos_classifier.fc2.bias"].float()
    scaling = float(z["speech_scaling_factor"]); bias = float(z["speech_bias_factor"])
    sched = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="cosine", prediction_type="v_prediction")

    def eos_logit(h):
        x = torch.relu(torch.nn.functional.linear(torch.from_numpy(h).float(), eos_w1, eos_b1))
        return torch.sigmoid(torch.nn.functional.linear(x, eos_w2, eos_b2))[0, 0].item()

    async def ddpm_sample(pos_c, neg_c, noise):
        sched.set_timesteps(a.ddpm_steps)
        condition = np.concatenate([pos_c, neg_c], axis=0).astype(np.float16)
        speech = torch.from_numpy(noise).float()
        for t in sched.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0).numpy().astype(np.float16)
            r = await head_fn(inputs={"noisy_images": rt.NDArray(np.ascontiguousarray(combined)),
                                      "timesteps": rt.NDArray(np.ascontiguousarray(np.full((2,), float(t), np.float16))),
                                      "condition": rt.NDArray(np.ascontiguousarray(condition))})
            eps = torch.from_numpy(r["eps"].numpy()).float()
            ce, ue = torch.split(eps, len(eps) // 2, dim=0)
            he = ue + a.cfg * (ce - ue)
            speech = sched.step(torch.cat([he, he], dim=0), t, speech).prev_sample
        return speech[: len(speech) // 2].numpy()

    async def gen_turn(seed_path, rng_seed):
        seed = np.load(ART / seed_path if not Path(seed_path).is_absolute() else seed_path)
        main = EngineBackbone(main_fn, 4, 2, 64, buf); main.seed(seed["lm_k"], seed["lm_v"], int(seed["main_prefill_len"][0]))
        tts = EngineBackbone(tts_fn, 20, 2, 64, buf); tts.seed(seed["tts_k"], seed["tts_v"], int(seed["tts_prefill_len"][0]))
        neg = EngineBackbone(neg_fn, 20, 2, 64, buf); neg.seed(seed["negtts_k"], seed["negtts_v"], 1)
        text_ids = seed["tts_text_ids"][0].tolist()
        neg_cond = seed["negtts_last_hidden"].astype(np.float32)
        rng = np.random.default_rng(rng_seed)
        tts_last, latents, win_idx, finished = None, [], 0, False
        while not finished:
            window = text_ids[win_idx * TW:(win_idx + 1) * TW]; win_idx += 1
            if len(window) > 0:
                for tok in window:
                    h = await main.step(embed_tokens[tok].reshape(1, 1, -1))
                    tts_last = await tts.step(h + type_emb[1].reshape(1, 1, -1))
            for _ in range(SW):
                latent = await ddpm_sample(tts_last[:, -1, :], neg_cond, rng.standard_normal((2, VAE_DIM)).astype(np.float32))
                latents.append(latent)
                emb = (await conn_fn(inputs={"features": rt.NDArray(np.ascontiguousarray(latent.reshape(1, 1, VAE_DIM).astype(np.float16)))}))["embed"].numpy()
                tts_last = await tts.step(emb + type_emb[0].reshape(1, 1, -1))
                neg_cond = (await neg.step(emb + type_emb[0].reshape(1, 1, -1)))[:, -1, :]
                if eos_logit(tts_last[:, -1, :]) > 0.5 or len(latents) >= DEC_T:
                    finished = True; break
        N = len(latents)
        padded = np.zeros((DEC_T, VAE_DIM), np.float32); padded[:N] = np.concatenate(latents, axis=0)
        scaled = (padded / scaling - bias).T[None].astype(np.float16)
        audio = (await dec_fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(scaled))}))["audio"].numpy()[0, 0, :N * 3200]
        print(f"  turn {seed_path}: {N} latents -> {N*3200/24000:.2f}s")
        return audio.astype(np.float32)

    gap = np.zeros(int(a.gap_ms * 24), np.float32)
    pieces = []
    for i, tp in enumerate(a.turns):
        pieces.append(await gen_turn(tp, rng_seed=1234 + i))
        pieces.append(gap)
    convo = np.concatenate(pieces)
    import soundfile as sf
    sf.write(a.out, convo, 24000)
    print(f"\n>>> {len(a.turns)}-speaker conversation: {len(convo)/24000:.2f}s -> {a.out}")


if __name__ == "__main__":
    asyncio.run(main())
