# Community port — NOT an Apple model.
"""VibeVoice-Realtime-0.5B host: replicate the upstream streaming generate() loop with the ported
components. Runs in the coreai venv (no vibevoice pkg). Two backends per component (--backend):
  * torch  — fp32 overlays / static-KV backbones (validate the loop LOGIC; expect ~1.0 vs oracle).
  * engine — Core AI .aimodel engines (int8 LMs, fp16 head/decoder; measures real quant drift).

Deterministic E2E gate: feed the oracle's captured per-token noise (randn{i}) so the whole trajectory
must reproduce the oracle latents + wav. Seeds the 3 KV streams from artifacts/e2e_seed.npz.

  PYTHONPATH=. <coreai-venv>/bin/python host_e2e.py --backend torch --ddpm-steps 5 --cfg 1.5
"""
from __future__ import annotations
import argparse, asyncio, glob, sys
from pathlib import Path
import numpy as np, torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402
from dpm_solver import DPMSolverMultistepScheduler
from torch_overlays import DiffusionHeadOverlay, ConnectorOverlay
from decoder_ref import DecoderOverlay
from backbone import load_backbone, build_kv_state, Qwen2Cfg

ART = HERE / "artifacts"
SNAP = hf_snapshot("microsoft/VibeVoice-Realtime-0.5B", "model.safetensors")
TW, SW = 5, 6  # text / speech window sizes
VAE_DIM = 64


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    n = min(a.numel(), b.numel())
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0).item()


class TorchBackbone:
    """fp32 static-KV backbone wrapper with its own KV buffers + position counter."""
    def __init__(self, sd, prefix, nl, buf, final_norm):
        self.bb = load_backbone(sd, prefix, nl, buf, final_norm=final_norm, dtype=torch.float32)
        self.kc, self.vc = build_kv_state(self.bb.cfg, buf, torch.float32)
        self.pos = 0

    def seed(self, K, V, prefill_len):
        # K,V are (nl,1,nkv,L,hd) — the upstream post-RoPE cached K/V — written straight into the buffer.
        self.kc[:, :, :, :prefill_len, :] = torch.from_numpy(K).float()
        self.vc[:, :, :, :prefill_len, :] = torch.from_numpy(V).float()
        self.pos = prefill_len

    def step(self, emb):  # emb (1,1,896) -> hidden (1,1,896)
        h = self.bb.decode(emb.float(), torch.tensor([self.pos], dtype=torch.int32), self.kc, self.vc)
        self.pos += 1
        return h


def run_torch(args):
    sd = load_file(SNAP)
    z = np.load(ART / "oracle_ref.npz")
    seed = np.load(ART / "e2e_seed.npz")
    buf = args.cache_len

    embed_tokens = sd["model.language_model.embed_tokens.weight"].float()  # (V,896)
    type_emb = sd["model.tts_input_types.weight"].float()                  # (2,896)
    eos_w1 = sd["tts_eos_classifier.fc1.weight"].float(); eos_b1 = sd["tts_eos_classifier.fc1.bias"].float()
    eos_w2 = sd["tts_eos_classifier.fc2.weight"].float(); eos_b2 = sd["tts_eos_classifier.fc2.bias"].float()
    scaling = float(z["speech_scaling_factor"]); bias = float(z["speech_bias_factor"])

    head = DiffusionHeadOverlay().float().eval().load_upstream(sd)
    conn = ConnectorOverlay().float().eval().load_upstream(sd)
    decoder = DecoderOverlay().float().eval().load_upstream(sd)

    main = TorchBackbone(sd, "model.language_model.", 4, buf, final_norm=False)
    tts = TorchBackbone(sd, "model.tts_language_model.", 20, buf, final_norm=True)
    neg = TorchBackbone(sd, "model.tts_language_model.", 20, buf, final_norm=True)
    main.seed(seed["lm_k"], seed["lm_v"], int(seed["main_prefill_len"][0]))
    tts.seed(seed["tts_k"], seed["tts_v"], int(seed["tts_prefill_len"][0]))
    neg.seed(seed["negtts_k"], seed["negtts_v"], 1)

    text_ids = seed["tts_text_ids"][0].tolist()
    neg_cond = torch.from_numpy(seed["negtts_last_hidden"]).float()  # (1,896)  first CFG-negative
    tts_last = None

    sched = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="cosine", prediction_type="v_prediction")

    def eos_logit(h):  # h (1,896)
        x = torch.relu(torch.nn.functional.linear(h, eos_w1, eos_b1))
        return torch.sigmoid(torch.nn.functional.linear(x, eos_w2, eos_b2))[0, 0].item()

    def ddpm_sample(pos_c, neg_c, noise):
        sched.set_timesteps(args.ddpm_steps)
        condition = torch.cat([pos_c, neg_c], dim=0).float()          # (2,896)
        speech = torch.from_numpy(noise).float()                      # (2,64)  captured init noise
        for t in sched.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = head(combined, t.repeat(combined.shape[0]).float(), condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + args.cfg * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = sched.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]                             # (1,64)

    latents, win_idx, noise_i, finished = [], 0, 0, False
    while not finished:
        window = text_ids[win_idx * TW:(win_idx + 1) * TW]
        win_idx += 1
        if len(window) > 0:
            main_hs = [main.step(embed_tokens[tok].reshape(1, 1, -1)) for tok in window]
            for h in main_hs:
                tts_last = tts.step(h + type_emb[1].reshape(1, 1, -1))
        for _ in range(SW):
            pos_cond = tts_last[:, -1, :]
            noise = z[f"randn{noise_i}"] if f"randn{noise_i}" in z else np.random.randn(2, VAE_DIM).astype(np.float32)
            latent = ddpm_sample(pos_cond, neg_cond, noise)          # (1,64)
            noise_i += 1
            latents.append(latent.detach().numpy())
            acoustic_embed = conn(latent.reshape(1, 1, VAE_DIM))     # (1,1,896)
            tts_last = tts.step(acoustic_embed + type_emb[0].reshape(1, 1, -1))
            neg_full = neg.step(acoustic_embed + type_emb[0].reshape(1, 1, -1))
            neg_cond = neg_full[:, -1, :]
            if eos_logit(tts_last[:, -1, :]) > 0.5:
                finished = True
                break
            if len(latents) >= int(z["num_latents"][0]):
                finished = True
                break

    # gate vs oracle latents + full non-streaming decode
    N = len(latents)
    print(f"host produced {N} latents (oracle {int(z['num_latents'][0])})")
    lat_cos = [cos(latents[i], z[f"latent{i}"]) for i in range(min(N, int(z['num_latents'][0])))]
    print(f"  latent cos: min={min(lat_cos):.6f} mean={np.mean(lat_cos):.6f}  first5={['%.4f'%c for c in lat_cos[:5]]}")

    lat_stack = torch.from_numpy(np.stack(latents, axis=2)).squeeze(0).unsqueeze(0).float()  # (1,64,N)
    scaled = lat_stack / scaling - bias
    with torch.inference_mode():
        audio = decoder(scaled).float().numpy()
    dz = np.load(ART / "dec_ref.npz")
    wc = cos(audio, dz["audio_full"][:, :, :N * 3200])
    print(f"  wav (host decode) vs oracle non-stream decode: cos={wc:.6f}")
    import soundfile as sf
    sf.write(str(ART / "host_e2e_torch.wav"), audio.reshape(-1), 24000)
    print(f"  -> artifacts/host_e2e_torch.wav")
    return min(lat_cos), wc


class EngineBackbone:
    """Core AI decode(q=1) engine wrapper with a seeded fp16 KV state buffer + position counter."""
    def __init__(self, fn, nl, nkv, hd, buf):
        self.fn = fn
        self.buf = buf
        self.k = np.zeros((nl, 1, nkv, buf, hd), np.float16)
        self.v = np.zeros((nl, 1, nkv, buf, hd), np.float16)
        self.pos = 0

    def seed(self, K, V, prefill_len):
        self.k[:, :, :, :prefill_len, :] = K.astype(np.float16)
        self.v[:, :, :, :prefill_len, :] = V.astype(np.float16)
        self.pos = prefill_len

    async def step(self, emb):  # emb (1,1,896) np -> hidden (1,1,896) np
        import coreai.runtime as rt
        state = {"keyCache": rt.NDArray(self.k), "valueCache": rt.NDArray(self.v)}
        r = await self.fn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(emb.astype(np.float16))),
                                  "pos": rt.NDArray(np.ascontiguousarray(np.array([self.pos], np.int32)))},
                          state=state)
        self.k = state["keyCache"].numpy(); self.v = state["valueCache"].numpy()  # mutated in place
        self.pos += 1
        return r["hidden"].numpy()


async def run_engine(args):
    import coreai.runtime as rt
    sd = load_file(SNAP)
    z = np.load(ART / "oracle_ref.npz")
    seed = np.load(ART / "e2e_seed.npz")
    buf = args.cache_len
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    def load(name):
        aim = glob.glob(str(ART / f"{name}" / f"{name}.aimodel"))[0]
        return rt.AIModel.load(aim, gpu)

    main_mode = args.main_mode or args.lm_mode
    tts_mode = args.lm_mode
    main_fn = (await load(f"vibevoice_mainlm_{main_mode}_decode_cl{buf}")).load_function("main")
    tts_fn = (await load(f"vibevoice_ttslm_{tts_mode}_decode_cl{buf}")).load_function("main")
    neg_fn = (await load(f"vibevoice_ttslm_{tts_mode}_decode_cl{buf}")).load_function("main")
    print(f"  LMs: main={main_mode} tts={tts_mode}")
    head_fn = (await load("vibevoice_diffusion_head_fp16")).load_function("main")
    conn_fn = (await load("vibevoice_connector_fp16")).load_function("main")
    dec_fn = (await load("vibevoice_decoder_fp16_t30")).load_function("main")
    DEC_T = 30

    embed_tokens = sd["model.language_model.embed_tokens.weight"].float().numpy()
    type_emb = sd["model.tts_input_types.weight"].float().numpy()
    eos_w1 = sd["tts_eos_classifier.fc1.weight"].float(); eos_b1 = sd["tts_eos_classifier.fc1.bias"].float()
    eos_w2 = sd["tts_eos_classifier.fc2.weight"].float(); eos_b2 = sd["tts_eos_classifier.fc2.bias"].float()
    scaling = float(z["speech_scaling_factor"]); bias = float(z["speech_bias_factor"])

    main = EngineBackbone(main_fn, 4, 2, 64, buf); main.seed(seed["lm_k"], seed["lm_v"], int(seed["main_prefill_len"][0]))
    tts = EngineBackbone(tts_fn, 20, 2, 64, buf); tts.seed(seed["tts_k"], seed["tts_v"], int(seed["tts_prefill_len"][0]))
    neg = EngineBackbone(neg_fn, 20, 2, 64, buf); neg.seed(seed["negtts_k"], seed["negtts_v"], 1)

    text_ids = seed["tts_text_ids"][0].tolist()
    neg_cond = seed["negtts_last_hidden"].astype(np.float32)  # (1,896)
    tts_last = None
    sched = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="cosine", prediction_type="v_prediction")

    def eos_logit(h):
        x = torch.relu(torch.nn.functional.linear(torch.from_numpy(h).float(), eos_w1, eos_b1))
        return torch.sigmoid(torch.nn.functional.linear(x, eos_w2, eos_b2))[0, 0].item()

    async def ddpm_sample(pos_c, neg_c, noise):
        sched.set_timesteps(args.ddpm_steps)
        condition = np.concatenate([pos_c, neg_c], axis=0).astype(np.float16)  # (2,896)
        speech = torch.from_numpy(noise).float()                              # (2,64)
        for t in sched.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0).numpy().astype(np.float16)
            tv = np.full((2,), float(t), np.float16)
            r = await head_fn(inputs={"noisy_images": rt.NDArray(np.ascontiguousarray(combined)),
                                      "timesteps": rt.NDArray(np.ascontiguousarray(tv)),
                                      "condition": rt.NDArray(np.ascontiguousarray(condition))})
            eps = torch.from_numpy(r["eps"].numpy()).float()
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + args.cfg * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = sched.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2].numpy()                            # (1,64)

    latents, win_idx, noise_i, finished = [], 0, 0, False
    while not finished:
        window = text_ids[win_idx * TW:(win_idx + 1) * TW]; win_idx += 1
        if len(window) > 0:
            for tok in window:
                h = await main.step(embed_tokens[tok].reshape(1, 1, -1))
                tts_last = await tts.step(h + type_emb[1].reshape(1, 1, -1))
        for _ in range(SW):
            pos_cond = tts_last[:, -1, :]
            noise = z[f"randn{noise_i}"] if f"randn{noise_i}" in z else np.random.randn(2, VAE_DIM).astype(np.float32)
            latent = await ddpm_sample(pos_cond, neg_cond, noise); noise_i += 1
            latents.append(latent)
            r = await conn_fn(inputs={"features": rt.NDArray(np.ascontiguousarray(latent.reshape(1, 1, VAE_DIM).astype(np.float16)))})
            acoustic_embed = r["embed"].numpy()
            tts_last = await tts.step(acoustic_embed + type_emb[0].reshape(1, 1, -1))
            neg_full = await neg.step(acoustic_embed + type_emb[0].reshape(1, 1, -1))
            neg_cond = neg_full[:, -1, :]
            if eos_logit(tts_last[:, -1, :]) > 0.5 or len(latents) >= int(z["num_latents"][0]):
                finished = True; break

    N = len(latents)
    print(f"host produced {N} latents (oracle {int(z['num_latents'][0])})")
    lat_cos = [cos(latents[i], z[f"latent{i}"]) for i in range(min(N, int(z['num_latents'][0])))]
    print(f"  latent cos: min={min(lat_cos):.6f} mean={np.mean(lat_cos):.6f}  first5={['%.4f'%c for c in lat_cos[:5]]}")

    # decode: pad latents to DEC_T (causal -> first N frames unaffected), trim to N*3200
    lat_stack = np.concatenate(latents, axis=0)                          # (N,64)
    padded = np.zeros((DEC_T, VAE_DIM), np.float32); padded[:N] = lat_stack
    scaled = (padded / scaling - bias).T[None].astype(np.float16)        # (1,64,DEC_T)
    r = await dec_fn(inputs={"latents": rt.NDArray(np.ascontiguousarray(scaled))})
    audio = r["audio"].numpy()[:, :, :N * 3200]
    dz = np.load(ART / "dec_ref.npz")
    wc = cos(audio, dz["audio_full"][:, :, :N * 3200])
    print(f"  wav (engine) vs oracle non-stream decode: cos={wc:.6f}")
    import soundfile as sf
    sf.write(str(ART / "host_e2e_engine.wav"), audio.reshape(-1).astype(np.float32), 24000)
    print(f"  -> artifacts/host_e2e_engine.wav")
    return min(lat_cos), wc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="torch", choices=["torch", "engine"])
    ap.add_argument("--ddpm-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--lm-mode", default="int8", choices=["fp16", "int8", "int4"], help="tts LM mode (feedback path)")
    ap.add_argument("--main-mode", default=None, choices=["fp16", "int8", "int4"], help="main LM mode (defaults to --lm-mode)")
    a = ap.parse_args()
    if a.backend == "torch":
        lat, wav = run_torch(a)
    else:
        lat, wav = asyncio.run(run_engine(a))
    ok = lat >= 0.99 and wav >= 0.99
    print(f"\n>>> host E2E {a.backend}: min latent cos={lat:.6f}  wav cos={wav:.6f} -> {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
