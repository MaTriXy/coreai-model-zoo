"""Pack everything VibeVoiceRunner (Swift) needs into a flat binary bundle it can read on-device,
so the runner needs neither the vibevoice pkg nor the 272 MB embed_tokens matrix.

Writes device_bundle/:
  meta.json                     shapes + scalars + DPMSolver++ schedule tables (validated cos 1.0)
  text_embeds.f16   [Nt,896]    main-LM input = embed_tokens[tts_text_ids] (pre-looked-up)
  type_emb.f16      [2,896]     tts_input_types (speech=0 / text=1)
  negtts_last.f16   [1,896]     first speech token's CFG-negative condition
  {main,tts,neg}_{k,v}.f16      seeded prefill KV (nl,1,2,L,64)
  eos_fc1_w.f32 [896,896] eos_fc1_b.f32 [896] eos_fc2_w.f32 [896] eos_fc2_b.f32 [1]
  noise.f16         [Ns,2,64]   per-speech-token DDPM init noise (oracle randn{i} -> deterministic gate)

  <coreai-venv>/bin/python pack_device_inputs.py --seed artifacts/e2e_seed.npz --out device_bundle
"""
import argparse, json
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402

HERE = Path(__file__).resolve().parent
SNAP = hf_snapshot("microsoft/VibeVoice-Realtime-0.5B", "model.safetensors")


def w16(path, arr):
    np.ascontiguousarray(arr, dtype=np.float16).tofile(path)


def w32(path, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default=str(HERE / "artifacts/e2e_seed.npz"))
    ap.add_argument("--oracle", default=str(HERE / "artifacts/oracle_ref.npz"))
    ap.add_argument("--out", default=str(HERE / "device_bundle"))
    ap.add_argument("--ddpm-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=1.5)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sd = load_file(SNAP)
    seed = np.load(a.seed)
    z = np.load(a.oracle)
    import sys; sys.path.insert(0, str(HERE))
    from dpm_solver import DPMSolverMultistepScheduler

    embed_tokens = sd["model.language_model.embed_tokens.weight"].float().numpy()  # (V,896)
    text_ids = seed["tts_text_ids"][0].tolist()
    text_embeds = embed_tokens[text_ids]                                            # (Nt,896)
    w16(out / "text_embeds.f16", text_embeds)
    w16(out / "type_emb.f16", sd["model.tts_input_types.weight"].float().numpy())
    w16(out / "negtts_last.f16", seed["negtts_last_hidden"])

    for nm, key in [("main", "lm"), ("tts", "tts"), ("neg", "negtts")]:
        w16(out / f"{nm}_k.f16", seed[f"{key}_k"])
        w16(out / f"{nm}_v.f16", seed[f"{key}_v"])

    w32(out / "eos_fc1_w.f32", sd["tts_eos_classifier.fc1.weight"].float().numpy())
    w32(out / "eos_fc1_b.f32", sd["tts_eos_classifier.fc1.bias"].float().numpy())
    w32(out / "eos_fc2_w.f32", sd["tts_eos_classifier.fc2.weight"].float().numpy())
    w32(out / "eos_fc2_b.f32", sd["tts_eos_classifier.fc2.bias"].float().numpy())

    # noise: one (2,64) per speech token (oracle randn{i}); pad to a generous max
    nmax = int(z["num_randn"][0])
    noise = np.stack([z[f"randn{i}"] for i in range(nmax)], axis=0)                  # (Ns,2,64)
    w16(out / "noise.f16", noise)

    # DPMSolver++ schedule tables at the chosen timesteps (validated cos 1.0 in numpy).
    s = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="cosine", prediction_type="v_prediction")
    s.set_timesteps(a.ddpm_steps)
    ts = [int(t) for t in s.timesteps]
    at = s.alpha_t.numpy().astype(np.float64); sg = s.sigma_t.numpy().astype(np.float64)
    lam = (np.log(at) - np.log(sg))
    sched = [{"t": t, "alpha": float(at[t]), "sigma": float(sg[t]), "lambda": float(lam[t])} for t in ts]

    meta = {
        "hidden": 896, "vae_dim": 64, "hop": 3200, "n_text": len(text_ids),
        "main_prefill_len": int(seed["main_prefill_len"][0]),
        "tts_prefill_len": int(seed["tts_prefill_len"][0]),
        "neg_prefill_len": 1,
        "text_window": 5, "speech_window": 6,
        "num_noise": nmax, "cfg": a.cfg, "ddpm_steps": a.ddpm_steps,
        "scaling": float(z["speech_scaling_factor"]), "bias": float(z["speech_bias_factor"]),
        "schedule": sched,  # per-step {t, alpha, sigma, lambda}; last step target is t=0 (alpha 1, sigma 0)
        "main_layers": 4, "tts_layers": 20, "n_kv": 2, "head_dim": 64,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"packed -> {out}")
    print(f"  n_text={len(text_ids)} num_noise={nmax} timesteps={ts}")
    print(f"  files: {sorted(p.name for p in out.iterdir())}")


if __name__ == "__main__":
    main()
