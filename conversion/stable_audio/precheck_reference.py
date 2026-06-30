# Community port — NOT a Stability model. Reference oracle run for the Stable Audio port.
"""Load stable-audio-open-small (stable-audio-tools reference) and generate one sample with a fixed
seed — the oracle for gating my exportable overlays. Captures: the generated latent, the decoded
waveform, and shapes/stats. CPU/fp32, deterministic.

  coreai-models/.venv/bin/python precheck_reference.py
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))

from stable_audio_tools.models.factory import create_model_from_config  # noqa: E402
from stable_audio_tools.models.utils import load_ckpt_state_dict  # noqa: E402
from stable_audio_tools.inference.generation import generate_diffusion_cond  # noqa: E402

import time  # noqa: E402

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = json.load(open(os.path.join(HERE, "model_config.json")))
SR = cfg["sample_rate"]
SS = cfg["sample_size"]
print(f"[ref] device={DEV} sample_rate={SR} sample_size={SS} (~{SS/SR:.1f}s)", flush=True)

model = create_model_from_config(cfg)
sd = load_ckpt_state_dict(os.path.join(HERE, "model.safetensors"))
miss, unexp = model.load_state_dict(sd, strict=False)
miss = [m for m in miss if "t5" not in m.lower()]  # T5 is fetched by the conditioner itself
print(f"[ref] loaded; missing(non-t5)={len(miss)} unexpected={len(unexp)}", flush=True)
model = model.to(DEV).float().eval()

COND = [{"prompt": "128 BPM tech house drum loop", "seconds_total": 11}]

with torch.inference_mode():
    # 1) sampler ONLY (T5 + 8-step DiT) — latent oracle
    t0 = time.time()
    lat = generate_diffusion_cond(model=model, steps=8, cfg_scale=1.0, conditioning=COND,
                                  sample_size=SS, sample_rate=SR, seed=0, device=DEV,
                                  return_latents=True)
    lat = lat[0] if isinstance(lat, (tuple, list)) else lat
    print(f"[ref] sampler 8-step: {time.time()-t0:.1f}s  latent={tuple(lat.shape)} "
          f"mean={lat.float().mean():.4f} std={lat.float().std():.4f}", flush=True)
    # 2) VAE decode ONCE — latent -> waveform
    t0 = time.time()
    audio = model.pretransform.decode(lat.to(DEV).float())
    print(f"[ref] VAE decode: {time.time()-t0:.1f}s  audio={tuple(audio.shape)} "
          f"min={audio.min():.3f} max={audio.max():.3f} absmean={audio.abs().mean():.4f}", flush=True)

torch.save({"latent": lat.cpu(), "audio": audio.cpu(), "cond": COND},
           os.path.join(HERE, "ref_oracle.pt"))
import torchaudio  # noqa: E402
wav = audio[0].cpu().to(torch.float32).clamp(-1, 1)
torchaudio.save(os.path.join(HERE, "ref_sample.wav"), wav, SR)
print("[ref] saved ref_oracle.pt + ref_sample.wav")
