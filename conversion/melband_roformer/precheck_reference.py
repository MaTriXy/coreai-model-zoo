"""Reference precheck for Mel-Band RoFormer (Kim Vocal, MIT).
Loads the lucidrains MelBandRoformer + Kim's checkpoint, separates a real song
with vocals, writes listenable vocals/instrumental, and saves a fixed 8s-chunk
I/O oracle (raw_audio -> vocals) for later engine gating.
Run: coreai-models/.venv/bin/python precheck_reference.py
"""
import os, sys, time
import numpy as np
import torch
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "_ref", "kim")
sys.path.insert(0, REF)

import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer

CFG = os.path.join(REF, "configs", "config_vocals_mel_band_roformer.yaml")
CKPT = os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt")
OUT = os.path.join(HERE, "_precheck")
os.makedirs(OUT, exist_ok=True)

with open(CFG) as f:
    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
print("model cfg:", dict(config.model))
CHUNK = config.inference.chunk_size  # 352800 = 8s @ 44100

model = MelBandRoformer(**dict(config.model))
sd = torch.load(CKPT, map_location="cpu")
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load: missing={len(missing)} unexpected={len(unexpected)}")
if missing:    print("  missing[:5]:", missing[:5])
if unexpected: print("  unexpected[:5]:", unexpected[:5])
model.eval()

# device: try MPS, fall back to CPU on any op gap
dev = "mps" if torch.backends.mps.is_available() else "cpu"
try:
    model = model.to(dev)
except Exception as e:
    print("to(mps) failed, cpu:", e); dev = "cpu"; model = model.to("cpu")
print("device:", dev)

# --- get a real song WITH vocals ---
import librosa
song = librosa.example("fishin")  # Karissa Hobbs - Let's Go Fishin' (vocals)
wav, sr = librosa.load(song, sr=44100, mono=False)
if wav.ndim == 1:
    wav = np.stack([wav, wav], 0)  # -> stereo [2, N]
print("song:", wav.shape, "sr", sr, "dur %.1fs" % (wav.shape[1] / sr))

# pick a musically-active 8s chunk (mid-song); clamp to length
off = min(int(45 * sr), max(0, wav.shape[1] - CHUNK))
chunk = wav[:, off:off + CHUNK]
if chunk.shape[1] < CHUNK:
    chunk = np.pad(chunk, ((0, 0), (0, CHUNK - chunk.shape[1])))
mix = torch.tensor(chunk, dtype=torch.float32)  # [2, CHUNK]

def run(x):
    with torch.no_grad():
        return model(x.unsqueeze(0).to(dev)).float().cpu()[0]  # [2, CHUNK] vocals

# warm + timed
try:
    t0 = time.time(); vocals = run(mix); dt = time.time() - t0
except Exception as e:
    print("MPS run failed, retry CPU:", e)
    dev = "cpu"; model = model.to("cpu")
    t0 = time.time(); vocals = run(mix); dt = time.time() - t0

inst = mix - vocals
print(f"\nchunk run {dt:.2f}s ({CHUNK/sr/dt:.1f}x realtime on {dev})")
print("mix     absmean %.4f rms %.4f" % (mix.abs().mean(), mix.pow(2).mean().sqrt()))
print("vocals  absmean %.4f rms %.4f" % (vocals.abs().mean(), vocals.pow(2).mean().sqrt()))
print("instrum absmean %.4f rms %.4f" % (inst.abs().mean(), inst.pow(2).mean().sqrt()))

sf.write(os.path.join(OUT, "mix_chunk.wav"), mix.T.numpy(), sr, subtype="FLOAT")
sf.write(os.path.join(OUT, "vocals_chunk.wav"), vocals.T.numpy(), sr, subtype="FLOAT")
sf.write(os.path.join(OUT, "instrumental_chunk.wav"), inst.T.numpy(), sr, subtype="FLOAT")

torch.save({"raw_audio": mix, "vocals": vocals, "chunk": CHUNK, "sr": sr},
           os.path.join(OUT, "ref_oracle.pt"))
print("\nsaved:", OUT, "(mix/vocals/instrumental wav + ref_oracle.pt)")
print("PRECHECK OK" if vocals.abs().mean() > 1e-4 else "PRECHECK WARN: near-silent vocals")
