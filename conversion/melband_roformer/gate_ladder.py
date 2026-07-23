"""Torch ladder gate: host_stft -> SepCore -> host_istft  vs  reference model.
Target cos >= 0.999 (should be ~1.0, same math). CPU (ref complex scatter needs CPU).
"""
import os, sys, time
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))
import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer
from export_core import SepCore, HostDSP

CFG = os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")
CKPT = os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt")

with open(CFG) as f:
    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
model = MelBandRoformer(**dict(config.model)).eval()
model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
raw = oracle["raw_audio"].unsqueeze(0)           # [1, 2, 352800]
vocals_ref = oracle["vocals"].unsqueeze(0)       # [1, 2, 352800]  (cached ref output)

core = SepCore(model).eval()
host = HostDSP(model)

def cosmax(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    cos = torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)
    return cos.item(), (a - b).abs().max().item()

with torch.no_grad():
    # sanity: re-run reference now to confirm oracle reproduces
    t0 = time.time(); ref2 = model(raw); t_ref = time.time() - t0
    c0, m0 = cosmax(ref2, vocals_ref)
    print(f"ref reproduce vs oracle: cos {c0:.7f} max|d| {m0:.2e}  ({t_ref:.2f}s)")

    t0 = time.time()
    stft_real = host.stft(raw)                    # [1, 2050, T, 2]
    masked = core(stft_real)                      # [1, 2050, T, 2]
    vocals_core = host.istft(masked, length=raw.shape[-1])
    t_core = time.time() - t0
    print("stft_real", tuple(stft_real.shape), "vocals_core", tuple(vocals_core.shape))

c, m = cosmax(vocals_core, ref2)
print(f"\nCORE pipeline vs reference: cos {c:.7f}  max|d| {m:.2e}  ({t_core:.2f}s)")
print("LADDER PASS" if c >= 0.999 else "LADDER FAIL")
