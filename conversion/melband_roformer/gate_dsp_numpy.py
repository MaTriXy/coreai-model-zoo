"""Verify a plain-numpy STFT/iSTFT recipe == torch.stft/istft (the HostDSP the
engine gate used), so the Swift/vDSP host can transcribe it with confidence.
Recipe (torch center=True, normalized=False): Hann-periodic win 2048, reflect-pad
n_fft//2 both sides, frame stride hop=441, rfft; iSTFT = irfft*win overlap-add /
sum(win^2 overlaps), trim pad, length C.
"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))
import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer
from export_core import SepCore, HostDSP

N_FFT, HOP, WIN = 2048, 441, 2048
PAD = N_FFT // 2
window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float64)  # Hann periodic

def np_stft(audio):  # audio [s, T] -> stft_real [ (f s), Tf, 2 ]  (f*2+s layout)
    s, T = audio.shape
    xp = np.stack([np.pad(audio[c], PAD, mode="reflect") for c in range(s)])  # [s, T+2*PAD]
    nfr = 1 + (xp.shape[1] - N_FFT) // HOP
    specs = np.zeros((s, N_FFT // 2 + 1, nfr), np.complex128)
    for c in range(s):
        for i in range(nfr):
            fr = xp[c, i * HOP:i * HOP + N_FFT] * window
            specs[c, :, i] = np.fft.rfft(fr)
    f = N_FFT // 2 + 1
    out = np.zeros((f * s, nfr, 2))
    for c in range(s):
        out[c::s, :, 0] = specs[c].real   # position f*s_stride... build (f s): idx=f*s+c
        out[c::s, :, 1] = specs[c].imag
    return out  # note idx = f*2 + c  == out[c::2]

def np_istft(stft_real, s, length):  # stft_real [(f s), Tf, 2] -> audio [s, length]
    f = stft_real.shape[0] // s
    nfr = stft_real.shape[1]
    total = N_FFT + HOP * (nfr - 1)
    out = np.zeros((s, total)); wsum = np.zeros(total)
    for c in range(s):
        comp = stft_real[c::s, :, 0] + 1j * stft_real[c::s, :, 1]   # [f, Tf]
        for i in range(nfr):
            fr = np.fft.irfft(comp[:, i], n=N_FFT) * window
            out[c, i * HOP:i * HOP + N_FFT] += fr
            if c == 0:
                wsum[i * HOP:i * HOP + N_FFT] += window ** 2
    wsum = np.maximum(wsum, 1e-8)
    out = out / wsum
    return out[:, PAD:PAD + length]

# ---- load model + host ref ----
with open(os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")) as fp:
    config = ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))
model = MelBandRoformer(**dict(config.model)).eval()
model.load_state_dict(torch.load(os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt"), map_location="cpu"), strict=False)
core = SepCore(model).eval(); host = HostDSP(model)
oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
raw = oracle["raw_audio"]          # [2, C]
C = raw.shape[1]

def cos(a, b):
    a, b = np.asarray(a).ravel().astype(np.float64), np.asarray(b).ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# 1) numpy STFT vs torch HostDSP.stft
sr_np = np_stft(raw.numpy())
sr_torch = host.stft(raw.unsqueeze(0))[0].numpy()
print(f"[1] np_stft   vs torch.stft : cos {cos(sr_np, sr_torch):.7f}  max|d| {np.abs(sr_np-sr_torch).max():.2e}")

# 2) numpy round-trip iSTFT(STFT(x)) ~ x
rt = np_istft(sr_np, 2, C)
print(f"[2] np istft(stft(x)) vs x  : cos {cos(rt, raw.numpy()):.7f}")

# 3) full: np_stft -> torch core -> np_istft  vs reference model vocals
with torch.no_grad():
    masked = core(torch.tensor(sr_np, dtype=torch.float32).unsqueeze(0))[0].numpy()
    voc_np = np_istft(masked, 2, C)
    voc_ref = model(raw.unsqueeze(0))[0].numpy()
print(f"[3] np-DSP + core vs reference vocals : cos {cos(voc_np, voc_ref):.7f}  max|d| {np.abs(voc_np-voc_ref).max():.2e}")
print("DSP RECIPE PASS" if cos(voc_np, voc_ref) >= 0.999 else "DSP RECIPE CHECK")
