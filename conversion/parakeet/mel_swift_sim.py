"""Simulate the EXACT ParakeetMelPreprocessor.swift algorithm in numpy (manual framing + DFT
matmul, centered Hann, constant pad, per-channel unbiased norm) and compare to the golden
oracle_30s input_features. If maxabs is tiny, the Swift port is numerically validated up front.

MAIN venv (numpy + librosa for the filterbank only). No GPU.
"""
import numpy as np, librosa
from pathlib import Path

HERE = Path(__file__).resolve().parent
N_FFT, WIN, HOP, NMELS = 512, 400, 160, 128
BUCKET = 2885
LOG_GUARD = 2.0 ** -24
EPS = 1e-5

# --- bundled filterbank: librosa-slaney [128,257] (= the dumped .bin) ---
mel_fb = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=NMELS, fmin=0.0, fmax=8000.0,
                             norm="slaney").astype(np.float32)  # [128,257]
# sanity: matches the dumped bundle bin?
binf = np.fromfile(HERE / "artifacts/bundle_assets/mel_filters_128x257_f32.bin",
                   dtype=np.float32).reshape(128, 257)
print(f"filterbank vs dumped .bin maxabs={np.abs(mel_fb - binf).max():.2e}")

# --- the libri1 clip (14.84 s) ---
wav, _ = librosa.load(librosa.example("libri1"), sr=16000, mono=True)
wav = wav[: int(16 * 16000)].astype(np.float32)


def swift_mel(samples: np.ndarray) -> np.ndarray:
    """Mirror the Swift algorithm exactly. Returns [128, 2885] f32 (mel-major)."""
    need = BUCKET * HOP                              # 461600
    x = samples
    if len(x) < need:
        x = np.concatenate([x, np.zeros(need - len(x), dtype=np.float32)])
    else:
        x = x[:need]
    # preemphasis 0.97: y[0]=x[0]; y[t]=x[t]-0.97 x[t-1]
    y = np.empty_like(x)
    y[0] = x[0]
    y[1:] = x[1:] - 0.97 * x[:-1]
    # centered Hann(400) inside a 512 window: 56 zeros | hann400 | 56 zeros
    n = np.arange(WIN)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / (WIN - 1))   # periodic=False -> /(WIN-1)
    w512 = np.zeros(N_FFT, dtype=np.float64)
    off = (N_FFT - WIN) // 2                          # 56
    w512[off:off + WIN] = hann
    # constant (zero) pad by n_fft//2 each side (center=True, pad_mode=constant)
    pad = N_FFT // 2                                  # 256
    padded = np.concatenate([np.zeros(pad), y.astype(np.float64), np.zeros(pad)])
    # exactly BUCKET frames, frame t = padded[t*hop : t*hop+512] * w512
    frames = BUCKET
    win = np.empty((N_FFT, frames))
    for t in range(frames):
        seg = padded[t * HOP: t * HOP + N_FFT]
        win[:, t] = seg * w512
    # DFT via cos/sin matmul (what Swift precomputes), one-sided 257 bins
    k = np.arange(N_FFT // 2 + 1)[:, None]            # [257,1]
    nn = np.arange(N_FFT)[None, :]                    # [1,512]
    ang = 2 * np.pi * k * nn / N_FFT
    cos_m = np.cos(ang); sin_m = np.sin(ang)          # [257,512]
    re = cos_m @ win; im = sin_m @ win                # [257,frames]
    power = re * re + im * im
    mel = mel_fb.astype(np.float64) @ power           # [128,frames]
    logmel = np.log(mel + LOG_GUARD)
    # per-channel (per mel bin) normalize over all BUCKET frames, unbiased (N-1)
    mean = logmel.mean(axis=1, keepdims=True)
    var = logmel.var(axis=1, ddof=1, keepdims=True)
    std = np.sqrt(var)
    norm = (logmel - mean) / (std + EPS)
    return norm.astype(np.float32)                    # [128,2885]


sim = swift_mel(wav)                                  # [128,2885]
d = np.load(HERE / "oracle_30s.npz")
oracle = d["input_features"][0].T                     # [128,2885] (oracle is [1,2885,128])
print(f"sim {sim.shape}  oracle {oracle.shape}")
diff = np.abs(sim - oracle)
print(f"swift-sim vs oracle: maxabs={diff.max():.3e}  mean={diff.mean():.3e}")
# where is the worst error?
i = np.unravel_index(diff.argmax(), diff.shape)
print(f"  worst at mel={i[0]} frame={i[1]}: sim={sim[i]:.4f} oracle={oracle[i]:.4f}")
