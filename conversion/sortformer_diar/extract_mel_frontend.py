"""Extract the Sortformer log-mel frontend tables (mel filterbank + Hann window) straight from a
freshly-constructed NeMo AudioToMelSpectrogramPreprocessor (same config as _nemo/model_config.yaml),
and PROVE the direct-construct preprocessor reproduces the captured golden mel (stream_ref.npz) from
the raw wav — so the Swift host mel can be gated against these exact tables.

Run in the NeMo oracle venv:  _sortformer_oracle/.venv/bin/python extract_mel_frontend.py
"""
import os
import numpy as np
import torch
import soundfile as sf
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor

HERE = os.path.dirname(os.path.abspath(__file__))
WAV = os.path.join(HERE, "test_multispk_16k.wav")

# exactly _nemo/model_config.yaml -> preprocessor
p = AudioToMelSpectrogramPreprocessor(
    normalize="NA", window_size=0.025, sample_rate=16000, window_stride=0.01,
    window="hann", features=128, n_fft=512, frame_splicing=1, dither=1.0e-05,
).eval()
feat = p.featurizer

fb = feat.fb.detach().cpu().numpy()          # mel filterbank
win = feat.window.detach().cpu().numpy()     # analysis window
print("fb", fb.shape, fb.dtype, "  window", win.shape, win.dtype)
print("preemph:", getattr(feat, "preemph", None), " log:", getattr(feat, "log", None),
      " pad_to:", getattr(feat, "pad_to", None), " normalize:", getattr(feat, "normalize", None),
      " mag_power:", getattr(feat, "mag_power", None))

# reproduce mel from the raw wav (dither is random; disable for a deterministic compare)
feat.dither = 0.0
wav, sr = sf.read(WAV)
wav_t = torch.tensor(wav, dtype=torch.float32)[None]
length = torch.tensor([wav_t.shape[1]])
with torch.no_grad():
    mel, mel_len = p(input_signal=wav_t, length=length)   # [1,128,T]
mel = mel[0].cpu().numpy()
print("mel", mel.shape, "mel_len", int(mel_len[0]))

# compare vs the captured golden mel (which was made WITH dither -> tolerate tiny diff)
ref = np.load(os.path.join(HERE, "stream_ref.npz"))["mel"][0]   # [128,T]
n = min(mel.shape[1], ref.shape[1])
a, b = mel[:, :n].reshape(-1), ref[:, :n].reshape(-1)
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
print(f"direct-construct mel vs golden: cos {cos:.6f}  maxΔ {np.abs(mel[:,:n]-ref[:,:n]).max():.4f}  "
      f"shapes {mel.shape} vs {ref.shape}")

# Ship the filterbank mel-major row [128,257] to match ParakeetMelPreprocessor's melFilters
# convention (row-major [nMels, nFreq]). NeMo fb here is [1,128,257].
fb = np.squeeze(fb)                                # -> [128,257] or [257,128]
if fb.shape == (257, 128):
    fb_mel_major = np.ascontiguousarray(fb.T)      # [128,257]
elif fb.shape == (128, 257):
    fb_mel_major = np.ascontiguousarray(fb)
else:
    raise SystemExit(f"unexpected fb shape {fb.shape}")

out = os.path.join(HERE, "_mel_tables")
os.makedirs(out, exist_ok=True)
fb_mel_major.astype(np.float32).tofile(os.path.join(out, "sortformer_mel_filters_128x257.f32"))
win.astype(np.float32).tofile(os.path.join(out, "sortformer_window.f32"))
print("saved fb [128,257] + window", win.shape, "->", out)
