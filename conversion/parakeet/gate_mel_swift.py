"""Phase 5 mel gate — validate the EXACT mel recipe the Swift frontend will port, end-to-end on
Core AI. Computes log-mel from the raw 16 kHz clip (no precomputed oracle), then runs
encoder.aimodel -> host TDT loop, and checks the transcript is token-exact vs the golden.

Two recipes are gated so we lock the right one for arbitrary-length app clips:
  Y (per-clip): normalize per-channel over the REAL clip's valid frames, then zero-pad the bucket
                to 2885 frames. This is HF's batched per-clip behavior (silence padding masked to 0).
  X (oracle):   pre-pad the audio to the bucket, normalize over speech+silence (reproduces how
                gen_oracle built oracle_30s — a known-good control).

Run (MAIN venv, _GPU_LOCK held):
    coreai-models/.venv/bin/python gate_mel_swift.py
"""
from __future__ import annotations
import argparse
import asyncio
from pathlib import Path
import numpy as np
import torch
import librosa
import coreai.runtime as rt

HERE = Path(__file__).resolve().parent
HID, NL = 640, 2
N_FFT, WIN, HOP, NMELS = 512, 400, 160, 128
BUCKET = 2885

MEL_FB = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=NMELS, fmin=0.0, fmax=8000.0,
                             norm="slaney").astype(np.float32)  # [128, 257]


def logmel_raw(wav: np.ndarray) -> torch.Tensor:
    """preemphasis -> STFT -> power -> mel -> log(.+2^-24). Returns [frames, 128] (pre-norm)."""
    x = torch.tensor(wav, dtype=torch.float32)
    xp = torch.cat([x[:1], x[1:] - 0.97 * x[:-1]])
    win = torch.hann_window(WIN, periodic=False)
    stft = torch.stft(xp, N_FFT, hop_length=HOP, win_length=WIN, window=win,
                      return_complex=True, pad_mode="constant")
    mag = torch.view_as_real(stft)
    mag = torch.sqrt(mag.pow(2).sum(-1)).pow(2)        # |stft|^2  [257, frames]
    mel = torch.from_numpy(MEL_FB) @ mag               # [128, frames]
    return torch.log(mel + 2**-24).T                   # [frames, 128]


def recipe_Y(wav: np.ndarray) -> np.ndarray:
    """Normalize over the real clip, then zero-pad frames to the bucket. -> [1,128,2885]."""
    lm = logmel_raw(wav)                               # [frames, 128]
    valid = len(wav) // HOP                            # features_lengths
    valid = min(valid, lm.shape[0])
    region = lm[:valid]
    mean = region.mean(0, keepdim=True)
    std = region.std(0, unbiased=True, keepdim=True)
    norm = (lm - mean) / (std + 1e-5)
    out = torch.zeros(BUCKET, NMELS)
    keep = min(valid, BUCKET)                          # only the valid frames; rest stay 0
    out[:keep] = norm[:keep]
    return out.T.unsqueeze(0).numpy().astype(np.float32)  # [1,128,2885]


def recipe_X(wav: np.ndarray) -> np.ndarray:
    """Pre-pad audio to the bucket sample-length, normalize over all valid frames (= oracle)."""
    need = BUCKET * HOP                                # 2885*160 = 461600
    if len(wav) < need:
        wav = np.concatenate([wav, np.zeros(need - len(wav), dtype=wav.dtype)])
    lm = logmel_raw(wav)                               # [>=2885, 128]
    valid = min(len(wav) // HOP, lm.shape[0])
    region = lm[:valid]
    mean = region.mean(0, keepdim=True)
    std = region.std(0, unbiased=True, keepdim=True)
    norm = (lm - mean) / (std + 1e-5)
    out = torch.zeros(BUCKET, NMELS)
    mask_len = min(valid, BUCKET)
    out[:mask_len] = norm[:mask_len]
    return out.T.unsqueeze(0).numpy().astype(np.float32)


def recipe_swift(wav: np.ndarray) -> np.ndarray:
    """The EXACT Swift algorithm: manual framing + cos/sin DFT matmul, centered Hann, constant
    pad, per-channel unbiased norm over all 2885 frames. -> [1,128,2885]."""
    need = BUCKET * HOP
    x = wav.astype(np.float32)
    x = np.concatenate([x, np.zeros(need - len(x), np.float32)]) if len(x) < need else x[:need]
    y = np.empty_like(x); y[0] = x[0]; y[1:] = x[1:] - 0.97 * x[:-1]
    n = np.arange(WIN)
    hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / (WIN - 1))
    w512 = np.zeros(N_FFT); off = (N_FFT - WIN) // 2; w512[off:off + WIN] = hann
    pad = N_FFT // 2
    padded = np.concatenate([np.zeros(pad), y.astype(np.float64), np.zeros(pad)])
    win = np.stack([padded[t * HOP: t * HOP + N_FFT] * w512 for t in range(BUCKET)], axis=1)
    k = np.arange(N_FFT // 2 + 1)[:, None]; nn = np.arange(N_FFT)[None, :]
    ang = 2 * np.pi * k * nn / N_FFT
    re = np.cos(ang) @ win; im = np.sin(ang) @ win
    power = re * re + im * im
    mel = MEL_FB.astype(np.float64) @ power
    logmel = np.log(mel + 2.0 ** -24)
    mean = logmel.mean(1, keepdims=True); std = np.sqrt(logmel.var(1, ddof=1, keepdims=True))
    norm = (logmel - mean) / (std + 1e-5)
    return np.expand_dims(norm.astype(np.float32), 0)


async def gpu(path):
    m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(
        rt.ComputeUnitKind.gpu()))
    return m.load_function("main")


async def run_e2e(enc_fn, pf, jf, mel_np, blank, durations) -> list[int]:
    r = await enc_fn({"mel": rt.NDArray(mel_np.astype(np.float16))})
    enc_proj = torch.from_numpy(r["enc_proj"].numpy().astype(np.float32))[0]
    T = enc_proj.shape[0]

    async def step_p(token, h, c):
        o = await pf({"token": rt.NDArray(token.numpy().astype(np.int32)),
                      "h": rt.NDArray(h.numpy()), "c": rt.NDArray(c.numpy())})
        return (torch.from_numpy(o["dec_out"].numpy().astype(np.float32)),
                torch.from_numpy(o["h_out"].numpy().astype(np.float32)),
                torch.from_numpy(o["c_out"].numpy().astype(np.float32)))

    async def step_j(dec, enc_frame):
        o = await jf({"dec_out": rt.NDArray(dec.numpy()), "enc_frame": rt.NDArray(enc_frame.numpy())})
        return (torch.from_numpy(o["token_logits"].numpy().astype(np.float32)),
                torch.from_numpy(o["dur_logits"].numpy().astype(np.float32)))

    h = torch.zeros(NL, 1, HID); c = torch.zeros(NL, 1, HID)
    dec, h, c = await step_p(torch.tensor([[blank]]), h, c)
    frame, emitted = 0, []
    while frame < T and len(emitted) < 12 * T:
        tl, dl = await step_j(dec, enc_proj[frame:frame + 1])
        token = int(tl.argmax()); dur = durations[int(dl.argmax())]
        if token == blank and dur == 0:
            dur = 1
        frame += dur
        if token != blank:
            emitted.append(token)
            dec, h, c = await step_p(torch.tensor([[token]]), h, c)
    return emitted


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="oracle_30s.npz")
    ap.add_argument("--artifacts", default="artifacts", help="bundle dir (relative to this script)")
    args = ap.parse_args()
    ART = HERE / args.artifacts

    d = np.load(HERE / args.oracle)
    gold = d["tokens"].tolist()
    blank = int(d["blank_id"])
    durations = (d["durations"].tolist() if "durations" in d
                 else list(range(int(d["n_durations"]))))

    # the natural libri1 clip (no manual silence pad) — the realistic app input
    wav, _ = librosa.load(librosa.example("libri1"), sr=16000, mono=True)
    wav = wav[: int(16 * 16000)]
    print(f"clip {len(wav)} samples = {len(wav)/16000:.3f}s   frames~{1+len(wav)//HOP}")

    enc_fn = await gpu(ART / f"parakeet_encoder_float16_L{BUCKET}.aimodel")
    pf = await gpu(ART / "parakeet_predict_float32.aimodel")
    jf = await gpu(ART / "parakeet_joint_float32.aimodel")

    for name, recipe in [("Y per-clip+zero-pad", recipe_Y), ("X oracle-style", recipe_X),
                         ("S manual-DFT swift-sim", recipe_swift)]:
        mel = recipe(wav)
        emitted = await run_e2e(enc_fn, pf, jf, mel, blank, durations)
        exact = emitted == gold
        same = sum(int(a == b) for a, b in zip(emitted, gold))
        txt = d["text"] if exact else "(differs)"
        print(f"[{name}] emitted {len(emitted)} (gold {len(gold)})  exact={exact}  agree {same}/{len(gold)}")
        if not exact:
            print(f"    first 12 emitted: {emitted[:12]}")
            print(f"    first 12 gold   : {gold[:12]}")


if __name__ == "__main__":
    asyncio.run(main())
