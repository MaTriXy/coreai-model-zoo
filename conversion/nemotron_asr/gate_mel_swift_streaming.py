"""Phase 8 mel gate — validate the EXACT streaming mel recipe the Swift frontend will port,
end-to-end on Core AI: raw 16 kHz samples pushed in arbitrary packets -> incremental log-mel
chunks (25 then 32 frames) -> streaming graphs -> host RNNT -> token-exact vs the oracle.

Nemotron mel (differs from Parakeet): NO normalization. preemphasis 0.97 (continuous across
the stream) -> centered Hann(400) in a 512 window -> one-sided DFT power -> librosa-slaney
mel [128,257] -> log(. + 2^-24).

Streaming form: mel frame t depends ONLY on samples [160t-200, 160t+200) (the 56-zero margins
of the padded window absorb both the center=True edge pad and the per-chunk preemphasis
boundary), so frame t is emitted as soon as sample 160t+200 arrives — packet size independent.
This class is the line-for-line spec for NemotronMelFrontend.swift.

Run (MAIN venv):  coreai-models/.venv/bin/python gate_mel_swift_streaming.py
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import librosa
import numpy as np
import torch

import coreai.runtime as rt

from export_encoder_streaming import CHUNK_FIRST, CHUNK_NEXT, CONV_K, HDIM, HEADS, HID, KV, Q, SPLIT, build_neg_mask

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
DEC_HID, DEC_NL, BLANK, MAX_SYMS = 640, 2, 13087, 10
N_FFT, WIN, HOP, NMELS = 512, 400, 160, 128
LOG_GUARD = 2.0 ** -24


class SwiftStreamingMel:
    """Push raw samples in ANY packet sizes; pop fixed mel chunks (25 frames, then 32)."""

    def __init__(self):
        n = np.arange(WIN)
        hann = 0.5 - 0.5 * np.cos(2 * np.pi * n / (WIN - 1))          # periodic=False
        self.w512 = np.zeros(N_FFT)
        self.w512[(N_FFT - WIN) // 2: (N_FFT - WIN) // 2 + WIN] = hann
        k = np.arange(N_FFT // 2 + 1)[:, None]
        nn = np.arange(N_FFT)[None, :]
        ang = 2 * np.pi * k * nn / N_FFT
        self.cos_m, self.sin_m = np.cos(ang), np.sin(ang)             # [257,512]
        self.mel_fb = librosa.filters.mel(sr=16000, n_fft=N_FFT, n_mels=NMELS,
                                          fmin=0.0, fmax=8000.0, norm="slaney").astype(np.float64)
        self.y = np.zeros(0)          # preemphasized stream (Swift: ring of last ~712 + frame idx)
        self.prev = None              # last raw sample of the previous packet
        self.next_frame = 0
        self.pending: list[np.ndarray] = []
        self.chunks_out = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Returns zero or more ready mel chunks [n_frames,128] (25 first, then 32 each)."""
        x = samples.astype(np.float64)
        first = x[:1] if self.prev is None else x[:1] - 0.97 * self.prev
        y = np.concatenate([first, x[1:] - 0.97 * x[:-1]])
        self.prev = x[-1:]
        self.y = np.concatenate([self.y, y])
        # frame t needs samples [160t-256, 160t+256) (zeros before stream start); the window's
        # 56-zero margins mean only [160t-200, 160t+200) matter -> ready once 160t+200 arrived
        frames = []
        while HOP * self.next_frame + WIN // 2 <= len(self.y):
            t = self.next_frame
            lo, hi = HOP * t - N_FFT // 2, HOP * t + N_FFT // 2
            seg = np.zeros(N_FFT)
            src_lo = max(lo, 0)
            seg[src_lo - lo: hi - lo if hi <= len(self.y) else len(self.y) - lo] = self.y[src_lo: min(hi, len(self.y))]
            w = seg * self.w512
            re, im = self.cos_m @ w, self.sin_m @ w
            mel = self.mel_fb @ (re * re + im * im)
            frames.append(np.log(mel + LOG_GUARD))
            self.next_frame += 1
        self.pending.extend(frames)
        out = []
        while len(self.pending) >= (CHUNK_FIRST if self.chunks_out == 0 else CHUNK_NEXT):
            n = CHUNK_FIRST if self.chunks_out == 0 else CHUNK_NEXT
            out.append(np.stack(self.pending[:n]).astype(np.float32))
            self.pending = self.pending[n:]
            self.chunks_out += 1
        return out


async def gpu(path):
    m = await rt.AIModel.load(str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    return m.load_function("main")


async def main():
    d = np.load(HERE / "oracle_stream_en_US.npz")
    mel_gold = d["mel"][0]                                            # [1465,128]
    one_hot = torch.from_numpy(d["one_hot"]).float()[None]
    gold_tokens = d["tokens"].tolist()
    n_chunks = int(d["n_chunks"])
    T = int(d["T"])

    wav, _ = librosa.load(librosa.example("libri1"), sr=16000, mono=True)
    wav = wav[: int(16.0 * 16000)].astype(np.float32)

    # ---- 1. sim mel vs oracle mel (packet size chosen to NOT align with chunk boundaries) ----
    fe = SwiftStreamingMel()
    chunks: list[np.ndarray] = []
    for s in range(0, len(wav), 1600):                                # 100 ms mic packets
        chunks.extend(fe.push(wav[s: s + 1600]))
    chunks = chunks[:n_chunks]
    sim = np.concatenate(chunks, axis=0)                              # [1465,128]
    diff = np.abs(sim - mel_gold[: sim.shape[0]])
    print(f"[mel sim] {len(chunks)} chunks, {sim.shape[0]} frames  "
          f"vs oracle maxabs={diff.max():.3e} mean={diff.mean():.3e}")

    # ---- 2. definitive: sim mel -> engine streaming pipeline -> tokens ----
    pf = await gpu(ART / "nemotron_asr_stream_pre_first_float16.aimodel")
    pr = await gpu(ART / "nemotron_asr_stream_pre_float16.aimodel")
    ca = await gpu(ART / "nemotron_asr_stream_conformer_a_float16.aimodel")
    cb = await gpu(ART / "nemotron_asr_stream_conformer_b_float16.aimodel")
    pd = await gpu(ART / "nemotron_asr_predict_float32.aimodel")
    jt = await gpu(ART / "nemotron_asr_joint_float32.aimodel")
    nd16 = lambda a: rt.NDArray(np.ascontiguousarray(a, dtype=np.float16))
    z16 = lambda *s: rt.NDArray(np.zeros(s, dtype=np.float16))

    async def step_p(token, h, c):
        o = await pd({"token": rt.NDArray(np.array([[token]], dtype=np.int32)),
                      "h": rt.NDArray(h), "c": rt.NDArray(c)})
        return o["dec_out"].numpy(), o["h_out"].numpy(), o["c_out"].numpy()

    async def step_j(dec, enc_frame):
        o = await jt({"dec_out": rt.NDArray(dec), "enc_frame": rt.NDArray(enc_frame)})
        return o["token_logits"].numpy()

    kA, vA = z16(SPLIT, HEADS, KV - Q, HDIM), z16(SPLIT, HEADS, KV - Q, HDIM)
    ccA = z16(SPLIT, HID, CONV_K - 1)
    kB, vB = z16(SPLIT, HEADS, KV - Q, HDIM), z16(SPLIT, HEADS, KV - Q, HDIM)
    ccB = z16(SPLIT, HID, CONV_K - 1)
    c0 = c1 = c2 = None
    h = np.zeros((DEC_NL, 1, DEC_HID), dtype=np.float32)
    c = np.zeros((DEC_NL, 1, DEC_HID), dtype=np.float32)
    dec, h, c = await step_p(BLANK, h, c)
    enc_buf: list[np.ndarray] = []
    frame, syms, steps, emitted = 0, 0, 0, []
    for i, chunk in enumerate(chunks):
        if i == 0:
            r = await pf({"mel": nd16(chunk[None])})
            e, c0, c1, c2 = (r[x] for x in ("embeds", "cache0", "cache1", "cache2"))
        else:
            r = await pr({"mel": nd16(chunk[None]), "cache0": c0, "cache1": c1, "cache2": c2})
            e, c0, c1, c2 = (r[x] for x in ("embeds", "cache0_out", "cache1_out", "cache2_out"))
        neg = nd16(build_neg_mask(i).numpy())
        r = await ca({"embeds": e, "neg_mask": neg, "k_cache": kA, "v_cache": vA, "conv_cache": ccA})
        xh, kA, vA, ccA = r["embeds_out"], r["k_out"], r["v_out"], r["conv_out"]
        r = await cb({"embeds": xh, "one_hot": nd16(one_hot.numpy()), "neg_mask": neg,
                      "k_cache": kB, "v_cache": vB, "conv_cache": ccB})
        kB, vB, ccB = r["k_out"], r["v_out"], r["conv_out"]
        enc_buf.extend(r["enc_proj"].numpy().astype(np.float32)[0])
        while frame < len(enc_buf) and steps < MAX_SYMS * T + 16:
            logits = await step_j(dec, enc_buf[frame][None])
            token = int(logits.argmax())
            steps += 1
            if token == BLANK:
                frame += 1; syms = 0
            else:
                emitted.append(token)
                dec, h, c = await step_p(token, h, c)
                syms += 1
                if syms >= MAX_SYMS:
                    frame += 1; syms = 0

    exact = emitted == gold_tokens
    same = sum(int(a == b) for a, b in zip(emitted, gold_tokens))
    print(f"[mel->e2e gpu] emitted {len(emitted)} (gold {len(gold_tokens)})  exact={exact}  "
          f"agree {same}/{len(gold_tokens)}")
    print(f"\n{'✅ PASS — Swift streaming mel recipe is locked' if exact else '❌ FAIL — recipe diverges'}")


if __name__ == "__main__":
    asyncio.run(main())
