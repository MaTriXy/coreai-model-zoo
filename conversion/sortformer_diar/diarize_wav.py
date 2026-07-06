"""Diarize an arbitrary 16 kHz mono wav with the eager Sortformer host loop (byte-equal to the
shipped graph) to preview how cleanly it splits speakers — for choosing a demo clip.

Run in the NeMo oracle venv:  _sortformer_oracle/.venv/bin/python diarize_wav.py <wav>
"""
import sys, os, numpy as np, torch, soundfile as sf
from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from sortformer_model import Sortformer, load_ckpt
from host_loop import run, eager_forward

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_nemo", "model_weights.ckpt")
FRAME = 0.08


def segments(frames, bridge=6):
    lab = [max(range(4), key=lambda s: f[s]) if max(f) > 0.5 else -1 for f in frames]
    segs, i = [], 0
    while i < len(lab):
        s = lab[i]
        if s < 0: i += 1; continue
        j = i + 1
        while j < len(lab) and lab[j] == s: j += 1
        segs.append([s, i, j]); i = j
    merged = []
    for s, a, b in segs:
        if merged and merged[-1][0] == s and a - merged[-1][2] <= bridge:
            merged[-1][2] = b
        else:
            merged.append([s, a, b])
    return merged


def main():
    wav_path = sys.argv[1]
    p = AudioToMelSpectrogramPreprocessor(
        normalize="NA", window_size=0.025, sample_rate=16000, window_stride=0.01,
        window="hann", features=128, n_fft=512, frame_splicing=1, dither=1.0e-05).eval()
    p.featurizer.dither = 0.0
    wav, sr = sf.read(wav_path)
    assert sr == 16000, f"want 16k, got {sr}"
    x = torch.tensor(wav, dtype=torch.float32)[None]
    with torch.no_grad():
        mel, mel_len = p(input_signal=x, length=torch.tensor([x.shape[1]]))
    model = Sortformer().eval(); load_ckpt(model, CKPT)
    total = run(eager_forward(model), mel, mel_len)[0]        # [nOut,4]
    frames = total.tolist()
    segs = segments(frames)
    spk_order, lbl = {}, {}
    dur = wav.shape[0] / 16000
    print(f"{os.path.basename(wav_path)}  {dur:.1f}s  frames={len(frames)}")
    for s, a, b in segs:
        if s not in lbl: lbl[s] = len(lbl) + 1
        print(f"  Speaker {lbl[s]}  {a*FRAME:6.2f}-{b*FRAME:6.2f}s  ({(b-a)*FRAME:.1f}s)")
    print(f"  -> {len(lbl)} distinct speakers, {len(segs)} turns")


if __name__ == "__main__":
    main()
