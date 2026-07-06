"""Capture a LONGER NeMo streaming reference (the demo wav x3 ~= 64.5 s) so the host loop's AOSC
speaker-cache compression is exercised over MULTIPLE chunks and gated end-to-end against NeMo
forward_streaming — the short 2-chunk clip never compresses in a way that changes the output.

Run in the NeMo env:  _sortformer_oracle/.venv/bin/python capture_stream_ref_long.py
"""
import os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
NEMO = os.path.join(HERE, "_dl", "diar_streaming_sortformer_4spk-v2.nemo")
WAV = os.path.join(HERE, "test_multispk_16k.wav")
REPS = 3
from nemo.collections.asr.models import SortformerEncLabelModel
model = SortformerEncLabelModel.restore_from(restore_path=NEMO, map_location="cpu").eval()

import soundfile as sf
wav, sr = sf.read(WAV)
wav = np.tile(wav, REPS)                        # concat x3
wav_t = torch.tensor(wav, dtype=torch.float32)[None]
length = torch.tensor([wav_t.shape[1]])
with torch.no_grad():
    mel, mel_len = model.preprocessor(input_signal=wav_t, length=length)   # [1,128,T]
    total_preds = model.forward_streaming(processed_signal=mel, processed_signal_length=mel_len)
print("wav", wav.shape, f"{wav.shape[0]/sr:.1f}s", "mel", tuple(mel.shape), "total_preds", tuple(total_preds.shape))
np.savez(os.path.join(HERE, "stream_ref_long.npz"),
         mel=mel.cpu().numpy(), mel_len=mel_len.cpu().numpy(), total_preds=total_preds.cpu().numpy())
print("saved stream_ref_long.npz")
