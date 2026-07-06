"""Capture NeMo's streaming reference for host-loop gating: the full-clip mel, and the streaming
total_preds from forward_streaming (the true streaming diarization output), plus the per-chunk
streaming-state trace (spkcache/mean_sil_emb/fifo lengths) so a Python/Swift host re-implementation
can be gated against the exact AOSC evolution.

Run in the NeMo env:  _sortformer_oracle/.venv/bin/python capture_stream_ref.py
"""
import os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
NEMO = os.path.join(HERE, "_dl", "diar_streaming_sortformer_4spk-v2.nemo")
WAV = os.path.join(HERE, "test_multispk_16k.wav")
from nemo.collections.asr.models import SortformerEncLabelModel
model = SortformerEncLabelModel.restore_from(restore_path=NEMO, map_location="cpu").eval()
sm = model.sortformer_modules

# trace streaming-state evolution around each streaming_update call
trace = {"spkcache_len": [], "mean_sil_norm": []}
orig_update = sm.streaming_update
def traced_update(streaming_state, chunk, preds, lc=0, rc=0):
    ss, chunk_preds = orig_update(streaming_state, chunk, preds, lc, rc)
    trace["spkcache_len"].append(int(ss.spkcache.shape[1]))
    trace["mean_sil_norm"].append(float(ss.mean_sil_emb.norm().item()))
    return ss, chunk_preds
sm.streaming_update = traced_update

import soundfile as sf
wav, sr = sf.read(WAV); wav_t = torch.tensor(wav, dtype=torch.float32)[None]
length = torch.tensor([wav_t.shape[1]])
with torch.no_grad():
    mel, mel_len = model.preprocessor(input_signal=wav_t, length=length)   # [1,128,T]
    total_preds = model.forward_streaming(processed_signal=mel, processed_signal_length=mel_len)

print("mel", tuple(mel.shape), "total_preds", tuple(total_preds.shape))
print("spkcache_len trace:", trace["spkcache_len"])
print("mean_sil_norm trace:", [f"{x:.3f}" for x in trace["mean_sil_norm"]])
np.savez(os.path.join(HERE, "stream_ref.npz"),
         mel=mel.cpu().numpy(), mel_len=mel_len.cpu().numpy(),
         total_preds=total_preds.cpu().numpy(),
         spkcache_len=np.array(trace["spkcache_len"]))
print("saved stream_ref.npz")
