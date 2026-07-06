"""Sortformer diarization reference precheck + oracle.
Load nvidia/diar_streaming_sortformer_4spk-v2 (.nemo) via NeMo, diarize the multi-speaker
test clip, print segments, and save the frame-level speaker-activity oracle.
Run: _sortformer_oracle/.venv/bin/python gen_oracle.py
"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
NEMO = os.path.join(HERE, "_dl", "diar_streaming_sortformer_4spk-v2.nemo")
WAV = os.path.join(HERE, "test_multispk_16k.wav")

from nemo.collections.asr.models import SortformerEncLabelModel
model = SortformerEncLabelModel.restore_from(restore_path=NEMO, map_location="cpu")
model.eval()
print("loaded:", type(model).__name__)
print("has diarize:", hasattr(model, "diarize"), "| streaming attrs:", [a for a in dir(model) if "stream" in a.lower()][:8])

with torch.no_grad():
    out = model.diarize(audio=[WAV], batch_size=1, include_tensor_outputs=True)
print("\ndiarize() returned type:", type(out))
# NeMo diarize returns a list; each item may be segments or a tensor of per-frame probs
res = out[0] if isinstance(out, (list, tuple)) else out
print("item type:", type(res))
try:
    print("item:", res if not hasattr(res, "shape") else res.shape)
except Exception as e:
    print("repr:", repr(res)[:500])
