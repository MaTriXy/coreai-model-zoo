"""Inspect Sortformer's built-in streaming export interface + capture the frame-level oracle."""
import os, sys, inspect, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
NEMO = os.path.join(HERE, "_dl", "diar_streaming_sortformer_4spk-v2.nemo")
WAV = os.path.join(HERE, "test_multispk_16k.wav")
from nemo.collections.asr.models import SortformerEncLabelModel
model = SortformerEncLabelModel.restore_from(restore_path=NEMO, map_location="cpu").eval()

print("=== streaming_input_examples ===")
try:
    ex = model.streaming_input_examples()
    def shp(x):
        if torch.is_tensor(x): return f"{tuple(x.shape)}/{x.dtype}"
        if isinstance(x, (list, tuple)): return [shp(y) for y in x]
        if isinstance(x, dict): return {k: shp(v) for k, v in x.items()}
        return type(x).__name__
    print(shp(ex))
except Exception as e:
    print("ERR", type(e).__name__, e)

print("\n=== input_types / output_types (neural-type interface) ===")
for attr in ["input_types", "output_types", "input_types_for_export", "output_types_for_export"]:
    try:
        v = getattr(model, attr, None)
        if v is not None: print(f"{attr}: {list(v.keys())}")
    except Exception as e:
        print(attr, "ERR", e)

print("\n=== forward signature ===")
try: print(inspect.signature(model.forward))
except Exception as e: print("ERR", e)

print("\n=== submodules ===")
for n, m in model.named_children():
    print(" ", n, type(m).__name__)

# frame-level oracle: run the model forward on the test clip to get per-frame [T,4] probs
print("\n=== forward -> frame probs (oracle) ===")
import soundfile as sf
wav, sr = sf.read(WAV); wav = torch.tensor(wav, dtype=torch.float32)[None]  # [1, N]
length = torch.tensor([wav.shape[1]])
try:
    with torch.no_grad():
        preds = model.forward(audio_signal=wav, audio_signal_length=length)
    p = preds[0] if isinstance(preds, (list, tuple)) else preds
    print("forward out:", tuple(p.shape) if torch.is_tensor(p) else type(p))
    if torch.is_tensor(p):
        np.save(os.path.join(HERE, "oracle_frames.npy"), p.cpu().numpy())
        act = (p[0].sigmoid() if p.max() > 1 or p.min() < 0 else p[0]).cpu().numpy()
        print("per-spk active frames (>0.5):", (act > 0.5).sum(0), "of", act.shape[0])
except Exception as e:
    import traceback; traceback.print_exc()
