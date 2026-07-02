# V-JEPA 2 (SSv2 action recognition) reference run — establish the oracle I/O for the Core AI export gate.
import os, json, numpy as np, torch
os.environ["HF_HUB_DISABLE_XET"] = "1"
from transformers import VJEPA2ForVideoClassification, AutoVideoProcessor

REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
HERE = os.path.dirname(os.path.abspath(__file__))

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
model = VJEPA2ForVideoClassification.from_pretrained(REPO, dtype=torch.float32).eval().to(DEV)
print("=== structure (top) === device:", DEV, flush=True)
print(model.__class__.__name__)
for n, _ in model.named_children():
    print("  child:", n)
nparam = sum(p.numel() for p in model.parameters())
print(f"params: {nparam/1e6:.0f}M  num_labels: {model.config.num_labels}")

# input: [B, T, C, H, W] = [1, 16, 3, 256, 256]. fixed seed -> reproducible oracle.
g = torch.Generator().manual_seed(0)
T = model.config.frames_per_clip
px = torch.rand(1, T, 3, 256, 256, generator=g)  # already 0..1, model does its own norm internally? check processor
print("running forward…", flush=True)
with torch.no_grad():
    out = model(pixel_values_videos=px.to(DEV))
logits = out.logits.float().cpu()
px = px.cpu()
top = torch.topk(logits[0], 5)
id2label = model.config.id2label
print("=== oracle ===")
print("logits shape:", tuple(logits.shape))
print("top5:", [(id2label.get(i.item(), i.item()), round(v.item(), 3)) for v, i in zip(top.values, top.indices)])

# save oracle for the export gate
np.save(os.path.join(HERE, "oracle_input.npy"), px.numpy())
np.save(os.path.join(HERE, "oracle_logits.npy"), logits.numpy())
print("saved oracle_input.npy", tuple(px.shape), "oracle_logits.npy", tuple(logits.shape))

# preprocessor (for the app-side normalization later)
try:
    vp = AutoVideoProcessor.from_pretrained(REPO)
    print("=== video processor ===")
    for k in ["crop_size", "image_mean", "image_std", "do_normalize", "do_rescale", "rescale_factor"]:
        if hasattr(vp, k): print(" ", k, "=", getattr(vp, k))
except Exception as e:
    print("video processor:", str(e)[:120])
