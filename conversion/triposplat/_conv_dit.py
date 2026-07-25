"""Net #3: convert DiT denoiser (LatentSeqMMFlowModel, one step) to Core AI + gate.

Loads ONLY the flow model (not the full pipeline) and feeds random conditioning at the real
runtime shapes — per-net conversion is gated converted-vs-eager on the SAME inputs, so random
inputs suffice and avoid the slow CPU DINOv3/VAE encode that wedged the full-pipeline version.

Conversion uses optimize=False: coreai-torch's optimize() pass on this 24-block / ~12.3k-token
attention graph runs >90 min and balloons to ~64 GB RAM, while the conversion itself is ~7 s.
On-device deployment runs its own AOT specialization (coreai-build), so the Python optimize()
is skipped here. (verify() forces optimize=True, hence the manual convert()+run() gate below.)

Two model.py fixes were required for this net (see git diff):
  1. Complex RoPE (torch.polar / view_as_complex) -> real cos/sin math: coreai has no complex ops.
  2. pos_embedder(pos_pe) is precomputed into a buffer: computing it in-graph makes coreai
     constant-fold sin/cos of huge args (Sobol * 2^16 * 2pi) at low precision (cos -> ~0.5).
"""
import sys
from pathlib import Path
import os, time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/ — coreai_kit
import torch, torch.nn as nn
import coreai_kit
from triposplat import load_flow_model

CK = "ckpts"
FEAT_LEN = 4101  # DINOv3 (cls+4 reg+patches) == padded VAE token length, see encode_image
OUT = "coreai_out/dit_fp32.aimodel"

print("loading flow_model ...", flush=True)
t0 = time.time()
fm = load_flow_model(f"{CK}/diffusion_models/triposplat_fp16.safetensors",
                     device="cpu", dtype=torch.float16).float().eval()
print(f"loaded in {time.time()-t0:.1f}s  cam_channels={fm.cam_channels}", flush=True)

gen = torch.Generator().manual_seed(0)
latent   = torch.randn(1, fm.q_token_length, fm.in_channels, generator=gen)
camera   = torch.randn(1, 1, fm.cam_channels, generator=gen)
t        = torch.tensor([1000.0])
feature1 = torch.randn(1, FEAT_LEN, fm.cond_channels,  generator=gen)
feature2 = torch.randn(1, FEAT_LEN, fm.cond2_channels, generator=gen)
print(f"latent={tuple(latent.shape)} camera={tuple(camera.shape)} "
      f"f1={tuple(feature1.shape)} f2={tuple(feature2.shape)}", flush=True)


class W(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, latent, camera, t, feature1, feature2):
        o = s.m({"latent": latent, "camera": camera}, t,
                {"feature1": feature1, "feature2": feature2})
        return o["latent"], o["camera"]


w = W(fm).eval()
ex = (latent, camera, t, feature1, feature2)
inames = ["latent", "camera", "t", "feature1", "feature2"]
onames = ["pred_latent", "pred_camera"]

with torch.no_grad():
    ref = w(*ex)
print("converting (optimize=False) ...", flush=True)
t0 = time.time()
coreai_kit.convert(w, ex, inames, onames, OUT, optimize=False)
print(f"converted in {time.time()-t0:.1f}s; running on Core AI ...", flush=True)
feed = {"latent": latent.numpy(), "camera": camera.numpy(), "t": t.numpy(),
        "feature1": feature1.numpy(), "feature2": feature2.numpy()}
outs = coreai_kit.run(OUT, feed, compute="cpu")
for on, r in [("pred_latent", ref[0]), ("pred_camera", ref[1])]:
    a = r.flatten().double(); b = torch.tensor(outs[on]).flatten().double()
    cos = float((a @ b) / (a.norm() * b.norm()))
    md = float((r - torch.tensor(outs[on])).abs().max())
    print(f"=== DiT GATE [{on}]: maxdiff={md:.4e} cos={cos:.6f} ===", flush=True)
