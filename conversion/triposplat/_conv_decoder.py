"""Net #4b: convert the Gaussian decoder (ElasticGaussianFixedlenDecoder, decoder.gs) to Core AI.

This is the per-point Gaussian-parameter transformer: 16 cross-blocks, fixed shape, run once.
Input: points (1,L,3) in [0,1] + cond=latent (1,8192,16). Output: features (1,L,480) where
480 = num_gaussians(32) * (xyz3 + dc3 + scaling3 + rot4 + opacity1 + offset_scale1).
The octree sampler that *produces* the points (decoder.octree + sample_probs) stays host-side
(data-dependent, dynamic seq-len) — see _conv_octree.py / Net #5.

L=8192 -> 8192*32 = 262144 gaussians (the pipeline default / max-quality output).
Conversion uses optimize=False (same rationale as the DiT).
"""
import sys, os, time
sys.path.insert(0, os.path.expanduser("~/Code/coreai"))
import torch, torch.nn as nn
import coreai_kit
from triposplat import load_decoder

CK = "ckpts"
OUT = "coreai_out/gs_decoder_fp32.aimodel"
L = 8192  # decoder tokens; gaussians = L * gaussians_per_point(32) = 262144

print("loading decoder ...", flush=True)
dec = load_decoder(f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors",
                   device="cpu", dtype=torch.float16)
gs = dec.gs.float().eval()
print(f"gaussians_per_point={dec.gaussians_per_point} out_channels={gs.out_channels}", flush=True)


class W(nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, points, cond):
        return s.m(x={"points": points}, cond=cond)["features"]


gen = torch.Generator().manual_seed(0)
points = torch.rand(1, L, 3, generator=gen)          # coords_norm in [0,1]
cond   = torch.randn(1, 8192, 16, generator=gen)     # DiT latent output
w = W(gs).eval()
ex = (points, cond)
inames, onames = ["points", "cond"], ["features"]

with torch.no_grad():
    ref = w(*ex)
print(f"eager out features={tuple(ref.shape)}; converting (optimize=False) ...", flush=True)
t0 = time.time()
coreai_kit.convert(w, ex, inames, onames, OUT, optimize=False)
print(f"converted in {time.time()-t0:.1f}s; running on Core AI ...", flush=True)
outs = coreai_kit.run(OUT, {"points": points.numpy(), "cond": cond.numpy()}, compute="cpu")
a = ref.flatten().double(); b = torch.tensor(outs["features"]).flatten().double()
cos = float((a @ b) / (a.norm() * b.norm()))
md = float((ref - torch.tensor(outs["features"])).abs().max())
print(f"=== GS-DECODER GATE: maxdiff={md:.4e} cos={cos:.6f} ===", flush=True)
