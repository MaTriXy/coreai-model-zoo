"""Ladder step 1: re-authored TimesFmCore vs HF oracle graph I/O (fp32).

Feeds oracle tokenizer_inputs through TimesFmCore with position_ids=arange (no padding,
CTX==512 full context) + pure causal mask, compares point/quantile projections.
Target: cos ~ 1.0 (same math, fp32).
"""
import numpy as np
import torch
from transformers import TimesFm2_5ModelForPrediction
from timesfm_core import TimesFmCore, rope_cos_sin, load_core_from_hf

z = np.load("oracle.npz", allow_pickle=True)
tok_in = torch.tensor(z["tok_in"])          # (3,16,64)
B, N, _ = tok_in.shape
cfg = dict(patch=32, horizon=128, hidden=1280, layers=20, heads=16, head_dim=80,
           inter=1280, q=9, oql=1024, eps=1e-6)

hf = TimesFm2_5ModelForPrediction.from_pretrained(
    "google/timesfm-2.5-200m-transformers").to(torch.float32).eval()
core = load_core_from_hf(hf, cfg)

pos = torch.arange(N, dtype=torch.float32).unsqueeze(0).expand(B, N)  # num_masked=0
cos, sin = rope_cos_sin(pos, cfg["head_dim"])
neg = torch.finfo(torch.float32).min
causal = torch.triu(torch.full((N, N), neg), diagonal=1)[None, None]  # (1,1,N,N)

with torch.no_grad():
    pp, pq = core(tok_in, cos, sin, causal)

def cos_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def report(name, got, ref):
    got = got.detach().numpy() if isinstance(got, torch.Tensor) else got
    c = cos_sim(got, ref)
    mae = float(np.abs(got - ref).mean())
    rng = float(np.abs(ref).mean() + 1e-9)
    print(f"  {name:20s} cos={c:.8f}  MAE={mae:.3e}  MAE/|ref|={mae/rng:.3e}  shape={tuple(got.shape)}")
    return c

print("== Ladder 1: TimesFmCore vs HF captured projections (fp32) ==")
c1 = report("proj_point", pp, z["proj_point_out"])
c2 = report("proj_q", pq, z["proj_q_out"])
ok = c1 > 0.9999 and c2 > 0.9999
print("RESULT:", "PASS" if ok else "FAIL", f"(min cos={min(c1,c2):.8f})")
