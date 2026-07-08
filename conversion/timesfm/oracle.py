"""TimesFM 2.5 oracle: run HF reference, dump final forecasts + graph-boundary I/O.

The Core AI graph boundary is the pure feed-forward transformer:
    tokenizer_inputs (B,N,2*patch_len) -> input_ff_layer -> 20 layers
      -> output_projection_point (B,N,H*Q) + output_projection_quantiles (B,N,Lq*Q)
Everything else (global RevIN, per-patch Welford RevIN, flip-invariance, continuous
quantile head, denorm, positivity clamp) is host DSP and is gated separately in numpy.

We capture the PRIMARY (non-flipped) pass I/O via hooks for the graph gate.
"""
import numpy as np
import torch
from transformers import TimesFm2_5ModelForPrediction

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--ctx", type=int, default=512)
_ap.add_argument("--out", default=None)
_ap.add_argument("--short", type=int, default=0, help="if >0, add a short (padded) series of this length")
_A = _ap.parse_args()

torch.manual_seed(0)
REPO = "google/timesfm-2.5-200m-transformers"
CTX = _A.ctx          # forecast_context_len
OUT = _A.out or f"oracle_{CTX}.npz"

model = TimesFm2_5ModelForPrediction.from_pretrained(REPO).to(torch.float32).eval()
cfg = model.config
print("config:", dict(patch=cfg.patch_length, horizon=cfg.horizon_length,
                      hidden=cfg.hidden_size, layers=cfg.num_hidden_layers,
                      heads=cfg.num_attention_heads, head_dim=cfg.head_dim,
                      inter=cfg.intermediate_size, q=len(cfg.quantiles),
                      oql=cfg.output_quantile_len, flip=cfg.force_flip_invariance,
                      pos=cfg.infer_is_positive, decode_index=cfg.decode_index))

# ---- fixed deterministic input series (3 shapes: seasonal, trend+season, walk) ----
t = np.arange(CTX, dtype=np.float64)
series = {
    "sine":  10.0 + 3.0*np.sin(2*np.pi*t/48.0),
    "trend": 5.0 + 0.02*t + 2.0*np.sin(2*np.pi*t/24.0),
    "walk":  50.0 + np.cumsum(np.sin(t*0.3) + 0.1*np.cos(t*0.05)),
}
if _A.short > 0:
    ts = np.arange(_A.short, dtype=np.float64)
    series["short"] = 20.0 + 4.0*np.sin(2*np.pi*ts/30.0)   # length < CTX -> HF front-pads
past = [torch.tensor(v, dtype=torch.float32) for v in series.values()]

# ---- hooks to capture PRIMARY-pass graph I/O ----
cap = {}
def once(key):
    def hook(mod, inp, out):
        if key not in cap:  # first (primary, non-flipped) call only
            cap[key+"_in"] = inp[0].detach().float().cpu().numpy()
            cap[key+"_out"] = (out if isinstance(out, torch.Tensor) else out[0]).detach().float().cpu().numpy()
    return hook
h = []
h.append(model.model.input_ff_layer.register_forward_hook(once("iff")))
h.append(model.output_projection_point.register_forward_hook(once("proj_point")))
h.append(model.output_projection_quantiles.register_forward_hook(once("proj_q")))

# capture last_hidden_state + context stats from the inner model (primary call)
inner = {}
def inner_hook(mod, inp, out):
    if "lhs" not in inner:
        inner["lhs"] = out.last_hidden_state.detach().float().cpu().numpy()
        inner["ctx_mu"] = out.context_mu.detach().float().cpu().numpy()
        inner["ctx_sigma"] = out.context_sigma.detach().float().cpu().numpy()
h.append(model.model.register_forward_hook(inner_hook))

with torch.no_grad():
    o = model(past_values=past, forecast_context_len=CTX)

for x in h: x.remove()

mean_pred = o.mean_predictions.float().cpu().numpy()      # (B, H)
full_pred = o.full_predictions.float().cpu().numpy()      # (B, H, Q)
print("mean_predictions:", mean_pred.shape, "full_predictions:", full_pred.shape)
print("captured:", sorted(cap.keys()), sorted(inner.keys()))
print("tokenizer_inputs (iff_in):", cap["iff_in"].shape,
      "last_hidden_state:", inner["lhs"].shape,
      "proj_point_out:", cap["proj_point_out"].shape,
      "proj_q_out:", cap["proj_q_out"].shape)

np.savez(OUT,
         ctx_len=np.int64(CTX),
         series=np.array([series[k].astype(np.float32) for k in series], dtype=object),
         series_names=np.array(list(series.keys())),
         mean_pred=mean_pred, full_pred=full_pred,
         tok_in=cap["iff_in"], iff_out=cap["iff_out"],
         lhs=inner["lhs"], ctx_mu=inner["ctx_mu"], ctx_sigma=inner["ctx_sigma"],
         proj_point_in=cap["proj_point_in"], proj_point_out=cap["proj_point_out"],
         proj_q_in=cap["proj_q_in"], proj_q_out=cap["proj_q_out"])
print("wrote", OUT)
# quick sanity print
for i, k in enumerate(series):
    print(f"  {k}: last ctx={series[k][-1]:.3f}  mean_fc[:5]={mean_pred[i,:5].round(3)}")
