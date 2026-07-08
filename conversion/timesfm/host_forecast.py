"""Independent host DSP + TimesFmCore graph -> final forecast.  Ladder 2.

Reproduces TimesFm2_5ModelForPrediction.forward host-side (everything except the
transformer, which is the exportable core). Validates the spec the Swift host will follow.
Only the default path: window_size=None, force_flip_invariance=True, truncate from config.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

TOL = 1e-6
DECODE_INDEX = 5
THETA = 10000.0


def _welford_stats(patched, masks_bool):
    """patched (B,N,P), masks_bool (B,N,P) True=invalid. Returns ctx_mu,ctx_sigma (B,N)
    = running (causal) mean/std over valid values across patches (Welford)."""
    B, N, P = patched.shape
    count = torch.zeros(B); mean = torch.zeros(B); std = torch.zeros(B)
    mus, sigmas = [], []
    for i in range(N):
        nv = patched[:, i, :]; mk = masks_bool[:, i, :]
        is_valid = (~mk).float()
        inc = is_valid.sum(-1)
        inc_safe = torch.where(inc == 0, torch.ones_like(inc), inc)
        im = (nv * is_valid).sum(-1) / inc_safe
        im = torch.where(inc == 0, torch.zeros_like(im), im)
        cen = nv - im.unsqueeze(-1)
        iv = ((cen * is_valid) ** 2).sum(-1) / inc_safe
        iv = torch.where(inc == 0, torch.zeros_like(iv), iv)
        isd = torch.sqrt(torch.clamp(iv, min=0.0))
        nc = count + inc
        nc_safe = torch.where(nc == 0, torch.ones_like(nc), nc)
        nm = (count * mean + im * inc) / nc_safe
        nm = torch.where(nc == 0, torch.zeros_like(nm), nm)
        nvar = (count * std**2 + inc * isd**2 + count * (mean - nm)**2 + inc * (im - nm)**2) / nc_safe
        nvar = torch.where(nc == 0, torch.zeros_like(nvar), nvar)
        count, mean, std = nc, nm, torch.sqrt(torch.clamp(nvar, min=0.0))
        mus.append(mean); sigmas.append(std)
    return torch.stack(mus, 1), torch.stack(sigmas, 1)


def _revin(x, loc, scale, reverse=False, mask=None):
    while loc.dim() < x.dim():
        loc = loc.unsqueeze(-1); scale = scale.unsqueeze(-1)
    if reverse:
        return x * scale + loc
    safe = torch.where(scale < TOL, torch.ones_like(scale), scale)
    normed = (x - loc) / safe
    if mask is not None:
        normed = torch.where(mask, torch.zeros_like(normed), normed)
    return normed


def _rope(pos, head_dim):
    inv = 1.0 / (THETA ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = pos.float().unsqueeze(-1) * inv.view(1, 1, -1)
    emb = torch.cat([freqs, freqs], -1)
    return emb.cos(), emb.sin()


def _run_graph(core, normalized_ts, input_padding, cfg):
    """Host replica of TimesFm2_5Model.forward, graph replaced by `core`.
    Returns point_forecast (B,H,Q), quantile_spreads (B,Lq,Q)."""
    B, L = normalized_ts.shape
    P = cfg["patch"]
    patched = normalized_ts.view(B, -1, P)
    masks_bool = input_padding[:, :L].view(B, -1, P) >= 0.5
    ctx_mu, ctx_sigma = _welford_stats(patched, masks_bool)
    normed = _revin(patched, ctx_mu, ctx_sigma, mask=masks_bool)
    tok_in = torch.cat([normed, masks_bool.float()], -1)          # (B,N,2P)
    patch_padding = masks_bool[..., -1]                            # (B,N)
    N = tok_in.shape[1]
    num_masked = patch_padding.int().sum(-1, keepdim=True)
    pos = torch.arange(N).unsqueeze(0) - num_masked                # (B,N)
    cos, sin = _rope(pos, cfg["head_dim"])
    # Single additive mask (fp16-safe fill): allowed = causal AND key-not-padded.
    # One combined mask (never add two fills -> no fp16 -inf overflow -> no all-masked-row NaN).
    NEG = -1e4
    i = torch.arange(N).view(N, 1)
    j = torch.arange(N).view(1, N)
    causal_ok = (j <= i)                                           # (N,N)
    key_ok = ~patch_padding                                        # (B,N)
    allowed = causal_ok[None] & key_ok[:, None, :]                 # (B,N,N)
    attn_bias = torch.where(allowed[:, None], torch.zeros(1), torch.full((1,), NEG))  # (B,1,N,N)
    with torch.no_grad():
        pp, pq = core(tok_in, cos, sin, attn_bias)
    Q = cfg["q"] + 1
    point = _revin(pp, ctx_mu, ctx_sigma, reverse=True).view(B, N, cfg["horizon"], Q)[:, -1]
    quant = _revin(pq, ctx_mu, ctx_sigma, reverse=True).view(B, N, cfg["oql"], Q)[:, -1]
    return point, quant


def forecast(core, series_1d, ctx_len, cfg, force_flip=True, truncate_neg=True):
    """series_1d: 1D torch tensor. Returns mean_pred (H,), full_pred (H,Q)."""
    ts = series_1d[-ctx_len:]
    input_min = ts.min()
    # _preprocess: pad front if short
    L = ts.shape[0]
    if L < ctx_len:
        pad = ctx_len - L
        input_ts = torch.cat([torch.zeros(pad), ts])[None]
        input_padding = torch.cat([torch.ones(pad), torch.zeros(L + cfg["horizon"])])[None]
    else:
        input_ts = ts[None]
        input_padding = torch.zeros(ctx_len + cfg["horizon"])[None]

    mu_g = input_ts.mean(1, keepdim=True)
    sigma_g = input_ts.std(1, keepdim=True)                        # unbiased (ddof=1)
    normalized = _revin(input_ts, mu_g, sigma_g)

    pf, qs = _run_graph(core, normalized, input_padding, cfg)
    if force_flip:
        fpf, fqs = _run_graph(core, -normalized, input_padding, cfg)
        def flipq(x):
            return torch.cat([x[..., :1], torch.flip(x[..., 1:], (-1,))], -1)
        pf = (pf - flipq(fpf)) / 2
        qs = (qs - flipq(fqs)) / 2

    H = min(cfg["horizon"], pf.shape[1])
    full = pf[:, :H, :].clone()
    mqh = min(H, qs.shape[1])
    for idx in range(1, cfg["q"] + 1):
        if idx == DECODE_INDEX:
            continue
        full[:, :mqh, idx] = qs[:, :mqh, idx] - qs[:, :mqh, DECODE_INDEX] + full[:, :mqh, DECODE_INDEX]

    full_pred = _revin(full, mu_g, sigma_g, reverse=True)          # (1,H,Q)
    mean_pred = full_pred[:, :, DECODE_INDEX]
    if truncate_neg and (input_min >= 0):
        full_pred = torch.clamp(full_pred, min=0.0)
        mean_pred = torch.clamp(mean_pred, min=0.0)
    return mean_pred[0], full_pred[0]


if __name__ == "__main__":
    from transformers import TimesFm2_5ModelForPrediction
    from timesfm_core import load_core_from_hf
    cfg = dict(patch=32, horizon=128, hidden=1280, layers=20, heads=16, head_dim=80,
               inter=1280, q=9, oql=1024, eps=1e-6)
    z = np.load("oracle.npz", allow_pickle=True)
    CTX = int(z["ctx_len"]); series = z["series"]; names = z["series_names"]
    hf = TimesFm2_5ModelForPrediction.from_pretrained(
        "google/timesfm-2.5-200m-transformers").to(torch.float32).eval()
    core = load_core_from_hf(hf, cfg)

    print("== Ladder 2: independent host DSP + core vs HF oracle final forecast ==")
    worst = 1.0
    for i, nm in enumerate(names):
        mp, fp = forecast(core, torch.tensor(series[i]), CTX, cfg)
        omp, ofp = z["mean_pred"][i], z["full_pred"][i]
        cm = float(mp.numpy().ravel() @ omp.ravel() / (np.linalg.norm(mp.numpy())*np.linalg.norm(omp)+1e-12))
        cf = float(fp.numpy().ravel() @ ofp.ravel() / (np.linalg.norm(fp.numpy())*np.linalg.norm(ofp)+1e-12))
        mae = float(np.abs(mp.numpy() - omp).mean())
        rel = mae / (np.abs(omp).mean() + 1e-9)
        worst = min(worst, cm, cf)
        print(f"  {str(nm):8s} mean cos={cm:.8f} full cos={cf:.8f} MAE={mae:.3e} rel={rel:.3e}")
    print("RESULT:", "PASS" if worst > 0.9999 else "FAIL", f"(min cos={worst:.8f})")
