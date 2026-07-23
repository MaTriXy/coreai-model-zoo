"""SepFull2 = STFT-matmul + SepCore + iSTFT-matmul, all in one Core AI graph.
The DFT is folded into constant basis matmuls (window baked in), so the Swift
host only does reflect-pad + framing + overlap-add (NO FFT, NO vDSP packing).

Graph I/O:  frames [1, 2, nfr=801, N=2048]  ->  recon [1, 2, nfr, N]
Host in :  reflect-pad raw[2,C] by N//2, slice frames at stride hop (unwindowed).
Host out:  overlap-add recon at stride hop, divide by wsum(window^2), trim pad.
"""
import os, sys, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))
from export_core import SepCore

N_FFT, HOP, F, PAD = 2048, 441, 1025, 1024
_win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)          # Hann periodic
_n = np.arange(N_FFT); _f = np.arange(F)
_ang = 2 * np.pi * np.outer(_n, _f) / N_FFT                              # [N,F]
Wr = (_win[:, None] * np.cos(_ang)).astype(np.float32)                   # [N,F]  analysis (win baked)
Wi = (-_win[:, None] * np.sin(_ang)).astype(np.float32)                  # [N,F]  rfft imag = -sin
_cf = np.ones(F); _cf[1:F - 1] = 2.0                                     # 2 interior, 1 DC/Nyquist
_ang2 = 2 * np.pi * np.outer(_f, _n) / N_FFT                             # [F,N]
IWr = (_win[None, :] * (_cf[:, None] / N_FFT) * np.cos(_ang2)).astype(np.float32)   # [F,N] synth (win baked)
IWi = (_win[None, :] * (_cf[:, None] / N_FFT) * (-np.sin(_ang2))).astype(np.float32)


class SepFull2(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.core = SepCore(model)
        for nme, arr in [("Wr", Wr), ("Wi", Wi), ("IWr", IWr), ("IWi", IWi)]:
            self.register_buffer(nme, torch.tensor(arr), persistent=False)

    def forward(self, frames):                       # frames [B,2,nfr,N]
        re = torch.einsum("bsfn,nk->bsfk", frames, self.Wr)   # [B,2,nfr,F]
        im = torch.einsum("bsfn,nk->bsfk", frames, self.Wi)
        B, S, nfr, Fk = re.shape
        comp = torch.stack([re, im], -1).permute(0, 3, 1, 2, 4)          # [B,F,2,nfr,2]
        stft_real = comp.reshape(B, 2 * Fk, nfr, 2)                      # idx = k*2 + s
        masked = self.core(stft_real)                                    # [B,2F,nfr,2]
        m = masked.reshape(B, Fk, 2, nfr, 2).permute(0, 2, 3, 1, 4)      # [B,2,nfr,F,2]
        cr, ci = m[..., 0], m[..., 1]                                    # [B,2,nfr,F]
        recon = torch.einsum("bsfk,kn->bsfn", cr, self.IWr) + torch.einsum("bsfk,kn->bsfn", ci, self.IWi)
        return recon                                                     # [B,2,nfr,N]


# host-side framing / overlap-add (numpy = spec the Swift host transcribes)
def frame_host(raw, nfr):                            # raw [2,C] -> frames [2,nfr,N]
    xp = np.stack([np.pad(raw[c], PAD, mode="reflect") for c in range(raw.shape[0])])
    return np.stack([[xp[c, i * HOP:i * HOP + N_FFT] for i in range(nfr)] for c in range(raw.shape[0])])

_wsum_cache = {}
def overlap_add_host(recon, length):                 # recon [2,nfr,N] -> audio [2,length]
    s, nfr, _ = recon.shape
    total = N_FFT + HOP * (nfr - 1)
    out = np.zeros((s, total))
    for c in range(s):
        for i in range(nfr):
            out[c, i * HOP:i * HOP + N_FFT] += recon[c, i]
    if total not in _wsum_cache:
        w = np.zeros(total)
        for i in range(nfr): w[i * HOP:i * HOP + N_FFT] += _win ** 2
        _wsum_cache[total] = np.maximum(w, 1e-8)
    out = out / _wsum_cache[total]
    return out[:, PAD:PAD + length]


if __name__ == "__main__":
    import yaml
    from ml_collections import ConfigDict
    from models.mel_band_roformer import MelBandRoformer
    import models.mel_band_roformer.attend as _att
    import torch.nn.functional as _F
    _att.Attend.flash_attn = lambda self, q, k, v: _F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

    with open(os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")) as fp:
        config = ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))
    C = config.inference.chunk_size
    model = MelBandRoformer(**dict(config.model)).eval()
    model.load_state_dict(torch.load(os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt"), map_location="cpu"), strict=False)
    oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
    raw = oracle["raw_audio"].numpy()
    nfr = 1 + (C + 2 * PAD - N_FFT) // HOP

    full = SepFull2(model).eval()
    frames = torch.tensor(frame_host(raw, nfr), dtype=torch.float32).unsqueeze(0)   # [1,2,nfr,N]
    with torch.no_grad():
        recon = full(frames)[0].numpy()
    voc = overlap_add_host(recon, C)
    with torch.no_grad():
        voc_ref = model(oracle["raw_audio"].unsqueeze(0))[0].numpy()
    def cos(a, b):
        a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"SepFull2 (matmul STFT/iSTFT) vs reference vocals: cos {cos(voc, voc_ref):.7f}  "
          f"max|d| {np.abs(voc - voc_ref).max():.2e}  frames{tuple(frames.shape)}")
    print("FULL2 PASS" if cos(voc, voc_ref) >= 0.999 else "FULL2 CHECK")
