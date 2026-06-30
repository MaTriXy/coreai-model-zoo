# Community port — NOT an Apple model.
"""S1b CPU check: the generalized BitNet ternary kernel's torch_defn + BitLinearMetal decomposition
reproduce the transformers reference F.linear(ActQuant(x), WeightQuant(W)) on the exact BitVLA
shapes that break the BitCPM kernel (down_proj K=6912; SigLIP K in {1152,4304}; fc1 N=4304 %32=16).
No GPU needed -- exercises packing + per-tensor scale + tail/pad logic, not the Metal dispatch.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitvla/validate_kernel_cpu.py
"""
from __future__ import annotations

import torch

from coreai_models.models.macos.bitnet_ternary_metal import (
    _NPAD, act_quant, bitnet_decompose, pack_tern_u32, _tern_torch_defn,
)


def hf_ref(x, W):  # transformers AutoBitLinear forward (fake-quant)
    mean_w = W.abs().mean().clamp(min=1e-5)
    Wq = (W / mean_w).round().clamp(-1, 1) * mean_w
    return torch.nn.functional.linear(act_quant(x), Wq)


def cpu_bitlinear(x, W, scale_dtype=torch.float16):  # mirror BitLinearMetal but via torch_defn
    N, K = W.shape
    codes, mean_w = bitnet_decompose(W)
    n_pad = (N + _NPAD - 1) // _NPAD * _NPAD
    if n_pad != N:
        codes = torch.cat([codes, torch.ones(n_pad - N, K, dtype=codes.dtype)], 0)
    qp = pack_tern_u32(codes)
    d = torch.full((1, n_pad), float(mean_w), dtype=scale_dtype)
    xq = act_quant(x)
    y = _tern_torch_defn(xq, qp, d)          # [1, N_pad]
    return y[:, :N]


SHAPES = [
    ("q_proj   ", 2560, 2560),
    ("kv_proj  ", 640, 2560),
    ("down_proj", 2560, 6912),    # K=6912 %512=256 -> broke BitCPM kernel
    ("vis_qkvo ", 1152, 1152),    # K=1152 %512=128
    ("vis_fc1  ", 4304, 1152),    # N=4304 %32=16
    ("vis_fc2  ", 1152, 4304),    # K=4304 %512=208
]


def main():
    torch.manual_seed(0)
    # fp16-scale err ~1e-3 is the storage rounding of mean_w (D is half, as on device);
    # the logic itself is bit-exact -- proven by the fp32-scale path (err==0). Gate metric is cos.
    print(f"{'name':10s} {'N':>5s} {'K':>5s}   fp16-err   fp32-err     cos")
    ok = True
    for name, N, K in SHAPES:
        W = torch.randn(N, K) * 0.05
        x = torch.randn(1, K)
        ref = hf_ref(x, W)
        got = cpu_bitlinear(x, W)
        got32 = cpu_bitlinear(x, W, torch.float32)
        err = (got - ref).abs().max().item()
        err32 = (got32 - ref).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        good = cos >= 0.9999 and err32 < 1e-5      # direction exact + fp32 path bit-exact
        ok &= good
        print(f"{name} {N:5d} {K:5d}   {err:.2e}   {err32:.2e}   {cos:.6f}  {'OK' if good else 'FAIL'}")
    print("\nALL OK (logic bit-exact via fp32; fp16 D = on-device storage, cos~1)" if ok else "\nMISMATCH")


if __name__ == "__main__":
    main()
