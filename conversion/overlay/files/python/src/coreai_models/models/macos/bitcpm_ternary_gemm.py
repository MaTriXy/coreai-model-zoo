"""Tiled ternary GEMM for Core AI — the M>4 prefill kernel the m4 template can't reach.

The shipped `bitcpm_ternary_metal` kernel and its m4 widening both cache x in REGISTERS
(16 K-values per lane x M), which caps M at 4 before the register file overflows. This is
the other mapping: lanes own an output sub-tile, K is a sequential loop, and both operands
stream through threadgroup memory — so M scales to a real prefill chunk (16/32/64).

Layout (MSL sees torch shapes reversed):
  A  torch [M, K]        -> A[k, m]
  QP torch [N, K/16] u32 -> QP[w, n]      16 ternary codes per uint32
  D  torch [N, K/256] f16-> D[g, n]       one fp16 scale per 256-K block
  C  torch [M, N]        -> C[n, m]

Tiles: BN=64 output columns, BK=64 K-step, BM=chunk (16/32/64). 256 threads (32x8) laid out
as a 16x16 thread grid, each owning TM=BM/16 rows x TN=4 columns. BK=64 with 64|k0 means a
step never straddles a 256-scale block, so the scale is a per-step scalar.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_torch import MetalParameter, TorchMetalKernel

from coreai_models.models.macos.bitcpm_ternary_metal import (
    _tern_torch_defn, pack_tern_u32, ternary_from_dequant,
)

_BN, _BK, _TN = 64, 64, 4
_TG = (32, 8, 1)          # 256 threads
_NTHREADS = 256

_GEMM_SRC = """
    const uint BM = __BM__, BN = __BN__, BK = __BK__, TM = __TM__, TN = __TN__;
    const uint K = A.get_extent(0);
    const uint t = tid.y * 32u + tid.x;          // 0..255
    const uint n_base = tgid.y * BN;

    threadgroup half xs[__BK__][__BM__];
    threadgroup half ws[__BK__][__BN__];

    const uint tm = t / 16u;                     // 0..15 -> rows [tm*TM, +TM)
    const uint tn = t % 16u;                     // 0..15 -> cols [tn*TN, +TN)

    float acc[__TM__][__TN__];
    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    for (uint k0 = 0; k0 < K; k0 += BK) {
        for (uint idx = t; idx < BK * BM; idx += 256u) {
            uint kk = idx / BM, mm = idx - kk * BM;
            xs[kk][mm] = half(A[k0 + kk, mm]);
        }
        {
            uint n = t % BN;                     // one (n, 16-code word) pair per thread
            uint wpart = t / BN;                 // 0..3  (BK/16)
            uint packed = uint(QP[(k0 >> 4) + wpart, n_base + n]);
            float d = float(D[k0 >> 8, n_base + n]);
            for (uint j = 0; j < 16; ++j) {
                int q = int((packed >> (j * 2u)) & 0x3u);
                ws[wpart * 16u + j][n] = half(float(q - 1) * d);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; ++kk) {
            float xr[__TM__], wr[__TN__];
            for (uint i = 0; i < TM; ++i) xr[i] = float(xs[kk][tm * TM + i]);
            for (uint j = 0; j < TN; ++j) wr[j] = float(ws[kk][tn * TN + j]);
            for (uint i = 0; i < TM; ++i)
                for (uint j = 0; j < TN; ++j) acc[i][j] += xr[i] * wr[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < TM; ++i)
        for (uint j = 0; j < TN; ++j)
            C[n_base + tn * TN + j, tm * TM + i] = TYPE(acc[i][j]);
"""

_PARAMS = [MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
           MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")]


def build_tern_gemm_kernel(bm: int, name: str | None = None) -> TorchMetalKernel:
    """Tiled ternary GEMM for a fixed chunk width ``bm`` (must be 16/32/64)."""
    if bm % 16 or not 16 <= bm <= 128:
        raise ValueError(f"BM={bm} must be a multiple of 16 in [16,128]")
    src = (_GEMM_SRC.replace("__BM__", str(bm)).replace("__BN__", str(_BN))
           .replace("__BK__", str(_BK)).replace("__TM__", str(bm // 16))
           .replace("__TN__", str(_TN)))
    return TorchMetalKernel(name or f"tern_gemm_m{bm}",
                            input_names=["A", "QP", "D"], result_names=["C"],
                            src=src, torch_defn=_tern_torch_defn,
                            metal_params=_PARAMS, template_dtypes={"A": "TYPE"})


class TernGemmLinear(nn.Module):
    """Ternary linear over a fixed M=bm prefill chunk. K % 64 == 0, N % 64 == 0."""

    def __init__(self, weight: torch.Tensor, kernel: TorchMetalKernel, bm: int) -> None:
        super().__init__()
        self.kernel, self.bm = kernel, bm
        self.N, K = int(weight.shape[0]), int(weight.shape[1])
        if self.N % _BN or K % _BK:
            raise ValueError(f"N={self.N} must be %{_BN}, K={K} must be %{_BK}")
        codes, d = ternary_from_dequant(weight)
        self.register_buffer("qp", pack_tern_u32(codes))
        self.register_buffer("d", d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, k = x.shape
        y = self.kernel(x.reshape(s, k), self.qp, self.d,
                        threads_per_grid=(32, 8 * (self.N // _BN), 1),
                        threads_per_thread_group=_TG,
                        result_shapes=[[s, self.N]])
        return y.reshape(b, s, self.N)
