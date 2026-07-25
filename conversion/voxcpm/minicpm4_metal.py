# Community port — NOT an Apple model.
"""Lossless fp16 Metal-kernel acceleration for the VoxCPM MiniCPM4 backbone decode (q=1).

The engine profile showed the backbone LM decode (base 24L + res 6L) is the bottleneck (~64% of
per-frame time), running at ~8-16% of peak bandwidth: it is NOT weight-bandwidth-bound (int8 was
SLOWER than fp16 — dequant overhead), it is launch/occupancy-bound — 24 layers x ~7 tiny matvec
dispatches/frame, each a single-thread-per-output GEMV the engine can't fill. The fix is the
gemma4 fp16 "simd" matvec kernel (one 32-lane SIMD-group per output column, split-K + simd_sum,
32x more threads in flight to hide memory latency) wrapping the bandwidth-dominant Linears.

LOSSLESS: same fp16 weights, fp32 accumulation, torch_defn = F.linear — numerically identical to
the stock matmul (gated cos >= 0.999 vs oracle). DECODE-ONLY: the kernel assumes M=1 (single row),
which is exactly the shipped path (prefill-via-decode runs the q=1 bundle per text token).

Covers the per-layer weight stream: MLP gate/up/down (84%) + attention q_proj/o_proj. k/v_proj
(N=128) stay fp16 nn.Linear — narrow-N matvecs never pay (Mac lesson), and the engine handles them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.gemma4_metal_mlp import Matvec

import minicpm4


class MetalMLP4(nn.Module):
    """MiniCPM4 SwiGLU MLP (down(silu(gate(x)) * up(x))) with the 3 projections on the matvec kernel."""

    def __init__(self, mlp: minicpm4.MLP, mv: Matvec) -> None:
        super().__init__()
        self.mv = mv
        self.register_buffer("gate_w", mv.store_weight(mlp.gate_proj.weight))
        self.register_buffer("up_w", mv.store_weight(mlp.up_proj.weight))
        self.register_buffer("down_w", mv.store_weight(mlp.down_proj.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape                       # decode: 1, 1, hidden
        xr = x.reshape(s, d)                    # [1, hidden]
        h = F.silu(self.mv(xr, self.gate_w)) * self.mv(xr, self.up_w)   # [1, inter]
        return self.mv(h, self.down_w).reshape(b, s, d)


class MetalLinear(nn.Module):
    """nn.Linear drop-in (y = x @ W.T) via the matvec kernel. M=1 decode only (attn q_proj/o_proj)."""

    def __init__(self, lin: nn.Linear, mv: Matvec) -> None:
        super().__init__()
        self.mv = mv
        self.register_buffer("w", mv.store_weight(lin.weight))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape                       # decode: 1, 1, in
        y = self.mv(x.reshape(s, d), self.w)    # [1, out]
        return y.reshape(b, s, -1)


def metalize_minicpm4(bb: "minicpm4.MiniCPM4Backbone", mv: Matvec | None = None,
                      do_attn: bool = True) -> object:
    """Swap each layer's MLP (+ attn q/o) for the fp16 matvec kernel. Returns the shared kernel
    (pass to export_to_coreai_with_kernels(custom_kernels=[kernel])). Additive: norms/SDPA/KV/k,v
    untouched."""
    if mv is None:
        mv = Matvec("simd")
    for layer in bb.layers:
        layer.mlp = MetalMLP4(layer.mlp, mv)
        if do_attn:
            layer.self_attn.q_proj = MetalLinear(layer.self_attn.q_proj, mv)
            layer.self_attn.o_proj = MetalLinear(layer.self_attn.o_proj, mv)
    return mv.kernel
