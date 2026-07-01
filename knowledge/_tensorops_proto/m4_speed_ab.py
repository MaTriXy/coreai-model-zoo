"""Gating de-risk for a Mac (M4 Max) prefill kernel: does the matrix-unit path (matmul2d, reused from
M0 — proven to compile+run on M4) BEAT coreai's default MPSGraph matmul on M4 Max at compute-bound
shapes? If matmul2d can't beat MPSGraph matmul here, a custom FlashAttention won't beat MPSGraph SDPA
either (same matrix ceiling). If it can, FlashAttention is worth building.

Run: ../../coreai-models/.venv/bin/python -u m4_speed_ab.py
"""
import asyncio, time
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter
from coreai.runtime import NDArray

MM_SRC = r"""
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(metal::dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
"""
def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b


kernel = TorchMetalKernel(
    name="mm2d_hh", input_names=["A", "B"], result_names=["C"], src=MM_SRC,
    torch_defn=_mm,
    metal_params=[MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")])


class MM2D(nn.Module):
    def forward(self, a, b):
        Mm = a.shape[0]; Nn = b.shape[1]
        return kernel(a, b, threads_per_grid=(128 * ((Nn + 31) // 32), (Mm + 63) // 64, 1),
                      threads_per_thread_group=(128, 1, 1), result_shapes=[[Mm, Nn]])


class MMref(nn.Module):
    def forward(self, a, b):
        return a @ b


def build(mod, a, b, kernels):
    ep = torch.export.export(mod.eval(), (a, b)).run_decompositions(get_decomp_table())
    conv = TorchConverter()
    if kernels:
        conv.register_custom_kernels(kernels)
    conv.add_exported_program(ep, input_names=["A", "B"], output_names=["C"])
    prog = conv.to_coreai(); prog.optimize()
    return prog


async def bench(prog, a, b, iters=30):
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "m.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            A = NDArray(a.numpy()); B = NDArray(b.numpy())
            for _ in range(4):
                out = await fn({"A": A, "B": B})
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter(); out = await fn({"A": A, "B": B}); ts.append((time.perf_counter()-t0)*1e3)
            g = out["C"].numpy().astype(np.float32)
    ms = float(np.median(sorted(ts)))
    return ms, g


async def main():
    torch.manual_seed(0)
    print("== matmul2d vs MPSGraph matmul on M4 Max GPU (compute-bound square shapes) ==")
    print(f"   {'M=K=N':>7} {'MPSGraph ms':>12} {'TFLOP/s':>8} | {'matmul2d ms':>12} {'TFLOP/s':>8} | {'ratio':>6} {'cos':>7}")
    for S in [1024, 2048, 4096]:
        a = torch.randn(S, S, dtype=torch.float16); b = torch.randn(S, S, dtype=torch.float16)
        flops = 2.0 * S * S * S
        ref_ms, ref_g = await bench(build(MMref(), a, b, None), a, b)
        k_ms, k_g = await bench(build(MM2D(), a, b, [kernel]), a, b)
        cos = float(k_g.reshape(-1) @ ref_g.reshape(-1) / (np.linalg.norm(k_g)*np.linalg.norm(ref_g)+1e-9))
        print(f"   {S:>7} {ref_ms:>12.3f} {flops/ref_ms/1e9:>8.2f} | {k_ms:>12.3f} {flops/k_ms/1e9:>8.2f} | "
              f"{ref_ms/k_ms:>5.2f}x {cos:>7.4f}")


if __name__ == "__main__":
    asyncio.run(main())
