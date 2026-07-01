"""M0: half x half -> half matmul2d via TorchMetalKernel, validated on M4 GPU vs torch.

Goal: prove coreai embeds + compiles + runs a matmul2d kernel, that the auto-generated
tensor<device T, dextents, tensor_handle> signature feeds matmul2d directly, and pin the
metal language version the runtime uses.
"""
import asyncio, os
from pathlib import Path
import tempfile
import numpy as np
import torch
import torch.nn as nn

from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter
from coreai.runtime import NDArray

M = int(os.environ.get("MM_M", 128))
K = int(os.environ.get("MM_K", 256))
N = int(os.environ.get("MM_N", 96))

# Metal tensor coords are TRANSPOSED vs numpy: torch[M,K] -> tensor extents [K,M]
# (dim0 = inner/contiguous). Verified with probe_dispatch: out[a,b] lands at numpy[b,a].
# So header-verbatim slicing is correct: tgid.x -> N tiles (step 32, tensor dim0),
# tgid.y -> M tiles (step 64, tensor dim1).
MM_SRC = r"""
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(metal::dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
"""

def torch_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

kernel = TorchMetalKernel(
    name="mm2d_hh",
    input_names=["A", "B"],
    result_names=["C"],
    src=MM_SRC,
    torch_defn=torch_mm,
    metal_params=[MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
)

class MM(nn.Module):
    def forward(self, a, b):
        Mm = a.shape[0]; Nn = b.shape[1]
        tg_n = (Nn + 31) // 32   # tgid.x tiles N (dim0, step 32)
        tg_m = (Mm + 63) // 64   # tgid.y tiles M (dim1, step 64)
        return kernel(a, b,
                      threads_per_grid=(128 * tg_n, tg_m, 1),
                      threads_per_thread_group=(128, 1, 1),
                      result_shapes=[[Mm, Nn]])

def build():
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randn(K, N, dtype=torch.float16)
    ep = torch.export.export(MM().eval(), (a, b)).run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["A", "B"], output_names=["C"])
    prog = conv.to_coreai(); prog.optimize()
    return prog, a, b

async def main():
    prog, a, b = build()
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "mm2d.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            out = await fn({"A": NDArray(a.numpy()), "B": NDArray(b.numpy())})
            gpu = out["C"].numpy().astype(np.float32)
    ref = (a.float() @ b.float()).numpy()
    g = gpu.reshape(-1); r = ref.reshape(-1)
    cos = float(g @ r / (np.linalg.norm(g) * np.linalg.norm(r) + 1e-12))
    rel = float(np.linalg.norm(g - r) / (np.linalg.norm(r) + 1e-12))
    print(f"M={M} K={K} N={N}  gpu={gpu.shape}")
    print(f"cos-sim = {cos:.6f}   rel-l2 = {rel:.6f}")
    # 2D correctness map over 64x32 tiles
    close = np.isclose(gpu, ref, rtol=2e-2, atol=1e-1)
    print(f"elemwise close frac = {close.mean():.3f}")
    for mi in range(0, M, 64):
        row = []
        for ni in range(0, N, 32):
            blk = close[mi:mi+64, ni:ni+32]
            row.append(f"{blk.mean():.2f}")
        print(f"  m-tile@{mi:4d}: " + " ".join(row))

if __name__ == "__main__":
    asyncio.run(main())
