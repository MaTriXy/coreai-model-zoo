"""M1a: half activations x int8 weights -> half, via matmul2d (native half x int8_t -> half).
Proves the quantized matmul2d path with zero reinterpret (int8_t is a native coreai dtype).
"""
import asyncio, os
from pathlib import Path
import tempfile
import numpy as np
import torch, torch.nn as nn
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter
from coreai.runtime import NDArray

M = int(os.environ.get("MM_M", 128))
K = int(os.environ.get("MM_K", 256))
N = int(os.environ.get("MM_N", 96))

# Same body as M0; only B's element dtype differs (int8_t via the auto signature).
SRC = r"""
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(metal::dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
"""

def torch_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # b is int8; matmul2d computes half x int8 -> half (raw integer weights, no scale)
    return (a.float() @ b.float()).to(torch.float16)

kernel = TorchMetalKernel(
    name="mm2d_hi8", input_names=["A", "B"], result_names=["C"],
    src=SRC, torch_defn=torch_mm,
    metal_params=[MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
)

class MM(nn.Module):
    def forward(self, a, b):
        Mm = a.shape[0]; Nn = b.shape[1]
        tg_n = (Nn + 31) // 32; tg_m = (Mm + 63) // 64
        return kernel(a, b,
                      threads_per_grid=(128 * tg_n, tg_m, 1),
                      threads_per_thread_group=(128, 1, 1),
                      result_shapes=[[Mm, Nn]])

async def main():
    a = torch.randn(M, K, dtype=torch.float16)
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    ep = torch.export.export(MM().eval(), (a, b)).run_decompositions(get_decomp_table())
    conv = TorchConverter(); conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["A", "B"], output_names=["C"])
    prog = conv.to_coreai(); prog.optimize()
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "mm2d_i8.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            out = await fn({"A": NDArray(a.numpy()), "B": NDArray(b.numpy())})
            gpu = out["C"].numpy().astype(np.float32)
    ref = (a.float() @ b.float()).numpy()
    g = gpu.reshape(-1); r = ref.reshape(-1)
    cos = float(g @ r / (np.linalg.norm(g)*np.linalg.norm(r)+1e-12))
    rel = float(np.linalg.norm(g-r)/(np.linalg.norm(r)+1e-12))
    print(f"[int8] M={M} K={K} N={N}  cos-sim={cos:.6f}  rel-l2={rel:.6f}")
    print(f"  gpu[0,:4]={gpu[0,:4]}  ref[0,:4]={ref[0,:4]}")

if __name__ == "__main__":
    asyncio.run(main())
