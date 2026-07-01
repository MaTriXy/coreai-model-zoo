"""M1b: half activations x UNIFORM int4 weights -> half via matmul2d.
Weights arrive as a packed uint8 tensor; reinterpret bytes as int4b_format inside the
kernel (tensor_inline from a device uchar* pointer) and feed matmul2d. N,K baked as
literals. No per-block scale (uniform int4) -- proves the uint8->int4 reinterpret path.
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
N = int(os.environ.get("MM_N", 128))

def make_src(N: int, K: int) -> str:
    # A: half tensor [K,M]; Wp bytes -> int4b_format tensor [N,K] (dim0=N packed 2/byte);
    # C: half tensor [N,M]. matmul2d half x int4 -> half.
    return f"""
    constexpr auto desc = matmul2d_descriptor(64, 32,
        static_cast<int>(metal::dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;

    device uchar* wptr = &Wp[0, 0];
    metal::dextents<int, 2> wext({N}, {K});
    tensor<device metal::int4b_format, metal::dextents<int, 2>, tensor_inline> Wi(wptr, wext);

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = Wi.slice(tgid.x * 32, 0);
    auto mC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(mA, mB, mC);
"""

def pack_int4(w: torch.Tensor) -> torch.Tensor:
    Kd, Nd = w.shape
    flat = (w.reshape(-1) & 0xF).to(torch.int32)
    lo = flat[0::2]; hi = flat[1::2]
    return (lo | (hi << 4)).to(torch.uint8).reshape(Kd, Nd // 2)

def build_kernel(N: int, K: int):
    def torch_mm(a: torch.Tensor, wp: torch.Tensor) -> torch.Tensor:
        return a.new_zeros((a.shape[0], N))
    return TorchMetalKernel(
        name="mm2d_i4", input_names=["A", "Wp"], result_names=["C"],
        src=make_src(N, K), torch_defn=torch_mm,
        metal_params=[MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
    )

async def main():
    kernel = build_kernel(N, K)

    class MM(nn.Module):
        def forward(self, a, wp):
            Mm = a.shape[0]; Nn = wp.shape[1] * 2
            tg_n = (Nn + 31) // 32; tg_m = (Mm + 63) // 64
            return kernel(a, wp,
                          threads_per_grid=(128 * tg_n, tg_m, 1),
                          threads_per_thread_group=(128, 1, 1),
                          result_shapes=[[Mm, Nn]])

    a = torch.randn(M, K, dtype=torch.float16)
    w = torch.randint(-8, 8, (K, N), dtype=torch.int32)
    wp = pack_int4(w)
    ep = torch.export.export(MM().eval(), (a, wp)).run_decompositions(get_decomp_table())
    conv = TorchConverter(); conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["A", "Wp"], output_names=["C"])
    prog = conv.to_coreai(); prog.optimize()
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "mm2d_i4.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            out = await fn({"A": NDArray(a.numpy()), "Wp": NDArray(wp.numpy())})
            gpu = out["C"].numpy().astype(np.float32)
    ref = (a.float() @ w.float()).numpy()
    g = gpu.reshape(-1); r = ref.reshape(-1)
    cos = float(g @ r / (np.linalg.norm(g)*np.linalg.norm(r)+1e-12))
    rel = float(np.linalg.norm(g-r)/(np.linalg.norm(r)+1e-12))
    print(f"[int4-uniform] M={M} K={K} N={N}  cos-sim={cos:.6f}  rel-l2={rel:.6f}")
    print(f"  gpu[0,:6]={gpu[0,:6]}")
    print(f"  ref[0,:6]={ref[0,:6]}")

if __name__ == "__main__":
    asyncio.run(main())
