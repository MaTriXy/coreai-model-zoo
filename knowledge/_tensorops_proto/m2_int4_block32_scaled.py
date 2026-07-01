"""M2: LLaDA-shaped block-32 fp16-scaled int4 matmul.
Per K-block-of-32: dequant int4 weights -> threadgroup half, scaled by the block's fp16
scale, then matmul2d half x half accumulate into float C. Exact fp16 scales (no E8M0).
Validates numerics vs a torch blockwise-dequant reference.
"""
import asyncio, os
from pathlib import Path
import tempfile
import numpy as np
import torch, torch.nn as nn
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter
from coreai.runtime import NDArray

M = int(os.environ.get("MM_M", 128))
K = int(os.environ.get("MM_K", 256))   # must be multiple of 32
N = int(os.environ.get("MM_N", 128))
BLK = 32

def make_src(N: int, K: int) -> str:
    return f"""
    constexpr int TILE_M = 64, TILE_N = 32, BLK = 32;
    constexpr auto descMul = matmul2d_descriptor(TILE_M, TILE_N, BLK, false, false, false,
        matmul2d_descriptor::mode::multiply);
    constexpr auto descAcc = matmul2d_descriptor(TILE_M, TILE_N, BLK, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<descMul, execution_simdgroups<4>> opMul;
    matmul2d<descAcc, execution_simdgroups<4>> opAcc;

    // raw packed weight bytes; decode signed int4 manually (unambiguous).
    // packing: linear int4 index = k*N + n (C-order, n fastest), 2/byte, low nibble = even.
    device uchar* wptr = &Wp[0, 0];

    threadgroup half wsh[BLK * TILE_N];   // dequantized weight block, layout [k][n] (n fastest)

    auto mC = C.slice(tgid.x * TILE_N, tgid.y * TILE_M);
    uint nblk = {K} / BLK;
    for (uint b = 0; b < nblk; ++b) {{
        // dequant this [k=32, n=32] weight block into threadgroup, scaled by fp16 block scale
        for (uint e = ltid.x; e < BLK * TILE_N; e += 128) {{
            uint kk = e / TILE_N;                 // 0..31 within-block k
            uint nn = e % TILE_N;                 // 0..31 within-tile n
            uint nglob = tgid.x * TILE_N + nn;
            uint kglob = b * BLK + kk;
            uint lin = kglob * {N}u + nglob;      // linear int4 index
            uchar byte = wptr[lin >> 1];
            int code = (lin & 1u) ? (int)(byte >> 4) : (int)(byte & 0xF);
            if (code > 7) code -= 16;             // signed int4 [-8,7]
            half s = Sc[nglob, b];                // Sc = tensor[N, K/32]
            wsh[kk * TILE_N + nn] = (half)code * s;
        }}
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        tensor<threadgroup half, metal::dextents<int, 2>, tensor_inline>
            Wsh((threadgroup half*)wsh, metal::dextents<int, 2>(TILE_N, BLK));   // [n=32, k=32]
        auto pA = A.slice(b * BLK, tgid.y * TILE_M);                            // [k=32, m=64]
        if (b == 0) opMul.run(pA, Wsh, mC);
        else        opAcc.run(pA, Wsh, mC);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }}
"""

def pack_int4(w: torch.Tensor) -> torch.Tensor:
    Kd, Nd = w.shape
    flat = (w.reshape(-1) & 0xF).to(torch.int32)
    lo = flat[0::2]; hi = flat[1::2]
    return (lo | (hi << 4)).to(torch.uint8).reshape(Kd, Nd // 2)

def build_kernel(N: int, K: int):
    def ref(a: torch.Tensor, wp: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        return a.new_zeros((a.shape[0], N), dtype=torch.float32)
    return TorchMetalKernel(
        name="mm2d_i4_bs", input_names=["A", "Wp", "Sc"], result_names=["C"],
        src=make_src(N, K), torch_defn=ref,
        metal_params=[
            MetalParameter("tgid", "uint2", "threadgroup_position_in_grid"),
            MetalParameter("ltid", "uint2", "thread_position_in_threadgroup"),
        ],
    )

async def main():
    kernel = build_kernel(N, K)

    class MM(nn.Module):
        def forward(self, a, wp, sc):
            Mm = a.shape[0]; Nn = wp.shape[1] * 2
            tg_n = (Nn + 31) // 32; tg_m = (Mm + 63) // 64
            return kernel(a, wp, sc,
                          threads_per_grid=(128 * tg_n, tg_m, 1),
                          threads_per_thread_group=(128, 1, 1),
                          result_shapes=[[Mm, Nn]])

    torch.manual_seed(0)
    a = torch.randn(M, K, dtype=torch.float16)
    w = torch.randint(-8, 8, (K, N), dtype=torch.int32)                 # int4 codes
    nblk = K // BLK
    scale = (torch.rand(nblk, N, dtype=torch.float16) * 0.1 + 0.02)     # fp16 per-block scale
    wp = pack_int4(w)
    ep = torch.export.export(MM().eval(), (a, wp, scale)).run_decompositions(get_decomp_table())
    conv = TorchConverter(); conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["A", "Wp", "Sc"], output_names=["C"])
    prog = conv.to_coreai(); prog.optimize()
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "mm2d_i4_bs.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            out = await fn({"A": NDArray(a.numpy()), "Wp": NDArray(wp.numpy()),
                            "Sc": NDArray(scale.numpy())})
            gpu = out["C"].numpy().astype(np.float32)
    # torch reference: dequant W[k,n] = code * scale[k//32, n]
    wdeq = w.float() * scale.float().repeat_interleave(BLK, dim=0)      # [K,N]
    ref = (a.float() @ wdeq).numpy()
    g = gpu.reshape(-1); r = ref.reshape(-1)
    cos = float(g @ r / (np.linalg.norm(g)*np.linalg.norm(r)+1e-12))
    rel = float(np.linalg.norm(g-r)/(np.linalg.norm(r)+1e-12))
    print(f"[int4 block-32 scaled] M={M} K={K} N={N}  cos-sim={cos:.6f}  rel-l2={rel:.6f}")
    print(f"  gpu[0,:6]={gpu[0,:6]}")
    print(f"  ref[0,:6]={ref[0,:6]}")

if __name__ == "__main__":
    asyncio.run(main())
