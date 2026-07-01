"""Export a matmul model to a persistent .aimodel for A19 AOT compile + PipelinedBench timing.
MODE=mm2d : the M0 matmul2d TorchMetalKernel (half x half -> half, neural-accel path)
MODE=ref  : plain torch a@b (coreai default matmul / MPSGraph) -- the A/B baseline
Same shape M(S) x K x N for both. Saves to OUTDIR/<name>.aimodel.
"""
import os, sys
from pathlib import Path
import torch, torch.nn as nn
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter

M = int(os.environ.get("MM_M", 128))
K = int(os.environ.get("MM_K", 4096))
N = int(os.environ.get("MM_N", 12288))
MODE = os.environ.get("MODE", "mm2d")
OUT = Path(os.environ.get("OUTDIR", "/tmp/mm_device"))
OUT.mkdir(parents=True, exist_ok=True)

RELAXED = "true" if os.environ.get("RELAXED", "0") == "1" else "false"
MT = int(os.environ.get("MTILE", 64))   # output tile rows (M)
NT = int(os.environ.get("NTILE", 32))   # output tile cols (N)
NS = int(os.environ.get("NSIMD", 4))    # simdgroups per threadgroup
MM_SRC = f"""
    constexpr auto desc = matmul2d_descriptor({MT}, {NT},
        static_cast<int>(metal::dynamic_extent), false, false, {RELAXED});
    matmul2d<desc, execution_simdgroups<{NS}>> op;
    auto mA = A.slice(0, tgid.y * {MT});
    auto mB = B.slice(tgid.x * {NT}, 0);
    auto mC = C.slice(tgid.x * {NT}, tgid.y * {MT});
    op.run(mA, mB, mC);
"""

def torch_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a @ b

def build_mm2d():
    # B is a RESIDENT weight (baked constant), only A streams in -- matches LLaDA (weights resident).
    kernel = TorchMetalKernel(
        name="mm2d_hh", input_names=["A", "B"], result_names=["C"],
        src=MM_SRC, torch_defn=torch_mm,
        metal_params=[MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
    )
    class MM(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("B", torch.randn(K, N, dtype=torch.float16))
        def forward(self, a):
            Mm = a.shape[0]; Nn = self.B.shape[1]
            tpg = 32 * NS
            tg_n = (Nn + NT - 1) // NT; tg_m = (Mm + MT - 1) // MT
            return kernel(a, self.B, threads_per_grid=(tpg * tg_n, tg_m, 1),
                          threads_per_thread_group=(tpg, 1, 1),
                          result_shapes=[[Mm, Nn]])
    a = torch.randn(M, K, dtype=torch.float16)
    ep = torch.export.export(MM().eval(), (a,)).run_decompositions(get_decomp_table())
    conv = TorchConverter(); conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["A"], output_names=["C"])
    return conv.to_coreai()

def build_ref():
    class MM(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("B", torch.randn(K, N, dtype=torch.float16))
        def forward(self, a):
            return a @ self.B
    a = torch.randn(M, K, dtype=torch.float16)
    ep = torch.export.export(MM().eval(), (a,)).run_decompositions(get_decomp_table())
    conv = TorchConverter()
    conv.add_exported_program(ep, input_names=["A"], output_names=["C"])
    return conv.to_coreai()

name = f"mm_{MODE}_s{M}_k{K}_n{N}"
prog = build_mm2d() if MODE == "mm2d" else build_ref()
prog.optimize()
asset_path = OUT / f"{name}.aimodel"
prog.save_asset(asset_path)
print(f"SAVED {asset_path}")
