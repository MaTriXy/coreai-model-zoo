"""Probe: which threadgroups actually run, and what tgid they see, under dispatchThreads.
Output [64,32] (known-good aligned shape). Thread 0 of each group marks out[tgid.x, tgid.y]."""
import asyncio, os
from pathlib import Path
import tempfile
import numpy as np
import torch, torch.nn as nn
from coreai_torch import TorchMetalKernel, TorchConverter, get_decomp_table, MetalParameter
from coreai.runtime import NDArray

TGX = int(os.environ.get("TGX", 2))   # desired threadgroups in x
TGY = int(os.environ.get("TGY", 3))   # desired threadgroups in y
TPTGX = int(os.environ.get("TPTGX", 128))

SRC = r"""
    if (ltid.x == 0 && ltid.y == 0) {
        out[tgid.x, tgid.y] = (float)(100 + tgid.x * 10 + tgid.y);
    }
"""

def probe_ref(x: torch.Tensor) -> torch.Tensor:
    return x.new_zeros((64, 32))

kernel = TorchMetalKernel(
    name="probe2",
    input_names=["x"],
    result_names=["out"],
    src=SRC,
    torch_defn=probe_ref,
    metal_params=[
        MetalParameter("tgid", "uint2", "threadgroup_position_in_grid"),
        MetalParameter("ltid", "uint2", "thread_position_in_threadgroup"),
    ],
)

class P(nn.Module):
    def forward(self, x):
        return kernel(x,
                      threads_per_grid=(TPTGX * TGX, TGY, 1),
                      threads_per_thread_group=(TPTGX, 1, 1),
                      result_shapes=[[64, 32]])

async def main():
    x = torch.zeros(1)
    ep = torch.export.export(P().eval(), (x,)).run_decompositions(get_decomp_table())
    conv = TorchConverter(); conv.register_custom_kernels([kernel])
    conv.add_exported_program(ep, input_names=["x"], output_names=["out"])
    prog = conv.to_coreai(); prog.optimize()
    with tempfile.TemporaryDirectory() as td:
        asset = prog.save_asset(Path(td) / "probe2.aimodel")
        async with asset.executable() as ai:
            fn = ai.load_function("main")
            out = await fn({"x": NDArray(x.numpy())})
            o = out["out"].numpy()
    print(f"requested threads_per_grid=({TPTGX*TGX},{TGY},1) tptg=({TPTGX},1,1) -> expect {TGX}x{TGY} groups")
    sub = o[:TGX+1, :TGY+1].astype(int)
    print("out[tgid.x, tgid.y] marks (value 100+10x+y where a group ran):")
    print(sub)

if __name__ == "__main__":
    asyncio.run(main())
