# Community port — NOT an Apple model.
"""A19 kernel probe: INT2 unpack variants at the real gemma4 mixed-bit shapes.

Exports one tiny bundle per variant — a chained stack of L int2 FFN halves
(gateup 1536->12288 + down 12288->1536, RANDOM packed codes) followed by the
262144-row int2 lm_head matvec — with input "A" [1,1536] fp16 and output
"C" [1,262144]. Runs UNCHANGED on the PipelinedBench PB_MM harness (it loads
fn "main", streams A, times the run): sideload to Documents/models/<name> and
launch with PB_MM=<name>.

Per-run weight read ~185 MB (~half a real token's int2 bytes), so ms directly
ranks the variants' effective bandwidth on A19. Variants:
  base    R=4 SGY=8, scalar shift/xor unpack (the shipped P3/P4 kernel)
  r8sgy4  R=8 SGY=4, scalar (tiling shape only)
  lutf4   R=4 SGY=8, threadgroup byte-LUT (256 x float4) + float4 FMA
  lutr2   R=2 SGY=8, same LUT body (register-pressure point)

Run (from the coreai-models checkout):
  .venv/bin/python ../coreai-models-community/conversion/export_gemma4_kprobe_int2.py [--variants a,b]
Then per variant:
  xcrun coreai-build compile exports/gemma4_kprobe_<v>/gemma4_kprobe_<v>.aimodel \
    --output exports/gemma4_kprobe_<v> --platform iOS --architecture h18p --expect-frequent-reshapes
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from torch import nn

from coreai_models.models.macos.gemma4_metal_mlp import export_to_coreai_with_kernels
from coreai_models.models.macos.gemma4_metal_mlp_int2 import qp2_from_packed

from coreai_torch import MetalParameter, TorchMetalKernel

DTYPE = torch.float16
K_DIM = 1536
N_FFN = 12288
HEAD_ROWS = 262144
LAYERS = 6

# ---- MSL bodies (same I/O contract as the shipped kernels) ---------------------------------------

_SCALAR_MATVEC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = uint(QP[w0, n]);
            float s = 0.0f;
            for (uint j = 0; j < 16; ++j) {
                uint q = (packed >> (2 * j)) & 0x3;
                s += xr[j] * float(int(q ^ 2u) - 2);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            C[n, 0] = TYPE(tot * float(SC[n]));
        }
    }
"""

_SCALAR_GATEUP = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float accg[__R__], accu[__R__];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(A[k0 + j, 0]);
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = uint(QPG[w0, n]);
            uint pu = uint(QPU[w0, n]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint j = 0; j < 16; ++j) {
                float x = xr[j];
                sgm += x * float(int(((pg >> (2 * j)) & 0x3) ^ 2u) - 2);
                sum += x * float(int(((pu >> (2 * j)) & 0x3) ^ 2u) - 2);
            }
            accg[r] += sgm;
            accu[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            uint n = base_row + r;
            float xg = tg * float(SCG[n]);
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n, 0] = TYPE(gel * (tu * float(SCU[n])));
        }
    }
"""

# byte -> 4 codes via a 4 KB threadgroup float4 LUT; inner loop = 4 gathers + 4 float4 FMAs
# per 32-bit word instead of 16 shift/xor/cvt chains. Same products in fp32 (order-only noise).
_LUT_MATVEC = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    threadgroup float4 lut[256];
    for (uint b = sg * 32 + lane; b < 256; b += 32 * SGY) {
        lut[b] = float4(float(int((b & 3u) ^ 2u) - 2),
                        float(int(((b >> 2) & 3u) ^ 2u) - 2),
                        float(int(((b >> 4) & 3u) ^ 2u) - 2),
                        float(int(((b >> 6) & 3u) ^ 2u) - 2));
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float acc[__R__];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float4 x0 = float4(float(A[k0 + 0, 0]), float(A[k0 + 1, 0]), float(A[k0 + 2, 0]), float(A[k0 + 3, 0]));
        float4 x1 = float4(float(A[k0 + 4, 0]), float(A[k0 + 5, 0]), float(A[k0 + 6, 0]), float(A[k0 + 7, 0]));
        float4 x2 = float4(float(A[k0 + 8, 0]), float(A[k0 + 9, 0]), float(A[k0 + 10, 0]), float(A[k0 + 11, 0]));
        float4 x3 = float4(float(A[k0 + 12, 0]), float(A[k0 + 13, 0]), float(A[k0 + 14, 0]), float(A[k0 + 15, 0]));
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint p = uint(QP[w0, n]);
            float4 s4 = x0 * lut[p & 0xffu]
                      + x1 * lut[(p >> 8) & 0xffu]
                      + x2 * lut[(p >> 16) & 0xffu]
                      + x3 * lut[(p >> 24) & 0xffu];
            acc[r] += s4.x + s4.y + s4.z + s4.w;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            C[n, 0] = TYPE(tot * float(SC[n]));
        }
    }
"""

_LUT_GATEUP = """
    const uint R = __R__, SGY = __SGY__;
    const uint K = A.get_extent(0);
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    threadgroup float4 lut[256];
    for (uint b = sg * 32 + lane; b < 256; b += 32 * SGY) {
        lut[b] = float4(float(int((b & 3u) ^ 2u) - 2),
                        float(int(((b >> 2) & 3u) ^ 2u) - 2),
                        float(int(((b >> 4) & 3u) ^ 2u) - 2),
                        float(int(((b >> 6) & 3u) ^ 2u) - 2));
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    float accg[__R__], accu[__R__];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float4 x0 = float4(float(A[k0 + 0, 0]), float(A[k0 + 1, 0]), float(A[k0 + 2, 0]), float(A[k0 + 3, 0]));
        float4 x1 = float4(float(A[k0 + 4, 0]), float(A[k0 + 5, 0]), float(A[k0 + 6, 0]), float(A[k0 + 7, 0]));
        float4 x2 = float4(float(A[k0 + 8, 0]), float(A[k0 + 9, 0]), float(A[k0 + 10, 0]), float(A[k0 + 11, 0]));
        float4 x3 = float4(float(A[k0 + 12, 0]), float(A[k0 + 13, 0]), float(A[k0 + 14, 0]), float(A[k0 + 15, 0]));
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = uint(QPG[w0, n]);
            uint pu = uint(QPU[w0, n]);
            float4 sg4 = x0 * lut[pg & 0xffu]
                       + x1 * lut[(pg >> 8) & 0xffu]
                       + x2 * lut[(pg >> 16) & 0xffu]
                       + x3 * lut[(pg >> 24) & 0xffu];
            float4 su4 = x0 * lut[pu & 0xffu]
                       + x1 * lut[(pu >> 8) & 0xffu]
                       + x2 * lut[(pu >> 16) & 0xffu]
                       + x3 * lut[(pu >> 24) & 0xffu];
            accg[r] += sg4.x + sg4.y + sg4.z + sg4.w;
            accu[r] += su4.x + su4.y + su4.z + su4.w;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            uint n = base_row + r;
            float xg = tg * float(SCG[n]);
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n, 0] = TYPE(gel * (tu * float(SCU[n])));
        }
    }
"""

VARIANTS = {
    "base": dict(r=4, sgy=8, mv=_SCALAR_MATVEC, gu=_SCALAR_GATEUP),
    "r8sgy4": dict(r=8, sgy=4, mv=_SCALAR_MATVEC, gu=_SCALAR_GATEUP),
    "lutf4": dict(r=4, sgy=8, mv=_LUT_MATVEC, gu=_LUT_GATEUP),
    "lutr2": dict(r=2, sgy=8, mv=_LUT_MATVEC, gu=_LUT_GATEUP),
}


def _unpack_u32_codes(qp: torch.Tensor) -> torch.Tensor:
    n, k16 = qp.shape
    p = qp.to(torch.int64)
    c = torch.stack([(p >> (2 * j)) & 0x3 for j in range(16)], dim=-1).reshape(n, k16 * 16)
    return torch.where(c >= 2, c - 4, c).float()


def _mv_torch_defn(x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
    w = _unpack_u32_codes(qp) * sc.float().unsqueeze(1)
    return torch.nn.functional.linear(x, w.to(x.dtype))


def _gu_torch_defn(x: torch.Tensor, qpg: torch.Tensor, scg: torch.Tensor,
                   qpu: torch.Tensor, scu: torch.Tensor) -> torch.Tensor:
    g = torch.nn.functional.linear(x, (_unpack_u32_codes(qpg) * scg.float().unsqueeze(1)).to(x.dtype))
    u = torch.nn.functional.linear(x, (_unpack_u32_codes(qpu) * scu.float().unsqueeze(1)).to(x.dtype))
    return torch.nn.functional.gelu(g, approximate="tanh") * u


def build_kernels(variant: str):
    v = VARIANTS[variant]
    mv = TorchMetalKernel(
        f"kprobe_mv_{variant}", input_names=["A", "QP", "SC"], result_names=["C"],
        src=v["mv"].replace("__R__", str(v["r"])).replace("__SGY__", str(v["sgy"])),
        torch_defn=_mv_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )
    gu = TorchMetalKernel(
        f"kprobe_gu_{variant}", input_names=["A", "QPG", "SCG", "QPU", "SCU"], result_names=["C"],
        src=v["gu"].replace("__R__", str(v["r"])).replace("__SGY__", str(v["sgy"])),
        torch_defn=_gu_torch_defn,
        metal_params=[MetalParameter("tid", "uint2", "thread_position_in_threadgroup"),
                      MetalParameter("tgid", "uint2", "threadgroup_position_in_grid")],
        template_dtypes={"A": "TYPE"},
    )
    return mv, gu, v["r"], v["sgy"]


def rand_packed(rows: int, cols: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    packed = torch.randint(0, 256, (rows, cols // 4), dtype=torch.uint8, generator=gen)
    scale = torch.full((rows,), 1.0e-3, dtype=torch.float32)
    return qp2_from_packed(packed.flatten(), rows, cols), scale


class ProbeStack(nn.Module):
    """L x (int2 gateup + int2 down) + the 262144-row int2 head. A [1,K] -> C [1,HEAD_ROWS]."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        self.mv, self.gu, self.r, self.sgy = build_kernels(variant)
        gen = torch.Generator().manual_seed(0)
        for i in range(LAYERS):
            for name, (rows, cols) in (("g", (N_FFN, K_DIM)), ("u", (N_FFN, K_DIM)),
                                       ("d", (K_DIM, N_FFN))):
                qp, sc = rand_packed(rows, cols, gen)
                self.register_buffer(f"l{i}_{name}_qp", qp)
                self.register_buffer(f"l{i}_{name}_sc", sc)
        qp, sc = rand_packed(HEAD_ROWS, K_DIM, gen)
        self.register_buffer("head_qp", qp)
        self.register_buffer("head_sc", sc)

    def _mv(self, x: torch.Tensor, qp: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        n = qp.shape[0]
        return self.mv(x, qp, sc, threads_per_grid=(32, n // self.r, 1),
                       threads_per_thread_group=(32, self.sgy, 1), result_shapes=[[1, n]])

    def _gu(self, x: torch.Tensor, qpg, scg, qpu, scu) -> torch.Tensor:
        n = qpg.shape[0]
        return self.gu(x, qpg, scg, qpu, scu, threads_per_grid=(32, n // self.r, 1),
                       threads_per_thread_group=(32, self.sgy, 1), result_shapes=[[1, n]])

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        x = A
        for i in range(LAYERS):
            h = self._gu(x, getattr(self, f"l{i}_g_qp"), getattr(self, f"l{i}_g_sc"),
                         getattr(self, f"l{i}_u_qp"), getattr(self, f"l{i}_u_sc"))
            x = self._mv(h, getattr(self, f"l{i}_d_qp"), getattr(self, f"l{i}_d_sc"))
        return self._mv(x, self.head_qp, self.head_sc)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out-dir", default="exports")
    args = ap.parse_args()

    for variant in args.variants.split(","):
        name = f"gemma4_kprobe_{variant}"
        print(f"=== {name} ===", flush=True)
        model = ProbeStack(variant).eval()
        ref = {"A": torch.zeros(1, K_DIM, dtype=DTYPE)}
        prog = export_to_coreai_with_kernels(
            model, reference_inputs=ref, custom_kernels=[model.mv, model.gu],
            input_names=("A",), output_names=("C",), state_names=(),
        )
        prog.optimize()
        out_dir = Path(args.out_dir) / name
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)

        import coreai.runtime as rt

        prog.save_asset(out_dir / f"{name}.aimodel", rt.AIModelAssetMetadata())
        print(f"saved {out_dir}/{name}.aimodel", flush=True)


if __name__ == "__main__":
    main()
