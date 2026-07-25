// Gemma4 E2B mixed-bit RAW-METAL decode loop — quantized matvec kernels (P1).
//
// Plain-MSL ports of the PROVEN DSL kernels in gemma4_metal_mlp{,_int2}.py
// (shipped in the ffnfused bundle, oracle-token-gated 3/3). Same math, same
// tiling (R=4 rows/SIMD-group, SGY=8 SIMD-groups/threadgroup), same fp32
// accumulate + fp16 I/O; only the DSL accessors are replaced by raw pointers.
//
// Layouts (row-major, contiguous; N % 32 == 0 everywhere in this model):
//   X    [K]        half   — the S=1 activation row
//   QP2  [N, K/16]  uint32 — INT2: 16 codes/word, code j at bits [2j+1:2j],
//                            signed two's complement ((q^2)-2 decode)
//   QP4  [N, K/8]   uint32 — INT4: 8 nibbles/word, low nibble first, UNSIGNED
//                            codes q_u = q_s+8 (affine hosting of symmetric)
//   SC2  [N]        float  — INT2 per-row symmetric scale
//   SC4  [N, K/G]   half   — INT4 affine scale  (G = 64), per-row constant
//   BI4  [N, K/G]   half   — INT4 affine bias = -8*scale (exact in fp16)
//   W8   [N, K]     char   — INT8 per-row symmetric codes (PLE projections)
//   SC8  [N]        float  — INT8 per-row scale
//   C    [N]        half   — output row
//
// Dispatch (all kernels): threads = [32, N/R], group_size = [32, SGY].

#include <metal_stdlib>
using namespace metal;

constant uint R   = 4;    // rows per SIMD-group
constant uint SGY = 8;    // SIMD-groups per threadgroup
constant uint G   = 64;   // INT4 affine group size along K

// interleave-4 pack layout (p2b_interleave_pack.py): QP words stored [N/4, kw, 4] so an
// R=4 SIMD-group fetches word w0 of all four rows with ONE uint4 load (A19 wide-load
// lane). Specialized via function constant — the default (flat) codegen is untouched.
// Per-row word VALUES are identical either way, so the dot order proof is unchanged.
#ifndef IL4
#define IL4 0
#endif

// word w0 of rows base..base+3 as a uint4 (il: single 16B load; flat: 4 row-major loads)
#define QP_ROWS4(QP, base, kw, w0)                                          \
    (IL4 ? ((device const uint4*)(QP))[((base) >> 2) * (kw) + (w0)]         \
         : uint4((QP)[(base) * (kw) + (w0)],                                \
                 (QP)[((base) + 1) * (kw) + (w0)],                          \
                 (QP)[((base) + 2) * (kw) + (w0)],                          \
                 (QP)[((base) + 3) * (kw) + (w0)]))

// ---- INT2 symmetric matvec (FFN L15-34 down + the 262144-row lm_head) -----------------
kernel void matvec_int2sym(
    device const half*  X   [[buffer(0)]],
    device const uint*  QP  [[buffer(1)]],
    device const float* SC  [[buffer(2)]],
    device half*        C   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;                          // uint32 words per row

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {           // block = 32 lanes * 16 codes
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(X[k0 + j]);
        uint w0 = (kb >> 4) + lane;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint packed = pk[r];
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
            C[n] = half(tot * SC[n]);
        }
    }
}

// ---- INT4 affine matvec (FFN L0-14 down, attn q/k/v/o all layers) ----------------------
kernel void matvec_int4aff(
    device const half* X   [[buffer(0)]],
    device const uint* QP  [[buffer(1)]],
    device const half* SC  [[buffer(2)]],
    device const half* BI  [[buffer(3)]],
    device half*       C   [[buffer(4)]],
    constant uint&     K   [[buffer(5)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;                          // uint32 words per row
    const uint ng = K / G;                           // affine groups per row

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {           // block = 32 lanes * 8 nibbles
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(X[k0 + j]);
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;                           // constant for this lane's 8 vals
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = pk[r];
            float sc = float(SC[n * ng + grp]);
            float bi = float(BI[n * ng + grp]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r] = half(tot);
    }
}

// ---- FUSED gate+up+gelu+mul, INT2 (L15-34 double-wide FFN first half) -------------------
kernel void gateup_int2sym(
    device const half*  X    [[buffer(0)]],
    device const uint*  QPG  [[buffer(1)]],
    device const float* SCG  [[buffer(2)]],
    device const uint*  QPU  [[buffer(3)]],
    device const float* SCU  [[buffer(4)]],
    device half*        C    [[buffer(5)]],
    constant uint&      K    [[buffer(6)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(X[k0 + j]);
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = QPG[n * kw + w0];
            uint pu = QPU[n * kw + w0];
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
            float xg = tg * SCG[n];
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n] = half(gel * (tu * SCU[n]));
        }
    }
}

// ---- FUSED gate+up+gelu+mul, INT4 (L0-14 FFN first half) --------------------------------
kernel void gateup_int4aff(
    device const half* X    [[buffer(0)]],
    device const uint* QPG  [[buffer(1)]],
    device const half* SCG  [[buffer(2)]],
    device const half* BIG  [[buffer(3)]],
    device const uint* QPU  [[buffer(4)]],
    device const half* SCU  [[buffer(5)]],
    device const half* BIU  [[buffer(6)]],
    device half*       C    [[buffer(7)]],
    constant uint&     K    [[buffer(8)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;
    const uint ng = K / G;

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(X[k0 + j]);
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = QPG[n * kw + w0];
            uint pu = QPU[n * kw + w0];
            float scg = float(SCG[n * ng + grp]), big = float(BIG[n * ng + grp]);
            float scu = float(SCU[n * ng + grp]), biu = float(BIU[n * ng + grp]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                float x = xr[j];
                sgm += x * (scg * float((pg >> (j * 4)) & 0xf) + big);
                sum += x * (scu * float((pu >> (j * 4)) & 0xf) + biu);
            }
            accg[r] += sgm;
            accu[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
            C[base_row + r] = half(gel * tu);
        }
    }
}

// ---- R=1 variants (N=1536 outputs: FFN down + attn o) -------------------------------------
// The R=4 tiling starves occupancy at N=1536 (384 SIMD-groups); R=1 quadruples the
// groups in flight (measured: int2 down 48->31us, int4 down ~2x). Per-row dot order
// is unchanged — bit-exact vs the R=4 kernels. Dispatch: threads = [32, N], tg [32, 8].
kernel void matvec_int2sym_r1(
    device const half*  X   [[buffer(0)]],
    device const uint*  QP  [[buffer(1)]],
    device const float* SC  [[buffer(2)]],
    device half*        C   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint n = tgid.y * SGY + tid.y;
    const uint kw = K >> 4;
    float acc = 0.0f;
    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(X[k0 + j]);
        uint packed = QP[n * kw + (kb >> 4) + lane];
        float s = 0.0f;
        for (uint j = 0; j < 16; ++j) {
            uint q = (packed >> (2 * j)) & 0x3;
            s += xr[j] * float(int(q ^ 2u) - 2);
        }
        acc += s;
    }
    float tot = simd_sum(acc);
    if (lane == 0) C[n] = half(tot * SC[n]);
}

kernel void matvec_int4aff_r1(
    device const half* X   [[buffer(0)]],
    device const uint* QP  [[buffer(1)]],
    device const half* SC  [[buffer(2)]],
    device const half* BI  [[buffer(3)]],
    device half*       C   [[buffer(4)]],
    constant uint&     K   [[buffer(5)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint n = tgid.y * SGY + tid.y;
    const uint kw = K >> 3;
    const uint ng = K / G;
    float acc = 0.0f;
    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(X[k0 + j]);
        uint packed = QP[n * kw + (kb >> 3) + lane];
        uint grp = k0 / G;
        float sc = float(SC[n * ng + grp]);
        float bi = float(BI[n * ng + grp]);
        float s = 0.0f;
        for (uint j = 0; j < 8; ++j) {
            uint q = (packed >> (j * 4)) & 0xf;
            s += xr[j] * (sc * float(q) + bi);
        }
        acc += s;
    }
    float tot = simd_sum(acc);
    if (lane == 0) C[n] = half(tot);
}

// ---- norm-prologue (_nx) variants --------------------------------------------------------
// The pre-attention / pre-FFN / final RMSNorms feed EXACTLY ONE matvec group each, and a
// 32-thread rmsnorm dispatch costs a full dependent-dispatch fence (~9-15us) — the raw
// loop's dominant tax. These variants fold the norm into the matvec prologue with
// BIT-IDENTICAL numerics: every SIMD-group redundantly recomputes the rmsnorm_glue
// reduction (same per-lane contiguous ssq order, same simd_sum tree, fp32 inv), then the
// K-loop applies  xr = float( half(float(X[k]) * inv) * NW[k] )  — the exact half-rounded
// value the standalone kernel would have materialized. Compute is free next to the fence.

#define NX_PROLOGUE                                                        \
    float ssq = 0.0f;                                                      \
    {                                                                      \
        const uint nept = K / 32;                                          \
        for (uint i = 0; i < nept; ++i) {                                  \
            float xv = float(X[lane * nept + i]);                          \
            ssq += xv * xv;                                                \
        }                                                                  \
    }                                                                      \
    const float ms  = simd_sum(ssq) / float(K);                            \
    const float inv = metal::precise::rsqrt(ms + 1e-6f);

kernel void matvec_int4aff_nx(
    device const half* X   [[buffer(0)]],
    device const half* NW  [[buffer(1)]],
    device const uint* QP  [[buffer(2)]],
    device const half* SC  [[buffer(3)]],
    device const half* BI  [[buffer(4)]],
    device half*       C   [[buffer(5)]],
    constant uint&     K   [[buffer(6)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;
    const uint ng = K / G;

    NX_PROLOGUE

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) {
            half hx = half(float(X[k0 + j]) * inv) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = pk[r];
            float sc = float(SC[n * ng + grp]);
            float bi = float(BI[n * ng + grp]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r] = half(tot);
    }
}

kernel void matvec_int2sym_nx(
    device const half*  X   [[buffer(0)]],
    device const half*  NW  [[buffer(1)]],
    device const uint*  QP  [[buffer(2)]],
    device const float* SC  [[buffer(3)]],
    device half*        C   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    NX_PROLOGUE

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) {
            half hx = half(float(X[k0 + j]) * inv) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 4) + lane;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint packed = pk[r];
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
            C[n] = half(tot * SC[n]);
        }
    }
}

kernel void gateup_int2sym_nx(
    device const half*  X    [[buffer(0)]],
    device const half*  NW   [[buffer(1)]],
    device const uint*  QPG  [[buffer(2)]],
    device const float* SCG  [[buffer(3)]],
    device const uint*  QPU  [[buffer(4)]],
    device const float* SCU  [[buffer(5)]],
    device half*        C    [[buffer(6)]],
    constant uint&      K    [[buffer(7)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    NX_PROLOGUE

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) {
            half hx = half(float(X[k0 + j]) * inv) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 4) + lane;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = QPG[n * kw + w0];
            uint pu = QPU[n * kw + w0];
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
            float xg = tg * SCG[n];
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n] = half(gel * (tu * SCU[n]));
        }
    }
}

kernel void gateup_int4aff_nx(
    device const half* X    [[buffer(0)]],
    device const half* NW   [[buffer(1)]],
    device const uint* QPG  [[buffer(2)]],
    device const half* SCG  [[buffer(3)]],
    device const half* BIG  [[buffer(4)]],
    device const uint* QPU  [[buffer(5)]],
    device const half* SCU  [[buffer(6)]],
    device const half* BIU  [[buffer(7)]],
    device half*       C    [[buffer(8)]],
    constant uint&     K    [[buffer(9)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;
    const uint ng = K / G;

    NX_PROLOGUE

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) {
            half hx = half(float(X[k0 + j]) * inv) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = QPG[n * kw + w0];
            uint pu = QPU[n * kw + w0];
            float scg = float(SCG[n * ng + grp]), big = float(BIG[n * ng + grp]);
            float scu = float(SCU[n * ng + grp]), biu = float(BIU[n * ng + grp]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                float x = xr[j];
                sgm += x * (scg * float((pg >> (j * 4)) & 0xf) + big);
                sum += x * (scu * float((pu >> (j * 4)) & 0xf) + biu);
            }
            accg[r] += sgm;
            accu[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
            C[base_row + r] = half(gel * tu);
        }
    }
}

// ---- residual-fold (_nxa/_nxaa) variants — the A19 dispatch-tax fusion lane -------------
// A19's per-dispatch cost is ~7-8 us on the GPU timeline (measured: dependent AND
// independent 32-thread chains both ~8 us/dispatch), so the standalone residual-add
// rmsnorm_glue dispatches (post_attn / post_ffw / post_ple, 3x35/token) are pure tax.
// These variants fold the residual add into the NEXT matvec's prologue with BIT-IDENTICAL
// numerics: every SIMD-group recomputes rmsnorm_glue's exact reduction over the delta D
// (same per-lane contiguous ssq order, same simd_sum tree), forms the fp16
//   x' = RES + half(float(D)*inv1) * PW        (the _aa variant appends * half(ls))
// in rmsnorm_glue's op order, stages x' in threadgroup memory (sg 0), then (for the
// normed variants) reduces ssq2 over x' exactly like NX_PROLOGUE. tg 0 / sg 0 also
// writes x' to XOUT — the canonical activation for later residual consumers. K = 1536.
//
// NXA  = fold + second norm (gateup: post_attn fold + pre_ffw norm)
// NXAA = fold with layer scalar + second norm (q/k/v: post_ple fold + pre_attn norm)
// RAW  = fold only, matvec reads x' unnormed (PLE gate: post_ffw fold)

// Only SIMD-group 0 runs the fold (the other groups would burn redundant ALU on a
// weak-ALU part — they instead idle at the barrier, freeing the core for neighbor
// threadgroups' main loops). inv2 is broadcast through threadgroup memory; every
// consumer sees sg0's exact reduction, so the numerics class is unchanged.
#define NXA_FOLD_BODY(XP_EXPR, DO_NORM2)                                   \
    threadgroup half xs[1536];                                             \
    threadgroup float tg_inv2;                                             \
    const uint nept = K / 32;                                              \
    if (sg == 0) {                                                         \
        float ssq1 = 0.0f;                                                 \
        for (uint i = 0; i < nept; ++i) {                                  \
            float dv = float(D[lane * nept + i]);                          \
            ssq1 += dv * dv;                                               \
        }                                                                  \
        const float inv1 =                                                 \
            metal::precise::rsqrt(simd_sum(ssq1) / float(K) + 1e-6f);      \
        float ssq2 = 0.0f;                                                 \
        for (uint i = 0; i < nept; ++i) {                                  \
            const uint k = lane * nept + i;                                \
            half nrm = half(float(D[k]) * inv1) * PW[k];                   \
            half xp = (XP_EXPR);                                           \
            xs[k] = xp;                                                    \
            if (tgid.y == 0) XOUT[k] = xp;                                 \
            if (DO_NORM2) { float xv = float(xp); ssq2 += xv * xv; }       \
        }                                                                  \
        const float s2 = simd_sum(ssq2);                                   \
        if (lane == 0)                                                     \
            tg_inv2 = DO_NORM2                                             \
                ? metal::precise::rsqrt(s2 / float(K) + 1e-6f)             \
                : 1.0f;                                                    \
    }                                                                      \
    threadgroup_barrier(mem_flags::mem_threadgroup);                       \
    const float inv2 = tg_inv2;

kernel void gateup_int2sym_nxa(
    device const half*  D    [[buffer(0)]],
    device const half*  PW   [[buffer(1)]],
    device const half*  RES  [[buffer(2)]],
    device half*        XOUT [[buffer(3)]],
    device const half*  NW   [[buffer(4)]],
    device const uint*  QPG  [[buffer(5)]],
    device const float* SCG  [[buffer(6)]],
    device const uint*  QPU  [[buffer(7)]],
    device const float* SCU  [[buffer(8)]],
    device half*        C    [[buffer(9)]],
    constant uint&      K    [[buffer(10)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    NXA_FOLD_BODY(RES[k] + nrm, true)

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) {
            half hx = half(float(xs[k0 + j]) * inv2) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 4) + lane;
        uint4 pg4 = QP_ROWS4(QPG, base_row, kw, w0);
        uint4 pu4 = QP_ROWS4(QPU, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = pg4[r];
            uint pu = pu4[r];
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
            float xg = tg * SCG[n];
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n] = half(gel * (tu * SCU[n]));
        }
    }
}

kernel void gateup_int4aff_nxa(
    device const half* D    [[buffer(0)]],
    device const half* PW   [[buffer(1)]],
    device const half* RES  [[buffer(2)]],
    device half*       XOUT [[buffer(3)]],
    device const half* NW   [[buffer(4)]],
    device const uint* QPG  [[buffer(5)]],
    device const half* SCG  [[buffer(6)]],
    device const half* BIG  [[buffer(7)]],
    device const uint* QPU  [[buffer(8)]],
    device const half* SCU  [[buffer(9)]],
    device const half* BIU  [[buffer(10)]],
    device half*       C    [[buffer(11)]],
    constant uint&     K    [[buffer(12)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;
    const uint ng = K / G;

    NXA_FOLD_BODY(RES[k] + nrm, true)

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) {
            half hx = half(float(xs[k0 + j]) * inv2) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        uint4 pg4 = QP_ROWS4(QPG, base_row, kw, w0);
        uint4 pu4 = QP_ROWS4(QPU, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint pg = pg4[r];
            uint pu = pu4[r];
            float scg = float(SCG[n * ng + grp]), big = float(BIG[n * ng + grp]);
            float scu = float(SCU[n * ng + grp]), biu = float(BIU[n * ng + grp]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                float x = xr[j];
                sgm += x * (scg * float((pg >> (j * 4)) & 0xf) + big);
                sum += x * (scu * float((pu >> (j * 4)) & 0xf) + biu);
            }
            accg[r] += sgm;
            accu[r] += sum;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
            C[base_row + r] = half(gel * tu);
        }
    }
}

// PLE gate with the post_ffw fold: x' is consumed RAW (no second norm), mode-1
// (round -> gelu -> half -> *MUL) epilogue hardcoded.
kernel void matvec_int8_gate_nxa(
    device const half*  D    [[buffer(0)]],
    device const half*  PW   [[buffer(1)]],
    device const half*  RES  [[buffer(2)]],
    device half*        XOUT [[buffer(3)]],
    device const char*  W8   [[buffer(4)]],
    device const float* SC   [[buffer(5)]],
    device const half*  MUL  [[buffer(6)]],
    device half*        C    [[buffer(7)]],
    constant uint&      K    [[buffer(8)]],
    constant uint&      moff [[buffer(9)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    NXA_FOLD_BODY(RES[k] + nrm, false)

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(xs[k0 + j]);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            // char4 pair loads (1B scalar loads are issue-bound on A19); j order kept
            device const char4* w4 = (device const char4*)(W8 + n * K + k0);
            const char4 wa = w4[0];
            const char4 wb = w4[1];
            float s = 0.0f;
            s += xr[0] * float(int(wa.x));
            s += xr[1] * float(int(wa.y));
            s += xr[2] * float(int(wa.z));
            s += xr[3] * float(int(wa.w));
            s += xr[4] * float(int(wb.x));
            s += xr[5] * float(int(wb.y));
            s += xr[6] * float(int(wb.z));
            s += xr[7] * float(int(wb.w));
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            half y = half(tot * SC[n]);
            float xf = float(y);
            float gel = 0.5f * xf * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xf + 0.044715f * xf * xf * xf)));
            C[n] = half(gel) * MUL[moff + n];
        }
    }
}

// q/k/v matvec with the post_ple fold (layers >= 1): x' = (RES + norm(P)*PW) * ls,
// then the pre_attn norm exactly as _nx. Bind XOUT = the canonical x buffer on the
// q dispatch only (k/v get a scratch buffer so the hazard tracker keeps them parallel).
kernel void matvec_int4aff_nxaa(
    device const half* D    [[buffer(0)]],
    device const half* PW   [[buffer(1)]],
    device const half* RES  [[buffer(2)]],
    device half*       XOUT [[buffer(3)]],
    device const half* NW   [[buffer(4)]],
    device const uint* QP   [[buffer(5)]],
    device const half* SC   [[buffer(6)]],
    device const half* BI   [[buffer(7)]],
    device half*       C    [[buffer(8)]],
    constant uint&     K    [[buffer(9)]],
    constant float&    ls   [[buffer(10)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;
    const uint ng = K / G;
    const half lsh = half(ls);

    NXA_FOLD_BODY((RES[k] + nrm) * lsh, true)

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) {
            half hx = half(float(xs[k0 + j]) * inv2) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = pk[r];
            float sc = float(SC[n * ng + grp]);
            float bi = float(BI[n * ng + grp]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r] = half(tot);
    }
}

// ---- fused q/k/v matvec (write layers): one dispatch over 10*hd rows ---------------------
// Rows [0, 8hd) = q, [8hd, 9hd) = k, [9hd, 10hd) = v. Row-segment boundaries are
// multiples of R=4, so each SIMD-group's 4 rows live in one segment and the weight/output
// selection is uniform per group. Per-row dot order identical to matvec_int4aff_nx(aa).
// _nx_qkv = layer 0 (plain pre_attn norm), _nxaa_qkv = layers >= 1 (post_ple fold first).
kernel void matvec_int4aff_nx_qkv(
    device const half* X    [[buffer(0)]],
    device const half* NW   [[buffer(1)]],
    device const uint* QPQ  [[buffer(2)]],
    device const half* SCQ  [[buffer(3)]],
    device const half* BIQ  [[buffer(4)]],
    device const uint* QPK  [[buffer(5)]],
    device const half* SCK  [[buffer(6)]],
    device const half* BIK  [[buffer(7)]],
    device const uint* QPV  [[buffer(8)]],
    device const half* SCV  [[buffer(9)]],
    device const half* BIV  [[buffer(10)]],
    device half*       CQ   [[buffer(11)]],
    device half*       CK   [[buffer(12)]],
    device half*       CV   [[buffer(13)]],
    constant uint&     K    [[buffer(14)]],
    constant uint&     hd   [[buffer(15)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint grow = (tgid.y * SGY + sg) * R;      // global row (q ++ k ++ v)
    const uint kw = K >> 3;
    const uint ng = K / G;

    device const uint* QP; device const half* SC; device const half* BI; device half* C;
    uint base_row;
    if (grow < 8 * hd)      { QP = QPQ; SC = SCQ; BI = BIQ; C = CQ; base_row = grow; }
    else if (grow < 9 * hd) { QP = QPK; SC = SCK; BI = BIK; C = CK; base_row = grow - 8 * hd; }
    else                    { QP = QPV; SC = SCV; BI = BIV; C = CV; base_row = grow - 9 * hd; }

    NX_PROLOGUE

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) {
            half hx = half(float(X[k0 + j]) * inv) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = pk[r];
            float sc = float(SC[n * ng + grp]);
            float bi = float(BI[n * ng + grp]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r] = half(tot);
    }
}

kernel void matvec_int4aff_nxaa_qkv(
    device const half* D    [[buffer(0)]],
    device const half* PW   [[buffer(1)]],
    device const half* RES  [[buffer(2)]],
    device half*       XOUT [[buffer(3)]],
    device const half* NW   [[buffer(4)]],
    device const uint* QPQ  [[buffer(5)]],
    device const half* SCQ  [[buffer(6)]],
    device const half* BIQ  [[buffer(7)]],
    device const uint* QPK  [[buffer(8)]],
    device const half* SCK  [[buffer(9)]],
    device const half* BIK  [[buffer(10)]],
    device const uint* QPV  [[buffer(11)]],
    device const half* SCV  [[buffer(12)]],
    device const half* BIV  [[buffer(13)]],
    device half*       CQ   [[buffer(14)]],
    device half*       CK   [[buffer(15)]],
    device half*       CV   [[buffer(16)]],
    constant uint&     K    [[buffer(17)]],
    constant uint&     hd   [[buffer(18)]],
    constant float&    ls   [[buffer(19)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint grow = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 3;
    const uint ng = K / G;
    const half lsh = half(ls);

    device const uint* QP; device const half* SC; device const half* BI; device half* C;
    uint base_row;
    if (grow < 8 * hd)      { QP = QPQ; SC = SCQ; BI = BIQ; C = CQ; base_row = grow; }
    else if (grow < 9 * hd) { QP = QPK; SC = SCK; BI = BIK; C = CK; base_row = grow - 8 * hd; }
    else                    { QP = QPV; SC = SCV; BI = BIV; C = CV; base_row = grow - 9 * hd; }

    NXA_FOLD_BODY((RES[k] + nrm) * lsh, true)

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) {
            half hx = half(float(xs[k0 + j]) * inv2) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / G;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            uint packed = pk[r];
            float sc = float(SC[n * ng + grp]);
            float bi = float(BI[n * ng + grp]);
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j) {
                uint q = (packed >> (j * 4)) & 0xf;
                s += xr[j] * (sc * float(q) + bi);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) C[base_row + r] = half(tot);
    }
}

// ---- constant-mem byte-LUT int2 decode variants (A19 ALU relief) -------------------------
// The int2 gateup measured 23.4 GB/s CACHE-HOT on A19 (vs int4 gateup 100+): the 2-bit
// unpack (~6 ops/code x 2 matrices) is ALU-bound there. This LUT replaces the per-code
// shift/xor/sub with one constant-cache byte lookup per 4 codes. Products x*c with
// c in {-2,-1,0,1} are EXACT in fp16 and the fp32 accumulation order (byte asc, code asc
// = code order) is unchanged -> bit-exact vs the arithmetic decode. (The BANNED variants
// were the threadgroup byte-LUT and shl-asr — this is the constant-memory lane from the
// session-1 Next list, previously unmeasured.)
constant half4 LUT2[256] = {
    half4(0.0h, 0.0h, 0.0h, 0.0h),
    half4(1.0h, 0.0h, 0.0h, 0.0h),
    half4(-2.0h, 0.0h, 0.0h, 0.0h),
    half4(-1.0h, 0.0h, 0.0h, 0.0h),
    half4(0.0h, 1.0h, 0.0h, 0.0h),
    half4(1.0h, 1.0h, 0.0h, 0.0h),
    half4(-2.0h, 1.0h, 0.0h, 0.0h),
    half4(-1.0h, 1.0h, 0.0h, 0.0h),
    half4(0.0h, -2.0h, 0.0h, 0.0h),
    half4(1.0h, -2.0h, 0.0h, 0.0h),
    half4(-2.0h, -2.0h, 0.0h, 0.0h),
    half4(-1.0h, -2.0h, 0.0h, 0.0h),
    half4(0.0h, -1.0h, 0.0h, 0.0h),
    half4(1.0h, -1.0h, 0.0h, 0.0h),
    half4(-2.0h, -1.0h, 0.0h, 0.0h),
    half4(-1.0h, -1.0h, 0.0h, 0.0h),
    half4(0.0h, 0.0h, 1.0h, 0.0h),
    half4(1.0h, 0.0h, 1.0h, 0.0h),
    half4(-2.0h, 0.0h, 1.0h, 0.0h),
    half4(-1.0h, 0.0h, 1.0h, 0.0h),
    half4(0.0h, 1.0h, 1.0h, 0.0h),
    half4(1.0h, 1.0h, 1.0h, 0.0h),
    half4(-2.0h, 1.0h, 1.0h, 0.0h),
    half4(-1.0h, 1.0h, 1.0h, 0.0h),
    half4(0.0h, -2.0h, 1.0h, 0.0h),
    half4(1.0h, -2.0h, 1.0h, 0.0h),
    half4(-2.0h, -2.0h, 1.0h, 0.0h),
    half4(-1.0h, -2.0h, 1.0h, 0.0h),
    half4(0.0h, -1.0h, 1.0h, 0.0h),
    half4(1.0h, -1.0h, 1.0h, 0.0h),
    half4(-2.0h, -1.0h, 1.0h, 0.0h),
    half4(-1.0h, -1.0h, 1.0h, 0.0h),
    half4(0.0h, 0.0h, -2.0h, 0.0h),
    half4(1.0h, 0.0h, -2.0h, 0.0h),
    half4(-2.0h, 0.0h, -2.0h, 0.0h),
    half4(-1.0h, 0.0h, -2.0h, 0.0h),
    half4(0.0h, 1.0h, -2.0h, 0.0h),
    half4(1.0h, 1.0h, -2.0h, 0.0h),
    half4(-2.0h, 1.0h, -2.0h, 0.0h),
    half4(-1.0h, 1.0h, -2.0h, 0.0h),
    half4(0.0h, -2.0h, -2.0h, 0.0h),
    half4(1.0h, -2.0h, -2.0h, 0.0h),
    half4(-2.0h, -2.0h, -2.0h, 0.0h),
    half4(-1.0h, -2.0h, -2.0h, 0.0h),
    half4(0.0h, -1.0h, -2.0h, 0.0h),
    half4(1.0h, -1.0h, -2.0h, 0.0h),
    half4(-2.0h, -1.0h, -2.0h, 0.0h),
    half4(-1.0h, -1.0h, -2.0h, 0.0h),
    half4(0.0h, 0.0h, -1.0h, 0.0h),
    half4(1.0h, 0.0h, -1.0h, 0.0h),
    half4(-2.0h, 0.0h, -1.0h, 0.0h),
    half4(-1.0h, 0.0h, -1.0h, 0.0h),
    half4(0.0h, 1.0h, -1.0h, 0.0h),
    half4(1.0h, 1.0h, -1.0h, 0.0h),
    half4(-2.0h, 1.0h, -1.0h, 0.0h),
    half4(-1.0h, 1.0h, -1.0h, 0.0h),
    half4(0.0h, -2.0h, -1.0h, 0.0h),
    half4(1.0h, -2.0h, -1.0h, 0.0h),
    half4(-2.0h, -2.0h, -1.0h, 0.0h),
    half4(-1.0h, -2.0h, -1.0h, 0.0h),
    half4(0.0h, -1.0h, -1.0h, 0.0h),
    half4(1.0h, -1.0h, -1.0h, 0.0h),
    half4(-2.0h, -1.0h, -1.0h, 0.0h),
    half4(-1.0h, -1.0h, -1.0h, 0.0h),
    half4(0.0h, 0.0h, 0.0h, 1.0h),
    half4(1.0h, 0.0h, 0.0h, 1.0h),
    half4(-2.0h, 0.0h, 0.0h, 1.0h),
    half4(-1.0h, 0.0h, 0.0h, 1.0h),
    half4(0.0h, 1.0h, 0.0h, 1.0h),
    half4(1.0h, 1.0h, 0.0h, 1.0h),
    half4(-2.0h, 1.0h, 0.0h, 1.0h),
    half4(-1.0h, 1.0h, 0.0h, 1.0h),
    half4(0.0h, -2.0h, 0.0h, 1.0h),
    half4(1.0h, -2.0h, 0.0h, 1.0h),
    half4(-2.0h, -2.0h, 0.0h, 1.0h),
    half4(-1.0h, -2.0h, 0.0h, 1.0h),
    half4(0.0h, -1.0h, 0.0h, 1.0h),
    half4(1.0h, -1.0h, 0.0h, 1.0h),
    half4(-2.0h, -1.0h, 0.0h, 1.0h),
    half4(-1.0h, -1.0h, 0.0h, 1.0h),
    half4(0.0h, 0.0h, 1.0h, 1.0h),
    half4(1.0h, 0.0h, 1.0h, 1.0h),
    half4(-2.0h, 0.0h, 1.0h, 1.0h),
    half4(-1.0h, 0.0h, 1.0h, 1.0h),
    half4(0.0h, 1.0h, 1.0h, 1.0h),
    half4(1.0h, 1.0h, 1.0h, 1.0h),
    half4(-2.0h, 1.0h, 1.0h, 1.0h),
    half4(-1.0h, 1.0h, 1.0h, 1.0h),
    half4(0.0h, -2.0h, 1.0h, 1.0h),
    half4(1.0h, -2.0h, 1.0h, 1.0h),
    half4(-2.0h, -2.0h, 1.0h, 1.0h),
    half4(-1.0h, -2.0h, 1.0h, 1.0h),
    half4(0.0h, -1.0h, 1.0h, 1.0h),
    half4(1.0h, -1.0h, 1.0h, 1.0h),
    half4(-2.0h, -1.0h, 1.0h, 1.0h),
    half4(-1.0h, -1.0h, 1.0h, 1.0h),
    half4(0.0h, 0.0h, -2.0h, 1.0h),
    half4(1.0h, 0.0h, -2.0h, 1.0h),
    half4(-2.0h, 0.0h, -2.0h, 1.0h),
    half4(-1.0h, 0.0h, -2.0h, 1.0h),
    half4(0.0h, 1.0h, -2.0h, 1.0h),
    half4(1.0h, 1.0h, -2.0h, 1.0h),
    half4(-2.0h, 1.0h, -2.0h, 1.0h),
    half4(-1.0h, 1.0h, -2.0h, 1.0h),
    half4(0.0h, -2.0h, -2.0h, 1.0h),
    half4(1.0h, -2.0h, -2.0h, 1.0h),
    half4(-2.0h, -2.0h, -2.0h, 1.0h),
    half4(-1.0h, -2.0h, -2.0h, 1.0h),
    half4(0.0h, -1.0h, -2.0h, 1.0h),
    half4(1.0h, -1.0h, -2.0h, 1.0h),
    half4(-2.0h, -1.0h, -2.0h, 1.0h),
    half4(-1.0h, -1.0h, -2.0h, 1.0h),
    half4(0.0h, 0.0h, -1.0h, 1.0h),
    half4(1.0h, 0.0h, -1.0h, 1.0h),
    half4(-2.0h, 0.0h, -1.0h, 1.0h),
    half4(-1.0h, 0.0h, -1.0h, 1.0h),
    half4(0.0h, 1.0h, -1.0h, 1.0h),
    half4(1.0h, 1.0h, -1.0h, 1.0h),
    half4(-2.0h, 1.0h, -1.0h, 1.0h),
    half4(-1.0h, 1.0h, -1.0h, 1.0h),
    half4(0.0h, -2.0h, -1.0h, 1.0h),
    half4(1.0h, -2.0h, -1.0h, 1.0h),
    half4(-2.0h, -2.0h, -1.0h, 1.0h),
    half4(-1.0h, -2.0h, -1.0h, 1.0h),
    half4(0.0h, -1.0h, -1.0h, 1.0h),
    half4(1.0h, -1.0h, -1.0h, 1.0h),
    half4(-2.0h, -1.0h, -1.0h, 1.0h),
    half4(-1.0h, -1.0h, -1.0h, 1.0h),
    half4(0.0h, 0.0h, 0.0h, -2.0h),
    half4(1.0h, 0.0h, 0.0h, -2.0h),
    half4(-2.0h, 0.0h, 0.0h, -2.0h),
    half4(-1.0h, 0.0h, 0.0h, -2.0h),
    half4(0.0h, 1.0h, 0.0h, -2.0h),
    half4(1.0h, 1.0h, 0.0h, -2.0h),
    half4(-2.0h, 1.0h, 0.0h, -2.0h),
    half4(-1.0h, 1.0h, 0.0h, -2.0h),
    half4(0.0h, -2.0h, 0.0h, -2.0h),
    half4(1.0h, -2.0h, 0.0h, -2.0h),
    half4(-2.0h, -2.0h, 0.0h, -2.0h),
    half4(-1.0h, -2.0h, 0.0h, -2.0h),
    half4(0.0h, -1.0h, 0.0h, -2.0h),
    half4(1.0h, -1.0h, 0.0h, -2.0h),
    half4(-2.0h, -1.0h, 0.0h, -2.0h),
    half4(-1.0h, -1.0h, 0.0h, -2.0h),
    half4(0.0h, 0.0h, 1.0h, -2.0h),
    half4(1.0h, 0.0h, 1.0h, -2.0h),
    half4(-2.0h, 0.0h, 1.0h, -2.0h),
    half4(-1.0h, 0.0h, 1.0h, -2.0h),
    half4(0.0h, 1.0h, 1.0h, -2.0h),
    half4(1.0h, 1.0h, 1.0h, -2.0h),
    half4(-2.0h, 1.0h, 1.0h, -2.0h),
    half4(-1.0h, 1.0h, 1.0h, -2.0h),
    half4(0.0h, -2.0h, 1.0h, -2.0h),
    half4(1.0h, -2.0h, 1.0h, -2.0h),
    half4(-2.0h, -2.0h, 1.0h, -2.0h),
    half4(-1.0h, -2.0h, 1.0h, -2.0h),
    half4(0.0h, -1.0h, 1.0h, -2.0h),
    half4(1.0h, -1.0h, 1.0h, -2.0h),
    half4(-2.0h, -1.0h, 1.0h, -2.0h),
    half4(-1.0h, -1.0h, 1.0h, -2.0h),
    half4(0.0h, 0.0h, -2.0h, -2.0h),
    half4(1.0h, 0.0h, -2.0h, -2.0h),
    half4(-2.0h, 0.0h, -2.0h, -2.0h),
    half4(-1.0h, 0.0h, -2.0h, -2.0h),
    half4(0.0h, 1.0h, -2.0h, -2.0h),
    half4(1.0h, 1.0h, -2.0h, -2.0h),
    half4(-2.0h, 1.0h, -2.0h, -2.0h),
    half4(-1.0h, 1.0h, -2.0h, -2.0h),
    half4(0.0h, -2.0h, -2.0h, -2.0h),
    half4(1.0h, -2.0h, -2.0h, -2.0h),
    half4(-2.0h, -2.0h, -2.0h, -2.0h),
    half4(-1.0h, -2.0h, -2.0h, -2.0h),
    half4(0.0h, -1.0h, -2.0h, -2.0h),
    half4(1.0h, -1.0h, -2.0h, -2.0h),
    half4(-2.0h, -1.0h, -2.0h, -2.0h),
    half4(-1.0h, -1.0h, -2.0h, -2.0h),
    half4(0.0h, 0.0h, -1.0h, -2.0h),
    half4(1.0h, 0.0h, -1.0h, -2.0h),
    half4(-2.0h, 0.0h, -1.0h, -2.0h),
    half4(-1.0h, 0.0h, -1.0h, -2.0h),
    half4(0.0h, 1.0h, -1.0h, -2.0h),
    half4(1.0h, 1.0h, -1.0h, -2.0h),
    half4(-2.0h, 1.0h, -1.0h, -2.0h),
    half4(-1.0h, 1.0h, -1.0h, -2.0h),
    half4(0.0h, -2.0h, -1.0h, -2.0h),
    half4(1.0h, -2.0h, -1.0h, -2.0h),
    half4(-2.0h, -2.0h, -1.0h, -2.0h),
    half4(-1.0h, -2.0h, -1.0h, -2.0h),
    half4(0.0h, -1.0h, -1.0h, -2.0h),
    half4(1.0h, -1.0h, -1.0h, -2.0h),
    half4(-2.0h, -1.0h, -1.0h, -2.0h),
    half4(-1.0h, -1.0h, -1.0h, -2.0h),
    half4(0.0h, 0.0h, 0.0h, -1.0h),
    half4(1.0h, 0.0h, 0.0h, -1.0h),
    half4(-2.0h, 0.0h, 0.0h, -1.0h),
    half4(-1.0h, 0.0h, 0.0h, -1.0h),
    half4(0.0h, 1.0h, 0.0h, -1.0h),
    half4(1.0h, 1.0h, 0.0h, -1.0h),
    half4(-2.0h, 1.0h, 0.0h, -1.0h),
    half4(-1.0h, 1.0h, 0.0h, -1.0h),
    half4(0.0h, -2.0h, 0.0h, -1.0h),
    half4(1.0h, -2.0h, 0.0h, -1.0h),
    half4(-2.0h, -2.0h, 0.0h, -1.0h),
    half4(-1.0h, -2.0h, 0.0h, -1.0h),
    half4(0.0h, -1.0h, 0.0h, -1.0h),
    half4(1.0h, -1.0h, 0.0h, -1.0h),
    half4(-2.0h, -1.0h, 0.0h, -1.0h),
    half4(-1.0h, -1.0h, 0.0h, -1.0h),
    half4(0.0h, 0.0h, 1.0h, -1.0h),
    half4(1.0h, 0.0h, 1.0h, -1.0h),
    half4(-2.0h, 0.0h, 1.0h, -1.0h),
    half4(-1.0h, 0.0h, 1.0h, -1.0h),
    half4(0.0h, 1.0h, 1.0h, -1.0h),
    half4(1.0h, 1.0h, 1.0h, -1.0h),
    half4(-2.0h, 1.0h, 1.0h, -1.0h),
    half4(-1.0h, 1.0h, 1.0h, -1.0h),
    half4(0.0h, -2.0h, 1.0h, -1.0h),
    half4(1.0h, -2.0h, 1.0h, -1.0h),
    half4(-2.0h, -2.0h, 1.0h, -1.0h),
    half4(-1.0h, -2.0h, 1.0h, -1.0h),
    half4(0.0h, -1.0h, 1.0h, -1.0h),
    half4(1.0h, -1.0h, 1.0h, -1.0h),
    half4(-2.0h, -1.0h, 1.0h, -1.0h),
    half4(-1.0h, -1.0h, 1.0h, -1.0h),
    half4(0.0h, 0.0h, -2.0h, -1.0h),
    half4(1.0h, 0.0h, -2.0h, -1.0h),
    half4(-2.0h, 0.0h, -2.0h, -1.0h),
    half4(-1.0h, 0.0h, -2.0h, -1.0h),
    half4(0.0h, 1.0h, -2.0h, -1.0h),
    half4(1.0h, 1.0h, -2.0h, -1.0h),
    half4(-2.0h, 1.0h, -2.0h, -1.0h),
    half4(-1.0h, 1.0h, -2.0h, -1.0h),
    half4(0.0h, -2.0h, -2.0h, -1.0h),
    half4(1.0h, -2.0h, -2.0h, -1.0h),
    half4(-2.0h, -2.0h, -2.0h, -1.0h),
    half4(-1.0h, -2.0h, -2.0h, -1.0h),
    half4(0.0h, -1.0h, -2.0h, -1.0h),
    half4(1.0h, -1.0h, -2.0h, -1.0h),
    half4(-2.0h, -1.0h, -2.0h, -1.0h),
    half4(-1.0h, -1.0h, -2.0h, -1.0h),
    half4(0.0h, 0.0h, -1.0h, -1.0h),
    half4(1.0h, 0.0h, -1.0h, -1.0h),
    half4(-2.0h, 0.0h, -1.0h, -1.0h),
    half4(-1.0h, 0.0h, -1.0h, -1.0h),
    half4(0.0h, 1.0h, -1.0h, -1.0h),
    half4(1.0h, 1.0h, -1.0h, -1.0h),
    half4(-2.0h, 1.0h, -1.0h, -1.0h),
    half4(-1.0h, 1.0h, -1.0h, -1.0h),
    half4(0.0h, -2.0h, -1.0h, -1.0h),
    half4(1.0h, -2.0h, -1.0h, -1.0h),
    half4(-2.0h, -2.0h, -1.0h, -1.0h),
    half4(-1.0h, -2.0h, -1.0h, -1.0h),
    half4(0.0h, -1.0h, -1.0h, -1.0h),
    half4(1.0h, -1.0h, -1.0h, -1.0h),
    half4(-2.0h, -1.0h, -1.0h, -1.0h),
    half4(-1.0h, -1.0h, -1.0h, -1.0h),
};

kernel void gateup_int2sym_nxa_lut(
    device const half*  D    [[buffer(0)]],
    device const half*  PW   [[buffer(1)]],
    device const half*  RES  [[buffer(2)]],
    device half*        XOUT [[buffer(3)]],
    device const half*  NW   [[buffer(4)]],
    device const uint*  QPG  [[buffer(5)]],
    device const float* SCG  [[buffer(6)]],
    device const uint*  QPU  [[buffer(7)]],
    device const float* SCU  [[buffer(8)]],
    device half*        C    [[buffer(9)]],
    constant uint&      K    [[buffer(10)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    NXA_FOLD_BODY(RES[k] + nrm, true)

    float accg[R], accu[R];
    for (uint r = 0; r < R; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) {
            half hx = half(float(xs[k0 + j]) * inv2) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 4) + lane;
        uint4 pg4 = QP_ROWS4(QPG, base_row, kw, w0);
        uint4 pu4 = QP_ROWS4(QPU, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uchar4 gb = as_type<uchar4>(pg4[r]);
            uchar4 ub = as_type<uchar4>(pu4[r]);
            float sgm = 0.0f, sum = 0.0f;
            for (uint b = 0; b < 4; ++b) {
                half4 gc = LUT2[gb[b]];
                half4 uc = LUT2[ub[b]];
                for (uint j = 0; j < 4; ++j) {
                    float x = xr[4 * b + j];
                    sgm += x * float(gc[j]);
                    sum += x * float(uc[j]);
                }
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
            float xg = tg * SCG[n];
            float gel = 0.5f * xg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (xg + 0.044715f * xg * xg * xg)));
            C[n] = half(gel * (tu * SCU[n]));
        }
    }
}

kernel void matvec_int2sym_lut(
    device const half*  X   [[buffer(0)]],
    device const uint*  QP  [[buffer(1)]],
    device const float* SC  [[buffer(2)]],
    device half*        C   [[buffer(3)]],
    constant uint&      K   [[buffer(4)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) xr[j] = float(X[k0 + j]);
        uint w0 = (kb >> 4) + lane;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uchar4 pb = as_type<uchar4>(pk[r]);
            float s = 0.0f;
            for (uint b = 0; b < 4; ++b) {
                half4 pc = LUT2[pb[b]];
                for (uint j = 0; j < 4; ++j)
                    s += xr[4 * b + j] * float(pc[j]);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            C[n] = half(tot * SC[n]);
        }
    }
}

kernel void matvec_int2sym_nx_lut(
    device const half*  X   [[buffer(0)]],
    device const half*  NW  [[buffer(1)]],
    device const uint*  QP  [[buffer(2)]],
    device const float* SC  [[buffer(3)]],
    device half*        C   [[buffer(4)]],
    constant uint&      K   [[buffer(5)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;
    const uint kw = K >> 4;

    NX_PROLOGUE

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 512) {
        uint k0 = kb + lane * 16;
        float xr[16];
        for (uint j = 0; j < 16; ++j) {
            half hx = half(float(X[k0 + j]) * inv) * NW[k0 + j];
            xr[j] = float(hx);
        }
        uint w0 = (kb >> 4) + lane;
        uint4 pk = QP_ROWS4(QP, base_row, kw, w0);
        for (uint r = 0; r < R; ++r) {
            uchar4 pb = as_type<uchar4>(pk[r]);
            float s = 0.0f;
            for (uint b = 0; b < 4; ++b) {
                half4 pc = LUT2[pb[b]];
                for (uint j = 0; j < 4; ++j)
                    s += xr[4 * b + j] * float(pc[j]);
            }
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            C[n] = half(tot * SC[n]);
        }
    }
}

// ---- INT8 per-row symmetric matvec (PLE gate/proj + per_layer_model_projection) ---------
// mode 0: C[n] = half(tot * sc)                                (plain)
// mode 1: C[n] = half(gelu_tanh(half(tot*sc))) * MUL[moff+n]   (the PLE gate epilogue:
//         round-then-gelu-then-half-mul, matching the reference op order exactly)
kernel void matvec_int8(
    device const half*  X    [[buffer(0)]],
    device const char*  W8   [[buffer(1)]],
    device const float* SC   [[buffer(2)]],
    device const half*  MUL  [[buffer(3)]],
    device half*        C    [[buffer(4)]],
    constant uint&      K    [[buffer(5)]],
    constant uint&      mode [[buffer(6)]],
    constant uint&      moff [[buffer(7)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * SGY + sg) * R;

    float acc[R];
    for (uint r = 0; r < R; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {           // block = 32 lanes * 8 bytes
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = float(X[k0 + j]);
        for (uint r = 0; r < R; ++r) {
            uint n = base_row + r;
            // char4 pair loads (1B scalar loads are issue-bound on A19); j order kept
            device const char4* w4 = (device const char4*)(W8 + n * K + k0);
            const char4 wa = w4[0];
            const char4 wb = w4[1];
            float s = 0.0f;
            s += xr[0] * float(int(wa.x));
            s += xr[1] * float(int(wa.y));
            s += xr[2] * float(int(wa.z));
            s += xr[3] * float(int(wa.w));
            s += xr[4] * float(int(wb.x));
            s += xr[5] * float(int(wb.y));
            s += xr[6] * float(int(wb.z));
            s += xr[7] * float(int(wb.w));
            acc[r] += s;
        }
    }
    for (uint r = 0; r < R; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            if (mode == 0) {
                C[n] = half(tot * SC[n]);
            } else {
                half y = half(tot * SC[n]);
                float xf = float(y);
                float gel = 0.5f * xf * (1.0f + metal::precise::tanh(
                    0.7978845608028654f * (xf + 0.044715f * xf * xf * xf)));
                C[n] = half(gel) * MUL[moff + n];
            }
        }
    }
}
