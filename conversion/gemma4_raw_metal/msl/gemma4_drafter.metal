// Gemma4 E2B mixed-bit RAW-METAL decode loop — MTP drafter kernels (P3).
//
// The drafter (Section-11 EAGLE-class head, 4 tiny layers) rides the SAME kernels as
// the main loop (qknorm_rope, rmsnorm_glue, flash_sdpa_decode_occ with H=4 cross-
// attending the main L13/L14 caches, matvec_int4aff, embed_gather, argmax) plus the
// two below, which fold its static int8 activation fake-quant (the QAT operating
// point — 18 calibrated scales; dropping them measurably drops draft acceptance):
//
//   act_q(x, s) = clamp(round(x / s), -128, 127) * s      (round = nearest-even)
//
// Drafter numerics are alpha-preserving, not bit-critical: drafts only steer WHICH
// positions verify computes; the (bit-exact) verify decides every emitted token.

#include <metal_stdlib>
using namespace metal;

constant uint DR   = 4;    // rows per SIMD-group
constant uint DSGY = 8;    // SIMD-groups per threadgroup
constant uint DG   = 64;   // INT4 affine group size

inline float act_q(float x, float s) {
    return clamp(metal::rint(x / s), -128.0f, 127.0f) * s;
}

// ---- INT8 per-row symmetric matvec with act-quant on the input --------------------------
// (pre_proj K=3072, q_proj, o_proj, post_proj). Dispatch: threads = [32, N/4], tg [32, 8].
kernel void matvec_int8_aq(
    device const half*  X    [[buffer(0)]],
    device const char*  W8   [[buffer(1)]],
    device const float* SC   [[buffer(2)]],
    device half*        C    [[buffer(3)]],
    constant uint&      K    [[buffer(4)]],
    constant float&     aqs  [[buffer(5)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * DSGY + sg) * DR;

    float acc[DR];
    for (uint r = 0; r < DR; ++r) acc[r] = 0.0f;

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = act_q(float(X[k0 + j]), aqs);
        for (uint r = 0; r < DR; ++r) {
            uint n = base_row + r;
            float s = 0.0f;
            for (uint j = 0; j < 8; ++j)
                s += xr[j] * float(int(W8[n * K + k0 + j]));
            acc[r] += s;
        }
    }
    for (uint r = 0; r < DR; ++r) {
        float tot = simd_sum(acc[r]);
        if (lane == 0) {
            uint n = base_row + r;
            C[n] = half(tot * SC[n]);
        }
    }
}

// ---- INT4 fused gate+up+gelu+mul with act-quant in (aq_gating) and out (aq_down) ---------
// C[n] = act_q( gelu_tanh(gate . act_q(x)) * (up . act_q(x)), aq_out )
kernel void gateup_int4aff_aq(
    device const half* X    [[buffer(0)]],
    device const uint* QPG  [[buffer(1)]],
    device const half* SCG  [[buffer(2)]],
    device const half* BIG  [[buffer(3)]],
    device const uint* QPU  [[buffer(4)]],
    device const half* SCU  [[buffer(5)]],
    device const half* BIU  [[buffer(6)]],
    device half*       C    [[buffer(7)]],
    constant uint&     K    [[buffer(8)]],
    constant float&    aq_in  [[buffer(9)]],
    constant float&    aq_out [[buffer(10)]],
    uint2 tid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]])
{
    const uint lane = tid.x;
    const uint sg = tid.y;
    const uint base_row = (tgid.y * DSGY + sg) * DR;
    const uint kw = K >> 3;
    const uint ng = K / DG;

    float accg[DR], accu[DR];
    for (uint r = 0; r < DR; ++r) { accg[r] = 0.0f; accu[r] = 0.0f; }

    for (uint kb = 0; kb < K; kb += 256) {
        uint k0 = kb + lane * 8;
        float xr[8];
        for (uint j = 0; j < 8; ++j) xr[j] = act_q(float(X[k0 + j]), aq_in);
        uint w0 = (kb >> 3) + lane;
        uint grp = k0 / DG;
        for (uint r = 0; r < DR; ++r) {
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
    for (uint r = 0; r < DR; ++r) {
        float tg = simd_sum(accg[r]);
        float tu = simd_sum(accu[r]);
        if (lane == 0) {
            float gel = 0.5f * tg * (1.0f + metal::precise::tanh(
                0.7978845608028654f * (tg + 0.044715f * tg * tg * tg)));
            C[base_row + r] = half(act_q(gel * tu, aq_out));
        }
    }
}
