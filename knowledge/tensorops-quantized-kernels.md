# TensorOps: quantized matmul, cooperative tensors, and the neural accelerator (WWDC26 330)

> Foundation note for the **GPU-now** track — the hand-tuned-kernel layer UNDER
> [`custom-metal-kernels.md`](custom-metal-kernels.md). `TorchMetalKernel` gets your MSL into the
> `.aimodel`; **TensorOps is what that MSL should be written WITH** when it does matmul-shaped work.
> Source: WWDC26 session 330 "Optimize custom ML operations with Metal tensors" (verbatim transcript:
> `ondevice/_wwdc330_transcript.txt`, timestamps below), Apple session page, Metal Performance
> Primitives docs. Nothing here is device-measured by this project yet — marked ⚗️ where untested.

## Why this matters here

The stack, per the talk [00:28]: Core AI / MLX → MPS → **Metal Performance Primitives + TensorOps**
(an MSL API for tensor ops — matmul, convolution — inside your kernel). Two facts make it relevant
to this zoo:

1. **TensorOps auto-uses hardware acceleration across GPU generations** [01:29] — including the
   **neural accelerator, a new HW block in each shader core of the M5 family** [01:47], which Apple's
   session page extends to **A19** GPUs. It targets "dense compute-bound work **such as the prefill
   stage of an LLM**" [01:58]. A hand-rolled scalar/simdgroup MSL matmul gets NONE of this; the same
   kernel written on TensorOps does, with no per-generation code.
2. **TensorOps natively eats quantized tensors** — you pass int4/int8/fp8 data (+ scales) straight
   into `matmul2d` and "TensorOps will handle dequantization for you" [07:51], on the HW path.

This project's shipped custom kernels (the gemma4 attn-qo int8 kernels, the int8 dequant-LUT matvec)
are hand-rolled MSL — correct, measured, but blind to the neural accelerator. That's fine for
**decode** (memory-bandwidth-bound; int8 already reads at int8 memory). It is NOT fine for
**prefill** (compute-bound) on M5/A19-class hardware, where TensorOps is the only sanctioned way to
reach the new silicon. ⚗️ Expected shape of the win: prefill/TTFT, not decode tok/s.

## Quantized dtype support matrix (OS-gated — check before shipping)

| dtype | TensorOps support | OS floor |
|---|---|---|
| fp16 / fp32 | always | 26 |
| **int4, int8** | native (data type on the tensor) | **26 (point update)** [03:24] |
| **fp4, fp8, int2** | native | **27** [03:31] |
| **MX scaling formats / FP8 E8M0 block scale factors** | native scale plane | **27** [03:37, 04:21] |
| coop tensor **directly as matmul input** | `get_left_input_cooperative_tensor` | **27** [12:20] (26 = store/reload via threadgroup memory) |

⚠️ The sub-byte dtypes carry **extra alignment requirements** vs the larger types [09:20] — check the
Metal docs per dtype before assuming a layout. ⚠️ Zoo models that must run on OS 26 cannot use the
27-only column; gate kernels accordingly.

> **📅 OS27 runtime now on-device (2026-07-01).** macOS 27 dev **beta 2** (`26A5368g`, 2026-06-22)
> is shipping and installed on this machine (`Darwin 27.0.0`). The **27** rows above move from
> "announced at WWDC26 330" to **runtime present** — the fp4/fp8/int2, MX·E8M0 scale-plane, and
> coop-tensor-as-matmul-input paths can now be **RUN** on device, not just compiled (compile was
> already confirmed 2026-07-01 with Metal Toolchain v27.1 — see §"The real `matmul2d` API"). ⇒ the
> A19/M5 A/B for the neural-accelerator **prefill lever** and the **fp4 `matmul2d` dequant** is
> **unblocked** — bench via `ondevice/PipelinedBench`. ⚠️ **Runtime / AOT-compile only on 27**; keep
> the *convert* box on **macOS 26.4** (the `coreai-core` wheel mis-converts on 27). This is the OS
> half of Stream D's FP4-runtime gate — now satisfied; its remaining blocker is the kernel work here.

## Scale planes: one MTLTensor = data + scales (OS 27)

In OS 27 a single `MTLTensor` carries the quantized data plane **plus an auxiliary scale plane**
[04:23]: FP8 E8M0 block-wise scales, `blockFactors` defining the block (e.g. **32×1 → 32 data
elements share one scale** [06:17]). Host side: scale-plane descriptor (`dataType`, `blockFactors`)
→ auxiliary-plane map (kind = scales) → attach to the main `tensorDescriptor` →
`newTensorWithDescriptor` [04:44–05:12]. Kernel side: declare the scales-plane type + the full
tensor type (dtype + scales plane), bind to a buffer binding point — or construct a `tensor_inline`
on the shader stack from raw pointers + metadata when you don't want a host-side MTLTensor [06:43].

`slice()` on such a tensor slices **data and scale planes simultaneously**, respecting the block
size [07:26] — so threadgroup tiling code doesn't change.

## Quantized matmul2d — and the custom-format escape hatch

The quantized path is the SAME `matmul2d_descriptor` / op setup as fp16 [07:32]: descriptor (tile
sizes), op (simdgroups per threadgroup), `run`. Feed quantized tensors directly; dequant happens
inside, HW-accelerated [07:59].

If your format is NOT one TensorOps understands (e.g. a palettized LUT codebook — this zoo's
int8 LUT matvec): dequantize yourself, but **into a cooperative tensor** (storage distributed
across the thread-private registers of the participating threads [08:50]) and pass THAT as the
matmul input — instead of staging through threadgroup memory [08:30–09:05]. Saves the extra
load/store round trip the naive approach pays.

## FlashAttention on TensorOps (the recipe, [09:31–13:25])

The talk builds fused attention (QK^T → softmax → ×V, one kernel) from four primitives:

1. **`execution_simdgroup` operation scope** — each simdgroup runs an independent matmul and owns
   **complete rows** of the intermediate matrix, so softmax needs no cross-simdgroup exchange
   [10:12–10:39]; slice input tiles by simdgroup ID.
2. **Cooperative tensor** holds the intermediate matrix — never written to device memory [10:43].
3. **`reduce_rows`** computes the per-row max (reduction op = max, init = −INFINITY) into a second,
   smaller cooperative tensor [11:00–11:33]; **`map_iterator`** maps each 2D element's iterator to
   its row-reduction element [11:41–12:02]; dereference both to compute softmax in place.
4. Second matmul ×V takes the cooperative tensor **directly as left input** (OS 27,
   `get_left_input_cooperative_tensor`) — but layouts vary by dtype/etc., so call
   **`is_compatible_as_left_input` / `..._right_input` first**; `false` ⇒ you must store/reload
   via threadgroup memory after all [12:47–13:16]. `op.run` is identical either way.

**Core AI integration is the already-documented path** — the talk wires this exact kernel into a
Sam3 segmentation model via `TorchMetalKernel` (MSL body as a Python string, swap the HF attention
impl, export) [13:35–14:33]. See [`custom-metal-kernels.md`](custom-metal-kernels.md); nothing new
is needed to ship a TensorOps kernel inside a `.aimodel`.

## What this changes for this zoo (assessment, ⚗️ until measured)

- **Calibration first** — Apple's own [MLX-on-M5 LLM measurements](https://machinelearning.apple.com/research/exploring-llms-mlx-m5)
  (same accelerator class as A19): **prefill/TTFT 3.33–4.06× vs M4; decode 1.19–1.27×**, i.e. almost
  exactly the 120→153 GB/s bandwidth delta alone. The accelerator moves compute-bound prefill and
  contributes ~nothing to BW-bound decode. Set expectations accordingly for any work below.
- **Prefill lever on M5/A19**: a TensorOps rewrite of the big matmuls (or a TensorOps FlashAttention
  for the prefill graph) is the only route to the neural accelerator. Today's static q16 int8
  prefill (147 tok/s on iPhone 17 Pro) is strong; the question is what A19's accelerator adds on
  top — and whether the prefill is compute-bound at all at small chunk sizes (this project measured
  chunk-32 == chunk-16 per-token cost, pointing at unroll/SDPA/KV-fill, not matmul compute). Probe
  the bottleneck before writing kernels. Decode: don't bother — BW-bound, int8lin already near ½
  device bandwidth.
- **int4 re-test**: this project's int4km NO-GO was a numerics failure of OUR hand-rolled dequant
  path. Native int4 tensors + E8M0 block scales with TensorOps' own dequant is a DIFFERENT numerics
  path — one conversion + PSNR gate on the 27 beta would settle it.
- **fp8/fp4 as new design points**: between int8 (ships, clean) and int4 (failed), OS 27 adds fp8
  (int8 memory, more dynamic range) and fp4 (int4 memory, likely friendlier numerics than int4
  uniform). Pairs with the `coreai-opt` side: FP4_E2M1 / FP8_E4M3FN/E5M2 already exist in the
  compression API ([`compression-reference.md`](compression-reference.md)) — the Metal-side support
  closes the loop.
- **Keep preferring native SDPA for decode attention** ([`custom-metal-kernels.md`](custom-metal-kernels.md)
  rule). The FlashAttention recipe is for cases SDPA doesn't cover: quantized-KV attention, fused
  prefill attention, or shapes where the packed-state RMW tax killed the stateful-kernel monolith.
- **Start from the TensorOps sample code** [15:53] + MPP programming guide — captions omit the
  on-screen code, so exact signatures come from there, not from the transcript.

## The real `matmul2d` API (extracted from the SDK headers, 2026-07-01)

The signatures the transcript omits are **shipped in the Metal Toolchain 27 / iPhoneOS 27 SDK**, not
just in a sample download. Authoritative source on this machine:
`…/iPhoneOS27.0.sdk/System/Library/Frameworks/MetalPerformancePrimitives.framework/Headers/`
→ `MPPTensorOpsMatMul2d.h` (+ `MPPTensorOpsConvolution2d.h` for encoder Conv, + `__impl/*Impl.h`).
Header guard: `#if defined(__METAL_VERSION__) && defined(__HAVE_TENSOR__)`, inside
`#pragma METAL internals : enable`, namespace `mpp::tensor_ops`. Verbatim API:

```cpp
// descriptor: output tile m×n, k=dynamic_extent => read K from the tensor; transpose flags pick NN/NT/TN/TT
constexpr auto desc = matmul2d_descriptor(64, 32, /*k=*/static_cast<int>(dynamic_extent),
                                          /*transpose_left=*/false, /*transpose_right=*/false,
                                          /*relaxed_precision=*/false);   // mode: multiply | multiply_accumulate
matmul2d<desc, execution_simdgroups<4>> op;                 // 4 SIMD-groups cooperate per threadgroup
auto mA = A.slice(0, tgid.y*64);                            // tensor<device half, dextents<int32_t,2>>
auto mB = B.slice(tgid.x*32, 0);
auto mC = C.slice(tgid.x*32, tgid.y*64);                    // C must be zero-initialized (computes C = A*B + C)
op.run(mA, mB, mC);                                         // static_slice<…> on inside tiles skips bounds checks
// dispatch: threadgroups=((M+63)/64,(N+31)/32,1), threadsPerTG=(threadExecutionWidth*4,1,1)
```

**The dtype table is the headline for this zoo** (left=activations, right=weights, dest=accum). Native
in-kernel dequant — pass the quantized tensor straight in as the right operand:
- `half × int8_t → half|float`, `half × int4b_format → half|float`  ⇐ **LLaDA / FLUX / any int4-or-int8
  weight × fp16 activation is a single matmul2d** (the zoo's whole quantized-prefill class).
- `half × metal_fp4_e2m1_format → half|float`, `half × metal_fp8_e4m3/e5m2 → half|float` (OS27) — the
  FP4/FP8 path Stream D needs, same op, just a different right-operand dtype.
- `int8 × int4b_format → int32`, `int2b_format`, all-fp4/all-fp8 — full sub-byte matrix present.

**Toolchain compile facts (validated on this machine 2026-07-01, Metal Toolchain v27.1.5194.15,
`metalfe-32023.917`, target air64-darwin27)** — `xcrun -sdk iphoneos metal -std=… -c` on a kernel that
includes the umbrella header and runs `matmul2d`:
- **`__HAVE_TENSOR__` (tensors at all) requires `-std=metal4.0`** — OFF at metal3.2. So any TensorOps
  kernel must be compiled at metal4.0+.
- **`matmul2d` + single-plane quantized operands compile at metal4.0**: both `half × half → float` and
  **`half × int4b_format → half`** (LLaDA's int4 weight × fp16 activation) produced AIR cleanly. Kernel
  shape = the header example verbatim (descriptor `matmul2d_descriptor(64,32,dynamic_extent)`,
  `matmul2d<desc, execution_simdgroups<4>>`, `A.slice(...)`/`B.slice(...)`/`C.slice(...)`, `op.run`).
  `int4b_format` = `metal::int4b_format` (sub-byte std type); `using namespace mpp::tensor_ops;`.
- **Block-wise scales (the per-block-32 int4 LLaDA actually ships) need `-std=metal4.1`**:
  `__HAVE_TENSOR_MULTIPLANE__` is **OFF at 4.0, ON at 4.1**. The scale-carrying MSL type is
  `metal::tensor_blockwise<PlaneTag, ElementType, BlockSizes...>` (MPPTensorOpsTraits.h, multiplane
  guard). So: bare `int4b_format` (uniform, no per-block scale) = 4.0; **blockwise-int4 / MX·E8M0 block
  scales = 4.1 + `tensor_blockwise`.** LLaDA = the 4.1 path.

**Open integration question to prototype on Mac (before the A19 A/B)**: the op consumes Metal
`tensor<device T, dextents<int32_t,2>>` (or `tensor_blockwise`) operands, but `TorchMetalKernel`
auto-generates a *plain buffer-pointer* signature. So the MSL body must build the operands from the raw
pointers — the `tensor_inline`-from-raw-pointer+metadata path the §330 talk mentions — inside
`helper_src`/`src`, with the MPP includes. **AND** coreai-torch must compile that embedded MSL at
**metal4.1** (else the blockwise scale plane won't be available) — confirm the embedded-kernel std
version is settable. That wrapper (raw `device half*` + int4 blocks + fp16 scales + shape → tensors →
`matmul2d`) is the reusable scaffolding to write and numerically-validate on M4 (correctness is
HW-independent; the neural-accel *speedup* only appears on M5/A19), then graft into the LLaDA forward
and A/B on device.

## Kernel-phase findings — SDK header re-read (2026-07-01, Session B)

Two facts read straight from the beta SDK headers on this machine that **change the integration plan**
(source: `…/MacOSX.sdk/…/MetalPerformancePrimitives.framework/Headers/`, verified against the installed
`coreai` 1.0.0b1 / `coreai_torch` 0.4.0 authoring code):

1. **`TorchMetalKernel` does NOT emit a raw buffer-pointer signature anymore — it emits Metal
   `tensor<...>` operands directly.** `coreai/authoring/metal.py:198-213` (`template_kernel_src`)
   generates, for every rank-≥1 IO, `tensor<device <dtype>, metal::dextents<int, N>, tensor_handle>
   <name> [[buffer(i)]]`, and the auto-prepended header is already
   `#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>` + `using namespace metal;` +
   `using namespace mpp::tensor_ops;` (metal.py:234-237). ⇒ **fp16 / int8 / uint8 operands feed
   `matmul2d` straight via `.slice(...)` — no `tensor_inline`-from-pointer needed for them.** The ONLY
   operand that still needs hand-construction is **int4**: `int4b_format` is NOT in coreai's
   `metal_type_mappings` (only bf16/f16/f32/si8/ui8/ui32/si32/i1), so int4 weights must arrive as a
   **uint8 buffer** (packed 2×int4/byte) and be **reinterpreted to `int4b_format` inside the kernel**.
2. **⛔ The native block-scale plane is E8M0-ONLY — it will NOT accept LLaDA's fp16 block-32 scales.**
   `MPPTensorOpsMatMul2dImpl.h:6247,6284` hard `static_assert`:
   `is_same_v<scaleType, metal::metal_fp8_ue8m0_format>` ("Scale data type must be
   metal_fp8_ue8m0_format"). Plus: block-0 size **must == 32**, block-1 size **must == 1** (6256/6262),
   and a **right operand carrying scales MUST be transposed** (`transpose_right==true`, 6302; left must
   NOT, 6265). So the "pass int4 + scales, TensorOps dequants for you" path only takes **power-of-2
   (E8M0) per-block-32 scales**. LLaDA ships **arbitrary fp16** per-block scales → the native
   `tensor_blockwise` path can't consume them losslessly (there's even a `// TODO: need to update the
   helper method` next to the assert, so Apple may widen this later — recheck each beta).

**⇒ Revised kernel strategy (preserves LLaDA numerics, needs only metal4.0):** do a **uniform
`half × int4b_format → half` `matmul2d` per K-block-of-32, and apply the fp16 per-block scale manually**
between blocks (accumulate `C[m,n] += scale[b,n] · Σ_{k∈block b} A[m,k]·Wq[k,n]`). This uses arbitrary
fp16 scales exactly, needs **only metal4.0 (uniform int4, no multiplane / no `tensor_blockwise`)**, and
sidesteps both the E8M0 restriction and the transpose-right requirement. The native E8M0 `tensor_blockwise`
path (metal4.1) is a **separate** de-risk — only worth it if a re-quant of LLaDA to power-of-2 block
scales passes the PSNR gate. Incremental M4 validation ladder: **M0** `half×half→half` matmul2d (proves
coreai embeds+compiles+runs matmul2d and the `tensor_handle` signature feeds it) → **M1** uniform
`half×int4b_format→half` (proves the uint8→int4 reinterpret) → **M2** manual block-32 fp16 scaling vs the
torch dequant reference.

## ✅ VALIDATED ON M4 MAX (macOS 27.0, 2026-07-01) — matmul2d runs inside a coreai .aimodel

All three integration unknowns the plan flagged are **RETIRED**. Prototypes (re-runnable):
`coreai-models-community/knowledge/_tensorops_proto/{m0_half_x_half, m1a_half_x_int8,
m1b_half_x_int4_uniform, probe_dispatch}.py` (run with `coreai-models/.venv/bin/python`).

**Results (cos-sim vs a torch reference, all shapes M/K/N that are multiples of 64/·/32):**
| kernel | operands | cos-sim | note |
|---|---|---|---|
| M0 | `half × half → half` | **1.000000** | matmul2d compiles+runs in coreai; `tensor_handle` sig feeds it directly |
| M1a | `half × int8_t → half` | **1.000000** | quantized path, zero reinterpret (int8_t is a native coreai dtype) |
| M1b | `half × int4b_format → half` (uniform) | **1.000000** | the uint8→int4 reinterpret works |

**The facts that make it work (all empirically pinned, not guessed):**
1. **coreai's runtime compiles the embedded `metal4_kernel` MSL at ≥ metal4.0** — `matmul2d`
   (guarded by `__HAVE_TENSOR__` = 4.0) compiled and RAN from inside a `.aimodel` on the Mac GPU. No
   std knob needed for the uniform-int4 path. (metal4.1/`tensor_blockwise` only needed for the E8M0
   native-scale path, which we don't use.)
2. **The auto-generated signature is `tensor<device T, metal::dextents<int,N>, tensor_handle>`** (not
   raw pointers): fp16/int8 operands feed `matmul2d` straight via `.slice(...)`.
3. **⚠️ Metal tensor coords are TRANSPOSED vs the numpy/torch buffer.** A torch `[M,K]` tensor becomes a
   Metal tensor with **extent(0)=K (inner/contiguous), extent(1)=M**. Verified with `probe_dispatch.py`:
   a thread writing `out[a,b]` lands at numpy `[b,a]`. ⇒ **Apple's `MPPTensorOpsMatMul2d.h` header
   example slicing is correct verbatim** (`mA=A.slice(0,tgid.y*64)`, `mB=B.slice(tgid.x*32,0)`,
   `mC=C.slice(tgid.x*32,tgid.y*64)`); a "natural row-major" slice is WRONG (gives cos≈0.2, only the
   origin tile correct). The working kernel body is exactly:
   ```cpp
   constexpr auto desc = matmul2d_descriptor(64,32,static_cast<int>(metal::dynamic_extent),false,false,false);
   matmul2d<desc, execution_simdgroups<4>> op;
   auto mA = A.slice(0, tgid.y*64);           // A = tensor[K,M]; dim1=M tiled by 64
   auto mB = B.slice(tgid.x*32, 0);           // B = tensor[N,K]; dim0=N tiled by 32
   auto mC = C.slice(tgid.x*32, tgid.y*64);   // C = tensor[N,M]
   op.run(mA, mB, mC);                        // default mode=multiply (no C-zeroing needed)
   ```
4. **Dispatch = `dispatchThreads` (TOTAL threads, non-uniform).** `threads_per_grid` is total threads,
   `threads_per_thread_group` is per-group; threadgroups = grid/group per dim. For the 4-SIMD matmul use
   `threads_per_thread_group=(128,1,1)` (=4 simdgroups × 32) and
   `threads_per_grid=(128·ceil(N/32), ceil(M/64), 1)` → tgid.x tiles N (step 32), tgid.y tiles M (step 64).
   `MetalParameter("tgid","uint2","threadgroup_position_in_grid")`.
5. **int4 reinterpret (unknown (b)) — the working recipe:** pass packed int4 as a **uint8** tensor `Wp`
   (2 int4/byte). Inside the body: `device uchar* wptr = &Wp[0,0];` (**taking `&Wp[0,0]` DOES compile and
   yields a usable base `device uchar*`**), then
   `tensor<device metal::int4b_format, metal::dextents<int,2>, tensor_inline> Wi(wptr, metal::dextents<int,2>(N,K));`
   and slice `Wi` as the right operand. `int4b_format`'s `data_handle_type` is `device uchar*` so no cast
   is needed. **Packing convention that matched exactly (cos=1.0):** C-order flatten of the `[K,N]` int4
   weight (n fastest → linear = k·N+n), 2 per byte, **low nibble = even linear index**
   (`byte = (v[2i]&0xF) | ((v[2i+1]&0xF)<<4)`, values two's-complement 4-bit). Signed decode is automatic.
6. **Bake `N,K` as MSL literals** (f-string the body) or read from tensor extents — do NOT pass them as
   runtime scalar inputs (awkward). And compute the dispatch grid from `wp.shape`/`a.shape` symints, not
   from tensor-valued inputs (that breaks `torch.export`).

**✅ M2 = block-32 fp16 scaling — VALIDATED (cos-sim 1.000000 on real LLaDA shapes).** Recipe that
worked (`_tensorops_proto/m2_int4_block32_scaled.py`): loop K in blocks of 32; per block, **dequant the
[k=32, n=32] weight tile into a `threadgroup half[32*32]` buffer, each element multiplied by its fp16
block scale** (128 threads, 8 elems each, then `threadgroup_barrier`), then
`matmul2d` **half(device A) × half(threadgroup Wsh) → float(device C)** with
`mode::multiply` on block 0 and `mode::multiply_accumulate` after, a second barrier before reuse. This
applies **arbitrary fp16 scales exactly** (no E8M0), uses the accelerator for the half×half matmul, and
keeps int4 in device memory (expanded only transiently in threadgroup). Validated on the attention
o-proj shape `M64×K4096×N4096` and the FFN up-proj `M128×K4096×N12288` — both **cos-sim = 1.000000**.
`matmul2d_descriptor(64,32,32,false,false,false, matmul2d_descriptor::mode::multiply[_accumulate])`;
right operand `tensor<threadgroup half, dextents<int,2>, tensor_inline> Wsh(ptr, dextents(32,32))` with
layout `wsh[k*32+n]` (n fastest) so `Wsh[n,k]` matches. (The A19 *speed* A/B still doesn't NEED M2 —
matmul2d throughput is scale-independent, so uniform-int4 M1b measures the accelerator lever; M2 is for
correct output.)

### Two gotchas that cost real debugging time (both now fixed in the protos)
- **⚠️ Reading an `int4b_format` tensor element-wise (`(int)Wi[n,k]`) returns the UNSIGNED nibble
  `[0,15]`, NOT the signed value.** `matmul2d` itself decodes signed int4 correctly (M1b cos=1.0), but a
  manual element read does not sign-extend. Fix: read the packed **bytes** yourself and sign-extend
  (`int c = (lin&1)?(byte>>4):(byte&0xF); if (c>7) c-=16;`) — M2 does this via the raw `device uchar*`,
  which also sidesteps the next gotcha.
- **⚠️ The packed-int4 path only works when the packed inner dim `N` is a MULTIPLE OF 128.** A
  `tensor<device int4b_format, ...., tensor_inline>` built with implied packed strides (and, it appears,
  coreai's own uint8 weight buffer) pads the row stride to a 64-byte / 128-int4 boundary, so a dense
  packing mismatches for N ∉ 128ℤ. Empirically: M1b/M2 give **cos=1.0 at N∈{128,256,384,512}** and
  **garbage (cos≈0) at N∈{64,96,160,192,224,352}**. **Every LLaDA-8B matmul output dim is a multiple of
  128** (d_model 4096, mlp_hidden 12288, head_dim 128, vocab 126464) so this is a non-issue for LLaDA;
  for a general kernel, pad N up to 128 (tile + mask the tail) or supply explicit strides where the
  element type allows it. `K` (contraction) only needs multiple-of-32 (block size); `M` tested at
  multiples of 64.

### ⛔ A19 DEVICE A/B — SPEED RESULT IS NEGATIVE (2026-07-01, the lever does not pay off)
Measured on iPhone 17 Pro (A19, OS27) via a `PB_MM` PipelinedBench mode (added this session): a single
`A[M,K]@B[K,N]->C[M,N]` with **B a resident baked weight** (only A streams, matching LLaDA), custom
`matmul2d` kernel vs coreai's **default matmul (MPSGraph)** at the same shape. FFN shape K=4096, N=12288.

| kernel | M | warm_med (ms) | TFLOP/s med | warm_min | note |
|---|---|---|---|---|---|
| **ref (MPSGraph default)** | 128 | **2.51** | **5.13** | 2.19 | scales up with M |
| ref (MPSGraph default) | 256 | **4.10** | **6.29** | 3.90 | |
| matmul2d 64×32×4 (naive) | 128 | 4.18 | 3.08 | 3.91 | flat ~3 TFLOP/s |
| matmul2d 64×32×4 (naive) | 256 | 8.64 | 2.98 | 8.24 | **2.1× slower than ref** |
| matmul2d `relaxed_precision=true` | 128 | 4.16 | 3.10 | — | no effect (fp32-accum not the cap) |
| matmul2d 128×64 tile, 8 simdgroups | 128 | 4.49 | 2.87 | **2.39 / 5.40** | peak ≈ ref, but high variance |

**Conclusion:** on A19, **coreai's default MPSGraph matmul already runs at ~5–6.3 TFLOP/s** at LLaDA
prefill shapes and scales with M. A custom `matmul2d` kernel is **at best comparable** (a bigger tile hits
a 5.4 TFLOP/s *peak*, matching ref, but with poor consistency) and **at worst ~2× slower** (naive tile,
flat ~3 TFLOP/s; `relaxed_precision` doesn't help). **Apple's "neural-accelerator → 3–4× prefill" (an
*M5* claim) does NOT materialize on A19 for EITHER path at these shapes** — ~6 TFLOP/s looks like the
practical matmul ceiling the default already reaches. ⇒ **Grafting `matmul2d` into LLaDA would not speed
it up — likely a regression**, since LLaDA's *existing* int4 matmul path already measured **6.7 TFLOP/s**
(faster than this kernel's median). **Stream B "custom matmul2d for LLaDA prefill" = negative ROI, PAUSE.**

Honest caveats: (1) the kernel isn't exhaustively tuned — a large tuning effort *might* eke out a small
win, but the default is already competitive so ROI is low; (2) tested half×half — the int4 kernel is
correctness-validated (M1b/M2) but not device-speed-tested, and it would share the same tiling ceiling
while LLaDA's existing int4 path is already 6.7, so int4 matmul2d would also lose; (3) custom kernels
still have value for **fusion** (collapsing several ops + their dispatch/memory traffic into one kernel) —
a different lever than "replace one matmul"; (4) this is **A19**; **M5** desktop silicon may show the
accelerator gain Apple advertised — untested here. Harness: `ondevice/_mm_device_probe.sh` + `PB_MM` mode
in `PipelinedBench`; exporter `scratchpad/export_mm_device.py` (B-resident, tile/simdgroup/relaxed knobs).

### On fp8 / fp4 matmul2d (Stream D) — NOT tested, NOT "failed", but read this
This A/B was **fp16×fp16 only**. fp8/fp4 `matmul2d` was **not measured** — do not record it as a failure.
But separate the two things they'd buy, because the prefill result above bears on one and not the other:
- **Prefill (compute-bound):** an fp8/fp4 `matmul2d` uses the **same tiling** as the fp16/int4 kernel here,
  so it would hit the **same ~3–5 TFLOP/s ceiling** and, like int4, is unlikely to beat the default
  matmul on A19. ⇒ the "fp4 `matmul2d` accelerates prefill" hope is **doubtful on A19** (pending an
  actual fp4 measurement). This is the part Stream D pinned on "Stream B builds the fp4 matmul2d path."
- **Decode (bandwidth-bound):** fp8/fp4's real payoff is **halving/quartering weight read bandwidth**,
  which is a DECODE win and is **independent of the matmul2d compute ceiling** — this result does NOT
  refute it. But decode is S=1 (matVEC, not matMUL2d's regime); the fp4 bandwidth win rides the existing
  int4/fp4 **matvec** kernel path, not `matmul2d`. So Stream D's decode-bandwidth thesis stands on its
  own and should be measured with a matvec/decode bench, not this prefill A/B.
⇒ **Re-scope Stream D:** its value is the fp4 *weight-footprint/decode-bandwidth* + *numerics* (fp4 ≈ int8
perplexity, already de-risked), NOT an fp4-`matmul2d`-prefill speedup. Measure fp4 decode directly before
betting on it.

The M4 correctness work (M0–M2, all cos=1.0) still stands as a **reusable, validated** asset and the
device toolchain path is proven (matmul2d **AOT-compiles + runs on A19**). What's refuted is only the
*speed premise* that a custom matmul2d beats the default on A19.
