# Tiled ternary GEMM + chunked prefill — lifting the S=1 tax off custom-kernel bundles

Every zoo model that ships a hand-written Metal kernel (BitCPM-8B, BitVLA, the gemma4 raw-Metal
family) is pinned to a **static S=1 export**, so the engine walks the prompt one token at a time.
This is the kernel's doing, not Core AI's, and it costs an order of magnitude on prefill. This
doc records the fix — a tiled ternary GEMM plus a two-entrypoint bundle — and the measurements
that bound what it buys.

**Result: BitCPM-8B prefill 57.7 → 364.7 tok/s on an M4 Max (6.32×), argmax-identical to the
sequential walk at every position, and the multifunction bundle AOT-compiles for iPhone h18p.**

## 1. Why S=1 was forced

`bitcpm_ternary_metal.py` is a matvec, not a GEMM:

```
46:  xr[j] = float(A[k0 + j, 0]);      // reads column 0 only
62:  C[base_row + r, 0] = TYPE(tot);   // writes column 0 only
133: result_shapes=[[1, N]]            // one output row, hardcoded
```

The dispatch grid is `(32, N/R, 1)`: the 32 lanes are spent on the **K reduction** (`simd_sum`)
and the y axis on **output rows**. No thread dimension is left for tokens, so `s > 1` doesn't
merely run slow — `A.get_extent(0)` returns `s` instead of `K` and the result is wrong.

A dynamic `input_ids` export would then hand the kernel a dynamic-row reshape, which MPSGraph
refuses to lower (`'mps_spi.copy_discarding_constraints' op input must have tensor constraints`),
so the export pins `input_ids` to `[1,1]` and the whole graph becomes S=1.

`gemma4_metal_mlp_m4.py` widens this to M=4 by caching x as a `float4` in registers — measured
**1.9–3.0×** on the ternary shapes, and it cannot go further: the 16-word x cache times M
overflows the register file (which is why its `R` drops 4 → 2).

## 2. The tiled GEMM (`bitcpm_ternary_gemm.py`)

The other mapping: **lanes own an output sub-tile, K is a sequential loop, and both operands
stream through threadgroup memory.**

| | m1 / m4 | gemm |
|---|---|---|
| 32 lanes | K reduction (`simd_sum`) | output tile (16×16 thread grid) |
| K | lane-parallel | sequential loop |
| x lives in | registers (caps M at 4) | threadgroup memory |
| weights | re-read per output | dequantized once per tile, reused across M |

Tiles: `BN=64`, `BK=64`, `BM` = chunk (16/32/64/128); 256 threads (32×8); each thread owns
`TM = BM/16` rows × `TN = 4` columns. Threadgroup memory is `xs[BK][BM] + ws[BK][BN]` half =
16 KB at BM=64.

**`BK=64` with `64 | k0` means a K-step never straddles a 256-element scale block**, so the
per-block fp16 scale is a per-step scalar and drops out of the inner loop entirely.

Constraints: `K % 64 == 0`, `N % 64 == 0`. BitCPM (4096/16384/256) and Qwen3.6-27B
(5120/17408/6144/1024) both qualify.

### Kernel-level throughput (M4 Max, one linear, `coreai.runtime`, vs the shipped M=1)

| K × N | m4 (M=4) | gemm M=16 | M=32 | M=64 | M=128 |
|---|---:|---:|---:|---:|---:|
| 4096 × 4096 (attn q/o) | 3.00× | 10.70× | 17.63× | **26.54×** | — |
| 4096 × 16384 (FFN gate/up) | 2.06× | 7.15× | 10.57× | **14.84×** | — |
| 16384 × 4096 (FFN down) | 1.90× | 6.69× | 9.74× | **13.49×** | — |
| 5120 × 17408 (Qwen3.6-27B FFN) | 1.97× | 6.79× | 9.46× | **12.70×** | 14.38× |

All `max|err| ≤ 0.0005` against the dense fp16 reference. BM=128 buys only +13% while doubling
threadgroup memory to 24 KB (halving occupancy) — **BM=64 is the kernel's practical point.**

## 3. The two-entrypoint bundle

`export_bitcpm8b_chunked_prefill.py` emits one bundle with `main` (S=1 matvec) and `prefill`
(S=C GEMM), weights shared. Two pieces were missing and are implemented there:

- **`export_to_coreai_multifunction_with_kernels`** — upstream `export_to_coreai_multifunction`
  has no custom-kernel hook and `export_to_coreai_with_kernels` is single-entrypoint. The fix is
  `converter.register_custom_kernels(...)` *before* the `add_pytorch_module` loop (that call
  validates each exported program against the converter's known lowerings).
- **`DualTernaryLinear`** — one weight pair, two kernels, selected by `s`. Both entrypoints are
  traced at a static query length so `s` is a Python int and **the branch resolves at trace time**;
  each function ends up holding exactly one kernel. 224 linears swapped for the full model.

⚠️ The traced `position_ids` length must satisfy the position `Dim`'s own `min` (= the query
length). Copying the S=1 script's `arange(65)` silently works for C ≤ 64 and fails at C=128 with
`65 not in range [128, 4095]`.

## 4. End-to-end, full 32-layer BitCPM-8B

Interleaved round-robin, same process, 5 rounds (the arms are stable to <0.3%; the **S=1
baseline** is the DVFS-sensitive one at ±11%, so per-run sweeps are not comparable — interleave):

| arm | median tok/s | vs S=1 | gate |
|---|---:|---:|---|
| S=1 (shipped) | 57.8 | 1.00× | — |
| C=16 | 205.7 | 3.56× | argmax 16/16 |
| C=32 | 265.0 | 4.59× | argmax 32/32 |
| **C=64** | **339.4** | **5.87×** | argmax 64/64 |
| C=128 | 365.4 | 6.32× | argmax 128/128 |

The S=1 median (57.7) matches the shipped BitCPM-8B record (62.65 tok/s decode, M4 Max), which
validates the harness. Logits differ at fp16 rounding level (`max|delta|` 0.02–0.03, mean 0.0016)
because the GEMM reduces in a different order; **argmax agrees at every position.**

`C=128` is fastest but only +8% over C=64 for double the per-chunk working set — **ship C=64**,
keep 128 for Mac-side headroom.

End-to-end (5.87×) is well under the kernel-level 13–15× because attention, norms, RoPE and KV
writes don't speed up. An 8-layer probe showed a *higher* 6.64× — the fp16 `lm_head`
(73472 × 4096 = 602 MB) is the single biggest beneficiary of batching, and it weighs more in a
shallow model. The 32-layer number is the honest one.

## 5. Traps hit

- **The engine sizes the logits buffer as `ceil(vocab/64)*64`.** BitCPM's 73448 (%64 = 40) aborts
  at engine warm-up with `MPSNDArray ... buffer is not large enough. Must be 146944 bytes`. Pad
  the head. (gemma4 262144 / qwen3 151936 / Qwen3.6-27B 248320 are all clean.)
- **JIT `.aimodel` loading OOMs the Mac GPU at 32 layers**
  (`kIOGPUCommandBufferCallbackErrorOutOfMemory`) — 8 layers was fine. Route through
  `coreai-build compile --platform macOS --architecture h16c --preferred-compute gpu
  --expect-frequent-reshapes` and load the `.aimodelc` with `SpecializationOptions.default()`,
  per [`coreai-env`](../../CLAUDE.md) — the documented workaround, and it works here.
- `_ANECompiler: ANECCompile() FAILED` appears on the Mac JIT path (the GPU-only kernel can't be
  ANE-placed) and is benign — it falls back to GPU. It does **not** appear on the iOS AOT path
  where `--preferred-compute gpu` is explicit.

## 6. iOS

`coreai-build compile --platform iOS --architecture h18p --preferred-compute gpu
--min-deployment-version 27.0` → **EXIT 0, 3.0 GB `.aimodelc`**, A19 / A19 Pro, iOS 27.0:

```
Function main       input_ids [1 × 1]     logits [1 × 1 × 73472]
Function prefill    input_ids [1 × 64]    logits [1 × 64 × 73472]
  States: keyCache/valueCache (Float16, 32 × 1 × 2 × ? × 128)   -- shared
```

**Custom Metal kernel × multifunction × iOS h18p had no precedent** (the `_pf64` iOS precedents —
GLM-OCR, the VL family — carry no custom kernel; the custom-kernel iOS precedents — BitCPM, BitVLA
— are single-function). It compiles.

Still open: the on-device run. The unknown is the **instantaneous footprint of one S=C chunk** on
A19 — the ~2.3 KB·p² scratch law measured for this bundle class was measured on the S=1 walk and
does not describe a batched chunk. C=32 exists as the fallback if C=64 doesn't fit.

## 7. What this says about speculative decoding

A verify pass is the same shape as a prefill chunk (M = k+1 drafted tokens), so the four widths
above measure verify cost directly. Expressed in decode-step equivalents (`C / speedup`):

| C | verify cost (decode steps) | per candidate |
|---:|---:|---:|
| 16 | 4.49 | 0.281 |
| 32 | 6.97 | 0.218 |
| 64 | 10.90 | 0.170 |
| 128 | 20.25 | 0.158 |

Least squares over the four points is clean and linear:

```
verify cost  ≈  2.28  +  0.140·C     decode steps
               ^^^^      ^^^^^
               fixed     marginal
```

**The 0.140 marginal is what the GEMM bought** — each extra drafted candidate costs a seventh of
a decode step. **The 2.28 fixed floor is what it did not**: attention, norms, RoPE, KV writes and
per-call overhead don't batch away. That floor is, almost certainly, the same thing PrismML
reports for Ternary Bonsai 27B — *"On Apple Silicon the batch-1 verification pass does not yet
amortize, so the drafter layer is not enabled by default on-device"* — while the same drafter is a
measured 1.34× win on H100.

Break-even (accepted tokens per round must exceed the verify cost):

| draft depth k | verify cost | must accept | as % of k |
|---:|---:|---:|---:|
| 4 | 2.83 | 2.83 | **71%** |
| 8 | 3.39 | 3.39 | 42% |
| 16 | 4.51 | 4.51 | 28% |
| 32 | 6.75 | 6.75 | 21% |

**The fixed floor makes wide drafting easier, not harder** — at k=4 you must accept 71% of what
you drafted, at k=32 only 21%. This inverts the usual instinct to keep k small, and it is a direct
consequence of paying 2.28 steps whether you verify 4 candidates or 128.

Two things are missing before this is actionable:

- **The kernel's narrowest tile is BM=16** (thread grid 16×16, `TM = BM/16`). Classic k=4–8 spec
  decode underfills it. A BM=8 variant (8×32 grid, TM=1/TN=2) is a re-arrangement of the same
  kernel, not new work — unwritten and unmeasured.
- **No ternary drafter exists.** The zoo's drafter asset is `gemma4_e2b_mtp_drafter` (different
  model, different quant), and the recorded verdict there was *"MTP loses on A19 — bandwidth-bound
  only"*. Ternary cuts per-token bytes ~8×, so **the premise behind that verdict may no longer
  hold**; re-running the MTP measurement on a ternary body is a cheap falsification and should
  come before any new drafter work.

⚠️ Apple bug **178056451** (custom Metal kernels — recorded as "the spec-decode ceiling itself")
is marked Fixed in the beta4 release notes, which this box has not installed. Beta4 also still
lists **177729331** (AOT failure) as a Known Issue, and everything above depends on AOT. Finish
the on-device gate on the current OS *before* updating, or a device failure can't be attributed.

## 8. Where this applies beyond BitCPM

The kernel is ternary-specific but the *shape* of the fix is not: any zoo bundle pinned to S=1 by
an M=1 kernel can take the same treatment (BitVLA reuses this exact ternary kernel; the gemma4
raw-Metal family has its own quant formats and would need the tiling applied to those). The
`export_to_coreai_multifunction_with_kernels` helper is format-agnostic.
