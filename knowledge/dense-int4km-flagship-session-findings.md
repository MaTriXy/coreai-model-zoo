# Dense-path int4km coverage + flagship Qwen3.6-35B — session findings & methods (2026-07-01)

> Companion to [[flagship-full-tuning-stack]] (the plan) and [[spec-decode-design]] (Axis-2). This doc
> records the **measured results, methods, and hard lessons** of the dense-path-coverage lever (kernel
> "Axis-1"), from de-risk to the flagship 2.18× measurement, so it's fully reproducible later.

## 1. The lever (what & why)

Flagship MoE decode is weight-bandwidth-bound. The shipped `metalize_moe` (gather_qmm) kernelizes ONLY the
routed-expert FFN; the **dense path stays on MPSGraph** (lm_head + attn q/o + shared expert, int8 or fp16).
**Lever = wire the proven fused int4km matvec (`gemma4_metal_mlp.build_fused_int4km_kernel` /
`gemma4_metal_attn_int4km.MetalInt4KMLinear`, gemma4-measured 2.9× int8 / +1.43× int4km, 8/8 EXACT,
AOT-surviving) into that dense path.** lm_head is the single biggest per-token matvec (vocab×hidden);
attn q/o are the next; k/v stay fp16 (N small — "small-N matvecs never pay", the Mac lesson).

## 2. Method — the export variants (reproducible)

- **LFM2.5-8B-A1B**: `conversion/export_lfm2_moe_dense_int4km_decode_pipelined.py <int8km|int4km>`
  (committed bab5fa7). = baseline gather_qmm + `metalize_dense_int4km(model, kernel)` on `model.lm_head`
  and full-attn `self_attn.{q_proj,out_proj}`.
- **Qwen3.6-35B-A3B**: `conversion/export_qwen3_6_moe_dense_int4km_decode_pipelined.py int4km`.
  = experts int4km (gather_qmm km4) + dense int4km on `model.lm_head` + full-attn `self_attn.{q_proj,o_proj}`
  (21 matvecs: lm_head + 10 full-attn layers × q/o). Key: add lm_head + attn q/o to the **int8-quant skip
  list** (keep fp16) so `MetalInt4KMLinear` can palettize them itself.
- **BUG FIXED (needed for both)**: lfm2/qwen attention forwards read `self.q_proj.weight.dtype` to cast
  the input; `MetalInt4KMLinear` had no `.weight` → export failed. Fix = added a `weight` property
  returning the codebook `cb` to `gemma4_metal_attn_int4km.MetalInt4KMLinear` (isolated with a toy repro:
  tie/head were innocent; the attn `.weight.dtype` access was the culprit).

## 3. Results (measured)

| evidence | number | method / file |
|---|---|---|
| lm_head int4km **per-op** vs fp16 @vocab=248K (flagship's exact shape) | **2.77×** | `ondevice/_dense_int4km_microbench.py` (single-op benches are round-trip-floor-confounded at small N; lm_head is big enough to dominate) |
| byte-audit ceiling (config-only) | Qwen3.6 **~1.97×**, GLM-4.7 ~1.34× | `ondevice/_flagship_dense_coverage_audit.py` (GLM cross-checks to its known 3.58 GB/tok) |
| **LFM-8B on-device (A19, PipelinedBench)** decode | **1.23× sustained / 1.43× avg** | thermally-matched PB_N=6; baseline int4-experts+dense-fp16 vs #2 int4-experts+dense-int4 |
| LFM-8B on-device **quality** | **PASS** (33/48 greedy match, coherent, 25+17=42 correct) | reasoning prompt via `oraclePrompt`; the numerics degenerate-prompt "1/24" was an artifact |
| **Flagship Qwen3.6-35B (Mac M4 Max GPU)** decode | **2.18×** (baseline 2.79 → #2 6.08 tok/s) | `ondevice/_qwen36_mac_bench.py`; #2 (experts+dense int4) vs shipped baseline (experts+dense int8, `qwen3_6_..._sym8_gather`) |

**The flagship 2.18× exceeds the ~1.97× audit projection** — and per-step fixed overhead *compresses* the
ratio, so the true byte-read win is ≥2.18×. This beats the prior gather_qmm result (Qwen3.6 2.1× over the
32× over-read; [Qiita](https://qiita.com/john-rocky/items/d687d2c6fcc4f3e70a82)) because #2 stacks
experts int8→int4 AND dense int8→int4 on top of gather_qmm.

## 4. Hard limits & methods learned (the expensive lessons)

- **⚠️ HF download slowness = `hf_xet` bug, NOT a rate-limit.** Symptom: starts ~14 MB/s then stalls
  (esp. near 99%). **Fix = `HF_HUB_DISABLE_XET=1`** (plain HTTP, stable full speed). HF's actual
  rate-limit is a 5-min fixed window (header `ratelimit-policy: fixed window;resolvers;q=12000;w=300`),
  quota 12000 req/5min — nowhere near hit. Restarting `snapshot_download` repeatedly LOSES `.incomplete`
  progress; let one run finish. [xet-core #789 / huggingface_hub #3580].
- **⚠️ On-device (A19) model-size ceiling ≈ 5-6 GB (int4-8B-class).** LFM-8B (5 GB) runs; **Qwen3.6-35B
  int4 (18 GB) → `signal 9` (jetsam OOM)** on the iPhone 17 Pro's ~12 GB RAM (killed during the ~26-min
  cold compile). Flagship 35B **cannot run on the phone** — bench it on the Mac.
- **✅ Mac-GPU method for these decode bundles (macOS 27):** raw `rt.AIModel.load(aimodel,
  SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()))`. It **spews
  `ANECCompile() FAILED / MLIR MPS to ANEC conversion failed` (dozens) — these are NON-FATAL**: MPSGraph
  falls back to GPU and runs. Earlier I killed a run on the first ANE error (wrong call). The Mac has the
  RAM the phone lacks. There is **no GPU-only spec** (`allowed_compute_unit_kinds` is a read-only
  property; only `default`/`cpu_only`/`from_preferred` exist), so you can't suppress the ANE attempts.
- **⚠️ Absolute tok/s from the quick Mac driver are ~10× too slow** (ANE-retry + per-step
  re-specialization overhead): 2.79/6.08 are NOT real speeds (the real Qwen3.6-35B is tens of tok/s via
  the proper engine). **Only the RATIO (2.18×) is valid** (both bundles share the same slow path).

## 5. Quality — the honest state

- **The dense-int4km lever itself is quality-safe** (LFM-8B: coherent, 25+17=42 correct, 33/48 greedy).
- **Flagship int4 degrades quality** — the shipped-recipe article already documents "non-QAT int4 quality
  degradation / ~12 token flip / 41" on Qwen3.6. This session's flagship greedy check was **inconclusive**
  (the quick Mac driver produced garbage for BOTH int8 and int4 = a driver bug, not int4). So the 2.18×
  is a **speed win at a known int4 quality cost**.
- **The quality-safe path to the flagship speedup = FP4 (E2M1) or QAT-int4** (the "int4 answer" —
  int4-speed at int8-quality; fp4 numerics de-risked: fp4 ≈ int8 perplexity, `project_quant_d_port`).
  **CORRECTION (per [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) §"A19 DEVICE A/B"
  + §"fp8/fp4"):** FP4's flagship win is DECODE-bandwidth (¼ the weight bytes, like this int4 lever), which
  **rides an fp4/int4 MATVEC kernel (like `int4km`), NOT the TensorOps `matmul2d`** — so it is **NOT
  blocked** by the A19 refutation of the matmul2d *prefill* speed lever (default MPSGraph already ≈6 TFLOP/s
  there; the "3–4× prefill" was an M5 claim that doesn't hold on A19). fp8/fp4 matvec is **UNVERIFIED
  on-device** (numerics de-risked, on-device speed/integration not measured yet). Concretely: **swap the
  int4km matvec for an fp4-E2M1 matvec in this same dense-coverage lever** → same ~2× decode bandwidth win,
  int8-like quality. That + QAT-int4 (OS26-shippable) is Stream D's lane (separate sessions).

## 6. Also-measured this session
- **Prefill FlashAttention is a Stream-B (TensorOps) problem, not a pure-MSL Mac win**: MPSGraph prefill
  SDPA is compute-bound (~22% of M4 fp16 peak, clean S² scaling, no crash); scalar q=1-decode kernels
  can't beat a matrix-tuned baseline. Needs simdgroup_matrix / cooperative-tensor. `ondevice/_prefill_sdpa_baseline.py`.

## 7. Reproduce (commands)
```
# 1. download the fp16 source (disable xet!)
HF_HUB_DISABLE_XET=1 python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.6-35B-A3B')"
# 2. export #2 (full Axis-1) and the baseline
cd coreai-models && .venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_moe_dense_int4km_decode_pipelined.py int4km
.venv/bin/python ../coreai-models-community/conversion/export_qwen3_6_moe_metal_decode_pipelined.py sym8
# 3. bench on the Mac GPU (phone jetsams the 35B) — ANE errors are non-fatal
.venv/bin/python ../ondevice/_qwen36_mac_bench.py exports/qwen3_6_35b_a3b_decode_int4km_gather_dense_int4km
.venv/bin/python ../ondevice/_qwen36_mac_bench.py exports/qwen3_6_35b_a3b_decode_sym8_gather
# LFM-8B on-device A/B (fits the phone): PipelinedBench + ondevice deploy script, PB_N=6, reasoning oraclePrompt
```
bundles are coreml (NOT committed); `ondevice/` isn't a git repo (scripts live there, documented here).
Nothing pushed to HF — USER-GATED.
