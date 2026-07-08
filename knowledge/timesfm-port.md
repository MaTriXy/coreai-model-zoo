# TimesFM 2.5 200M — port knowledge

The zoo's first **time-series forecasting foundation model**. `google/timesfm-2.5-200m-transformers`
(Apache-2.0). Archetype = **stateless single graph + host DSP** (like Kokoro / Parakeet), but the
graph is a decoder-only LLM-shaped transformer and the "tokens" are numeric patches.

## What it is

Decoder-only transformer over **patches** of the input series (patch_length 32 → one token). Given
a context of up to 16384 points it emits, at each patch position, a **128-step horizon** of **10
quantiles** (deciles 0.1…0.9 + mean). We forecast from the **last** patch. Config: hidden 1280, 20
layers, 16 heads, head_dim 80, intermediate 1280, rms_norm_eps 1e-6, RoPE θ 10000,
`force_flip_invariance=True`, `infer_is_positive=True`, `decode_index=5`,
`use_continuous_quantile_head=True`, `output_quantile_len=1024`.

Needs **transformers ≥ 5.x** for `TimesFm2_5ModelForPrediction` (config says `5.3.0.dev0`); 4.57 only
has TimesFM 2.0 (`TimesFmModelForPrediction`). Keep a separate oracle venv on transformers-main.

## Architecture — the decoder is zoo-standard, the I/O is new

The transformer core is **familiar LLM machinery** the export stack already handles:
- **Gemma sandwich-norm**: 4 RMSNorm/layer — `input`, `post_attention`, `pre_feedforward`,
  `post_feedforward`; residual adds *after* the post-norms (`x = post_norm(sub(x)) + residual`).
- **Qwen-style QK-norm**: per-head RMSNorm on Q and K, applied **after RoPE**.
- **Learnable per-dim attention scale**: `scale = softplus(scaling) · (log2(e)/√head_dim)`,
  multiplied into Q; the attention interface is then called with `scaling=1.0`. (`log2(e)=1.442695`
  — folds a base-2 softmax convention; replicate exactly.)
- **Plain (non-gated) MLP**: `fc1 → swish → fc2`, `intermediate == hidden == 1280`. No gate_proj.
- Full MHA (num_kv_heads == num_heads == 16), all Linears **bias-free** except `input_ff_layer`.
- Input embed = `TimesFm2_5ResidualBlock`: `Linear(64→1280)→swish→Linear(1280→1280) + Linear(64→1280)`
  skip. Input = `concat([normed_patch(32), mask_bool_float(32)])` = 64-dim.
- Output heads = two ResidualBlocks off the hidden: `output_projection_point` (→ 128·10) and
  `output_projection_quantiles` (→ 1024·10), both bias-free.

The **novel part is I/O**, and it's all host DSP (below).

## Graph boundary (what we export)

Pure feed-forward transformer over patch tokens — **no KV cache, no data-dependent control flow,
static shapes**:

```
tok_in[1,N,64] , cos[1,N,80] , sin[1,N,80] , attn_bias[1,1,N,N]
  → input_ff_layer → 20 decoder layers → output_projection_point[1,N,1280] , output_projection_quantiles[1,N,10240]
```

N = ctx_len / 32 (ship bundle **N=64, ctx 2048**). RoPE cos/sin and the additive mask are computed
**host-side** and passed in, so the graph never sees position_ids or padding logic. Everything else
is host arithmetic.

## Host DSP (the exact spec — `host_forecast.py`)

Two **nested** RevIN, run twice for flip-invariance:

1. **Global RevIN** (`ModelForPrediction.forward`): truncate to ctx, front-pad short series to ctx
   (mask=1 on pad). `mu_g = input.mean()`, `sigma_g = input.std()` (**unbiased, ddof=1**);
   `normalized = (input − mu_g)/sigma_g`.
2. **Per-patch causal RevIN** (`Model.forward`): patch the normalized series; compute **running
   (causal) mean/std per patch with Welford** over valid points → `ctx_mu, ctx_sigma [B,N]`;
   `normed_patch = (patch − ctx_mu)/ctx_sigma`, masked→0. `tok_in = concat([normed_patch, mask])`.
3. Position: `pos = arange(N) − num_masked_patches`; RoPE from that. Mask = causal **AND**
   key-not-padded, as **one** additive tensor.
4. Graph → point/quantile projections; **un-RevIN with per-patch `ctx_mu/ctx_sigma`** (reverse),
   reshape, take the **last** patch → `pf[B,128,10]`, `qs[B,1024,10]`.
5. **Flip-invariance**: repeat 1–4 on `−normalized`; combine `pf = (pf − flipq(pf⁻))/2` where
   `flipq` reverses the quantile axis except the median (`cat([x[...,:1], flip(x[...,1:])])`).
6. **Continuous-quantile head**: `full[...,idx] = qs[...,idx] − qs[...,median] + full[...,median]`
   for each non-median quantile (median index = `decode_index` = 5).
7. **Global denorm**: `full_pred = full·sigma_g + mu_g`. `mean_pred = full_pred[...,5]`.
8. **Positivity clamp**: if the input min ≥ 0, clamp outputs ≥ 0.

The flip pass means **2 graph calls per forecast**. All of 1–8 are deterministic → identical on Mac
and device; only the transformer runs on the Core AI engine.

## Gate ladder (all green, 2026-07-08)

1. Re-authored `TimesFmCore` vs HF captured projections (fp32): **cos 1.0000000**, MAE ~1e-6.
2. Independent host DSP + core vs HF final forecast (fp32): **cos 1.0000000**, rel ~1e-8.
3. Core AI export fp32 CPU: **cos 1.0000000**; fp16 CPU/GPU: **cos ≥ 0.99998**.
4. E2E host DSP + fp16 GPU engine vs HF forecast: **cos 0.9999999**, values match to 2–3 dp,
   including a front-padded short-context (300→2048) case.

fp16 bundle 462.9 MB. Mac GPU ~7 ms/graph → **~14 ms per 128-step forecast**. iOS h18p AOT: clean.

## Gotchas (hard-won)

- **Weight names**: the raw `model.safetensors` stores MLP as `mlp.ff0`/`mlp.ff1`; transformers
  remaps to `fc1`/`fc2` on load. Load from safetensors → map `ff0→fc1`, `ff1→fc2` (order matters).
- **fp16 mask fill**: `torch.finfo(float32).min` (−3.4e38) becomes **−inf in fp16**, and
  *adding* causal + key-padding fills overflows to −inf too → an all-masked query row → softmax NaN →
  `0 · NaN` propagates into valid positions. Fix: build **one** additive mask
  (`allowed = causal ∧ key_valid`) with a **finite** fill (−1e4). A finite fill makes an all-masked
  row a uniform (harmless) average; the padded query positions are discarded anyway (we read the
  last patch). This is invisible until you feed a **short (front-padded)** series.
- **Two RevIN levels are both real** and nested — the per-patch one denormalizes the *projections*
  (inside `_decode_and_project`), the global one denormalizes the *final* forecast. Don't collapse them.
- **`std` is unbiased** (ddof=1) for the global stats; Welford uses population variance per patch.
- **One bundle, many contexts**: fix N to the max (64/ctx-2048); the host front-pads + masks shorter
  series. No need for multi-bucket exports (weights would just be duplicated).

## Files

`conversion/timesfm/`: `timesfm_core.py` (export-clean graph + safetensors loader + `EngineCore`),
`host_forecast.py` (host DSP), `oracle.py` (HF dump, `--ctx/--short`), `gate_core.py` (ladder 1),
`export_timesfm.py` (export + ladder 3, `--ctx/--dtype`), `e2e_engine_gate.py` (E2E). HF DL used
`HF_HUB_DISABLE_XET=1`. Oracle env = fresh venv, transformers-main; export env = `coreai-models/.venv`
(loads weights from safetensors, no transformers dep).
