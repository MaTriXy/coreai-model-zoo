# BitCPM-8B — 1.58-bit ternary on Core AI (the zoo's first sub-int8 kernel)

Lessons from porting [`openbmb/BitCPM-CANN-8B`](https://huggingface.co/openbmb/BitCPM-CANN-8B) — the
MiniCPM4-8B architecture quantization-aware trained to **ternary** ({-1, 0, +1}) — to Apple Core AI
with a **custom 2-bit packed-GEMM Metal kernel**, the zoo's first sub-int8 kernel. It runs on the
iPhone 17 Pro GPU at **17 tok/s decode in ~2.1 GB resident** (an 8B at a 4B's footprint).

## 1. The weights: TQ2_0, and extracting ternary from it

`BitCPM-CANN-8B`'s `main` repo ships the **bf16 latent master** (standard MiniCPM modeling, no
BitLinear) — running it as-is is full-precision, not ternary. The ship-ternary truth is the
**`bitcpm4-8b-tq2_0.gguf`** (2.37 GB). TQ2_0 (llama.cpp) = per **256-element block** along the
reduction axis K: each weight is a 2-bit code in {0,1,2} → value `(code−1)` ∈ {-1,0,+1}, times one
fp16 scale `d` per block. `gguf.quants.dequantize` handles TQ2_0 / Q4_K / Q6_K directly (no hand
de-interleave). To recover the kernel inputs from a dequantized weight `W[N,K]` (== `d·(q−1)`):
`d_block = max(|W| in block)` (exact — the nonzero magnitude **is** `d`), `code = round(W/d)+1`.

Only the **224 transformer linears** (q/k/v/o + gate/up/down × 32 layers) are TQ2_0. The
**embedding is Q4_K** and the **untied LM head is Q6_K** — BitNet practice keeps those higher-precision.

## 2. The kernel: simpler than int4 k-means

`bitcpm_ternary_metal.py` is the int4-k-means matvec (`gemma4_metal_mlp.py`) minus the codebook:

- **Pack 16 ternary codes per uint32** (2 bits each). Decode block = **512 K** (32 lanes × 16
  codes/lane); a lane's 16 codes sit inside one 256-scale block (16 | 256).
- Dequant is `(code−1)` ∈ {-1,0,+1} — no codebook gather, no LUT. The matvec is a sign-add/subtract.
- **Per-lane scale before `simd_sum`.** Dequant is linear:
  `Σ_k x_k·d_b·(q_k−1) = Σ_b d_b·Σ_{k∈b} x_k·(q_k−1)`, and each lane's 16 codes are in one block, so a
  lane multiplies its partial by its own block's `d` then the simd-group reduces. Multi-row `R=4`,
  `SGY=8` (32 output rows/threadgroup) from the int4 kernel carry over.
- Constraints: `K % 512 == 0` (4096/16384 ✓), `N % 32 == 0` (256/4096/16384 ✓).

Numerics: the kernel's `torch_defn` is **bit-identical** to the gguf dequant (maxerr 0). The full
8B decode graph generates "The capital of France is" → "Paris." and the engine output is
**token-identical** to the torch reference (3/3 prompts, greedy) on M4 Max (62.7 tok/s).

## 3. The decode contract: `M=1` kernel ⇒ `S=1` static-ids export (the key trap)

The ternary kernel is **M=1** (single-row decode matvec), like every gemma4-metal kernel. A
**dynamic-`input_ids`** export lets the prefill run **`S>1`**; the kernel's `x.reshape(s,k)` then
produces a dynamic-row tensor MPSGraph can't constrain, and lowering fails at engine-compile:

```
error: 'mps_spi.copy_discarding_constraints' op input must have tensor constraints   (op_id ~48, early)
```

Fix: export **`--static-ids`** — `input_ids` pinned to `[1,1]` (S=1 always), `position_ids` + KV
dynamic. Prefill then runs as pipelined **S=1 steps** under `COREAI_CHUNK_THRESHOLD=1` (no prefill
loss; "prompt tok/s ≈ decode tok/s"). With S=1 the kernel compiles, AOT-survives, and runs.
`position_ids` must carry the **full length** (the attention offset = `seq_len − 1` = the KV write slot).

## 4. Gating an M=1 static bundle (harness gotchas)

- **`llm-runner` cannot drive it.** `--raw-tokens` / `--prompt` always try a multi-token prefill
  (256-tok specialization) vs the static `[1,1]` → `NDArrayDescriptor: Shape at dimension 1 of 256 is
  not a valid substitution for source shape 1`. This fails even on a known-good static decode bundle.
- **`llm-benchmark` runs it** (speed only): `COREAI_CHUNK_THRESHOLD=1 llm-benchmark --model <dir>`.
- **Token gate = the Python `coreai.runtime` API.** Load the **`.aimodel` directly** (a bundle dir's
  hand-written outer metadata lacks `assetVersion`; the inner `.aimodel`/`.aimodelc` has it):
  `m = await rt.AIModel.load(aimodel, SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.gpu()))`,
  `fn = m.load_function("main")`. The decode contract from `fn.desc`: inputs `[input_ids, position_ids]`,
  **state `[keyCache, valueCache]`**, output `[logits]`. Drive an S=1 loop:
  `await fn(inputs={input_ids:[[t]], position_ids:arange(pos+1)[None]}, state={keyCache, valueCache})`
  — the state NDArrays mutate in place across steps.

## 5. iPhone deploy

- **AOT** for h18p (`xcrun coreai-build compile … --preferred-compute gpu --architecture h18p`,
  EXIT 0): a `TorchMetalKernel` graph survives AOT for the iPhone GPU. ANE is unsupported (GPU-only
  kernel). See [`aot-and-specialization.md`](aot-and-specialization.md).
- **Device-disk traps** (sideloading the 3 GB bundle): the AOT load stages the precompiled MPSGraph
  package into `Library/Caches/coreai-cache` — needs ~3 GB free; a near-full device fails ENOSPC, and
  the **partial stage pollutes the content-keyed cache** → next launch fails `Code=2` (No such file).
  `devicectl` has no file-remove; the clean reset is **uninstall → reinstall → re-copy → relaunch**.
- **Measured (iPhone 17 Pro, CoreAIChat pipelined GPU, greedy):** decode **17 tok/s**, prefill
  **13 tok/s**, resident **~2.1 GB** during gen, headroom 4.3 GB, no jetsam; cold load 9 s.

## 6. Why this model

The 2026 MLX surge on Apple Silicon makes "match MLX on a Mac" a moving target. The durable Core AI
edge is a kernel MLX **structurally lacks** (its quant is 4/8-bit affine — there is no 2-bit ternary
GEMM) on a device MLX **doesn't ship to**. 1.58-bit on iPhone is both — and the architecture
(MiniCPM4-8B) was already in the zoo's path, so the whole novelty is the one kernel.
