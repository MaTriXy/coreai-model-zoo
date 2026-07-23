# Nanbeige4.2-3B on Core AI: support and optimization report

This report records the implementation and acceptance work for
[`Nanbeige/Nanbeige4.2-3B`](https://huggingface.co/Nanbeige/Nanbeige4.2-3B) at immutable revision
`5ff54fb7ed86ce8e216d78bff5417ab9981de3d4`. Conversion and Mac execution are implemented. The verified int8
bundle is [published on Hugging Face](https://huggingface.co/ukint-vs/Nanbeige4.2-3B-CoreAI/tree/5864ec7a5581940958e58354a6b6c46c8f06891e)
at immutable repository revision `5864ec7a5581940958e58354a6b6c46c8f06891e`; published-LanguageBundle
chat-template review, iPhone acceptance, and CoreAIKit enrollment remain pending.

Port by [Vadim Smirnov (@ukint-vs)](https://github.com/ukint-vs); tracked in
[`john-rocky/coreai-model-zoo#5`](https://github.com/john-rocky/coreai-model-zoo/issues/5).

## Model architecture

The [pinned released config](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/5ff54fb7ed86ce8e216d78bff5417ab9981de3d4/config.json)
and [vendor forward implementation](https://huggingface.co/Nanbeige/Nanbeige4.2-3B/blob/5ff54fb7ed86ce8e216d78bff5417ab9981de3d4/modeling_nanbeige.py)
show that Nanbeige4.2 is not a 44-weight-layer Llama. It owns **22 physical Llama blocks** and executes the same
blocks twice, applying the final RMS norm after each pass. Each pass needs independent attention history, so the
Core AI decoder has **44 logical KV-cache layers** while retaining only 22 trainable block instances:

```text
embedding
  → physical blocks 0…21 → norm       (cache slots 0…21)
  → physical blocks 0…21 → norm       (cache slots 22…43)
  → untied language-model head
```

The authoring overlay registers `model_type = "nanbeige"`, reuses the Llama embedding, fused QKV attention,
MLP, RMSNorm, RoPE, head, weight loader, and quantization traversal, and adds only recurrent execution plus a
cache-offset wrapper. The released configuration is accepted; unsupported vendor options fail by name. This
includes wrong loop counts, skipped loop norms, loop-loss metadata, attention or MLP bias, QK norm, shared loop
KV, N-gram features, split loops, hyper-connections, mHC, and depth attention.

The checkpoint config SHA-256 is
`f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19`. The implementation follows the
recurrent semantics in [mlx-lm commit `308bc1f`](https://github.com/ml-explore/mlx-lm/commit/308bc1f68fafd41756a5973e446215729f5ca7fe);
mlx-lm is a semantic oracle, not a runtime dependency.

## Conversion and runtime integration

`conversion/export_nanbeige41_decode_pipelined.py` remains backward-compatible with Nanbeige4.1. It selects
Llama or Nanbeige authoring from the checkpoint `model_type`, forwards `--revision` to model and tokenizer
downloads, and records the revision in bundle metadata. Nanbeige4.2 allocates K/V state as
`[44, 1, 8, max_context, 128]`; Nanbeige4.1 keeps its original shape.

The exported interface is unchanged:

```text
input_ids, position_ids, mutable k_cache, mutable v_cache → logits
```

The shipping candidate is static-S=1 `int8hu --head-sym --static-ids`. The runtime patch adds descriptor-driven
single-token prompt chunking and warmup plus a static logits-capacity fix. Those fixes are required for reliable
decode with the current pipelined runtime.

Apple describes states as the mechanism for updating KV caches in place and avoiding full-history recomputation,
and Core AI specialization as the way to compile model shapes for the target hardware
([Meet Core AI](https://developer.apple.com/videos/play/wwdc2026/324/)). The implementation uses those native
paths rather than adding a separate recurrence runtime.

## Correctness and quality gates

The synthetic suite verifies:

- exactly 22 unique physical blocks and 111 physical linear modules;
- exactly 44 cache slots;
- execution order `22 blocks → norm → same 22 blocks → norm`;
- cached and full-prompt float32 logits within `rtol=1e-4`, `atol=1e-4`;
- a one-layer truncated model still runs twice and uses two cache slots;
- every unsupported released-config alternative fails explicitly.

Against the isolated official checkpoint, full-prompt float32 logits have maximum absolute error
`1.01566e-4`, cached logits `2.09808e-5`, and the 32-token greedy continuation is identical. The int8 authoring
gate reports top-1 8/8, greedy 32/32, cosine `0.9997768`, and deterministic output. The exported Core AI int8
bundle is token-exact against the fp32 oracle for:

- “The capital of France is” — 24 tokens;
- “Water freezes at 0 degrees Celsius, which is” — 16 tokens;
- “Which is bigger, 9.11 or 9.8? Explain your answer.” — the full 64-token smoke, correctly reasoning that 9.8
  is greater than 9.11.

The engine bundle loads, mutates all 44 cache layers, and produces the same deterministic continuation on a
repeat run.

## Quantization results

| Candidate | Size | Quality status | M4 Max, prompt 128 / generation 256 / 3 runs | 4K boundary |
|---|---:|---|---:|---:|
| int8hu block-32 symmetric | 4.59 GiB | **Pass; shipping baseline** | 47.37 prefill / 46.35 decode tok/s | 29.83 / 32.80 tok/s |
| int4hu block-32 symmetric | 3.14 GiB | **No-go** | 58.91 prefill / 56.07 decode tok/s after warmup | 45.47 / 44.46 tok/s |
| mixed int4/int8 (layers 3/4/5 int4) | 4.41 GiB | **No-go** | not benchmarked; quality gate first | not run |

The int8 4K run used a 3,840-token prompt plus 256 generated tokens, peaked at 9.17 GiB RSS, and recorded zero
swaps. Int4 peaked at 6.24 GiB. Int4 is faster and smaller but is not shippable: Paris diverges after 2/24
tokens, the freezing prompt after 3/16, and the reasoning prompt diverges at the decisive first prediction with
margin `0.8237`. Determinism does not compensate for failed oracle parity.

Mixed precision was screened by changing physical layers, so each selected weight was still stored once and
used by both loop passes. All four layer quartiles failed decisive eager checks. Twelve layers were harmless in
isolation, but accumulated error made the combined set fail. The largest greedy eager survivor used int4 for
physical layers 3, 4, 5, and 10; its 4.3 GiB Core AI bundle matched the 24-token Paris probe and took only an
allowed fp32 knife-edge branch on the freezing probe, but failed the 64-token decimal-reasoning probe at token
zero with a decisive fp32 margin of `0.8542`. Removing layer 10 produced the 4.41 GiB candidate shown above and
failed at the same position and margin. Other two-layer combinations failed eager checks; the surviving
two-layer mix would put only 2/22 physical blocks in int4—too little bandwidth reduction to justify
another shipping mode after the larger Core AI failures. The exporter and recipe therefore remain unchanged:
int8 is the only accepted body quantization.

A second sweep selected int4 by projection role across all 22 physical layers while leaving every other body
linear and the head at the accepted int8 settings. `qkv_proj`, `o_proj`, `gate_proj`, and `down_proj` all
produced decisive teacher-forced errors; their worst accepted-int8 margins were `0.4620`, `0.1370`, `0.4620`,
and `0.3115`, respectively. `up_proj` was the only eager survivor: Paris and freezing were exact, and its sole
reasoning mismatch was a `0.0302` tie. The resulting 4.27 GiB Core AI bundle then passed both factual fp32
engine gates exactly but diverged on reasoning token zero at margin `0.8542`. The temporary mode and bundle were
removed. Selecting by projection role therefore does not recover a shippable mixed int4/int8 configuration.

All accepted Mac results were measured on an M4 Max with 36 GB RAM, macOS 27.0 (`26A5378n`), Xcode 27 beta 4,
runtime `aff0bb2`, AC power, and High Power Mode. A battery-saving run reached only 21.83 decode tok/s, so power
state is part of the benchmark record.

## Kernel optimization investigation

Core AI Instruments showed decode function intervals around 15–17 ms under tracing. Decode falls from 46.35
tok/s at the 128/256 workload to 32.80 tok/s at 4K, identifying the growing attention scan as the only
context-dependent target. The existing graph already preserves SDPA as a compiler-recognized composite.
Apple explicitly recommends its prepackaged fast primitives for operations such as SDPA before moving to a
custom kernel ([Core AI authoring and optimization](https://developer.apple.com/videos/play/wwdc2026/325/?time=0),
[coreai-torch documentation](https://apple.github.io/coreai-torch/)).

Two inline Metal alternatives were nevertheless built and run through the full model:

| Kernel experiment | 128/256 result | 3,840+256 result | Quality |
|---|---:|---:|---|
| one SIMD group/head, parity-oriented multi-pass softmax | 41.82 prefill / 28.96 decode tok/s | not pursued | reasoning gate fails |
| G8 sequence-split online softmax | 46.96 prefill / 44.84 decode tok/s | 28.52 / 20.44 tok/s | reasoning gate fails |

A standalone GPU probe proved that the custom kernel’s GQA mapping is correct: output cosine versus stock SDPA
was approximately 1.0 and maximum absolute error was `9.765625e-4`. Only 55.19% of fp16 elements were
bit-identical, however. Those sub-ULP/one-ULP differences are amplified by 44 recurrent passes and select a
different, high-margin reasoning branch. Changing sequence splitting, moving the scale into fp32, and replacing
online softmax with a closer multi-pass formulation did not restore end-to-end parity.

The custom kernels also lose throughput. The likely reason is that they replace Core AI’s optimized SDPA with
scalar/SIMD reductions while repeating key scans; the G8 variant additionally pays synchronization and merge
costs. Apple’s advanced route is TensorOps FlashAttention using cooperative tensor matmuls
([Optimize custom machine learning operations with Metal tensors](https://developer.apple.com/videos/play/wwdc2026/330/)),
so that route was implemented as a standalone q=1 GQA probe before touching the full model:

| Standalone attention implementation | 257-token cache | 4,096-token cache | Numerical result |
|---|---:|---:|---|
| Core AI externalized SDPA | **0.347 ms** | **0.379 ms** | reference |
| four-way sequence-split TensorOps FlashAttention | 0.422 ms | 0.596 ms | max absolute error `7.63e-6` |

The fair comparison loaded both assets in one process, warmed each, forced output materialization, and used the
median of twelve 100-call batches. An earlier one-SIMD-group TensorOps version was slower still because only
eight SIMD groups were active. Four-way sequence splitting raised occupancy and preserved fp32 online-softmax
state, but the partial-result staging and merge could not beat Apple’s composite. Since it lost the isolated
kernel gate by 22% at short context and 57% at 4K, no full-model candidate was exported and no experimental
kernel remains in the overlay.

The short-context weight path was tested separately with a dense S=1 symmetric-int8 Metal GEMV specialized
from the existing MoE gather kernel. The probe consumed the exact signed-int8 bytes and fp16 block-32 scales
emitted by the shipping quantizer, so it changed only execution, not weights. Each asset was warmed and timed
for 200 calls on the same M4 Max:

| Nanbeige projection shape | Stock int8 | Custom GEMV | Stock/custom |
|---|---:|---:|---:|
| output `[3072, 6144]` | **0.2797 ms** | 0.2900 ms | 0.965x |
| fused QKV `[8192, 3072]` | **0.3149 ms** | 0.3223 ms | 0.977x |
| MLP gate/up `[10752, 3072]` | **0.3459 ms** | 0.3590 ms | 0.964x |
| MLP down `[3072, 10752]` | **0.3519 ms** | 0.3599 ms | 0.978x |
| LM head `[166144, 3072]` | **1.7532 ms** | 1.7603 ms | 0.996x |

Outputs retained cosine approximately `1.0` with maximum absolute error at most `9.77e-4`, but every real
shape was slower. A fused gate+up/SwiGLU variant also failed to improve: row/threadgroup tunings `4×8`, `2×16`,
`2×8`, and `8×4` measured `0.962x`, `0.999x`, `0.891x`, and `0.993x` versus the stock pair. The best result is
statistical parity, not a deployable gain, so both probes were deleted.

**Decision:** retain the built-in externalized SDPA and stock quantized linears. They are faster and simpler.
No custom Nanbeige kernel is included in the conversion recipe.

## Loop-aware complexity and pipeline analysis

The loop changes storage, scheduling, and numerical sensitivity, but it does not create independent work that
can be reordered. Pass 1 layer 0 depends on the normalized output of pass 0 layer 21. Pass 1 keys and values are
therefore functions of a different hidden state and cannot reuse pass 0 cache entries. Splitting the passes into
two host-invoked functions would add a synchronization boundary without allowing overlap inside one
autoregressive stream. The current single graph is the correct minimal schedule.

The useful complexity distinction is:

| Resource | Decode complexity | Released 3B consequence |
|---|---:|---|
| Weight storage / quantization | `O(22 physical layers)` | one copy of 111 physical linear modules |
| Weight reads / transformer compute | `O(2 × 22)` per token | the shared stack must still execute twice |
| Attention work | `O(44 × context)` per token | the only term that grows during decode |
| KV-cache storage | `O(44 × context)` | 44 independent logical cache layers |
| Full-prompt attention | `O(44 × context²)` | long prefill remains quadratic |

With 8 KV heads, head dimension 128, fp16 K and V, the logical cache costs exactly 176 KiB per token: 704 MiB
at 4,096 tokens and 44 GiB at the advertised 262,144-token maximum. The latter does not fit the tested 36 GB
Mac even before weights and runtime memory, which is another reason the release claim remains 4K. The int8 body
also has a lower-bound weight stream of roughly 6.3 GB per generated token from executing the 3.15B-parameter
physical transformer twice, plus the 166K-vocabulary head. This explains why short-context decode is chiefly a
quantized-matvec/weight-bandwidth problem rather than an SDPA-kernel problem.

A loop-native speculative pipeline was also evaluated. Pass 0 was exposed as a self-draft and pass 1 as the
verifier along the exact full-model greedy path. Draft/final top-1 agreement was only 9/24 on the Paris probe,
5/16 on the freezing probe, and 27/64 on the decimal-reasoning probe: 41/104 tokens (39.4%) overall. No draft
top-1 probability reached 0.5. This is consistent with `loop_loss_weights = []`: the released intermediate state
is not a calibrated language-model exit. At that acceptance rate, pass-0 drafting plus pass-1 verification cannot
amortize its cache rollback, head, and function-boundary costs. Early exit and self-speculation are therefore
no-gos for the released checkpoint; they would need an explicitly trained intermediate head.

The remaining credible optimization order is:

1. Keep the compiler-recognized SDPA, fused QKV authoring, single two-pass graph, shared physical weights, and
   44 disjoint cache slots already implemented.
2. Keep int8 as the shipping body. The quality-guided layer and projection-role mixed int4/int8 sweeps above
   were the smallest remaining weight-bandwidth experiments; every meaningful candidate failed a decisive
   eager or Core AI gate.
3. If contexts beyond 4K become a requirement, evaluate quantized KV state. Int8 KV would halve the dominant
   cache capacity and long-context traffic, but it requires a quantization-aware attention path and must beat
   the stock composite plus all recurrent quality gates. It is not justified for the current verified 4K target.
4. Revisit pass-0 speculative decoding only with a distilled intermediate head or loop-loss training. That is a
   new model artifact, not a kernel-only optimization of this immutable checkpoint.

## Device status

Mac acceptance passes. iPhone acceptance does not: the connected iPhone 16 Pro is `h17p` on iOS 26.6, so Xcode
27 cannot mount a compatible developer image, and the release criterion requires an iOS 27 `h18p` device. No
iPhone throughput or memory claim is extrapolated from Mac measurements. Local `h18p` AOT compile acceptance
also stops before compilation because Xcode 27's optional Metal Toolchain component is not installed. Int4
cannot be used as a fallback because it failed the same quality gate required of int8.

## Reproduction

From a `coreai-models` checkout with the zoo overlay applied:

```sh
python3 ../coreai-model-zoo/conversion/zoo_convert.py run nanbeige4.2-3b --dry-run
python3 ../coreai-model-zoo/conversion/export_nanbeige41_decode_pipelined.py \
  int8hu --head-sym --static-ids \
  --hf-id Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4
python3 ../coreai-model-zoo/_smoke/verify_nanbeige42_checkpoint.py \
  --official-python /path/to/nanbeige-oracle/bin/python
python3 ../coreai-model-zoo/conversion/coreai_gate.py \
  exports/nanbeige4_2_3b_decode_int8hu_block32_sym_s1 \
  Nanbeige/Nanbeige4.2-3B \
  --revision 5ff54fb7ed86ce8e216d78bff5417ab9981de3d4 --arch nanbeige -n 24
```

The advertised 262K context is not claimed. The verified recipe remains 4K until larger contexts are measured on
both target classes. No bundle has been uploaded and no external CoreAIKit catalog has been changed. Publication
requires separate approval and successful iOS 27 `h18p` acceptance. The pinned vendor template is bundled and
renders correctly with thinking enabled and disabled; both modes still need published-LanguageBundle review.
