# Recurrent / linear-attention LLMs on Core AI (RWKV-7, and the no-KV playbook)

> Foundation note for porting **pure-recurrent / linear-attention LLMs** (RWKV-7, RWKV-6, Mamba-2,
> gated-delta-net, future SSMs) to Core AI + iPhone. The durable edge of these architectures is
> **O(1)-per-token decode with NO KV cache** (a fixed-size recurrent state, constant memory,
> unbounded context) — a structural win over every transformer in the zoo. This note captures the
> reusable mechanics, gates, and the one real integration wall.
> First established by the RWKV7-Goose-World3-1.5B port (SHIPPED 2026-07-01, HF
> `mlboydaisuke/RWKV7-Goose-1.5B-CoreAI`, iPhone 17 Pro **25.2 tok/s** decode).
> Sources: `conversion/rwkv7/{validate_ref,gate,prep_vocab}.py`, `conversion/export_rwkv7_decode.py`,
> the `models/macos/rwkv7.py` overlay, `apps/CoreAIChat/Sources/{RWKV7Backend,RWKVWorldTokenizer}.swift`;
> fla `fla/ops/rwkv7/fused_recurrent.py` (decode kernel) + `fla/layers/rwkv7.py`; cf.
> [[compute-units-and-authoring]], [[aot-and-specialization]], [[custom-metal-kernels]].

## 1. The decode recurrence lowers to STANDARD ops — no custom Metal kernel

The headline surprise: a matrix-state linear-attention recurrence, unrolled to a single decode step
(M=1), is a handful of small per-head tensor ops (rank-1 delta + diagonal decay + outer-product write
+ matvec read). It **lowers to standard Core AI ops** and JIT/AOT-compiles cleanly — no hand-written
MSL, unlike BitVLA's ternary GEMM (which hit the "custom kernel can't JIT on device" wall). The
Triton/chunked kernel in the upstream repo is **prefill-only**; for decode you loop M=1.

RWKV-7 WKV7 per head (`S = [K,V] = [head_dim, head_dim]`), transcribed 1:1 from fla's decode kernel:
```
S = diag(exp w)·S − (kk⊙a)·(kkᵀ S);   S += k⊗v;   o = rᵀ S
```
Precedent that this exports fine: `granite4h.py`'s Mamba-2 single-step SSM update is also pure torch.
**Rule of thumb:** if the recurrence is expressible as standard tensor ops at M=1, skip the custom
kernel — export it directly. (The chunked-prefill kernel is a separate, later speed lever.)

## 2. State model: fixed-size, no KV, fused writes

Wire the recurrent state as Core AI **states** (the `SSMState` shape `[num_layers, batch, *dims]` +
`update_states(layer_idx)` via `mutable_slice_update` in `primitives/macos/cache.py`), NOT a KVCache.
RWKV-7 carries exactly two, both fixed-shape (the O(1) win — no seq dim, no growth):
- `recState [layers, 1, heads, K, V]` — the matrix state `S`.
- `shiftState [layers, 1, 2, hidden]` — token-shift previous hidden (time-mix + channel-mix slots).

**GPU-delegate gotcha (macOS-27 beta, baked in — the lfm2 lesson):** per-layer `update_states` calls
compile to a read/slice_update/write round trip on the SAME state handle, and the delegate **silently
DROPS all but one** when there is more than one write. So each layer must RETURN its new state slices
and the model does **ONE fused `mutable_slice_update` per state tensor at the end** of the step
(disjoint slots, each layer reads only its own slot before any write → identical semantics).

The exported decode graph is **fully static** (input_ids `[1,1]`, no dynamic seq dim anywhere). Keep
a `position_ids [1,1]` input for engine parity even though RWKV is positionless (the runtime accepts
it unused — the parity gate confirmed it).

## 3. The integration wall: the pipelined engine can't drive a no-KV model

`CoreAIPipelinedEngine` **hard-assumes** `stateNames[0]=keyCache`, `stateNames[1]=valueCache`,
ALWAYS allocates a growing KV cache, and advances a position index. Its "extra states" patch allows
≤2 fixed-shape states but only *beyond* the KV pair. A model with only `recState`/`shiftState` and no
KV **cannot ride it**. This is the one real blocker for any no-KV architecture.

**The fix (reusable):** drive the low-level `InferenceFunction.run(inputs:, states:, outputViews:)`
directly in a **custom backend** — states bind **by NAME** via `MutableViews.insert(&arr, for:
"recState")`, so any state naming works. Mirror `BitVLABackend.swift`: `AIModel(contentsOf: url,
options: .init(preferredComputeUnitKind: .gpu))` → `loadFunction("main")` → build NDArrays with
`descriptor.resolvingDynamicDimensions(shape)` + `fillNDArray(&a, as: Float16.self, with:)` → loop
S=1, mutating the state NDArrays in place, host argmax over logits. RWKV is *simpler* than BitVLA (no
vision tower, no KV growth, no position). Register a mode case + a branch in the app's load/generate
dispatch. (Custom backends live in the CoreAIChat working tree and stay **untracked**, like BitVLA /
SpecDecode — the app is not committed piecemeal.)

CPU delegate note: these graphs currently **fail to load on the CPU delegate** (`CoreAICompiler error
3`); use the **GPU** compute unit (serialize under `~/code/coreai/_GPU_LOCK` on the beta driver).

## 4. Quantization: protect the recurrence, quantize the bulk

The recurrence runs in fp32 internally, but its inputs come from the `r/k/v/o` projections, and the
delta-rule is **precision-sensitive**. Ship recipe (`int8keepproj`): weight-only **int8** per-block-32
on the **FFN + LM head** (the weight bulk), keep `r/k/v/o_proj` + all LoRA factors + norms + embeddings
**fp16**. Gate (teacher-forced top-1 vs the torch ref): fp16 **127/127**, int8keepproj **127/127**
(~2 GB), int8-everywhere **126/127** (one near-tie flip — int8 on the projections into the fp32
recurrence is borderline). **Lesson:** for recurrent models, exclude the state-projection matmuls from
quantization.

## 5. Parity gate + tokenizer

- Gate **teacher-forced top-1**, not free-running greedy: a single borderline EOS near-tie flips one
  argmax and forks the whole tail, giving false "mismatches". Feed both engine and reference the SAME
  fixed token stream and compare per-step argmax + `max|Δlogits|`.
- State dtype at runtime must match the export dtype (fp16 export ⇒ fp16 state NDArrays, else
  `CoreAIRuntime error 3` at the call; only the fp32 export uses fp32 states).
- **RWKV World tokenizer** is a custom byte trie (greedy longest match), NOT BPE. `AutoTokenizer`
  pulls in the config's `auto_map` → `modeling_rwkv7` → `import fla` (Triton, unavailable on Mac);
  instantiate the tokenizer class directly instead. For device, bake a clean `id\tbase64(bytes)` vocab
  (`prep_vocab.py`, byte-exact verified) so the Swift port avoids Python-repr escape parsing.

## 6. On-device numbers + open levers

iPhone 17 Pro, RWKV-7 1.5B int8keepproj (h18p AOT): **decode 25.2 tok/s**, O(1) 2-state, no KV,
output byte-identical to the reference. Engine load ~12 s (cold GPU specialization). **Prefill is the
weak spot** — the M=1 loop makes prompt ingestion ~decode-rate (plus one-time warmup); the upstream
**chunked-prefill kernel** is the obvious future lever if long prompts matter (decode, the number that
sells the O(1) story, is already strong). In-app remote download: publish the device bundle under
`h18p/<dir>/` in the HF repo; point `modelPaths` remote at `h18p/<dir>` → `Documents/models/<dir>`.
