# Static-shape stateful decode + stock-runtime VLM (Unlimited-OCR)

Lessons from porting [`baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR) (DeepseekV2
R-SWA MoE doc-OCR) to Core AI on the **stock `coreai.runtime`** — no engine patch, no static-input
hook. Transferable to any sliding-window / bounded-cache decoder and any `inputs_embeds`-driven VLM.

## 1. A growing decode shape *faults* on Metal 4 — make the graph fully static

The classic flat-latency goal: R-SWA / sliding-window attention keeps the attended set constant, so
decode latency should be flat. The trap is that the obvious implementations all keep a **dynamic
shape** that the runtime re-specializes per step:

- a growing `position_ids [1, seq_len]` input, or
- the standard KV fetch `cache[0:seq_len]` (grows), or
- a dynamic-start narrow `cache.narrow(seq, seq_len−W, W)` for a windowed gather.

Any of these recompiles the Metal shader as `seq_len` grows. On older stacks this is the
**Qwen3-Coder-Next freeze** (periodic stalls). On **Metal 4 / macOS 26 it is worse — the runtime
faults** (`Failed to import MPS module` + `MTL4CommandQueueErrorDomain error 1`, command-buffer dies)
on the **2nd distinct shape**: the first shape compiles and runs, the second recompile crashes.

**Fix — a fully-static decode graph.** Inputs are `inputs_embeds [1,1,H]` + **`pos [1]` (int32, the
absolute position as a runtime *value*, not a shape)**. Then:

- **Data-driven KV write.** Build the `mutable_slice_update` `begin`/`end` from the `pos` tensor
  (`begin = cat([layer, 0, 0, pos, 0])`). A data-dependent slice offset **lowers and runs on the GPU
  with no recompile across offset values** (verified: 4 calls, offsets 5/9/9/20, constant shapes, all
  correct). This is the key enabler — the write position moves without any shape moving.
- **Full fixed-buffer read + mask.** Read the *whole* `StaticKVCache` `[L,1,Hkv,buf,d]` (constant)
  and apply the R-SWA visibility mask `(j≤i)&((j<Lm)|(j>i−W))` over `[0, buf_len)`. Math-identical to
  a prefix∪tail gather (masked-out / unwritten-zero slots contribute nothing), but no dynamic slice.

Result: no tensor shape ever changes → the engine compiles **once** → flat ~12.7 ms/token
(`max/median 1.22×`). Cost: the SDPA attends the full `buf_len` (e.g. 2048) every step instead of
`Lm+W` (~243); negligible vs the MoE FFN, and worth it for stock-runtime stability.

### What did NOT work

- **`torch.sym_max`** (for clamping a dynamic narrow start `max(seq_len−W, 0)`): traces under
  `torch.export`, but the Core AI converter has **no lowering** → `Unsupported ATen op: sym_max`.
- **Dead-prepend trick** (shift writes by W so the tail narrow start = `seq_len`, dodging the clamp):
  exports, but the dynamic-start narrow still **recompiles/faults** per step. Avoid dynamic narrow
  starts entirely.
- The math of the dynamic constant-window *gather* was proven correct in eager torch (cos 1.0 incl.
  steady-state), so it's purely an export/runtime limitation, not a numerics issue.

## 2. SDPA can't be externalized when you need a runtime mask

The engine-native (externalized) SDPA op takes `scale` / `is_causal` / `window_size` as **attributes**
— it does **not** accept an arbitrary **runtime mask tensor**. R-SWA (global prefix + sliding window)
isn't expressible as `is_causal`+`window_size`, so you must feed a custom bool mask → **don't
externalize SDPA** (it lowers as plain matmul/softmax/mask). Keep RMSNorm externalized. Passing a
runtime mask to an externalized SDPA produces a malformed graph that command-buffer-faults.

## 3. Driving a stateful, inputs_embeds VLM on the stock runtime (no patch)

The zoo's other VLMs ride the **pipelined engine + `apps/coreai-pipelined-static-inputs.patch`**
(image embeds in a bound MTLBuffer, extension ids `V+slot`, `engine.generate(tokens)`). That needs
the patch and a token-driven graph. The pure-export alternative, which works on **stock
`coreai.runtime`**:

- The decoder takes **`inputs_embeds`** directly (host assembles visual + text embeddings). Python:
  `function(inputs={...}, state={keyCache, valueCache})`. Swift: `AIModel` +
  `InferenceFunction.run(inputs:states:outputViews:)` with `InferenceFunction.MutableViews()` for the
  KV cache (the `CoreAISequentialEngine` pattern; `GraphModel` rejects stateful graphs).
- **Multi-function bundle**: stage two `add_pytorch_module(..., entrypoint_name=)` (prefill + decode)
  into one `TorchConverter`; `to_coreai()` **shares/dedups the weights** (one 3.2 GB bundle, not
  6.4 GB) and both functions share the state by name.
- The decode kernel (`gather_qmm`, MoE) requires `b*s==1`; for the q>1 prefill use
  `metalize_moe_batched` (`BatchedMetalSwitchGLU`) — it serves both (q=1 falls back to the q=1 path),
  so prefill + decode share one sym8 weight set.
- **Gate it directly** (custom `coreai.runtime` script), not `llm-runner` — a VLM is embeds-driven,
  not token-driven. Seed the prefix in torch (fp16) or via the engine prefill function; K/V are
  MoE-quant-independent, so an fp16-prefilled cache is valid for sym8 decode.
- `cpu_only` won't compile this graph (`CoreAICompiler error 2`); GPU works. The sym8 metal MoE
  kernel runs fine in `coreai.runtime` direct **once shapes are static** (earlier "Failed to import
  MPS" was the recompile, not the kernel).

## 4. Quantization consistency + greedy repetition

- **Consistent sym8 throughout** (prefix prefill *and* decode both sym8) is the honest ship config:
  **0 argmax flips** vs the fp32 oracle. A mixed fp16-prefix + sym8-decode shows artificially higher
  cos (0.9993) but isn't what ships; full sym8 dips a couple early-decode steps to cos 0.9989 (the
  sym8 prefix) with identical output. Gate on **flip≤2**, not a tight cos≥0.999.
- **Greedy derails** into degenerate repeats on dense content (e.g. long `↑↑↑` runs in a benchmark
  table). Match the model's reference decode: **no_repeat_ngram (n=35)** + a **consecutive-run cap**
  (ban a token that would be the K-th identical in a row). O(V)/step (one scan, no full sort).

## 5. The visual-token arrangement is host-side and exact

Base mode (image_size 640, `crop_mode=false`): vision `.aimodel` → 100 patches → `view(10,10)` →
append `image_newline` after each row (→110) → append `view_seperator` (→111) → `masked_scatter`
into `embed_tokens(input_ids)` at the `<image>` (id 128815) positions, where
`input_ids = [BOS, <image>×111, "document parsing."]` (Lm = 115). Reconstructs the reference prefix
**exactly** (cos 1.000000, |Δ| = 0). Ship `embed_tokens` + `image_newline` + `view_seperator` +
`prompt_input_ids` as raw tensors so the host (Swift) does the arrangement without the full model.

See [`models/unlimited-ocr/README.md`](../models/unlimited-ocr/README.md), [`conversion/unlimited_ocr/`](../conversion/unlimited_ocr),
[`apps/CoreAIOCR`](../apps/CoreAIOCR).
