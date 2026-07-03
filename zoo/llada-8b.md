# LLaDA-8B (diffusion LM) — Core AI

[`d3LLM/d3LLM_LLaDA`](https://huggingface.co/d3LLM/d3LLM_LLaDA) (distilled from
[`GSAI-ML/LLaDA-8B-Instruct`](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct), MIT) — the
zoo's **first diffusion LLM**. The reply is not written left-to-right: a masked canvas
denoises in place, tokens committing **in parallel, lowest-entropy first**, semi-AR block by
block. One static **bidirectional** forward (`input_ids[1,S] → logits[1,S,V]`, no KV cache);
the denoising loop is host code.

Bundle: [🤗 mlboydaisuke/LLaDA-8B-dLLM-CoreAI](https://huggingface.co/mlboydaisuke/LLaDA-8B-dLLM-CoreAI)
— macOS (int4 per-block-32 body + int8 head, **4.9 GB**, S=256 canvas). Mac-only today
(the canvas graph wants an AOT export before iPhone). Catalog id: **`llada-8b`**.

<!-- gen-cards:use-it begin id=llada-8b (managed by scripts/gen-cards — edit cards.json / QuickStart.swift, not this block) -->
## Use it

▶️ **Run it (source)** — the [DiffuseChat runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DiffuseChat)
(GUI + CLI, one app for every diffusion LM in the catalog):

```bash
git clone https://github.com/john-rocky/coreai-kit
open coreai-kit/Examples/DiffuseChat/DiffuseChat.xcodeproj
# → Run, then pick "LLaDA-8B (diffusion)" in the model picker

# agents / headless (macOS):
cd coreai-kit/Examples/DiffuseChat
swift run diffuse-cli --model llada-8b --prompt "What is the capital of France?"
```

💻 **Build with it** — complete; the glue is kit API, copy-paste runs:

```swift
import CoreAIKit

let dlm = try await KitDiffusionLM(catalog: "llada-8b")
let reply = try await dlm.reply(to: prompt)
// reply: the denoised answer — pass onStep: to watch the canvas fill in per forward
// (still-masked positions as ░), in parallel, not left-to-right
```

The take-home is [`Examples/DiffuseChat/Sources/QuickStart.swift`](https://github.com/john-rocky/coreai-kit/blob/main/Examples/DiffuseChat/Sources/QuickStart.swift)
— this exact code as one typed function, no UI; the CLI is an argument shell over it, and
the GUI renders the same live canvas.
The canvas is fixed (S=256 ≈ 210 generated tokens) and the whole history must fit — no
KV cache. `reply(messages:)` takes role/content turns and drops the oldest first.
Pass `onStep: nil` if you only want the final text.

**Integration checklist**

- SPM: `https://github.com/john-rocky/coreai-kit` → product **CoreAIKit**
- Info.plist: none needed
- Entitlements: none needed
- First run downloads the model — 5.3 GB (Mac) — then it loads from the
  local cache (Application Support; progress via the `downloadProgress` callback)
- Measure in Release — Debug is ~3× slower on per-token host work
<!-- gen-cards:use-it end -->

## Measured (M4 Max, GPU)

Distillation (d3LLM) commits ~8 tokens per forward; the entropy `threshold` trades step
count (NFE) against caution at constant ms/forward: `0.5 → NFE 19`, `1.0 → NFE 11 ≈ 38–40
tok/s`, `1.5 → NFE 8 ≈ 53 tok/s` (≥ ~2.5 degrades). The shipped bundle's metadata carries
the validated defaults; `KitDiffusionLM` reads them.

## Parity

- Engine vs the official `LLaDAModelLM`: **cos ≈ 1.0** per layer + logits
  (`conversion/dllm/gate_llada_torch.py`).
- Full decode vs the official `generate` at temperature 0: matched
  (`conversion/dllm/gate_llada_decode.py`). int8 is lossless; int4 per-block-32 diverges
  in token choice occasionally but stays correct (see
  [`knowledge/diffusion-llms-dllm.md`](../knowledge/diffusion-llms-dllm.md) — the "20/64
  token match ≠ broken" lesson).

## Canvas limits (honest)

S=256 fits ≈ 210 generated tokens and the **whole history must fit in the canvas** (no KV
cache): `KitDiffusionLM` drops oldest turns first, so short multi-turn works but long memory
does not. A delayed-KV-cache decode is the known next lever.

## Run

The kit's [DiffuseChat runner](https://github.com/john-rocky/coreai-kit/tree/main/Examples/DiffuseChat)
(GUI + `diffuse-cli`, live ░-canvas view), or the zoo's
[CoreAIChatMac](../apps/CoreAIChatMac) app (pick **LLaDA-8B d3LLM**, same denoising view).
