# int8 LM head + measuring decode speed honestly

**TL;DR: for a big-vocab decoder the fp16 `lm_head` is ~half the per-token memory read — untie it (if
tied) and quantize to int8 (per-block-32 **symmetric / absmax**, never clipping) for a large decode win
at no quality cost. Then measure the win on the bundle you actually ship, in the actual app, as a
same-conditions back-to-back A/B — controlled benchmarks, VLM image buffers, thermal, and the app's own
per-token UI work each move the number by tens of percent.**

## The int8 head lever
A 100k–250k-row `lm_head` is read on **every** decoded token, and on a sub-2B transformer that matrix is
roughly **half** of the per-token weight traffic. Quantizing it int8 ≈ halves that half.

- If the head is tied to the embedding, untie first, then quantize only the head:
  `model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())` → quant config
  `module_name_configs = {r".*lm_head$": head_quant_spec()}`.
- **Use plain `symmetric` (absmax), NOT `symmetric_with_clipping`.** Big-vocab head rows are fat-tailed;
  clipping the outliers flips next-token argmax (measured: qwen3.5-2B 6/16 oracle flips with clipping;
  absmax keeps cos ≈ 0.9999, argmax == HF). Per-block-32.
- Cost: untying adds the head's weights to the bundle (they were shared) — ~+0.2 GB at 250k vocab.
- This is the `int8hu` mode; most zoo decode exports already carry a `head_quant_spec()`. See
  [`compression-reference.md`](compression-reference.md) and [`pipelined-engine.md`](pipelined-engine.md).

## The same model gives different decode numbers — don't quote across them
Worked example: MiniCPM-V-4.6 (qwen3.5-0.8B class), iPhone 17 Pro, int8 head.

- **Controlled bench (PipelinedBench, pure compute), text core, A/B isolating the head:** 46 → 68 tok/s = **+48%**.
- **The shipped VLM bundle is slower than the text core** because it binds a static `image_embeds[64,1024]`
  buffer **every step** (extra per-token traffic). Same head, VLM bundle, same-conditions in-app A/B:
  51.5 → 70.0 tok/s = **+36%**. → *Measure the bundle you ship, not the text-core proxy.*
- **Thermal: the ratio is robust, the absolute is not.** Cool (post-reboot) ≈ 70; warm interactive ≈ 64.
  Old and new scale together — quote the **%** from a back-to-back A/B; if you cite an absolute, state the
  thermal condition and match it to whatever a demo video shows.
- **Rule:** measure an **in-app, same-conditions, back-to-back A/B** before quoting a speedup. Never reuse a
  different bundle's or a pure-compute number for the shipping config. (Mixing a cool/headless *old* with a
  warm/interactive *new* is fine only as a deliberate conservative lower bound.)

## The app can be the bottleneck, not the model
A SwiftUI chat that re-decodes the **whole** token list with the tokenizer on every token
(`tokenizer.decode(fullSequence)`) and pushes a view update is **O(n²)** and serializes against the model.
It dragged both the measured and the experienced rate well below the model's true decode speed, and got
worse the longer the output (a 343-token reasoning trace measured ~10 tok/s low).

- Fix: **throttle the live refresh to ~25 fps**, and **exclude the UI-callback time from the decode timer**
  so reported tok/s reflects model compute (matches the benchmark). After this, in-app ≈ model rate.
- True incremental decode (decode only the new token, O(1)) would remove the last residual but **risks `�`
  mid-character glitches for CJK** — a character spans multiple BPE tokens — so the throttle is the safe pick.

## Op note: a wedged GPU looks like a code regression
After a day of repeated 1 GB+ bundle loads + GPU-heavy work, an iPhone can wedge: a model load hangs at
cold-compile and `devicectl` reports the console connection invalidated. **Reboot the device** — it clears
the Metal/GPU state and the load returns to normal. Before blaming code, confirm the load path is unchanged (here the only edits
were UI-layer; the engine/backend were untouched) — heavy on-device sessions need an occasional reboot.
