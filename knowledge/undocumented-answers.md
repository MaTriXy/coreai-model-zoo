# Questions Apple's Core AI documentation doesn't answer — with measured answers

Apple documents the API surface well: what `state_names` is for, what `MutableViews` binds, how
`AIModel.specialize` and `coreai-build compile` are invoked. What it does not document is what
happens when you run the thing — the thresholds, the platform differences, the failure modes.

The questions below are ones a careful reader reaches and cannot resolve from the official
documentation. Each answer here is measured, on stated hardware, with a link to the note or the
run that produced it. Where the honest answer is "it depends, and here is the shape of the
dependence", it says that rather than inventing a number.

---

## When do I need to ahead-of-time compile, instead of letting the model specialize on device?

Apple says AOT is "optional but recommended for large models" and publishes no threshold.
Measured rule of thumb, from ports that shipped:

- **≥ 1 GB → AOT.** On-device JIT specialization of a graph that size stalls or gets killed.
- **≤ ~50 MB → JIT is fine.** No meaningful first-load penalty.
- **In between → try it**, and measure first load on the oldest device you support.

For a 4B-class GPU bundle, AOT is not optional in practice: it must be compiled **per device
class** and shipped as `.aimodelc`
(`xcrun coreai-build compile … --preferred-compute gpu --architecture h18p`, where `h18p` is the
iPhone 17 Pro architecture name — it tracks the device identifier, not the marketing name).

Source: [`PORTING.md` §8](../PORTING.md), [`aot-and-specialization.md`](aot-and-specialization.md).

## Does iOS behave differently from macOS for a dynamically-sized KV cache?

**Yes, and it is a correctness difference, not a performance one.** On iOS the on-device
compiler miscompiles dynamically-sized-KV specializations at sequence length ≥ 2048: the output
is corrupt from the first generated token, not degraded gradually.

The workaround in the shipped engine is to cap a pipelined turn's KV pre-grow at capacity 1024
on iOS for those bundles, with the app clamping its own response budget as a second line. Both
guards come out when the compiler fix lands.

Tracked in [coreai-kit#5](https://github.com/john-rocky/coreai-kit/issues/5); upstream
[apple/coreai-models#124](https://github.com/apple/coreai-models/issues/124).

## Can I run a 4B model on the Neural Engine?

At that size, measured: no. The ANE bundle for a 4B model **static-loads** — 31 ANE regions,
about 518 s cold — and then the warmup *inference* fails with `com.apple.appleneuralengine` /
`ANECompilerService` `Code=4097` ("ANE compile failed"). The GPU AOT bundle is the only
on-device path at that size.

This is not an argument against the ANE in general — it is an energy story at smaller sizes and
under different authoring rules (static shapes, palettized weights, per-head layouts). See
[`compute-units-and-authoring.md`](compute-units-and-authoring.md) for which rules apply where,
and [`performance-ceiling.md`](performance-ceiling.md) for why ANE is a tokens-per-joule play
rather than a speed one.

Source: on-device runs, 2026-06-27, [`aot-and-specialization.md`](aot-and-specialization.md).

## What does the chunk threshold actually change?

`llm-runner --help` hints "use 128 for MoE", which reads like a speed setting. It is a **memory
dial**. Measured on a 128 GB M4 Max, gpt-oss-20B, 4096-token prefill, 3 trials:

| `COREAI_CHUNK_THRESHOLD` | Prefill tok/s | Peak dirty footprint |
|---|---|---|
| 128 (the MoE hint) | 766 | **1.7 GB** |
| 1024 (default) | 1237 | not measured |
| 8192 (no chunking) | **1439** | **18.0 GB** |

Unchunked MoE prefill allocates very large expert activations. On a 16–32 GB Mac that swaps or
gets jetsammed, which is what the hint protects against; on a big-RAM Mac, raising the threshold
buys ~16% prefill over the default for free. **Decode is unaffected** either way (~76–78 tok/s
across all three).

Source: [`apple-models-bench.md`](apple-models-bench.md), with the repro command.

## Why does my hand-written Metal kernel force a single-token export?

A kernel written for `M=1` implies an `S=1` export: `input_ids` pinned to `[1,1]` for every
step. The engine then walks the prompt one token at a time, and prefill pays for it — this is
the kernel's constraint, not Core AI's.

It is escapable. Tiling the kernel and emitting a second entry point for chunked prefill lifted
that tax by **5.87×** on an 8B ternary model (57.7 → 338.4 prefill tok/s, argmax identical).

Source: [`bitcpm-ternary-1.58bit.md`](bitcpm-ternary-1.58bit.md) §3,
[`ternary-chunked-prefill.md`](ternary-chunked-prefill.md).

---

## What this page is not

It is not a substitute for Apple's documentation, and it does not correct it — everything here
sits *downstream* of the official API. If you are looking for what a symbol does, read Apple's
reference. (Its pages are client-rendered, so a plain fetch returns no body; the same content is
served as JSON from `developer.apple.com/tutorials/data/documentation/<path>.json`, which is
worth knowing if you are reading it with a tool.)

Every claim above is dated and attributed because a measurement without a machine, an OS build
and a date is a rumour. Where a beta fixes one of these, the note changes and this page follows.
