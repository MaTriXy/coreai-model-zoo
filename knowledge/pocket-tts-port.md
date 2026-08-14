# pocket-tts (Kyutai) on Core AI — port notes

Engineering notes from porting [`kyutai/pocket-tts-without-voice-cloning`](https://huggingface.co/kyutai/pocket-tts-without-voice-cloning)
to Core AI: the zoo's **first Kyutai model** and its **first Mimi conversion**. Ported by
[Rahul Rachuri](https://github.com/RahulRachuri) ([PR #12](https://github.com/john-rocky/coreai-model-zoo/pull/12));
card at [`models/pocket-tts/README.md`](../models/pocket-tts/README.md), host at
[pocket-tts-swift](https://github.com/RahulRachuri/pocket-tts-swift). Drafted here from his card
and PR comments, so corrections from him overrule this file.

License: **CC-BY-4.0**, inherited from the ungated checkpoint. Upstream weights are not mirrored —
the host reads Kyutai's own `model.safetensors`, `tokenizer.model` and per-voice embeddings.

## Shape of the port

| stage | what it is | Core AI form |
|---|---|---|
| **flow-LM** | AR transformer over Mimi latents at 12.5 Hz | one asset, **two functions over one shared rank-5 KV state**: `prefill` and `step` |
| **flow decoder** | one-step flow matching, cond + noise → latent[1,32] | stateless graph |
| **Mimi decoder** | streaming codec → 24 kHz PCM | graph with **12 in-graph state tensors**, rescale + k=1 quantizer folded in |
| host | sentencepiece, chunker, EOS compare, seeded noise, AR loop | Swift |

## `S_MAX` is derived from the model, not configured

The KV cache has no context window, so the capacity is a property of the model's own generation
bound: worst legal chunk is 162 voice + 50 text + 234 generated = 446, rounded to **512**. A
fixture-sized 256 looks fine on every short test and **silently truncates real sentences** — only
an end-to-end sweep caught it. When a family has no window, compute the bound; do not pick a
number that passes the fixtures you have.

## Prefill as a windowed second function — a second escape from the SDPA lowering crash

Prefill here is a second function over the shared state at a **static width of 16 tokens**,
applied as a sliding window. That is a design choice for state sharing, but it also means the
query block never gets large enough to stress the coreai SDPA composite, so the
`AICode→MPS lowering failed` crash from [`lfm2audio-port.md`](lfm2audio-port.md) never fires.

So there are two escapes from that crash, not one: rewrite as raw matmul-softmax with an explicit
additive mask, **or** keep the query block small by windowing. A 50-token chunk costs four prefill
calls and prefill is 2.1 % of wall time, so the windowing costs nothing worth reclaiming.

## Fixed-capacity KV caches must be zero-initialised, never NaN-initialised

The finding this port contributes, and the one most likely to bite the next Moshi-class cache.

Upstream fills the KV cache with `float("NaN")` in two places and gets away with it by **slicing
the unwritten tail off before attention**. A fixed-capacity graph cannot slice. A masked SDPA
still multiplies V by a zero weight, and `0 * NaN = NaN`, so the poison spreads through the cache
instead of being masked away. It fails silently rather than loudly, which is what makes it
expensive. FluidAudio hit the same thing independently on their Core ML route and scrub with
`where(isnan(keys), 0, keys)`.

**This is a different mechanism from the NaN entries already in this knowledge base**, which are
all fp16 overflow. Same symptom, different cause, different fix: overflow wants a precision or a
clamp, this wants an initialiser.

## `preferredComputeUnitKind: .cpu` is a partitioning hazard, not a compute-unit choice

Verified twice, on unrelated architectures. **The trigger is a preference expressed over a
heterogeneous allowed set, not an op.**

- `.gpu` and `.cpu` report **identical** `allowedComputeUnitKinds` of `[cpu, gpu, neuralEngine]`
  and differ only in which unit is *preferred*. `cpu_only()` collapses the set to `[cpu]` and is
  exact. Confirmed here in the Python bindings, where the three options print
  `Allowed: [CPU, GPU, Neural Engine] / Preferred: CPU`, the same with `Preferred: GPU`, and
  `Allowed: [CPU] / Preferred: None`.
- It scales with **how much there is to partition**, not with size or dtype. In the pocket-tts
  repro a 6-layer prefill over 16 positions and a multi-stage convolutional decoder both fail,
  while the same transformer at 1 position and a small stateless function are correct.
- What settles placement over arithmetic: under `.cpu` the first four samples of a 1920-sample
  frame are **bit-identical** to `.gpu` and `.cpuOnly`, and the divergence starts later in the
  same frame. Precision loss does not leave a bit-identical prefix.

**Blast radius varies enormously.** On pocket-tts's fp32 assets it reaches cosine mean 0.501,
min 0.222, max abs 2.117, with 27 of 32 end-of-sequence decisions flipped — anti-correlated, not
imprecise. On the zoo's own Kokoro the same path stays at rounding scale: of the three published
graphs only the **predictor** diverges (max|Δ| 5e-5 to 3.3e-4 on `duration`, min cosine 1.000000),
and the prosody and vocoder graphs are bit-identical. Kokoro's predictor is the unrolled-bi-LSTM
graph — hundreds of tiny sequential ops, so the most partition boundaries — while the two larger
graphs are convolutional. That is the scaling claim confirmed on a second model: **fragmentation,
not size.** It does not move with content either; 16 real tokens and 120 give the same order.

Kokoro is not mis-shipping on this: `duration` feeds an integer frame-count rounding, the largest
delta measured is 1.9e-5, the closest element sits 4.97e-3 from a rounding boundary, and the
rounded frame counts come back equal. Two orders of magnitude of headroom.

**Rule.** Use `SpecializationOptions.cpu_only()` for parity work, never `preferred .cpu`, and do
not ship `.cpu` in a host. Collapsing the allowed set to one unit is the only thing that reliably
closes this. Note that CoreAIKit's `GraphModel(computeUnits:)` currently offers only
`.neuralEngine / .gpu / .cpu`, so `cpuOnly` is not expressible through the kit at all — see
[`compute-units-and-authoring.md`](compute-units-and-authoring.md).

**Possibly the same defect as the `ANERegionFormationPass` abort below**: it fires under the same
condition and also disappears under `.cpuOnly`. Both symptoms vanish once the allowed set
collapses to a single unit.

## The other Core AI defects this port found

All verified with minimal repros. Two are filed with Apple (1, 2); the rest are written up.

1. **`ConvTranspose1d` is numerically wrong on the `cpu_only` delegate** at stride ≥ 8 and
   kernel ≥ 16 — the same asset is correct on gpu. Mimi's k=32/s=16/groups=512 upsample hit it at
   cos −0.01. Escape: at T=1 with `groups == channels` the op degenerates to a per-channel outer
   product, and re-authoring it that way is bit-exact on `cpu_only`. *(Filed.)*
2. **The Python `coreai.runtime` bindings leak one IOSurface per call** — with this pipeline's
   state sizes the process dies at ~2,250 calls, climbing 1.9 MB per call to ~3 GB. Run Python
   harnesses in subprocess workers on a call budget. The Swift `CoreAI` framework does not leak:
   4,773 calls flat at 300 MB. *(Filed.)*
3. **coreai-torch lowers ConvTranspose `output_padding` by padding the input**, silently producing
   the wrong output length. A converter defect, distinct from the *symbolic* output length
   [`kokoro-tts.md`](kokoro-tts.md) records for the same op.
4. **fp32 graphs with in-graph state abort under the default `SpecializationOptions`** — SIGABRT
   in `ANERegionFormationPass` instead of falling back. State an explicit `gpu` or `cpuOnly`
   preference on every load; see the placement section above.
5. `.cpu` numerics — above.

## Worth copying: gate the audio, not the tensors

Tensor similarity passes on audio nobody can understand, so this port's acceptance ladder ends in
an **ASR round trip** rather than a cosine: 0.00 % WER on the oracle prompt, 1.38 % on a 148-word
paragraph, 4.27 % across a 302-sentence sweep over all eight voices, 2.53 % on a ten-minute
long-form run. The short-utterance tail is upstream's, shown by reproducing it at the same rate in
seed-matched PyTorch — which is the part that turns "our port is fine" into evidence.
