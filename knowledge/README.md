# Core AI knowledge base

Hard-won, verified notes on Apple's Core AI (iOS/macOS 27) — what the docs don't spell out.

## Orientation
- [`coreai-overview.md`](coreai-overview.md) — what Core AI is, the 3 Apple repos, the `.aimodel`
  format, the PyTorch → `.aimodel` → Swift-runtime pipeline.
- [`conversion-guide.md`](conversion-guide.md) — converting a PyTorch model to `.aimodel`: the
  canonical `TorchConverter` API + the gotchas that cost real time.
- [`compute-units-and-authoring.md`](compute-units-and-authoring.md) — **ANE vs GPU vs CPU**: the
  static/BC1S/Conv2d/per-head/fp16 ANE rules vs the dynamic/fused/custom-kernel GPU rules, the
  macOS↔iOS export split, and the PSNR verification gates. Read this to choose a target.
- [`performance-ceiling.md`](performance-ceiling.md) — **reality check**: where Core AI LLM decode tops out
  (Mac GPU near its ceiling, MLX gap structural, fusion closed), what AOT does/doesn't do, and why ANE is an
  energy play not a speed one. Read before chasing a "dramatic speed" win.

## GPU-now track (speed)
- [`pipelined-engine.md`](pipelined-engine.md) — **read this first for decode speed**: riding Apple's
  `coreai-pipelined` engine = 3.5× over a hand-rolled per-token loop with ZERO custom kernels
  (qwen3.5: Mac 204 tok/s, iPhone 50.3–51.5). The decode-only loop-free export, the extra-states
  engine patch, the chunk=1 / warmup-256 traps, LUT-vs-linear int8, oracle gating, and what
  fits/doesn't (Gemma 4's PLE doesn't — yet).
- [`fm-provider.md`](fm-provider.md) — **zoo models behind Apple's `LanguageModelSession`**
  (WWDC 339): `CoreAILanguageModel(resourcesAt:)` = the whole integration (verified, incl.
  hybrid/SSM bundles on the patched pipelined engine), plus the own-conformance recipe that
  adds **tool calling** (verified round trip) and the protocol traps (dead prewarm, no guided
  generation on pipelined, re-prefill tax).
- [`custom-metal-kernels.md`](custom-metal-kernels.md) — `TorchMetalKernel` (WWDC 325): the API, the
  register-then-add order, MSL embedded in the `.aimodel`, GPU-only, what to (and not to) kernelize.
  Still the tool when a model CAN'T ride the pipelined engine (e.g. Gemma 4).
- [`tensorops-quantized-kernels.md`](tensorops-quantized-kernels.md) — the layer UNDER custom kernels
  (WWDC 330): TensorOps quantized matmul (int4/int8 = OS 26; fp4/fp8/int2 + E8M0 scale planes = OS 27),
  cooperative tensors, the FlashAttention recipe, and the M5/A19 GPU **neural accelerator** — the
  compute-bound/prefill lever hand-rolled MSL can't reach.

## ANE-later track (when the beta KV-write bug lifts + int4 head + AOT)
- [`aot-and-specialization.md`](aot-and-specialization.md) — specialization, `AIModelCache` /
  `AIModel.specialize()`, and AOT compile (`xcrun coreai-build compile` → `.aimodelc`,
  `--preferred-compute neural-engine`). The first-run-latency mitigation path.
- [`compression-reference.md`](compression-reference.md) — `coreai-opt` quantization & palettization
  API reference (int4/int8, granularity, mixed-precision, joint); the LM-head/embedding lever.
- [`coreai-beta-mpsgraph-kvwrite-bug.md`](coreai-beta-mpsgraph-kvwrite-bug.md) — the data-indexed
  in-graph KV write SIGSEGV (FB23024751 / apple#5): platform-agnostic (GPU too), host-cache workaround.

## Agent & app layer (FM framework / App Intents / Evaluations / security)
- [`spotlight-rag-third-party.md`](spotlight-rag-third-party.md) — running Apple's WWDC26
  `SpotlightSearchTool` (local RAG as one Tool) behind a **third-party zoo model** via
  `KitLanguageModel`: only `.toolCalling` is needed (not guided generation), the tool returns
  metadata-not-body (hydrate with a companion `fetch_note` tool), guidance level is a token gate,
  and the thinking-model `/no_think` mitigation. Verified example: `coreai-kit/Examples/SpotlightChat`.
- [`dynamic-profiles-local-models.md`](dynamic-profiles-local-models.md) — WWDC26 `DynamicProfile`
  (242) routing between **two local zoo models** (0.6B triage ↔ 4B expert) in one
  `LanguageModelSession`, fully on-device/airplane-mode — the config Apple's on-device↔PCC demo
  doesn't show. The body-purity rule, switch re-prefill cost, two-resident-model footprint, and
  why the model-decision channel must be guided-gen (not a tool) on the stock engine. Example:
  `agent-demos/DualProfileChat`.
- [`visual-intelligence-third-party-model.md`](visual-intelligence-third-party-model.md) — running
  YOUR own model (CLIP / RF-DETR) behind the system **Visual Intelligence** camera/screenshot search
  (WWDC26 297): `IntentValueQuery` + `SemanticContentDescriptor`, model-agnostic by construction (no
  model param, no capability, no entitlement), and the real gate — running a model in the query's
  background-launch memory budget. Example: `coreai-kit/Examples/VisualIntel`.
- [`agentic-security-checklist.md`](agentic-security-checklist.md) — pre-ship checklist for
  on-device LLM agent apps (WWDC 347+343): indirect prompt injection, the Lethal Trifecta,
  `.onToolCall`/`.historyTransform` guardrails, App-Intents risk-based confirmation +
  `authenticationPolicy` + `OwnershipProvidingEntity`.
- [`evaluations-framework.md`](evaluations-framework.md) — Apple's Evaluations framework
  (WWDC 298/299/335) mapped to this project's oracle/margin gates; the `disallowed`-trajectory
  injection test; a Vault-style on-device eval suite.

## Runtime & decode internals
- [`compression.md`](compression.md) — this project's LLM-specific empirical compression notes
  (int8 floor, per-subsystem sensitivity); pairs with `compression-reference.md`.
- [`stateful-kv-cache.md`](stateful-kv-cache.md) — stateful decode export, dual/hybrid KV state,
  the sliding-window ring buffer, the dynamic prefill+decode graph.
- [`swift-runtime.md`](swift-runtime.md) — the Core AI Swift API, driving `.aimodel` from Swift,
  non-standard architectures, macOS/Xcode 27 setup (incl. running Xcode 27 beta without sudo).

## Audio (the new modality — model-specific port notes)
- [`qwen2.5-omni-audio-understanding.md`](qwen2.5-omni-audio-understanding.md) — Qwen2.5-Omni's
  **Thinker** as on-device **audio understanding** (describes sounds, *not* ASR): de-dynamizing the
  Whisper-style encoder (ragged `cu_seqlens` chunk attention → one batched fixed attention, bit-exact),
  TMRoPE collapsing to 1-D, a bit-exact vDSP mel (DFT-as-matmul), audio embeds on a static buffer, and
  the two payoffs — **AOT clean-mmap weights dodge the iOS jetsam dirty limit** (4.5 GB decoder, 5930 MB
  free), and a **fixed-shape encoder runs on the ANE** (0.99 cos → byte-identical text) where the
  dynamic decoder can't.
- [`kokoro-tts.md`](kokoro-tts.md) — Kokoro-82M (StyleTTS2 + iSTFTNet), the zoo's first **text-to-speech**:
  3 bundles cut at the one data-dependent length + host DSP, the `weight_norm`-random-init bug
  (non-determinism → suspect weight loading), variable length via **masked-unrolled bi-LSTM +
  frame-masked InstanceNorm + host STFT**, the Core AI op rewrites (ConvTranspose1d/iSTFT → zero-insert
  + conv1d), and why an unrolled-LSTM model runs **faster on the CPU than the GPU**.
- [`adcsr-super-resolution.md`](adcsr-super-resolution.md) — AdcSR, the zoo's first **super-resolution**
  (one-step diffusion-GAN, pruned SD-2.1 + half VAE decoder): why fp16 NaNs (SD-2.1 attention overflow,
  `upcast_attention` ignored by the SDPA processor) so **fp32 ships**, the **global** (not per-tile)
  color-match, host-side tiling with an input-size cap, and the CoreGraphics `bytesPerRow`/no-y-flip traps.

Primary official sources behind these notes: the open repos (`coreai-torch`, `coreai-optimization`,
`coreai-models` incl. its agent skills), the WWDC26 talks **324 / 325 / 326 / 330** (verbatim transcripts in
`ondevice/_wwdc{324,325,326,330}_transcript.txt`), and `developer.apple.com/core-ai/`. Verified against
Hugging Face references (convert + numeric parity); on-device iOS 27 notes marked where still in bring-up.
