# MinerU2.5-Pro → Core AI — port notes

**Model**: [`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B)
(Apache-2.0, 1.2B). Whole-page document parsing: layout detection → per-region recognition →
`json2md`, all in one **stock Qwen2-VL** (`Qwen2VLForConditionalGeneration`, no custom code). The zoo's
third doc-OCR, and the first that folds whole-page auto-structuring into the model weights (per-region
OCRs [Unlimited-OCR](../zoo/unlimited-ocr.md) / [GLM-OCR](../zoo/glm-ocr.md) leave layout to a separate
detector).

## What made it easy

It is a plain Qwen2-VL, so it rides the shipped [GLM-OCR](../zoo/glm-ocr.md) / Qwen3-VL rider contract
(`image_embeds` + `rope_shift_start`/`rope_shift_amount` static-input hook) with the Qwen2 specifics
swapped in. Checkpoint keys map almost 1:1: the decoder is stock `model.*` Qwen2, the vision tower is
`visual.*` with only `patch_embed.proj` (Conv3d) → `patch_proj` (Linear) reshaped.

`coreai_models/models/macos/mineru.py` = `qwen3_vl.py`/`glm_ocr.py` with these diffs:

- **Decoder (Qwen2, not Qwen3/GLM)**: separate q/k/v **with bias**, **no q/k-norm**, `o_proj` no bias;
  **sectioned M-RoPE `[8,12,12]`** (sum = head_dim/2 = 32) applied **split-half** — build the per-axis
  freqs `cat([f_t[:8], f_h[8:20], f_w[20:32]])`, then `emb = cat([freqs, freqs])` and Qwen standard
  `rotate_half` (`[-x2, x1]`). This is *not* GLM's interleaved-pair rotate and *not* Qwen3-VL's `j%3`
  interleave. Standard 2-norm block, silu SwiGLU, `tie_word_embeddings=True` (no `lm_head.weight` in
  the checkpoint — confirmed).
- **Vision (Qwen2-VL ViT)**: LayerNorm blocks, fused qkv (bias), no q/k-norm, split-half 2D rope
  (baked); MLP is **non-gated `fc1 → quick_gelu → fc2`** (`x*sigmoid(1.702x)`); merger is the standard
  Qwen2-VL `PatchMerger` (`ln_q` over embed_dim → `view(embed_dim·merge²)` → `Linear → GELU → Linear`,
  out = decoder hidden). **No deepstack, no learned pos-embed.** ViT working dim is `embed_dim` (1280),
  *not* `hidden_size` (896 = merger output = decoder hidden); vision head_dim = 1280/16 = 80.

## Gate (single venv, no dump bridge)

`coreai-models/.venv` carries **both** `coreai_torch` and transformers 4.57.6 with `Qwen2VL`, so the
torch ladder runs the HF reference and the re-authored model side by side in one process — no
reference-dump bridge (`_mineru_ladder_check.py`). Result vs HF fp32 (ReadDoc/sample.png): text-only +
vision + full-VLM argmax **706/706**, max logit diff **0.0001**, generated token bit-identical. (The
flattened `cosine_similarity` over the 107M-element logits reads 1.07 — a reduction artifact; per-position
cos min is 0.999987.)

## macOS 27 decode gate — the AOT / `expect-frequent-reshapes` lever

Raw Python `rt.AIModel.load` of the dynamic-shape S=1 decode graph **wedges** on macOS 27: GPU-preferred
load routes to `ANECCompile` → *"MLIR MPS to ANEC conversion failed"* → repeated `MTL4CommandQueueError`
(the 90 s-watchdog hazard; kill immediately). `cpu_only()` fails fast with `CoreAICompiler error 2`.

Root cause (matches [`qwen3.6`](../zoo/qwen3.6.md)): the Swift engine loads dynamic graphs with
`SpecializationOptions(preferredComputeUnitKind: .gpu)` **plus `expectFrequentReshapes = true`**, which
steers the graph off the ANE path — but the **Python runtime binding doesn't expose that flag**, so raw
Python load can't replicate it.

Fix: **AOT-compile with the flag baked in** —
`xcrun coreai-build compile decoder.aimodel --architecture h16c --preferred-compute gpu --expect-frequent-reshapes`
(h16c = M-series Mac, h18p = iPhone 17 Pro). The precompiled `.aimodelc` embeds the h16c GPU MPSGraph
delegate; loading it runs on Mac GPU with no re-specialization and no wedge. Teacher-forced over the
706-position reference: **text region 24/24 exact**, generated token exact; **211.7 tok/s** decode; real
autoregressive greedy reproduces the whole page correctly.

## Quantization

`int8lin` (per-block-32 symmetric int8 body; embedding + tied `lm_head` stay fp16) is the ship floor:
**0 text-region argmax flips vs fp16** (all 13/706 flips at visual-token positions, never generated),
greedy generation byte-identical to fp16. int4 not shipped (cliff risk on 1.2B without QAT).

## 2-stage structured pipeline (the real value) — three non-obvious gotchas

The whole-page *structuring* (tables as `<table>` HTML) is a host-orchestrated 2-stage over the same
weights. Getting it onto Core AI's **fixed-grid** export surfaced three traps that each looked fatal:

1. **Layout needs a 1036² *square* grid, not the 768 recognition grid.** `Layout Detection:` at 768
   portrait returns garbage (dozens of bogus `header` blocks); at 1036² square (37×37 = 1369 tokens,
   the reference `layout_image_size`) it returns the correct blocks. The page is **stretched** (not
   letterboxed) into the square, so the 0–1 boxes map linearly onto the original page — crop directly,
   no letterbox inverse. → a **separate export** (`--grid-h 37 --grid-w 37`).
2. **`Table Recognition:` emits OTSL, not text or HTML.** The cells come back as **special tokens**
   `<fcel>` (cell) / `<ecel>` (empty) / `<nl>` (row). Decoding with `skip_special_tokens=true` silently
   drops them → concatenated cell text with no structure. Keep the special tokens, then convert OTSL →
   HTML (`mineru-vl-utils`'s `convert_otsl_to_html`; `<table>` is **post-processed**, not model-emitted).
3. **Recognition stays on the 768 grid.** A small region letterboxed into the 1036² square becomes
   unreadable (text too small) → garbage. Text + tables both read correctly at 768. So the reader loads
   **two bundles** (layout 1036² + recognition 768) and routes each stage to its own.

Verified end-to-end in the ReadDoc **Mac** app (`KitMineruReader.readStructured`, two `VLRuntime`s):
an 8-block page → title/paragraphs + a 5-row `<table>`, byte-identical to the reference. Single-pass
(768 only) still runs on iPhone; the 1036² layout grid makes the 2-stage Mac-tier.

## Artifacts

- Model: `coreai_models/models/macos/mineru.py`; export: `conversion/export_mineru_pipelined.py`.
- Bundles: recognition 768 (`vision/` + `decoder/`) + layout 1036² (`layout/`), vision fp16 + decoder
  int8lin, `pf64` chunked prefill; AOT `h16c` (Mac) + `h18p` (iPhone).
- Host 2-stage (layout → crop → recognition → `json2md`) + OTSL→HTML: source of truth =
  [`mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils) (`two_step_extract`,
  `post_process.otsl2html`); Swift `KitMineruReader.readStructured` is the app port.
