# MinerU2.5-Pro (1.2B) — Core AI whole-page document parsing

**On-device whole-page document structuring on Core AI.** A port of
[`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B)
(**Apache-2.0**, 1.2B) — a small, SOTA-quality document parser (OmniDocBench v1.6 **95.69**). Unlike a
per-prompt recognizer, MinerU2.5 does the *whole page in one model*: a **layout** pass finds the
blocks (title / text / table / formula / figure) and their reading order, then a **recognition** pass
reads each region into plain text, HTML tables (`<table>…`), or LaTeX — assembled into structured
Markdown / JSON. This is the piece the zoo's per-region OCRs ([Unlimited-OCR](unlimited-ocr.md),
[GLM-OCR](glm-ocr.md)) leave to a separate layout detector; MinerU folds it into the same weights.

MinerU2.5 is **stock Qwen2-VL** (`Qwen2VLForConditionalGeneration`): a Qwen2-VL ViT vision tower + a
Qwen2-0.5B text decoder, no custom code. This port is the shipped [Qwen3-VL](qwen3-vl.md) /
[GLM-OCR](glm-ocr.md) vision idiom (the `image_embeds` + rope-shift static-input hook) with a Qwen2
text decode — **no deepstack, no MoE, no MLA**. The vision tower runs once; its `image_embeds` are
injected at the image-placeholder positions (`V + slot`, row-major over the merged grid) and the text
decodes on top. The host runs the model twice per page (layout prompt, then recognition prompts).

**⬇️ Converted `.aimodel` bundles:
[mlboydaisuke/MinerU2.5-Pro-CoreAI](https://huggingface.co/mlboydaisuke/MinerU2.5-Pro-CoreAI)** —
`vision/` + `decoder/` (768-token **recognition** grid, Qwen2-VL ViT fp16 + int8lin decoder, pf64
chunked prefill) + `layout/` (a 1036² **square** recognition/detection grid for the 2-stage) +
`tokenizer/`. Apache-2.0.

## Architecture

- **Vision (Qwen2-VL ViT)**: embed 1280 / 32 L / 16 heads (head 80) / patch 14 / temporal 2 /
  spatial-merge 2, out 896. **LayerNorm** blocks, fused qkv (bias), **no q/k-norm**, non-gated
  `fc1 → quick_gelu → fc2` MLP, and the standard Qwen2-VL `PatchMerger`
  (`ln_q → view(merge²) → Linear → GELU → Linear`). **No deepstack, no learned pos-embed** — just
  baked 2D-rope constants. Exported as one fp16 `.aimodel`; `N` (visual tokens) is fixed by the
  export grid.
- **Decoder (Qwen2-0.5B)**: hidden 896 / 24 L / GQA 14-2 / head_dim 64 / vocab 151936, `tie_word_embeddings`.
  Separate q/k/v **with bias**, **no q/k-norm**, standard 2-norm block, silu SwiGLU, **sectioned
  M-RoPE `[8,12,12]`** applied split-half (Qwen standard `rotate_half` — not GLM's interleave). Driven
  on the pipelined-engine S=1 contract: `input_ids [1,1]`, `position_ids`, static `image_embeds [N,896]`
  + `rope_shift_start` + `rope_shift_amount`. Zero embeds + `shift_start = 1<<30` → a plain Qwen2 text
  decoder.

## Verified (M4 Max + iPhone 17 Pro, GPU, Core AI pipelined engine)

- **iPhone 17 Pro, on-device, whole page read correctly** — the h18p AOT bundles run through
  `KitMineruReader` in a real app (ReadDoc): a page photo → structured text (headings, paragraphs,
  tables) with nothing leaving the device. **~4 s/page** (warm) via **chunked prefill**: the `pf64`
  multifunction bundle feeds the 768 image tokens in **static S=64 chunks** (the engine auto-discovers
  the `prefill` function) then decodes S=1 — the S=1 prefill (~10 s) drops to **~2.9 s**, no token cut.
- **End-to-end real generation on the engine (shipped portrait 768 config): the sample page read
  verbatim** — letterbox → CLIP-norm → non-square patchify → GPU vision `.aimodel` → `image_embeds`
  → AOT-compiled (h16c) int8lin S=1 decoder, autoregressive greedy: title + paragraph + the **full
  table reconstructed row-by-row** (*"Quarterly Report / On-device inference shipped across all
  product lines this quarter… / Whisper 809M 0.18 s/token / …"*), matching the fp32 reference.
  **211.7 tok/s** decode (int8lin S=1, AOT h16c GPU). Portrait vision vs HF: per-token cos min
  **0.9975**.
- Torch ladder vs HF fp32: text-only + vision + full-VLM argmax **706/706 exact**, max logit diff
  **0.0001**, the generation-driving token bit-identical.
- Engine gate (Mac GPU): vision `image_embeds` cos **1.0002**; AOT int8lin decoder teacher-forced over
  706 positions — **text region 24/24 exact**, generated token exact.
- **int8lin vs fp16: 13 / 706 argmax flips, all at visual-token positions (0 in the text region)** —
  the OCR text is preserved. Greedy generation byte-identical to fp16.

## Run in app — `KitMineruReader` (Mac; iPhone single-pass)

Both grids ride the kit's VL rope-shift runtime (`VLRuntime`). Pages are CLIP-normalized and
patchified (the `VLImagePreprocessor` non-square + `.stretch`/`.aspectFitPad` paths added for
Qwen2-VL). Two entry points:

```swift
// Single-pass (768 grid): whole page → plain text (reading order). Runs on iPhone (chunked prefill).
let reader = try await KitMineruReader(catalog: "mineru2.5-pro")
let text = try await reader.read(imageAt: documentURL)

// 2-stage (structured Markdown, tables as <table> HTML). Needs the layout bundle too.
let reader = try await KitMineruReader(
    visionDir: rec.vision, decoderDir: rec.decoder,
    layoutVisionDir: layout.vision, layoutDecoderDir: layout.decoder)
let markdown = try await reader.readStructured(imageAt: documentURL)
```

**2-stage** (`readStructured`, mirrors `mineru-vl-utils` `two_step_extract` + `json2md`):
1. **Layout** on the page **stretched into 1036² square** (`VLArchitecture.mineruLayout`, 37×37 = 1369
   tokens) → `Layout Detection:` emits `<|box_start|>…<|ref_start|>type…` blocks. A 768 portrait grid
   mis-detects here — the square high-res grid is required. Boxes are 0–1 of the square, so they map
   linearly onto the original page.
2. **Recognition** per region on the **768 grid** (`.mineru`): each block is cropped from the original
   and read by type — `Table Recognition:` emits **OTSL** (`<fcel>`/`<nl>`), kept (not skipped) and
   converted to `<table>` HTML; `Formula Recognition:` → LaTeX; else plain text.
3. **json2md** joins the region contents in reading order.

Verified end-to-end in the ReadDoc Mac app: an 8-block page → title/paragraphs + a 5-row `<table>`,
byte-identical to the reference. ~11 s load (both bundles) + ~11 s/page. The single-pass path also
runs on iPhone; the 2-stage's 1036² layout grid is Mac-tier (two bundles, heavy).

## Pipeline (host side)

```
page image → letterbox into 672×896 (aspect-fit + white pad) → CLIP-normalize
           → non-square patchify [3072, 1176] (block-major, 64×48 patches)
           → vision .aimodel → image_embeds [768, 896]
           → prompt: [ <|vision_start|>, <image>×768, <|vision_end|>, "Text Recognition:" ]
             (image ids → V+slot; shift_start = img_start+768, shift_amount = 768−32 = 736)
           → decoder S=1 pipelined greedy decode → tokens → detokenize → markdown / <table>HTML / LaTeX
```

The full 2-stage whole-page mode (layout boxes → per-region crop → recognition → `json2md`) follows
[`opendatalab/mineru-vl-utils`](https://github.com/opendatalab/mineru-vl-utils)
(`MinerUClient.two_step_extract` + `post_process.json2md`) — prompts by type: text →
`"Text Recognition:"` · table → `"Table Recognition:"` · formula → `"Formula Recognition:"` ·
figure → `"Image Analysis:"`.

## Use / reproduce

- **Convert**: [`conversion/export_mineru_pipelined.py`](../conversion/export_mineru_pipelined.py)
  (`fp16` / `int8lin` / `int8hu`; vision stays fp16).
- **Run (Mac)**: drive the S=1 decoder bundle on the pipelined engine with three static inputs
  (`image_embeds` + `rope_shift_start` + `rope_shift_amount`) and `COREAI_CHUNK_THRESHOLD=1`; feed the
  prompt with the image placeholders rewritten to `V+slot`. Large decode graphs need AOT on macOS 27
  (`xcrun coreai-build compile … --architecture h16c --preferred-compute gpu --expect-frequent-reshapes`);
  h18p bundles are prepared for iPhone.
- **Knowledge**: [`knowledge/mineru-port.md`](../knowledge/mineru-port.md).

## Notes

- **Whole-page structuring is in the model** (layout + per-region recognition, `json2md` reading-order
  assembly) — the value over per-region OCRs. The 2-stage orchestration is host-side (Python
  `mineru-vl-utils` is the source of truth; a Swift host is the app-integration step).
- **License**: base `MinerU2.5-2509` is AGPL-3.0 — this port uses **`MinerU2.5-Pro-2605` (Apache-2.0)**.
- **Appropriate input**: single-page documents; layout runs on a downsampled page, recognition on
  native-res crops (pick a larger export grid for dense small text).
- int4 not shipped (weight-only int4 without QAT risks a quality cliff on a 1.2B model). iPhone (h18p)
  throughput pending device verification.
- *Community port — not affiliated with Apple or OpenDataLab.*
