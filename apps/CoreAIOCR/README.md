# CoreAIOCR — Unlimited-OCR on-device document OCR (macOS)

On-device document → structured-markdown OCR (tables / LaTeX / read-order) running the
**baidu/Unlimited-OCR** (3B-A0.5B MoE, MIT) model entirely on Core AI. Drop a document
image, get markdown.

Drives the **stock `coreai.runtime` directly** (no engine patch): a stateless DeepEncoder
vision `.aimodel` (`CoreAIKitVision.GraphModel`) + a stateful, `inputs_embeds`-driven
decoder (`CoreAI.AIModel`, unified `prefill`/`decode` functions, KV-cache state via
`InferenceFunction.MutableViews` — the `CoreAISequentialEngine` pattern). The pipeline is a
1:1 transcription of the verified Python reference `_unlimited_ocr/_ocr_pipeline.py`.

Pipeline: `CGImage → preprocess(640, norm .5) → vision .aimodel [1,100,1280] → arrange
(10×10 + image_newline rows + view_seperator) → scatter into embed_tokens(prompt) →
prefix [1,115,1280] → prefill + greedy decode → detokenize → markdown`.

## Build (on device — the model pipeline is validated; the app needs a Mac build)

```sh
cd apps/CoreAIOCR
xcodegen generate          # needs xcodegen
open CoreAIOCR.xcodeproj    # Xcode-beta (macOS 27 SDK); set DEVELOPMENT_TEAM, build & run
```

## Assets — stage into `~/Library/Application Support/CoreAIOCR/`

Generate from the port (`_unlimited_ocr/`, run in `coreai-models/.venv`):

| file | produced by | size |
|---|---|---|
| `_vision_export.aimodel` | `_ocr_vision_export.py` | 762 MB |
| `_dec_unified.aimodel` (functions: `prefill`, `decode`) | `_ocr_decoder_export.py --unified` | 3.2 GB |
| `embed_tokens.f16` / `image_newline.f16` / `view_seperator.f16` / `prompt_input_ids.i32` | `_ocr_arrange_verify.py` (→ `out/_swift_assets/`) | 331 MB + tiny |
| `tokenizer/tokenizer.json` | copy from `ckpt/` | — |

(For shipping, download these from an HF repo instead of staging by hand — follow the
`ModelCatalog`/`ModelDownloader` pattern in `CoreAIChatMac`.)

## Validation status (all on stock `coreai.runtime`, GPU)

- Vision `.aimodel`: engine cos 1.00000 vs fp32 oracle.
- Decoder: engine gate flips 0/9, decode **flat ~12.7 ms/token**; real autoregressive
  generation = fp32-oracle-identical structured OCR (`<table><tr><td>…`, all numbers).
- Arrangement recipe: reconstructs the oracle prefix exactly (cos 1.000000, |Δ|=0).
- Full image→markdown via engine: 386 tokens / 7.8 s, correct.

## Notes for the device build

- `CoreAI` is the system runtime framework (no SPM package); `CoreAIShared`
  (`fillNDArray`/`readNDArray`) comes from `coreai-models`, `GraphModel`/`TensorValue`
  from `coreai-kit`, `Tokenizers` from `swift-transformers`.
- Verify `Preprocess.toCHW640` once against `out/_vision_oracle` (`sam_in0`): the engine
  pipeline is exact, so any small drift would be in that CPU resampling only.
- `recognize()` uses pure greedy. The oracle used `no_repeat_ngram=35`; add the same
  block in the decode loop if you want byte-identical table markup on repetitive tables.
