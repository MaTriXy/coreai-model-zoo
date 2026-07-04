# GLM-OCR → Core AI: port notes

GLM-OCR (`zai-org/GLM-OCR`, 0.9B, MIT) is a small OCR variant of **GLM-4.V** (`Glm4v`) — transformers'
`modular_glm_ocr.py` inherits every class from `glm4v`. So the port is the shipped
[Qwen3-VL](../zoo/qwen3-vl.md) vision idiom + a GLM text decode; no R-SWA, no MoE, no MLA.

## Model definition (`coreai_models/models/macos/glm_ocr.py`)

Adapted from `qwen3_vl.py`. The differences that matter:

- **Vision (CogViT)**: `RMSNorm` blocks (Qwen uses LayerNorm) + per-head q/k-norm + **gated-SiLU** MLP
  (Qwen: fc1/fc2 gelu-tanh) + Conv3d patch → linear bake + Conv2d `downsample` (2×2) → linear bake +
  a GLM patch merger (proj→LN→GELU→gated). **No deepstack, no learned pos-embed** — 2D-rope constants
  only. Output = merged `image_embeds [N,1536]`.
- **Decoder (GLM text)**: **sandwich norm** (4 RMSNorm/layer; Qwen: 2), separate q/k/v/o (no fused
  qk-norm), **fused `gate_up_proj`** split at load into the MLP primitive, and — the subtle one —
  **sectioned M-RoPE `[16,24,24]`** (chunk i → axis i%3) with **GLM interleaved rotate**
  (`rotate_half` on even/odd, `cos[:64].repeat_interleave(2)`). Qwen3-VL's `mrope_masks` (j%3
  interleave, split-half rotate) is a *different* layout — not reusable, needs its own implementation.
- **Loading**: keys under `model.language_model.` / `model.visual.`; the MTP layer (`layers.16`,
  `num_nextn_predict_layers=1`) is dropped; `lm_head` untied (vocab 59392).

Torch ladder (vs HF, fp32): decoder logits cos 1.000020, vision cos 1.000061, **full-VLM argmax
694/694**.

## Export + quantization

`conversion/export_glm_ocr_pipelined.py` (a `export_qwen3_vl_pipelined.py` clone minus deepstack:
single vision output `image_embeds`). Two artifacts: the fixed-grid vision `.aimodel` (fp16) and the
text decoder pipelined bundle (dynamic + S=1). int8hu = body int8 per-block-32 + untied head absmax;
vision stays fp16. Tokenizer: `AutoTokenizer.save_pretrained` throws (`TokenizersBackend`) — copy the
source `tokenizer.json`/`tokenizer_config.json` into the bundle's `tokenizer/` (metadata
`embedded_tokenizer: true` → `<bundle>/tokenizer/tokenizer.json`).

int8hu quality (torch ladder, weight-only PTQ): **7/694 argmax flips, all at visual-token positions,
0 in the text region, the generation-driving position exact** — the OCR text is preserved. int8lin:
9/694. int4 not shipped (cliff risk without QAT on 0.9B).

## The gotcha: an S=1 bundle can't be gated by a raw Python decode loop

The decode-loop gate teacher-forces the S=1 bundle over the reference (`input_ids [1,1]`,
`position_ids = arange(i+1)` growing, `image_embeds` injected). Two traps:

1. **Python can't set `expectFrequentReshapes`** (Swift-only `SpecializationOptions` property). So the
   growing `position_ids` makes MPSGraph **JIT-recompile per shape**, spilling ~1.5 GB each to
   `$TMPDIR/com.apple.MetalPerformanceShadersGraph` — a full disk kills the run in ~7 steps. Numerics
   are exact right up to the crash (pure infra, not a model bug). The `ANECCompile FAILED` log flood
   is harmless (GPU fallback).
2. The **engine** (CoreAILM / `EngineFactory`) sets `expectFrequentReshapes` → **one specialization**,
   no per-shape recompile, disk-safe. That's the right way to drive S=1 decode. But feed the S=1
   bundle a multi-token prefill and the engine errors (`Shape … of 694 is not a valid substitution for
   source shape 1`): the S=1 `input_ids` is static `[1,1]`. **Set `COREAI_CHUNK_THRESHOLD=1`** to force
   one-token-at-a-time prefill (as `PipelinedBench`/`GlmOcrBench` do).

So: gate quantization at the **torch** level (argmax flips), and gate the engine end-to-end with the
**pipelined engine** (`GlmOcrBench`, chunk=1) — not a hand-rolled Python S=1 loop.

## End-to-end engine gate

`ondevice/GlmOcrBench` (a Qwen3-VL-bench clone minus deepstack; hidden 1536, N 682, 3 static inputs
`image_embeds` + `rope_shift_start` + `rope_shift_amount`) drives the S=1 int8hu bundle on the
pipelined engine: 694-token prefill + 40-token greedy decode with the real `image_embeds` injected.
Result: **NUMERICS engine-vs-HF = 40/40 PASS** (verbatim OCR of the sample doc), **~375 tok/s** decode
on M4 Max, engine ready 0.1 s. This is the real-generation gate the Python teacher-forced loop can't
reach.

Host contract (same as Qwen3-VL, deepstack removed): image placeholders → `V + slot` (row-major over
the merged grid); `rope_shift_start = img_start + N`; `rope_shift_amount = N − max(gh, gw)`. For the
sample (grid 22×31, N 682, img_start 5): shift 687 / 651.

## iPhone

Decoder AOT-compiles clean for iOS h18p (`coreai-build compile … --platform iOS --architecture h18p
--preferred-compute gpu --expect-frequent-reshapes`) — the multifunction-AOT breakage seen elsewhere
doesn't apply to this single-function bundle. `PipelinedBench` enrolls GLM-OCR as a `vlGlm` ModelSpec
(image_embeds + rope-shift, no deepstack) for on-device tok/s.
