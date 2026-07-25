# EfficientSAM3 → Core AI — port learnings (NOT shipped)

**Outcome: DROPPED (KEEP-LOCAL). The port worked end-to-end and was device-verified, but it was
redundant: the zoo already ships the full official Meta SAM 3** (`mlboydaisuke/sam3-CoreAI-official`,
text-prompt open-vocab segmentation, iPhone-verified, `official/` folder, 2026-06-23) — which does
text→segment with real coverage. This distilled edge version is strictly worse (narrow concept
coverage), so it adds nothing. This doc keeps the transferable Core AI techniques; the bundles /
conversion scripts / kit code / demo app were discarded.

`Simon7108528/EfficientSAM3` (Apache-2.0) = Meta **SAM 3** (Promptable Concept Segmentation)
progressively distilled to edge encoders (RepViT/EfficientViT/TinyViT vision + MobileCLIP text).

## Why it was dropped (the value gate, not a technical failure)
Coverage sweep on the **HF reference model** (its own `Sam3Processor`, threshold 0.05, all raw scores) —
checkpoint `efficient_sam3_repvit_m1.1_mobileclip_s1_ft` (RepViT-M1.1 + MobileCLIP-S1):
- **Fires**: `a wheel`, `a tire`, `a window`, `a shoe`, `a sneaker`, `a foot` (the parts/objects it was
  distilled on).
- **Empty (all scores ~0), even with an article**: `a car`, `a truck`, `a vehicle`, `bottle`, `banana`,
  `apple`, `person`, and everything on a cluttered groceries photo.
- **Article matters**: `a window` fires but bare `window` → empty; `a tire` fires but `tire` → empty.

So this **distilled** checkpoint segments a narrow, part-level concept set — the "segment anything by
text" pitch does not hold for it. The coverage loss comes from shrinking SAM 3's text encoder 354M →
MobileCLIP-S1 42.5M. **The teacher, full Meta SAM 3 (`facebook/sam3`, non-distilled), detects car/
truck/etc. properly — and it is ALREADY SHIPPED in this zoo** (`mlboydaisuke/sam3-CoreAI-official`,
`official/`, iPhone-verified). So this port was redundant from the start (a `/next-target` miss:
the shipped official SAM 3 was overlooked). → KEEP-LOCAL, discard artifacts.

---

## The port (all validated before dropping — the techniques are the value)

Ran as **3 stateless Core AI graphs + host glue** (like rf-detr-seg + a text encoder + open-vocab
scoring). **121.8M params** (the 2.36 GB `_ft.pth` carries EMA/extra state; the live model is small).

**Build**: `build_efficientsam3_image_model(device="cpu", backbone_type="repvit", model_name="m1.1",
text_encoder_type="MobileCLIP-S1", text_encoder_context_length=77)` — ctx **77** (matches the ft
checkpoint pos-embed; 16 → size mismatch). `Sam3Processor(model, device="cpu")` (else MPS → device
mismatch). Extra deps: `pycocotools psutil iopath`.

**Graph split**
| # | graph | in → out |
|---|---|---|
| 1 | vision = `backbone.forward_image` (RepViT+neck) | image (1,3,1008,1008) [-1,1] → backbone_fpn 288²/144²/72²@256 (`vision_features == fpn[2]`) |
| 2 | text = MobileCLIP encoder | token ids (1,77) int → text_memory (77,1,256) + embeds (77,1,512) |
| 3 | grounding = `forward_grounding` (DETR enc/dec + FPN seg head + dot-product scoring) | fpn×3 + pos_enc×3 + text_memory/mask/embeds → pred_boxes (1,200,4), pred_logits (1,200,1), pred_masks (1,200,288²), presence (1,1) |

**Host**: image preproc (resize 1008² + Normalize 0.5/0.5); CLIP-BPE tokenize (`SimpleTokenizer`, ctx 77);
attention mask `(tok!=0).ne(1)`; post-proc `probs = σ(logits)·σ(presence)`, keep>thr, box cxcywh→xyxy×[W,H],
mask `interpolate(→H,W).σ>0.5`.

### The transferable techniques (why this doc exists)
1. **Cross-venv export**: `torch.export` the sub-module in the model's venv → `torch.export.save` → load +
   `TorchConverter` in the coreai venv (same torch 2.9.0; ExportedProgram transfers; no model pkg needed
   there). **Engine int inputs must be int32** (int64 → CoreAIError 3).
2. **DETR data-dependent guard** → set `torch.compiler.is_dynamo_compiling = lambda: True`
   (+ `torch._dynamo.is_compiling`) before export, so `decoder._get_rpb_matrix` takes the trace-friendly
   cached branch instead of comparing symbolic feat-sizes (`Eq(u0,1)` guard failure).
3. **Geometry-encoder stub**: for text-only PCS the geometry encoder emits a constant dummy token but its
   box/point paths use `torchvision.roi_align` + `grid_sample` + `scatter` — which Core AI can't lower and
   which fail to *deserialize* in the coreai venv (no torchvision). Capture the dummy output once, replace
   `model.geometry_encoder` with a stub returning those constants → those ops never enter the graph.
4. **pos_enc = data resource, not a graph output**: pos_enc is a fixed constant for the 1008² input
   (~111 MB). Emitting it from a graph bloats the bundle (+115 MB) AND broke GPU load; ship it as a
   `pe{0,1,2}.bin` resource fed to the grounding graph host-side.
5. **`lang_mask` = float input, not bool**: Core AI `TensorValue` has no bool. Export the grounding graph
   taking `lang_mask` as float32 and cast `>0.5` internally (bool graph input → device `dtypeMismatch`).
6. **tokenizer.json from the model's own vocab**: sam3's `SimpleTokenizer` is standard OpenAI CLIP BPE
   (vocab 49408, sot 49406 / eot 49407). Generate `tokenizer.json` from its `encoder` (vocab) + `bpe_ranks`
   (merges) so a Swift CLIP tokenizer reads the exact vocab (parity by construction). Note: it **pads with
   0**, not eot, so the `(tok!=0)` mask works.
7. **Swift mask overlay**: draw each 288² mask as a **tinted-RGBA CGImage** (transparent elsewhere) and
   `ctx.draw` it — NOT `clip(to:mask:)`+fill-full-rect (washes the whole frame). In a top-left-flipped
   context, **flip the mask rows** when building the RGBA or `ctx.draw` mirrors it vs the box coords.

## Gates that passed (before drop)
- Per-graph vs HF oracle: cos 1.0 (all 3). Core AI fp32 convert: vision cos 1.0; grounding
  boxes/logits/presence cos 1.0, masks 0.9999999.
- **End-to-end (3 Core AI engines + host) vs HF**: `truck`+"wheel" → 2 detections, scores exact, mask
  IoU 1.0 — **CPU and Mac GPU**. Ship dtype fp32 (rf-detr precedent; DETR resists a clean `.half()`).
- **iPhone 17 Pro, in-app**: "a wheel" → 2 wheels, scores ≈ Mac (±0.001 fp16), warm ~738 ms, iOS h18p AOT.

So the **Core AI path for a distilled-SAM3 / DETR-seg model is fully proven** — reusable if a
wider-coverage SAM 3 variant (or the full `facebook/sam3`) is ported later.
