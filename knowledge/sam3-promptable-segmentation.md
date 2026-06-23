# SAM 3 promptable segmentation on the stock runtime

Lessons from porting [`facebook/sam3`](https://huggingface.co/facebook/sam3) (Meta, gated, 848 M)
to Core AI via Apple's official `models/sam3/export.py` and the official `CoreAIImageSegmenter`
Swift runtime — **stock runtime, no engine patch**. This is the cheapest class of port: a
single-forward vision model with a ready Apple runtime.

## 1. It's a segmenter *bundle*, not a bare `.aimodel`

`export.py` writes a **directory bundle** (`metadata.json` schema 0.2 + the `.aimodel` + a CLIP
`tokenizer/`):

```
sam3_float16/
  metadata.json            kind: "segmenter", assets.main → sam3_float16.aimodel
  sam3_float16.aimodel/     main.mlirb + main.hash + metadata.json
  tokenizer/                tokenizer.json + tokenizer_config.json (CLIP text)
```

The graph is `pixel_values [1,3,1008,1008] + input_ids → pred_masks, pred_boxes, pred_logits,
presence_logits, semantic_seg`. It's **text-prompt, open-vocabulary** (give it `"cat"`, not a fixed
class list). The whole runtime is one call:

```swift
import CoreAIImageSegmenter
let seg = try await ImageSegmenter(resourcesAt: "<bundle-dir>")   // loads metadata + tokenizer
let r = try await seg.segment(image: cgImage, prompt: "cat")      // r.segments: [Segment] (.mask/.box/.score)
```

`segment(image:prompt:)` tokenizes (CLIP), runs, decodes masks/boxes; score =
`sigmoid(pred_logit) × sigmoid(presence_logit)`. (Same runtime serves EfficientSAM via
`segment(image:pointQuery:)`.)

## 2. float16 is faithful — verify with the CLI `--parity-test` or a same-image diff

Ship **float16** (~1.7 GB, iOS-friendly). Verify with the official `image-segmenter` CLI: export
fp16 and fp32, run both on a test image, compare top segments. Measured (COCO two-cats, prompt
`cat`): top-2 scores differ **≤ 1e-4** (0.9746 vs 0.9747), boxes within **1 px**, top-3 masks have
**identical** foreground pixel counts; inference 0.55 s warm on M4 Max. The CLI also has a
`--parity-test <dir> --psnr-floor --cosine-floor` mode that compares raw outputs against reference
`.npy` tensors.

## 3. Gated model + license: redistribution is allowed, bundle the LICENSE

`facebook/sam3` is **gated** — accept the license on HF and `hf auth login` before the recipe can
download the checkpoint. The **SAM License permits redistribution** of the materials and derivative
works *under the same terms, provided you include a copy of the Agreement* (§1.b.i). So a converted
bundle can be hosted publicly **if you ship the `LICENSE` file in the bundle** and state the license
on the card. Hosting the converted artifact also lets others run it **without** the gated checkpoint
or a conversion environment — the point of the `official/` pre-converted folder.

## 4. Overlay rendering: mind the box origin

`Segment.box` is in pixel coordinates with a **platform-dependent origin** — bottom-left on macOS
(AppKit), top-left on iOS (UIKit). Compositing masks into a bottom-up `CGContext` needs the iOS box
y-flipped and the row-major mask flipped vertically. See
[`apps/CoreAISegment`](../apps/CoreAISegment/) (the sample app, macOS + iOS).
