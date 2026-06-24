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

## 4. iOS needs the AOT bundle — JIT-compiling the graph aborts on device

The hosted float16 bundle is a **JIT** `.aimodel`. It loads fine on macOS, but on iOS
`ImageSegmenter(resourcesAt:)` **crashes (SIGABRT) during load** — `MetalPerformanceShadersGraph`
JIT-compiles the SAM 3 graph and, while constant-folding a transpose for matmul canonicalization,
`BumpMmapResourceAllocator::allocateResource` throws `bad_alloc` (the folded constant overruns the
per-process budget) → uncaught → abort. It is **not** a clean jetsam (that would be `EXC_RESOURCE`),
and **not** fixed by the `increased-memory-limit` entitlement reliably (and that entitlement forces a
custom provisioning profile that fails to sign headless).

**Fix: ship/sideload the AOT-compiled bundle for iOS** — compilation moves to the Mac (plenty of
memory), the device just mmaps the precompiled MPSGraph package (no JIT spike; AOT clean-mmap also
dodges the jetsam dirty limit):

```bash
xcrun coreai-build compile sam3_float16.aimodel --output out \
    --platform iOS --preferred-compute gpu --architecture h18p --expect-frequent-reshapes
```

The output is `sam3_float16.h18p.aimodelc` (~3 GB — bigger than the 1.6 GB JIT bundle, but mmapped).
Rename it to the asset name the app expects and point `metadata.json` `assets.main` at it; the
runtime detects AOT from the inner `main-h18p.mlirb` + `…-delegates/MPSGraph/mpsExecutable.mpsgraphpackage`.
For debugging, `devicectl device copy to --domain-type appDataContainer --domain-identifier <bundleID>
--source <bundle> --destination Documents/<app-bundle-dir> --remove-existing-content true` lands it in
the app's container so an in-app downloader that checks file existence skips the network fetch.
Device-verified on iPhone 17 Pro.

## 5. Overlay rendering: mask orientation + box origin

`Segment.box` is in pixel coordinates with a **platform-dependent origin** — bottom-left on macOS
(AppKit), top-left on iOS (UIKit) — so the box is y-flipped into a bottom-up `CGContext` on iOS. The
**mask**, by contrast, must **not** be flipped: `CGContext.draw` renders a top-down `CGImage`
right-side-up (the same way it draws the base image), so building the mask `CGImage` from the
row-major mask in source order is correct — an extra flip double-flips and the masks come out
upside-down. See [`apps/CoreAISegment`](../apps/CoreAISegment/).
