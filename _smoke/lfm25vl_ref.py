#!/usr/bin/env python3
"""Dump the LFM2.5-VL fp32 oracle: every tensor the Core AI port has to reproduce.

The port re-authors two things -- the SigLIP2 vision tower and the 2-layer
projector -- and splices their output into the already-shipped LFM2 decoder.
This script runs the HF model on a fixed image + prompt and saves the
intermediates each stage must match, so the gates compare against the model
rather than against a note about the model.

Two things about this checkpoint family that the weight shapes give away, and
that a port written from the MiniCPM-V SigLIP recipe would get wrong:

  * `embeddings.patch_embedding.weight` is [768, 768] -- a Linear over
    flattened 16x16x3 patches, not a Conv2d. This is the SigLIP2 NaFlex form.
  * `embeddings.position_embedding.weight` is [256, 768] -- a 16x16 grid that
    gets resized to whatever patch grid the image actually produced.

**Requires transformers >= 5.** Not a packaging preference -- 4.x is wrong for
this checkpoint. `Lfm2VlMultiModalProjector.forward` in 4.57.6 applies a
LayerNorm unconditionally, while these configs set `projector_use_layernorm:
false` and ship no such weights. torch's LayerNorm default init (weight 1,
bias 0) means the stray normalization produces no warning at generation time
and no obvious garbage -- just a quietly different oracle. 5.14.1 gates it on
the flag. An oracle that is subtly wrong is worse than none, so this refuses to
run rather than let that reach a gate.

Run (a transformers-5 interpreter, e.g. the LiteRT VL venv):
    ~/code/litertlm-convert/.venv-vl093/bin/python _smoke/lfm25vl_ref.py \
        [--hf-id LiquidAI/LFM2.5-VL-450M] [--out ...]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

FP32 = torch.float32
DEFAULT_ID = "LiquidAI/LFM2.5-VL-450M"
# The canonical two-cats-on-a-couch COCO image the transformers vision tests use.
# Fixed URL, fixed bytes, so the reference is reproducible from the script alone.
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = "What is in this image?"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=DEFAULT_ID)
    ap.add_argument("--out", default=None, help="output .npz (default: _smoke/<slug>_ref.npz)")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    import transformers

    if int(transformers.__version__.split(".")[0]) < 5:
        raise SystemExit(
            f"transformers {transformers.__version__} applies the projector LayerNorm "
            "unconditionally; this checkpoint sets projector_use_layernorm=false and has "
            "no such weights. Run this on a transformers>=5 interpreter (see the docstring)."
        )

    import requests
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    slug = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    out = Path(args.out or Path(__file__).parent / f"{slug}_ref.npz")

    image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=60).raw).convert("RGB")
    print(f"image {image.size} from {IMAGE_URL}")

    processor = AutoProcessor.from_pretrained(args.hf_id)
    print(f"loading {args.hf_id} fp32 ...")
    model = AutoModelForImageTextToText.from_pretrained(args.hf_id, dtype=FP32)
    model.eval()

    conversation = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")

    saved: dict[str, np.ndarray] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            saved[k] = v.to(FP32).numpy() if v.is_floating_point() else v.numpy().astype(np.int32)

    # Hooks, not a re-implementation: record what this checkpoint does, not what
    # we believe it does. These are the four seams the port is gated at.
    # transformers 5 hangs embeddings/encoder/post_layernorm straight off the
    # Siglip2VisionModel; 4.x nested them under an extra `.vision_model`.
    tower = model.model.vision_tower
    tower = getattr(tower, "vision_model", tower)
    grabs: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_m, _i, o):
            grabs[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    handles = [
        tower.embeddings.register_forward_hook(grab("vision_embeddings")),
        tower.encoder.layers[0].register_forward_hook(grab("vision_layer0")),
        tower.encoder.layers[len(tower.encoder.layers) // 2].register_forward_hook(grab("vision_layer_mid")),
        tower.post_layernorm.register_forward_hook(grab("vision_post_layernorm")),
        model.model.multi_modal_projector.register_forward_hook(grab("image_features")),
    ]

    with torch.no_grad():
        outputs = model(**inputs)
    for h in handles:
        h.remove()

    saved["logits_last"] = outputs.logits[0, -1].to(FP32).numpy()
    for name, tensor in grabs.items():
        saved[name] = tensor.to(FP32).numpy()
        print(f"  {name}: {tuple(tensor.shape)}")

    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    new_ids = gen[0, inputs["input_ids"].shape[1]:]
    saved["gen_ids"] = new_ids.numpy().astype(np.int32)
    print(f"\ngreedy: {processor.decode(new_ids, skip_special_tokens=True)!r}")

    saved["_meta_hf_id"] = np.array(args.hf_id)
    saved["_meta_transformers"] = np.array(transformers.__version__)
    saved["_meta_image_url"] = np.array(IMAGE_URL)
    saved["_meta_prompt"] = np.array(PROMPT)

    np.savez(out, **saved)
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    print("keys:", ", ".join(k for k in sorted(saved) if not k.startswith("_meta")))


if __name__ == "__main__":
    main()
