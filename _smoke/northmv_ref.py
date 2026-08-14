#!/usr/bin/env python3
"""Dump the North-Micro-Vision fp32 oracle: every tensor the Core AI port has to reproduce.

Cohere's `cohere_compass` — a 400M SigLIP2-SO400M-derived tower (hidden 1152 /
intermediate 4304 / 27 layers, the same body shape as LFM2.5-VL-3B) feeding a 2B
Cohere decoder with a 262 144 tied vocab. Native-resolution: the processor picks a
patch grid per image, so the FIRST thing this has to report is what grid the
fixture actually produced -- that decides the export's baked grid.

**Requires transformers >= 5.16 (git main at the time of writing).** 5.15.0 does
not know `cohere_compass` at all, which at least fails loudly; the trap this
family taught (LFM2.5-VL) is the version that loads and is quietly wrong, so
this refuses to run on anything that cannot name the architecture.

Unlike the LFM2.5-VL oracle this does not hardcode module paths: it walks the
loaded model and hooks the tower's embeddings / first / middle / last block and
whatever projects into the decoder, printing what it found. A new architecture's
seam names are not knowledge worth guessing when the model itself has them.

Run:
    ~/code/litertlm-convert/.venv-vl0930-t515/bin/python _smoke/northmv_ref.py \\
        [--hf-id CohereLabs/North-Micro-Vision-Instruct] [--resize WxH]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

FP32 = torch.float32
DEFAULT_ID = "CohereLabs/North-Micro-Vision-Instruct"
IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
PROMPT = "What is in this image?"


def find_module(model, *needles):
    """First named module whose name ends with one of `needles`."""
    for needle in needles:
        for name, mod in model.named_modules():
            if name.endswith(needle):
                return name, mod
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default=DEFAULT_ID)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--resize", default=None, metavar="WxH",
                    help="pre-resize the fixture with the processor's own resampler, so the "
                         "oracle is captured through the grid the bundle will bake")
    args = ap.parse_args()

    import transformers

    major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    if (major, minor) < (5, 16):
        raise SystemExit(
            f"transformers {transformers.__version__} does not know `cohere_compass` "
            "(5.15.0 raises on AutoConfig). Install git main: "
            "pip install git+https://github.com/huggingface/transformers.git"
        )

    import requests
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    slug = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    suffix = f"_{args.resize}" if args.resize else ""
    out = Path(args.out or Path(__file__).parent / f"{slug}_ref{suffix}.npz")

    processor = AutoProcessor.from_pretrained(args.hf_id)
    image = Image.open(requests.get(IMAGE_URL, stream=True, timeout=60).raw).convert("RGB")
    print(f"image {image.size} from {IMAGE_URL}")
    if args.resize:
        resample = int(getattr(processor.image_processor, "resample", 2))
        w, h = (int(v) for v in args.resize.lower().split("x"))
        image = image.resize((w, h), resample)
        print(f"  pre-resized to {image.size} (PIL resample {resample}; the host does this)")

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
    saved: dict[str, np.ndarray] = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
            saved[k] = v.to(FP32).numpy() if v.is_floating_point() else v.numpy().astype(np.int32)

    # Seams, discovered rather than assumed.
    grabs: dict[str, torch.Tensor] = {}

    def grab(name):
        def hook(_m, _i, o):
            grabs[name] = (o[0] if isinstance(o, tuple) else o).detach()
        return hook

    handles = []
    tower_name, tower = find_module(model, "vision_tower", "vision_model", "visual")
    print(f"tower: {tower_name} ({type(tower).__name__ if tower is not None else '-'})")
    if tower is not None:
        inner = getattr(tower, "vision_model", tower)
        emb_name, emb = find_module(inner, "embeddings", "patch_embedding")
        if emb is not None:
            handles.append(emb.register_forward_hook(grab("vision_embeddings")))
        blocks = None
        for attr in ("layers", "blocks"):
            enc = getattr(getattr(inner, "encoder", inner), attr, None)
            if enc is not None:
                blocks = enc
                break
        if blocks is not None:
            print(f"  blocks: {len(blocks)}")
            handles.append(blocks[0].register_forward_hook(grab("vision_layer0")))
            handles.append(blocks[len(blocks) // 2].register_forward_hook(grab("vision_layer_mid")))
            handles.append(blocks[-1].register_forward_hook(grab("vision_layer_last")))
        post_name, post = find_module(inner, "post_layernorm", "norm", "ln_post")
        if post is not None:
            handles.append(post.register_forward_hook(grab("vision_post_layernorm")))
    proj_name, proj = find_module(model, "multi_modal_projector", "merger", "connector", "mm_projector")
    print(f"projector: {proj_name} ({type(proj).__name__ if proj is not None else '-'})")
    if proj is not None:
        handles.append(proj.register_forward_hook(grab("image_features")))

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
