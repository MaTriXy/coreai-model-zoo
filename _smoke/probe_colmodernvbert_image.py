# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "colpali-engine>=0.3.13",
#     "transformers>=5.5",
#     "pillow",
#     "numpy",
# ]
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Probe: inspect ColModernVBertProcessor.process_images output so we can design the static
# document-encoder graph (tile count, image-token layout, sequence length, pixel_values shape).
import numpy as np
import torch
from PIL import Image

MODEL_ID = "ModernVBERT/colmodernvbert"


def describe(name, inputs):
    print(f"\n===== {name} =====")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:22s} {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k:22s} {type(v)} = {v}")
    ids = inputs.get("input_ids")
    if ids is not None:
        seq = ids[0].tolist()
        print(f"  seq_len = {len(seq)}")
        n_img = sum(1 for t in seq if t == 50407)
        print(f"  image_token(50407) count = {n_img}")
        print(f"  first 40 ids: {seq[:40]}")
        print(f"  last 20 ids:  {seq[-20:]}")


def main():
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
    processor = ColModernVBertProcessor.from_pretrained(MODEL_ID)
    ip = processor.image_processor
    print("[image_processor] do_image_splitting =", getattr(ip, "do_image_splitting", "?"),
          "max_image_size =", getattr(ip, "max_image_size", "?"),
          "size =", getattr(ip, "size", "?"))

    # A synthetic 'page' (white with a black bar) at a doc-ish aspect ratio.
    page = Image.new("RGB", (1240, 1754), "white")  # ~A4 @150dpi
    for x in range(200, 1040):
        for y in range(300, 360):
            page.putpixel((x, y), (0, 0, 0))

    # default (splitting ON)
    di = processor.process_images([page])
    describe("process_images (default, splitting ON)", di)

    # splitting OFF -> single tile, the simplest static graph
    try:
        ip.do_image_splitting = False
        si = processor.process_images([page])
        describe("process_images (splitting OFF)", si)
    except Exception as e:
        print(f"[WARN] splitting OFF failed: {e}")

    # model forward output shape (splitting OFF inputs)
    model = ColModernVBert.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                           trust_remote_code=True, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        out = model(**si)
    print("\n[model output] splitting-OFF doc embeddings:",
          tuple(out.shape) if isinstance(out, torch.Tensor) else type(out))


if __name__ == "__main__":
    main()
