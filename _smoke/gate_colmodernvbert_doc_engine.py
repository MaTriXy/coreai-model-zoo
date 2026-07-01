# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
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
# Engine gate for the ColModernVBERT document/image encoder .aimodel (Phase 2).
#
# Loads the built static .aimodel on the GPU delegate, reproduces the exact pixel_values from
# the saved test page (process_images, splitting OFF), feeds (pixel_values, pixel_attention_mask)
# to the engine, and compares the engine's per-token 128-d doc multi-vector against the torch
# reference (reference_doc.json). Per-token cosine: mean > 0.99, min > 0.98.
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coreai.runtime as rt
from PIL import Image

MODEL_ID = "ModernVBERT/colmodernvbert"


async def gate(model_path: Path, ref: dict, ref_dir: Path, dtype: str, compute: str):
    from colpali_engine.models import ColModernVBertProcessor
    processor = ColModernVBertProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False

    page = Image.open(ref_dir / ref["test_image"]).convert("RGB")
    di = processor.process_images([page])
    pv = di["pixel_values"].to(torch.float16 if dtype == "float16" else torch.float32)
    pam = di["pixel_attention_mask"].to(torch.int32)

    cu = rt.ComputeUnitKind.cpu() if compute == "cpu" else rt.ComputeUnitKind.gpu()
    m = await rt.AIModel.load(
        str(model_path),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(cu),
    )
    fn = m.load_function("main")
    res = await asyncio.wait_for(fn(inputs={
        "pixel_values": rt.NDArray(np.ascontiguousarray(pv.numpy())),
        "pixel_attention_mask": rt.NDArray(np.ascontiguousarray(pam.numpy())),
    }), timeout=300)
    eng = torch.from_numpy(res["doc_embeddings"].numpy().astype(np.float32))[0]   # [Sd,128]
    refv = torch.tensor(ref["doc_embeddings"], dtype=torch.float32)               # [Sd,128]
    Sd = refv.shape[0]
    cos = [float(F.cosine_similarity(eng[t], refv[t], dim=0)) for t in range(Sd)]
    cmin, cmean = min(cos), sum(cos) / len(cos)
    print(f"[GATE] doc engine vs torch reference (per-token cosine), Sd={Sd}:")
    print(f"          min={cmin:.6f}  mean={cmean:.6f}")
    assert cmean > 0.99, f"engine mean cosine too low ({cmean})"
    assert cmin > 0.98, f"engine min per-token cosine too low ({cmin})"
    print("[PASS] ColModernVBERT document encoder engine gate (GPU) green.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--compute", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    ref_path = Path(args.reference)
    ref = json.loads(ref_path.read_text())
    asyncio.run(gate(Path(args.model), ref, ref_path.parent, args.dtype, args.compute))


if __name__ == "__main__":
    main()
