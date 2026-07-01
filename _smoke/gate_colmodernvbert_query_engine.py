# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "colpali-engine>=0.3.13",
#     "transformers>=5.5",
#     "numpy",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Engine gate for the ColModernVBERT query/text encoder .aimodel (Phase 1).
#
# Loads the built static .aimodel on the GPU delegate, feeds each reference query (the SAME
# process_texts + right-pad-to-grid path the export used), and compares the engine's per-token
# 128-d multi-vector against the torch-wrapper reference (reference_query.json). The reference
# vectors ARE the fp32 wrapper output the .aimodel must reproduce, so engine-vs-reference
# per-token cosine is the gate (mean > 0.99, min > 0.98; fp16 compute noise is expected).
import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import coreai.runtime as rt

MODEL_ID = "ModernVBERT/colmodernvbert"


def pad_to_grid(input_ids, attention_mask, seq_len, pad_id):
    cur = input_ids.shape[1]
    if cur > seq_len:
        return input_ids[:, :seq_len].contiguous(), attention_mask[:, :seq_len].contiguous()
    if cur < seq_len:
        pad_n = seq_len - cur
        ids_pad = torch.full((1, pad_n), pad_id, dtype=input_ids.dtype)
        mask_pad = torch.zeros((1, pad_n), dtype=attention_mask.dtype)
        input_ids = torch.cat([input_ids, ids_pad], dim=1)
        attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
    return input_ids.contiguous(), attention_mask.contiguous()


async def gate(model_path: Path, ref: dict):
    from colpali_engine.models import ColModernVBertProcessor
    processor = ColModernVBertProcessor.from_pretrained(MODEL_ID)
    seq_len = ref["seq_len"]
    pad_id = ref["pad_token_id"]

    m = await rt.AIModel.load(
        str(model_path),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()),
    )
    fn = m.load_function("main")

    worst_min, worst_mean = 1.0, 1.0
    print(f"[GATE] engine vs torch reference (per-token cosine), grid={seq_len}:")
    for key, q in ref["queries"].items():
        ti = processor.process_texts([q])
        ids, mask = pad_to_grid(
            ti["input_ids"].to(torch.int32), ti["attention_mask"].to(torch.int32),
            seq_len, pad_id,
        )
        res = await asyncio.wait_for(fn(inputs={
            "input_ids": rt.NDArray(np.ascontiguousarray(ids.numpy())),
            "attention_mask": rt.NDArray(np.ascontiguousarray(mask.numpy())),
        }), timeout=300)
        eng = torch.from_numpy(res["query_embeddings"].numpy().astype(np.float32))[0]   # [S,128]
        refv = torch.tensor(ref["query_embeddings"][key], dtype=torch.float32)          # [S,128]
        n_real = min(ref["real_lens"][key], seq_len)
        cos = [float(F.cosine_similarity(eng[t], refv[t], dim=0)) for t in range(n_real)]
        cmin, cmean = min(cos), sum(cos) / len(cos)
        worst_min, worst_mean = min(worst_min, cmin), min(worst_mean, cmean)
        print(f"          {key:14s} n_real={n_real:3d}  min={cmin:.6f}  mean={cmean:.6f}")
    print(f"[GATE] worst min per-token cos = {worst_min:.6f}  worst mean = {worst_mean:.6f}")
    assert worst_mean > 0.99, f"engine mean cosine too low ({worst_mean})"
    assert worst_min > 0.98, f"engine min per-token cosine too low ({worst_min})"
    print("[PASS] ColModernVBERT query encoder engine gate (GPU) green.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to .aimodel bundle")
    parser.add_argument("--reference", required=True, help="path to reference_query.json")
    args = parser.parse_args()
    ref = json.loads(Path(args.reference).read_text())
    asyncio.run(gate(Path(args.model), ref))


if __name__ == "__main__":
    main()
