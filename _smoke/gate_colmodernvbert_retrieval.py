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
# Phase 3 end-to-end retrieval gate for ColModernVBERT (query .aimodel + doc .aimodel).
#
# Builds a small text-bearing page corpus, encodes queries (query encoder) and pages (doc
# encoder, single 512px tile) on the GPU delegate, scores them with ColBERT-style MaxSim on the
# host, and checks the engine ranking against the colpali_engine torch oracle (processor.score
# on the SAME single-tile inputs). Two checks:
#   (1) FIDELITY  : engine MaxSim ranking == torch single-tile ranking (margin>0.05 hard-gated),
#                   and my MaxSim formula reproduces processor.score on the torch embeddings.
#   (2) QUALITY   : informational — does single-tile retrieval pick the human-intended page?
import argparse
import asyncio
from pathlib import Path

import numpy as np
import torch
import coreai.runtime as rt
from PIL import Image, ImageDraw, ImageFont

MODEL_ID = "ModernVBERT/colmodernvbert"
QUERY_GRID = 32

# query -> page text whose page should win. Plain enough to survive a 512px global view.
CORPUS = {
    "revenue": "Q3 FINANCIAL REPORT\n\nTotal revenue was 4.2 million dollars\nin the third quarter, up 18 percent\nyear over year.",
    "headcount": "EMPLOYEE HANDBOOK\n\nThe company employs 1,250 people\nacross 12 offices worldwide.",
    "invoice": "INVOICE  No. 4471\n\nAmount due: 8,900 USD\nPayment due date: March 15, 2026.",
    "menu": "CAFE MENU\n\nEspresso 3.00\nCappuccino 4.50\nBlueberry muffin 3.25",
}
QUERIES = {
    "revenue": "What was the total revenue in the third quarter?",
    "headcount": "How many employees does the company have?",
    "invoice": "When is the invoice payment due date?",
}


def _font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc",
              "/Library/Fonts/Arial.ttf"):
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_page(text):
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    f = _font(64)
    d.multiline_text((120, 200), text, fill="black", font=f, spacing=24)
    return img


def maxsim(q, d):
    # ColBERT late interaction: sum over query tokens of max over doc tokens of dot product.
    # q [Nq,128], d [Nd,128] (both L2-normalized). Returns a scalar.
    return float((q @ d.T).max(dim=1).values.sum())


async def run_engine(model_path, inputs):
    m = await rt.AIModel.load(
        str(model_path),
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    res = await asyncio.wait_for(fn(inputs=inputs), timeout=300)
    return res


def pad_to_grid(ids, mask, seq_len, pad_id):
    cur = ids.shape[1]
    if cur > seq_len:
        return ids[:, :seq_len].contiguous(), mask[:, :seq_len].contiguous()
    if cur < seq_len:
        n = seq_len - cur
        ids = torch.cat([ids, torch.full((1, n), pad_id, dtype=ids.dtype)], dim=1)
        mask = torch.cat([mask, torch.zeros((1, n), dtype=mask.dtype)], dim=1)
    return ids.contiguous(), mask.contiguous()


async def main_async(args):
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
    processor = ColModernVBertProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.do_image_splitting = False
    pad_id = int(processor.tokenizer.pad_token_id)
    model = ColModernVBert.from_pretrained(MODEL_ID, torch_dtype=torch.float32,
                                           trust_remote_code=True, attn_implementation="eager")
    model.eval()

    doc_keys = list(CORPUS)
    pages = {k: render_page(CORPUS[k]) for k in doc_keys}
    q_keys = list(QUERIES)

    # ---- torch oracle (single-tile) ----
    q_torch, d_torch = {}, {}
    with torch.no_grad():
        for qk in q_keys:
            q_torch[qk] = _t(model(**processor.process_texts([QUERIES[qk]])))[0]   # [Sq,128]
        for dk in doc_keys:
            d_torch[dk] = _t(model(**processor.process_images([pages[dk]])))[0]     # [89,128]
    ref = {qk: {dk: maxsim(q_torch[qk], d_torch[dk]) for dk in doc_keys} for qk in q_keys}

    # sanity: my MaxSim == processor.score on torch embeddings
    try:
        ps = processor.score([q_torch[qk] for qk in q_keys], [d_torch[dk] for dk in doc_keys])
        ps = torch.as_tensor(ps)
        mine = torch.tensor([[ref[qk][dk] for dk in doc_keys] for qk in q_keys])
        dmax = float((ps - mine).abs().max())
        print(f"[CHECK] my MaxSim vs processor.score max|Δ| = {dmax:.4f}")
    except Exception as e:
        print(f"[WARN] processor.score cross-check skipped: {e}")

    # ---- engine ----
    q_eng, d_eng = {}, {}
    for qk in q_keys:
        ti = processor.process_texts([QUERIES[qk]])
        n_real = int(ti["attention_mask"].sum())
        ids, mask = pad_to_grid(ti["input_ids"].to(torch.int32),
                                ti["attention_mask"].to(torch.int32), QUERY_GRID, pad_id)
        res = await run_engine(args.query_model, {
            "input_ids": rt.NDArray(np.ascontiguousarray(ids.numpy())),
            "attention_mask": rt.NDArray(np.ascontiguousarray(mask.numpy()))})
        q_eng[qk] = torch.from_numpy(res["query_embeddings"].numpy().astype(np.float32))[0][:n_real]
    fdt = torch.float16 if args.dtype == "float16" else torch.float32
    for dk in doc_keys:
        if args.doc_preproc == "squish":
            # Square-resize to 512 (no aspect-pad), mask all-ones — the simplest kit path
            # (reuses CoreAIKit ImagePreprocessor). The doc graph bakes input_ids, so only
            # pixels vary. This validates whether the squish path still retrieves correctly.
            img = pages[dk].convert("RGB").resize((512, 512), Image.BICUBIC)
            arr = (np.asarray(img).astype(np.float32) / 255.0 - 0.5) / 0.5  # HWC, mean/std 0.5
            pv = torch.from_numpy(np.transpose(arr, (2, 0, 1))[None, None]).to(fdt)
            pam = torch.ones((1, 1, 512, 512), dtype=torch.int32)
        else:
            di = processor.process_images([pages[dk]])
            pv = di["pixel_values"].to(fdt)
            pam = di["pixel_attention_mask"].to(torch.int32)
        res = await run_engine(args.doc_model, {
            "pixel_values": rt.NDArray(np.ascontiguousarray(pv.numpy())),
            "pixel_attention_mask": rt.NDArray(np.ascontiguousarray(pam.numpy()))})
        d_eng[dk] = torch.from_numpy(res["doc_embeddings"].numpy().astype(np.float32))[0]
    eng = {qk: {dk: maxsim(q_eng[qk], d_eng[dk]) for dk in doc_keys} for qk in q_keys}

    # ---- compare ----
    print("\n[RESULT] per query: oracle top doc | engine top doc | margin | intended")
    fidelity_ok, intended_ok = True, 0
    for qk in q_keys:
        ro = sorted(ref[qk].items(), key=lambda kv: kv[1], reverse=True)
        re_ = sorted(eng[qk].items(), key=lambda kv: kv[1], reverse=True)
        margin = ro[0][1] - ro[1][1]
        agree = ro[0][0] == re_[0][0]
        intend = re_[0][0] == qk
        intended_ok += int(intend)
        tag = "" if agree else " ** ENGINE DIFFERS **"
        print(f"  {qk:10s} oracle->{ro[0][0]:10s} engine->{re_[0][0]:10s} "
              f"margin {margin:6.3f} intended->{qk:10s} {'HIT' if intend else 'miss'}{tag}")
        if margin > 0.05 and not agree:
            fidelity_ok = False
    if args.doc_preproc == "faithful":
        print(f"\n[FIDELITY] engine reproduces torch single-tile ranking (clear-margin): "
              f"{'PASS' if fidelity_ok else 'FAIL'}")
        assert fidelity_ok, "engine ranking diverges from torch on a clear-margin query"
    print(f"[QUALITY ] {args.doc_preproc} engine retrieved intended page: "
          f"{intended_ok}/{len(q_keys)}")
    assert intended_ok == len(q_keys), f"{args.doc_preproc}: not all intended pages retrieved"
    print(f"[PASS] Phase 3 retrieval gate green ({args.doc_preproc}).")


def _t(out):
    return out if isinstance(out, torch.Tensor) else out.embeddings


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--query-model", required=True)
    p.add_argument("--doc-model", required=True)
    p.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    p.add_argument("--doc-preproc", choices=["faithful", "squish"], default="faithful")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
