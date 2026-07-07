# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "gliner2[local]",
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "numpy",
# ]
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# GLiNER2 (fastino/gliner2-privacy-filter-PII-multi) -> Core AI: zoo's FIRST NER / token-
# classification / schema-driven structured-extraction model, and zoo's FIRST DeBERTa-v3
# (disentangled-attention) port. On-device information extraction (PII redaction is the flagship
# use). Competitor GLiNER2Swift is macOS-CPU/MLX only -> iPhone/ANE is uncontested here.
#
# FUSED STATIC GRAPH — schema-agnostic (keeps GLiNER's zero-shot arbitrary-schema power):
#   forward(input_ids[1,S], attention_mask[1,S], text_word_idx[1,T], schema_idx[1,MMAX+1])
#       -> span_scores[1, MMAX, T, K]
# Host (collator) tokenizes text + linearizes the user's schema into input_ids, and supplies the
# gather indices (text-word first-token positions; schema special-marker positions). The graph
# runs mDeBERTa-v3 -> "first" word pooling gather -> SpanMarkerV0 -> CountLSTM(count=1) -> einsum
# -> sigmoid. count is fixed to 1 because entity extraction reads only count-step 0 (byte-identical
# to the model). Padded schema fields / word slots produce scores the host ignores via real M / len.
# Host decode = per-field threshold + confidence-desc greedy NMS (== GLiNER2._format_spans).
#
# Gates: (1) fp32 wrapper span_scores == GLiNER2 internal path (cos 1.0), (2) fp32 decoded entities
# == ext.extract, (3) Core AI convert, (4) GPU engine span_scores cos + decoded entities.
import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"
DEMO_LABELS = ["person", "email", "phone number", "credit card number",
               "social security number", "address", "date of birth", "organization"]
DEMO_TEXTS = [
    "Contact Dr. Sarah Johnson at sarah.j@acme.com or +1-415-555-0142.",
    "Wire to account holder James Lee, SSN 123-45-6789, card 4111 1111 1111 1111.",
    "Send the invoice to 500 Market St, San Francisco and cc maria@corp.io.",
    "Patient DOB 1985-07-14, treated at Mercy General Hospital by nurse Alan Poe.",
]


# ------------------------------------------------------------------ fused graph
class GLiNER2Extractor(torch.nn.Module):
    def __init__(self, ext, s, t, mmax, k):
        super().__init__()
        self.encoder = ext.encoder
        self.span_rep = ext.span_rep
        self.count_embed = ext.count_embed
        self.s, self.t, self.mmax, self.k = s, t, mmax, k
        starts = torch.arange(t).unsqueeze(1).expand(-1, k)
        ends = starts + torch.arange(k).unsqueeze(0)
        valid = ends < t
        ss = torch.where(valid, starts, torch.zeros_like(starts)).reshape(-1)
        se = torch.where(valid, ends, torch.zeros_like(ends)).reshape(-1)
        self.register_buffer("safe_spans", torch.stack([ss, se], -1).unsqueeze(0))  # [1,T*K,2]

    def forward(self, input_ids, attention_mask, text_word_idx, schema_idx):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[0]
        token_embs = hidden.index_select(0, text_word_idx[0].to(torch.long))    # [T,768]
        schema_vecs = hidden.index_select(0, schema_idx[0].to(torch.long))      # [MMAX+1,768]
        span_rep = self.span_rep(token_embs.unsqueeze(0), self.safe_spans)      # [1,T,K,768]
        struct_proj = self.count_embed(schema_vecs[1:], 1)                      # [1,MMAX,768]
        span_scores = torch.sigmoid(torch.einsum("lkd,bpd->bplk", span_rep[0], struct_proj))
        return span_scores                                                      # [1,MMAX,T,K]


# ------------------------------------------------------------------ host helpers
def collate(ext, text, labels):
    schema = ext.create_schema().entities(labels)
    b = ext.processor.collate_fn_inference([(text, schema)])
    return {
        "ids": b.input_ids[0].tolist(),
        "twi": b.text_word_indices[0].tolist()[: b.text_word_counts[0]],
        "ssi": list(b.schema_special_indices[0][0]),           # [count_marker, field_0, ...]
        "smap": b.start_mappings[0],
        "emap": b.end_mappings[0],
    }, schema


def pad(vals, n, v):
    return (list(vals) + [v] * n)[:n]


def make_inputs(c, s, t, mmax):
    ids = c["ids"]
    input_ids = torch.tensor([pad(ids, s, 0)], dtype=torch.int32)
    attn = torch.tensor([[1] * len(ids) + [0] * (s - len(ids))], dtype=torch.int32)
    word_idx = torch.tensor([pad(c["twi"], t, 0)], dtype=torch.int32)
    # schema_idx = [count_marker] + field positions, padded with the count-marker pos (safe gather)
    ssi = c["ssi"]
    schema_idx = torch.tensor([pad(ssi, mmax + 1, ssi[0])], dtype=torch.int32)
    return input_ids, attn, word_idx, schema_idx


def decode(scores, labels, text, smap, emap, thr):
    # scores [MMAX,T,K]; replicate GLiNER2 _find_spans + _format_spans (conf-desc greedy NMS).
    out = {}
    tl = len(smap)
    for idx, name in enumerate(labels):
        s = scores[idx]
        cand = []
        for start in range(min(tl, s.shape[0])):
            for width in range(s.shape[1]):
                end = start + width + 1
                if end > tl:
                    continue
                conf = float(s[start, width])
                if conf >= thr:
                    cs, ce = smap[start], emap[end - 1]
                    span = text[cs:ce].strip()
                    if span:
                        cand.append((span, conf, cs, ce))
        cand.sort(key=lambda x: x[1], reverse=True)
        sel = []
        for span, conf, cs, ce in cand:
            if not any(not (ce <= s2 or cs >= e2) for _, _, s2, e2 in sel):
                sel.append((span, conf, cs, ce))
        out[name] = [x[0] for x in sel]
    return out


def reference_scores(ext, text, labels):
    schema = ext.create_schema().entities(labels)
    b = ext.processor.collate_fn_inference([(text, schema)])
    with torch.no_grad():
        hidden = ext.encoder(input_ids=b.input_ids, attention_mask=b.attention_mask).last_hidden_state
        tok, sch = ext.processor.extract_embeddings_from_batch(hidden, b.input_ids, b)
        span_info = ext.compute_span_rep(tok[0])
        embs = torch.stack(sch[0][0])
        struct_proj = ext.count_embed(embs[1:], 1)
        scores = torch.sigmoid(torch.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj))
    return scores[0].float()  # [M, tl, K]


def ref_entities(ext, text, labels, thr):
    schema = ext.create_schema().entities(labels)
    r = ext.extract(text, schema, threshold=thr, include_confidence=False, include_spans=False)
    return {k: (v if isinstance(v, list) else [v] if v else []) for k, v in r.get("entities", {}).items()}


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("-S", type=int, default=256)
    ap.add_argument("-T", type=int, default=96)
    ap.add_argument("--mmax", type=int, default=16)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-convert", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    S, T, MMAX, THR = args.S, args.T, args.mmax, args.threshold

    from gliner2 import GLiNER2
    ext = GLiNER2.from_pretrained(MODEL_ID)
    ext.eval()
    k = ext.span_rep.span_rep_layer.max_width
    assert len(DEMO_LABELS) <= MMAX
    module = GLiNER2Extractor(ext, S, T, MMAX, k).eval()
    print(f"[INFO] S={S} T={T} MMAX={MMAX} K={k} dtype={args.dtype}")

    # GATE-1: raw span_scores == GLiNER2 internal path
    print("\n[GATE-1] fp32 span_scores vs GLiNER2 internal path:")
    worst = 1.0
    for text in DEMO_TEXTS:
        c, _ = collate(ext, text, DEMO_LABELS)
        assert len(c["ids"]) <= S and len(c["twi"]) <= T and len(c["ssi"]) <= MMAX + 1, "grow S/T/MMAX"
        ii, am, wi, si = make_inputs(c, S, T, MMAX)
        tl = len(c["twi"])
        M = len(DEMO_LABELS)
        with torch.no_grad():
            ss = module(ii, am, wi, si)[0][:M, :tl, :]
            ref = reference_scores(ext, text, DEMO_LABELS)
        # Only spans with start+width < tl are ever read by the host decoder; padded-word span
        # entries (start+width >= tl) are garbage in both paths (differently), so mask them out.
        starts = torch.arange(tl).unsqueeze(1)
        widths = torch.arange(k).unsqueeze(0)
        vmask = ((starts + widths) < tl).unsqueeze(0).expand(M, -1, -1)         # [M,tl,K]
        cos = F.cosine_similarity(ss[vmask].reshape(-1), ref[vmask].reshape(-1), dim=0).item()
        worst = min(worst, cos)
        print(f"   cos={cos:.6f} | {text[:44]!r}")
    print(f"[GATE-1] worst cos={worst:.6f}")
    assert worst > 0.9999, "span_scores diverge"

    # GATE-2: decoded entities == ext.extract
    print("\n[GATE-2] fp32 decoded entities vs ext.extract:")
    ok = True
    for text in DEMO_TEXTS:
        c, _ = collate(ext, text, DEMO_LABELS)
        ii, am, wi, si = make_inputs(c, S, T, MMAX)
        with torch.no_grad():
            ss = module(ii, am, wi, si)[0]
        mine = {kk: vv for kk, vv in decode(ss, DEMO_LABELS, text, c["smap"], c["emap"], THR).items() if vv}
        ref = {kk: vv for kk, vv in ref_entities(ext, text, DEMO_LABELS, THR).items() if vv}
        same = mine == ref
        ok &= same
        print(f"   {'OK ' if same else 'DIFF'} | {text[:40]!r} -> {mine}")
        if not same:
            print(f"        ref={ref}")
    assert ok, "decoded entities diverge from ext.extract"
    print("[PASS] fp32 self-gates green.")

    # reference fixture (first demo text) for the engine/Swift gate — captured at fp32
    c0, _ = collate(ext, DEMO_TEXTS[0], DEMO_LABELS)
    fi = make_inputs(c0, S, T, MMAX)
    with torch.no_grad():
        ref_scores = module(*fi)[0].float()
    # capture the fixture's reference entities BEFORE half() mutates the shared ext submodules
    ref_ent0 = {k: v for k, v in ref_entities(ext, DEMO_TEXTS[0], DEMO_LABELS, THR).items() if v}

    if args.dtype == "float16":
        module.half()
        _keep_fp32_buffers(module)

    if args.no_convert:
        print("[SKIP] convert"); return

    from coreai_torch import TorchConverter, get_decomp_table
    from coreai.runtime import AIModelAssetMetadata
    out_dir = Path(args.output_dir)
    model_path = out_dir / f"gliner2-pii_{args.dtype}_s{S}_t{T}_m{MMAX}.aimodel"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"{model_path} exists; --overwrite")
    print("\n[INFO] torch.export + convert...")
    exported = torch.export.export(module, args=(), kwargs={
        "input_ids": fi[0], "attention_mask": fi[1], "text_word_idx": fi[2], "schema_idx": fi[3]})
    exported = exported.run_decompositions(get_decomp_table())
    conv = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "attention_mask", "text_word_idx", "schema_idx"],
        output_names=["span_scores"])
    prog = conv.to_coreai(); prog.optimize()
    print("[INFO] === CONVERT OK ===")
    if model_path.exists():
        shutil.rmtree(model_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = AIModelAssetMetadata()
    meta.author = "fastino (GLiNER2-PII); mDeBERTa-v3 base (Microsoft, MIT)"
    meta.license = "Apache-2.0"
    meta.model_description = ("GLiNER2 schema-driven information extraction (PII) — mDeBERTa-v3 + "
                              "SpanMarker + CountLSTM, fused static graph.")
    meta.creation_date = int(time.time())
    prog.save_asset(model_path, meta)
    print(f"[INFO] saved {model_path}")
    ext.processor.tokenizer.save_pretrained(out_dir / "tokenizer")
    _finalize_tokenizer_and_extractor(ext, out_dir, S, T, MMAX, k, THR)
    (out_dir / "reference.json").write_text(json.dumps({
        "model": MODEL_ID, "S": S, "T": T, "MMAX": MMAX, "K": k, "threshold": THR,
        "demo_labels": DEMO_LABELS, "demo_text": DEMO_TEXTS[0],
        "input_ids": fi[0][0].tolist(), "attention_mask": fi[1][0].tolist(),
        "text_word_idx": fi[2][0].tolist(), "schema_idx": fi[3][0].tolist(),
        "start_map": c0["smap"], "end_map": c0["emap"],
    }, indent=2, ensure_ascii=False))

    asyncio.run(engine_gate(model_path, fi, ref_scores, DEMO_TEXTS[0], c0, DEMO_LABELS, THR, ref_ent0))


def _finalize_tokenizer_and_extractor(ext, out_dir, S, T, MMAX, k, THR):
    # Make the exported tokenizer folder consumable by swift-transformers, and emit the small
    # config the Swift/kit collator needs.
    #
    # (1) Route DebertaV2Tokenizer -> XLMRobertaTokenizer. Both are the same SentencePiece/Unigram
    #     family; swift-transformers has no DebertaV2 class and its non-strict fallback is *BPE*
    #     (wrong for an SPM model), so we alias to a Unigram-mapped class name. Verified per-word
    #     tokenize() byte-identical to HF for URLs/emails/accents/hyphens/numbers.
    # (2) GLiNER's special markers ([P],[E],...) live in added_tokens ABOVE the 250101-entry Unigram
    #     vocab, so convertTokensToIds returns [UNK] for them in Swift. The collator emits their ids
    #     directly from this map instead. `pool_ids` = {[P],[C],[E],[R],[L]} triggers schema pooling.
    proc = ext.processor
    tok = proc.tokenizer
    cfg_path = out_dir / "tokenizer" / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["tokenizer_class"] = "XLMRobertaTokenizer"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    marker_ids = {t: tok.convert_tokens_to_ids(t)
                  for t in (proc.P_TOKEN, proc.E_TOKEN, proc.SEP_TEXT, proc.SEP_STRUCT)}
    pool_ids = [tok.convert_tokens_to_ids(t)
                for t in (proc.P_TOKEN, proc.C_TOKEN, proc.E_TOKEN, proc.R_TOKEN, proc.L_TOKEN)]
    (out_dir / "extractor.json").write_text(json.dumps({
        "S": S, "T": T, "MMAX": MMAX, "K": k, "threshold": THR,
        "prompt_token": "entities", "marker_ids": marker_ids, "pool_ids": pool_ids,
    }, indent=2, ensure_ascii=False))
    print(f"[INFO] tokenizer_class -> XLMRobertaTokenizer; wrote extractor.json {marker_ids}")


def _keep_fp32_buffers(module):
    n = 0
    for m in module.modules():
        for bn, buf in list(m.named_buffers(recurse=False)):
            if buf is not None and buf.dtype == torch.float16 and ("rel_embeddings" in bn or "position" in bn):
                setattr(m, bn, buf.float()); n += 1
    if n:
        print(f"[INFO] kept {n} rel-pos buffer(s) fp32")


async def engine_gate(path, fi, ref_scores, text, c0, labels, thr, ref_ent):
    import coreai.runtime as rt
    print("\n[INFO] engine (GPU)...")
    m = await rt.AIModel.load(
        str(path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    res = await asyncio.wait_for(fn(inputs={
        "input_ids": rt.NDArray(np.ascontiguousarray(fi[0].numpy())),
        "attention_mask": rt.NDArray(np.ascontiguousarray(fi[1].numpy())),
        "text_word_idx": rt.NDArray(np.ascontiguousarray(fi[2].numpy())),
        "schema_idx": rt.NDArray(np.ascontiguousarray(fi[3].numpy())),
    }), timeout=300)
    eng = torch.from_numpy(res["span_scores"].numpy().astype(np.float32))[0]
    cos = F.cosine_similarity(eng.reshape(-1), ref_scores.reshape(-1), dim=0).item()
    print(f"[GATE-3] engine-vs-fp32 span_scores cos={cos:.6f} max|Δ|={(eng-ref_scores).abs().max():.4f}")
    eng_ent = {k: v for k, v in decode(eng, labels, text, c0["smap"], c0["emap"], thr).items() if v}
    print(f"[GATE-3] engine entities = {eng_ent}")
    print(f"[GATE-3] ref    entities = {ref_ent}")
    print("[PASS] engine gate green" if (cos > 0.999 and eng_ent == ref_ent) else "[WARN] look")


if __name__ == "__main__":
    main()
