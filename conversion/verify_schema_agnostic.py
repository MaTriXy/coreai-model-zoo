# /// script
# requires-python = ">=3.11"
# dependencies = ["gliner2[local]", "coreai-core==1.0.0b2", "numpy"]
# [tool.uv]
# index-url = "https://pypi.org/simple"
# prerelease = "allow"
# index-strategy = "unsafe-best-match"
# ///
# Prove the converted fp16 bundle is TRULY schema-agnostic: feed several DIFFERENT runtime label
# sets (never seen at export time) through the SAME engine and check decoded entities == ext.extract.
import asyncio, sys
from pathlib import Path
import numpy as np, torch
import coreai.runtime as rt
from gliner2 import GLiNER2

BUNDLE = sys.argv[1]
S, T, MMAX, THR = 256, 96, 16, 0.5
CASES = [
    ("Reset creds: user=jdoe pass=Hunter2! key=sk-9f2a at 10.0.0.4.",
     ["username", "password", "api key", "ip address"]),
    ("Acme Corp paid $4,200 to Globex on 2026-03-01 in Berlin.",
     ["organization", "money", "date", "location"]),
    ("Reach me: bob@x.io, +44 20 7946 0958, or @bobby on socials.",
     ["email", "phone number", "social media handle"]),
    ("Nothing sensitive here, just a walk in the park.",
     ["person", "email", "credit card number"]),
]


def pad(v, n, x): return (list(v) + [x] * n)[:n]


def decode(scores, labels, text, smap, emap, thr):
    out, tl = {}, len(smap)
    for idx, name in enumerate(labels):
        s = scores[idx]; cand = []
        for st in range(min(tl, s.shape[0])):
            for w in range(s.shape[1]):
                if st + w + 1 > tl: continue
                cf = float(s[st, w])
                if cf >= thr:
                    cs, ce = smap[st], emap[st + w]; sp = text[cs:ce].strip()
                    if sp: cand.append((sp, cf, cs, ce))
        cand.sort(key=lambda x: x[1], reverse=True); sel = []
        for sp, cf, cs, ce in cand:
            if not any(not (ce <= s2 or cs >= e2) for _, _, s2, e2 in sel): sel.append((sp, cf, cs, ce))
        out[name] = [x[0] for x in sel]
    return {k: v for k, v in out.items() if v}


async def main():
    ext = GLiNER2.from_pretrained("fastino/gliner2-privacy-filter-PII-multi"); ext.eval()
    m = await rt.AIModel.load(BUNDLE, rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    fn = m.load_function("main")
    allok = True
    for text, labels in CASES:
        schema = ext.create_schema().entities(labels)
        b = ext.processor.collate_fn_inference([(text, schema)])
        ids = b.input_ids[0].tolist(); twi = b.text_word_indices[0].tolist()[:b.text_word_counts[0]]
        ssi = list(b.schema_special_indices[0][0]); smap = b.start_mappings[0]; emap = b.end_mappings[0]
        assert len(ids) <= S and len(twi) <= T and len(ssi) <= MMAX + 1
        ii = np.array([pad(ids, S, 0)], np.int32); am = np.array([[1]*len(ids)+[0]*(S-len(ids))], np.int32)
        wi = np.array([pad(twi, T, 0)], np.int32); si = np.array([pad(ssi, MMAX+1, ssi[0])], np.int32)
        res = await asyncio.wait_for(fn(inputs={
            "input_ids": rt.NDArray(np.ascontiguousarray(ii)), "attention_mask": rt.NDArray(np.ascontiguousarray(am)),
            "text_word_idx": rt.NDArray(np.ascontiguousarray(wi)), "schema_idx": rt.NDArray(np.ascontiguousarray(si)),
        }), timeout=300)
        eng = torch.from_numpy(res["span_scores"].numpy().astype(np.float32))[0]
        mine = decode(eng, labels, text, smap, emap, THR)
        r = ext.extract(text, schema, threshold=THR)
        ref = {k: (v if isinstance(v, list) else [v]) for k, v in r.get("entities", {}).items() if v}
        ok = mine == ref; allok &= ok
        print(f"{'OK  ' if ok else 'DIFF'} labels={labels}")
        print(f"      text: {text}")
        print(f"      engine: {mine}")
        if not ok: print(f"      ref   : {ref}")
    print("\n[PASS] schema-agnostic engine verified" if allok else "\n[FAIL] mismatch")


asyncio.run(main())
