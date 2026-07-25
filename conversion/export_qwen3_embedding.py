# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.1",
#     "sentence-transformers>=5.0",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Export Qwen/Qwen3-Embedding-0.6B (full SentenceTransformer pipeline: Qwen3 backbone ->
# last-token (EOS) pooling -> L2 normalize) as a single static Core AI graph:
#   (input_ids [1,S] int32, attention_mask [1,S] int32) -> embedding [1, 1024] (unit vector)
#
# This is an ENCODER, not a generator: one forward over the (right-padded) input -> one
# pooled, L2-normalized vector. No autoregressive loop, no KV cache, no LM head, no sampling.
# It is exported / run exactly like the vision encoders (plain .aimodel via AIModel.run),
# NOT the pipelined generate engine.
#
# Why right-padding + fixed grid: last-token pooling under a causal mask is padding-safe on
# the RIGHT (real tokens never attend to trailing pads, so the last real token's hidden state
# is identical with or without pads). A fixed grid avoids per-length respecialization on device.
# sentence-transformers' last-token Pooling locates the last real token from the attention mask
# (index of the first 0, minus 1), so the gather is baked into the graph.
#
# The wrapper is verified against SentenceTransformer.encode() BEFORE export (per-text cosine,
# the full N x N retrieval-similarity matrix, and MRL-truncated rankings), and reference
# embeddings are dumped to JSON for the Swift / engine parity test.
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from sentence_transformers import SentenceTransformer

MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
EMBED_DIM = 1024
# MRL truncation dims to validate (Matryoshka: truncate the 1024-d vector then re-L2-normalize).
MRL_DIMS = [1024, 512, 256, 128]

# Reference texts: 3 EN query/doc pairs + 1 JA pair. Each query's matching doc must rank
# highest -> the cosine matrix is the retrieval-order anchor across every code path.
REFERENCE_TEXTS = {
    "q_capital": ("query", "What is the capital of Japan?"),
    "q_beesting": ("query", "How do I treat a bee sting?"),
    "q_mllang": ("query", "best programming language for machine learning"),
    "q_fuji_ja": ("query", "富士山の高さはどのくらいですか？"),
    "d_tokyo": ("document", "Tokyo is the capital and largest city of Japan."),
    "d_beesting": ("document", "For a bee sting, remove the stinger, wash the area with soap "
                               "and water, then apply a cold pack to reduce swelling."),
    "d_python": ("document", "Python is the most widely used programming language for machine "
                             "learning and data science."),
    "d_fuji_ja": ("document", "富士山は標高3,776メートルで、日本で最も高い山です。"),
}
# The human-intended query -> document pairing (for labeling only). The GATE is port
# fidelity (mine reproduces the OFFICIAL model's ranking), not agreement with this guess:
# e.g. the official model answers the JA capital query with the *English* Tokyo doc (a
# legitimate cross-lingual near-tie), and a faithful port must reproduce exactly that.
INTENDED_PAIRS = {
    "q_capital": "d_tokyo",
    "q_beesting": "d_beesting",
    "q_mllang": "d_python",
    "q_fuji_ja": "d_fuji_ja",
}
QUERY_KEYS = [k for k in REFERENCE_TEXTS if k.startswith("q_")]
# Clear-margin threshold (top1 - top2 cosine): rankings above it are gated hard; near-ties
# below it are reported, not failed (cf. the RF-DETR / argmax-margin rule for near-duplicates).
MARGIN = 0.05


class EmbeddingModule(torch.nn.Module):
    # Runs the SentenceTransformer module chain (Transformer -> last-token Pooling ->
    # Normalize) on a features dict, returning only the final L2-normalized sentence
    # embedding. The explicit re-normalize is a no-op when the chain already normalizes
    # (re-normalizing a unit vector is identity) and a safety net otherwise.
    def __init__(self, st: SentenceTransformer):
        super().__init__()
        self.stages = torch.nn.ModuleList(list(st))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        for stage in self.stages:
            features = stage(features)
        return F.normalize(features["sentence_embedding"], p=2, dim=1)


def tokenize(st: SentenceTransformer, text: str, seq_len: int | None):
    # Right-pad to a fixed grid (last-token pooling under causal attention is right-pad safe).
    # seq_len=None -> no padding (the apples-to-apples reference for st.encode).
    if seq_len is None:
        tok = st.tokenizer(text, return_tensors="pt")
    else:
        tok = st.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt",
            padding_side="right",
        )
    return tok["input_ids"].to(torch.int32), tok["attention_mask"].to(torch.int32)


def prompted(st: SentenceTransformer, kind: str, text: str) -> str:
    # Use the prompts shipped in the ST config so the host (Swift) can mirror them exactly:
    # queries get the "Instruct: ...\nQuery:" prefix, documents get no prefix.
    prompts = st.prompts or {}
    return prompts.get(kind, "") + text


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0))


def sim_matrix(emb: dict) -> dict:
    keys = list(emb)
    out = {}
    for a in keys:
        for b in keys:
            if a < b:
                out[f"{a}|{b}"] = cos(emb[a], emb[b])
    return out


def ranked_docs(emb: dict, query_key: str) -> list:
    docs = [k for k in emb if k.startswith("d_")]
    return sorted(((d, cos(emb[query_key], emb[d])) for d in docs),
                  key=lambda kv: kv[1], reverse=True)


def topdoc(emb: dict, query_key: str) -> str:
    return ranked_docs(emb, query_key)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    torch.manual_seed(0)

    print("[INFO] Sourcing model (CPU, fp32; torch.export traces on CPU)...")
    # Force fp32: the checkpoint is bf16, but the oracle/reference must be clean fp32, and
    # the fp16 export uses autocast over fp32 weights (autocast targeting fp16 clashes with
    # bf16 weights in Qwen3's RMSNorm fp32-roundtrip -> _assert_tensor_metadata mismatch).
    st = SentenceTransformer(MODEL_NAME, device="cpu",
                             model_kwargs={"torch_dtype": torch.float32})
    st.eval()
    print(f"[INFO] ST modules: {[type(m).__name__ for m in st]}")
    print(f"[INFO] ST prompts: {st.prompts}")

    module = EmbeddingModule(st)
    module.eval()

    # ---- fp32 oracle + wrapper self-check ----
    # mine     = fixed-grid right-padded wrapper (the exact path the device runs).
    # mine_nopad = unpadded wrapper (apples-to-apples vs st.encode's single-text encode).
    # The padded-vs-unpadded gap isolates the SDPA mask-path fp noise (real tokens never
    # attend to trailing pads under the causal mask, but the padded path picks a different
    # SDPA kernel) from any pooling/gather bug — which would show up as nopad drift too.
    oracle, mine, mine_nopad = {}, {}, {}
    with torch.no_grad():
        for key, (kind, raw) in REFERENCE_TEXTS.items():
            text = prompted(st, kind, raw)
            oracle[key] = torch.tensor(st.encode([text], normalize_embeddings=True)[0])
            ids, mask = tokenize(st, text, args.seq_len)
            mine[key] = module(ids, mask)[0]
            ids_n, mask_n = tokenize(st, text, None)
            mine_nopad[key] = module(ids_n, mask_n)[0]
    per_text = {k: cos(oracle[k], mine[k]) for k in REFERENCE_TEXTS}
    per_text_nopad = {k: cos(oracle[k], mine_nopad[k]) for k in REFERENCE_TEXTS}
    worst = min(per_text.values())
    print("[CHECK] wrapper-vs-st.encode per-text cosine (padded grid | unpadded):")
    for k in REFERENCE_TEXTS:
        print(f"          {k:14s} {per_text[k]:.6f} | {per_text_nopad[k]:.6f}")
    print(f"[CHECK] worst per-text cosine = {worst:.6f} (padded) "
          f"/ {min(per_text_nopad.values()):.6f} (unpadded)")
    # >0.999 is the embedder ship bar (cf. EmbeddingGemma); the unpadded path should be
    # near-exact, confirming the gather/pooling is correct and the gap is pad-path fp noise.
    assert worst > 0.999, f"wrapper diverges from SentenceTransformer (worst {worst})"
    assert min(per_text_nopad.values()) > 0.9999, "unpadded wrapper drift => pooling/gather bug"

    # ---- retrieval port fidelity: mine reproduces the OFFICIAL model's top-1 ranking ----
    # (absolute cosine values may drift ~1e-3 from the pad-path noise; what must hold is the
    # retrieval ORDER. Clear-margin queries are gated hard; near-ties are reported.)
    m_oracle, m_mine = sim_matrix(oracle), sim_matrix(mine)
    max_dsim = max(abs(m_oracle[k] - m_mine[k]) for k in m_oracle)
    print(f"[CHECK] N x N similarity matrix max |oracle - mine| = {max_dsim:.6f} (informational)")
    mine_top, margins = {}, {}
    print("[CHECK] retrieval top-1 (oracle vs mine | margin = top1-top2 of mine):")
    for qk in QUERY_KEYS:
        ro, rm = ranked_docs(oracle, qk), ranked_docs(mine, qk)
        mine_top[qk] = rm[0][0]
        margins[qk] = rm[0][1] - rm[1][1]
        clear = margins[qk] > MARGIN
        agree = rm[0][0] == ro[0][0]
        print(f"          {qk:14s} oracle->{ro[0][0]:11s} mine->{rm[0][0]:11s} "
              f"margin {margins[qk]:.3f}{'' if clear else ' (near-tie)'}  "
              f"intended {INTENDED_PAIRS[qk]:11s} {'OK' if agree else '** PORT DIFFERS **'}")
        # Port fidelity is mandatory for clear-margin queries; near-ties may flip on fp noise.
        if clear:
            assert agree, f"port changes the ranking for clear-margin query {qk}"

    # ---- MRL: truncate + re-normalize; clear-margin rankings preserved at every dim ----
    print("[CHECK] MRL truncation rankings (vs mine full-dim top-1):")
    for dim in MRL_DIMS:
        trunc = {k: F.normalize(v[:dim], p=2, dim=0) for k, v in mine.items()}
        flips = [qk for qk in QUERY_KEYS if topdoc(trunc, qk) != mine_top[qk]]
        clear_flips = [qk for qk in flips if margins[qk] > MARGIN]
        print(f"          dim {dim:4d}  flips: {flips or 'none'}"
              f"{'' if not clear_flips else f'  CLEAR-MARGIN FLIPS {clear_flips}'}")
        assert not clear_flips, f"MRL dim {dim} reorders a clear-margin query: {clear_flips}"

    # Reference embeddings for the Swift / engine parity test — computed in fp32 (the
    # CoreAI-path wrapper output, the value the .aimodel must reproduce on device).
    vectors = {k: [float(x) for x in mine[k]] for k in REFERENCE_TEXTS}

    # ---- export ----
    ids, mask = tokenize(st, prompted(st, "query", REFERENCE_TEXTS["q_capital"][1]), args.seq_len)
    # fp16: cast the module to true fp16 weights. Qwen3's RMSNorm upcasts activations to fp32
    # internally and casts back, so the norm stays numerically safe in an fp16 graph. (autocast
    # to fp16 instead trips torch.export's _assert_tensor_metadata on that fp32 roundtrip; and
    # Qwen3-0.6B activations, unlike Gemma3, do not overflow fp16 -- verified by the engine gate.)
    module.to(dtype)
    print("[INFO] Running torch export with decompositions...")
    example = {"input_ids": ids.clone(), "attention_mask": mask.clone()}
    exported = torch.export.export(module, args=(), kwargs=example)
    exported = exported.run_decompositions(get_decomp_table())

    print("[INFO] Converting to Core AI...")
    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "attention_mask"],
        output_names=["embedding"],
    )
    coreai_program = converter.to_coreai()
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    out_dir = Path(args.output_dir)
    model_path = out_dir / f"qwen3-embedding-0.6b_{args.dtype}_s{args.seq_len}_static.aimodel"
    if model_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{model_path} exists; pass --overwrite")
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = AIModelAssetMetadata()
    metadata.author = "Alibaba Qwen"
    metadata.license = "Apache-2.0"
    metadata.model_description = (
        "Qwen3-Embedding-0.6B text embedding model (Qwen3-0.6B backbone, last-token pooling, "
        "L2-normalized 1024-d, MRL-truncatable 32-1024). Source: "
        "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"
    )
    metadata.creation_date = int(time.time())
    coreai_program.save_asset(model_path, metadata)
    print(f"[INFO] Saved {model_path}")

    ref_payload = {
        "model": MODEL_NAME,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "embed_dim": EMBED_DIM,
        "mrl_dims": MRL_DIMS,
        "prompts": st.prompts,
        "pad_token_id": int(st.tokenizer.pad_token_id),
        "padding_side": "right",
        "texts": {k: {"kind": v[0], "text": v[1]} for k, v in REFERENCE_TEXTS.items()},
        "embeddings": vectors,
        "cosines": m_mine,
        "intended_pairs": INTENDED_PAIRS,
        "expected_topdoc": mine_top,        # the torch ranking the engine must reproduce
        "topdoc_margin": margins,           # top1-top2 cosine; gate hard only when > margin
        "margin": MARGIN,
        "selfcheck_worst_cos": worst,
    }
    ref_path = out_dir / "reference.json"
    ref_path.write_text(json.dumps(ref_payload, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved {ref_path}")

    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    st.tokenizer.save_pretrained(tok_dir)
    print(f"[INFO] Saved tokenizer to {tok_dir}")
    print("[DONE] All torch-ladder gates passed; bundle written.")


if __name__ == "__main__":
    main()
