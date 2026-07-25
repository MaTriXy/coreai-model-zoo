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
# Export google/embeddinggemma-300m (full SentenceTransformer pipeline: transformer ->
# mean pooling -> dense stack -> normalize) as a single static Core AI graph:
#   (input_ids [1,S] int32, attention_mask [1,S] int32) -> embedding [1, 768]
# The wrapper is verified against SentenceTransformer.encode() BEFORE export, and
# reference embeddings are dumped to JSON for the Swift-side parity test.
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from sentence_transformers import SentenceTransformer

MODEL_NAME = "google/embeddinggemma-300m"

REFERENCE_TEXTS = {
    "query_bike": ("query", "red bicycle parked at the beach"),
    "query_capital": ("query", "what is the capital of Japan"),
    "doc_bike": ("document", "A crimson bike leaning against a palm tree by the sea."),
    "doc_tokyo": ("document", "Tokyo is the capital and largest city of Japan."),
}


class EmbeddingModule(torch.nn.Module):
    # Runs the SentenceTransformer module chain on a features dict, exactly like
    # SentenceTransformer.forward, returning only the final sentence embedding.
    def __init__(self, st: SentenceTransformer):
        super().__init__()
        self.stages = torch.nn.ModuleList(list(st))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        for stage in self.stages:
            features = stage(features)
        return features["sentence_embedding"]


def tokenize(st: SentenceTransformer, text: str, seq_len: int):
    tok = st.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    return tok["input_ids"].to(torch.int32), tok["attention_mask"].to(torch.int32)


def prompted(st: SentenceTransformer, kind: str, text: str) -> str:
    # Use the prompts shipped in the ST config so Swift can mirror them exactly.
    prompts = st.prompts or {}
    if kind == "query":
        template = prompts.get("query", "task: search result | query: ")
    else:
        template = prompts.get("document", "title: none | text: ")
    return template + text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    print("[INFO] Sourcing model...")
    # CPU throughout: torch.export traces on CPU, and a mixed CPU/MPS pipeline crashes.
    st = SentenceTransformer(MODEL_NAME, device="cpu")
    st.eval()
    print(f"[INFO] ST prompts: {st.prompts}")

    module = EmbeddingModule(st)
    module.eval()

    # Self-check in fp32 BEFORE export: wrapper must reproduce st.encode().
    text = prompted(st, "query", "red bicycle parked at the beach")
    ref = torch.tensor(st.encode([text], normalize_embeddings=True))
    ids, mask = tokenize(st, text, args.seq_len)
    with torch.no_grad():
        mine = module(ids, mask)
    cos = torch.nn.functional.cosine_similarity(ref, mine.float()).item()
    print(f"[CHECK] wrapper vs st.encode cosine = {cos:.6f}")
    assert cos > 0.999, f"wrapper does not match SentenceTransformer ({cos})"

    # Reference embeddings for the Swift parity test — computed in fp32 BEFORE the dtype
    # cast (CPU half inference produces NaN in torch; fp32-vs-fp16 cosine drift is well
    # inside the test tolerance).
    vectors = {}
    with torch.no_grad():
        for key, (kind, raw) in REFERENCE_TEXTS.items():
            ids_r, mask_r = tokenize(st, prompted(st, kind, raw), args.seq_len)
            emb = module(ids_r, mask_r)
            vectors[key] = [float(x) for x in emb[0]]
    if args.dtype != "float16":
        module.to(dtype)
    # float16: keep fp32 weights and let autocast put matmuls in fp16 at trace time.
    # A full .to(float16) overflows Gemma3 activations — the graph emits NaN embeddings
    # (verified on-device via the Swift parity test).
    cosines = {}
    for a in REFERENCE_TEXTS:
        for b in REFERENCE_TEXTS:
            if a < b:
                va = torch.tensor(vectors[a])
                vb = torch.tensor(vectors[b])
                cosines[f"{a}|{b}"] = float(
                    torch.nn.functional.cosine_similarity(va, vb, dim=0))
    print(f"[CHECK] reference cosines: {json.dumps(cosines, indent=2)}")

    print("[INFO] Running torch export with decompositions...")
    example = {
        "input_ids": ids.clone(),
        "attention_mask": mask.clone(),
    }
    if args.dtype == "float16":
        with torch.autocast(device_type="cpu", dtype=dtype):
            exported = torch.export.export(module, args=(), kwargs=example)
    else:
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
    model_path = out_dir / f"embeddinggemma-300m_{args.dtype}_static.aimodel"
    if model_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{model_path} exists; pass --overwrite")
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = AIModelAssetMetadata()
    metadata.author = "Google DeepMind"
    metadata.license = "Gemma"
    metadata.model_description = (
        "EmbeddingGemma-300m text embedding model (mean pooling + dense projection, "
        "L2-normalized 768-d). Source: https://huggingface.co/google/embeddinggemma-300m"
    )
    metadata.creation_date = int(time.time())
    coreai_program.save_asset(model_path, metadata)
    print(f"[INFO] Saved {model_path}")

    ref_payload = {
        "model": MODEL_NAME,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "prompts": st.prompts,
        "texts": {k: {"kind": v[0], "text": v[1]} for k, v in REFERENCE_TEXTS.items()},
        "cosines": cosines,
    }
    ref_path = out_dir / "reference.json"
    ref_path.write_text(json.dumps(ref_payload, indent=2))
    print(f"[INFO] Saved {ref_path}")

    # Ship the tokenizer next to the model for the Swift side.
    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    st.tokenizer.save_pretrained(tok_dir)
    print(f"[INFO] Saved tokenizer to {tok_dir}")


if __name__ == "__main__":
    main()
