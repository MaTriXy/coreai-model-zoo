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
# Export Qwen/Qwen3-Reranker-0.6B (cross-encoder: same Qwen3-0.6B backbone + LM head) as a
# single static Core AI graph:
#   (input_ids [1,S] int32, attention_mask [1,S] int32) -> probs [1, 2] = softmax([no, yes])
# The relevance score is probs[0, 1] = P(yes).
#
# Unlike the embedder (which pools a vector), the reranker reads the LM's next-token logits at
# the LAST real token and compares the "yes" vs "no" tokens. We bake the whole tail into the
# graph: gather the last real token's hidden (mask-based, right-pad safe under the causal mask),
# apply the LM head to THAT ONE position only (cheap; full-sequence logits are never needed),
# select the {no, yes} logits, softmax -> [P_no, P_yes]. One .aimodel forward = one score.
#
# Host formats the pair exactly like the official model card:
#   prefix + "<Instruct>: {instr}\n<Query>: {q}\n<Document>: {d}" + suffix
# then right-pads to the grid. The graph does the rest.
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"

# Official prompt scaffolding (Qwen3-Reranker model card).
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the "
          "Query and the Instruct provided. Note that the answer can only be \"yes\" or "
          "\"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"

# 3 relevant + 3 irrelevant query/doc pairs (EN + JA). Relevant must score P(yes) > 0.5 and
# every relevant pair must outrank the irrelevant pairs sharing its query (the ranking anchor).
PAIRS = {
    "rel_capital": (True, "What is the capital of Japan?", "Tokyo is the capital and largest city of Japan."),
    "rel_beesting": (True, "How do I treat a bee sting?", "Remove the stinger, wash with soap and water, then apply a cold pack to reduce swelling."),
    "rel_fuji_ja": (True, "富士山の高さはどのくらいですか？", "富士山は標高3,776メートルで、日本で最も高い山です。"),
    "irr_capital": (False, "What is the capital of Japan?", "Python is the most widely used programming language for machine learning."),
    "irr_beesting": (False, "How do I treat a bee sting?", "Tokyo is the capital and largest city of Japan."),
    "irr_fuji_ja": (False, "富士山の高さはどのくらいですか？", "The recipe calls for two eggs and a cup of flour."),
}
# Each relevant pair shares a query with the irrelevant pair of the same topic -> rank check.
RANK_GROUPS = {"capital": ("rel_capital", "irr_capital"),
               "beesting": ("rel_beesting", "irr_beesting"),
               "fuji_ja": ("rel_fuji_ja", "irr_fuji_ja")}


def format_instruction(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


class RerankerModule(torch.nn.Module):
    # Qwen3 backbone forward -> last-real-token hidden -> LM head (one position) ->
    # softmax over {no, yes} -> [1, 2] probabilities. P(yes) = output[:, 1].
    def __init__(self, model, no_id: int, yes_id: int):
        super().__init__()
        self.backbone = model.model      # Qwen3Model
        self.lm_head = model.lm_head     # tied to embeddings
        self.register_buffer("yn_ids", torch.tensor([no_id, yes_id], dtype=torch.long))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).last_hidden_state          # [B, S, H]
        bsz, _, hdim = hidden.shape
        idx = attention_mask.to(torch.long).sum(dim=1) - 1                 # [B] last real pos
        gather_idx = idx.view(bsz, 1, 1).expand(bsz, 1, hdim)              # [B, 1, H]
        last_hidden = hidden.gather(1, gather_idx).squeeze(1)              # [B, H]
        logits = self.lm_head(last_hidden)                                 # [B, vocab]
        yn = logits.index_select(1, self.yn_ids)                           # [B, 2] = [no, yes]
        return F.softmax(yn, dim=-1)                                       # [B, 2]


def build_ids(tok, instruction, query, doc, seq_len, pad):
    body = format_instruction(instruction, query, doc)
    ids = (tok.encode(PREFIX, add_special_tokens=False)
           + tok.encode(body, add_special_tokens=False)
           + tok.encode(SUFFIX, add_special_tokens=False))
    real = len(ids)
    if pad:
        if real > seq_len:
            raise ValueError(f"pair needs {real} tokens > grid {seq_len}; raise --seq-len")
        ids = ids + [tok.pad_token_id] * (seq_len - real)               # right-pad
    mask = [1] * real + [0] * (len(ids) - real)
    return (torch.tensor([ids], dtype=torch.int32),
            torch.tensor([mask], dtype=torch.int32), real)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    torch.manual_seed(0)

    print("[INFO] Sourcing model (CPU, fp32)...")
    # fp32 throughout (clean oracle/reference); the fp16 graph is produced by module.half()
    # below, NOT autocast (autocast-fp16 collides with Qwen3 RMSNorm's fp32 roundtrip).
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float32).eval()
    no_id = tok.convert_tokens_to_ids("no")
    yes_id = tok.convert_tokens_to_ids("yes")
    print(f"[INFO] yes={yes_id} no={no_id} pad={tok.pad_token_id} vocab={model.config.vocab_size}")

    module = RerankerModule(model, no_id, yes_id).eval()

    # ---- oracle (official scoring, per pair, no padding -> logits[:, -1]) + self-check ----
    @torch.no_grad()
    def official_score(query, doc):
        ids, mask, _ = build_ids(tok, DEFAULT_INSTRUCTION, query, doc, args.seq_len, pad=False)
        logits = model(input_ids=ids.long(), attention_mask=mask.long()).logits[:, -1, :]
        yn = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1)
        return float(F.softmax(yn, dim=1)[0, 1])

    oracle, mine, lengths = {}, {}, {}
    with torch.no_grad():
        for key, (_, q, d) in PAIRS.items():
            oracle[key] = official_score(q, d)
            ids, mask, real = build_ids(tok, DEFAULT_INSTRUCTION, q, d, args.seq_len, pad=True)
            mine[key] = float(module(ids, mask)[0, 1])
            lengths[key] = real
    max_diff = max(abs(oracle[k] - mine[k]) for k in PAIRS)
    print("[CHECK] P(yes): official | wrapper(padded grid) | |diff|  (len)")
    for k, (rel, _, _) in PAIRS.items():
        tag = "REL" if rel else "irr"
        print(f"          {k:14s} {tag}  {oracle[k]:.4f} | {mine[k]:.4f} | "
              f"{abs(oracle[k]-mine[k]):.5f}  ({lengths[k]} tok)")
    print(f"[CHECK] max |official - wrapper| = {max_diff:.5f}")
    assert max_diff < 1e-3, f"wrapper diverges from official scoring ({max_diff})"

    # ---- relevance gate: relevant > 0.5 > irrelevant, and relevant outranks same-query irr ----
    for key, (rel, _, _) in PAIRS.items():
        side = mine[key] > 0.5
        print(f"[CHECK] {key:14s} P(yes)={mine[key]:.4f} -> {'relevant' if side else 'irrelevant'}"
              f" {'OK' if side == rel else '** WRONG SIDE **'}")
        assert side == rel, f"{key} on wrong side of 0.5"
    for topic, (rk, ik) in RANK_GROUPS.items():
        print(f"[CHECK] rank[{topic}] rel {mine[rk]:.4f} > irr {mine[ik]:.4f} "
              f"{'OK' if mine[rk] > mine[ik] else '** INVERTED **'}")
        assert mine[rk] > mine[ik], f"ranking inverted for {topic}"

    # ---- export ----
    ids0, mask0, _ = build_ids(tok, DEFAULT_INSTRUCTION, *PAIRS["rel_capital"][1:], args.seq_len, pad=True)
    module.to(dtype)   # true fp16 weights (Qwen3 RMSNorm upcasts internally; safe)
    print("[INFO] Running torch export with decompositions...")
    example = {"input_ids": ids0.clone(), "attention_mask": mask0.clone()}
    exported = torch.export.export(module, args=(), kwargs=example)
    exported = exported.run_decompositions(get_decomp_table())

    print("[INFO] Converting to Core AI...")
    converter = TorchConverter().add_exported_program(
        exported_program=exported,
        input_names=["input_ids", "attention_mask"],
        output_names=["probs"],
    )
    coreai_program = converter.to_coreai()
    coreai_program.optimize()
    print("[INFO] Model optimized.")

    out_dir = Path(args.output_dir)
    model_path = out_dir / f"qwen3-reranker-0.6b_{args.dtype}_s{args.seq_len}_static.aimodel"
    if model_path.exists():
        if not args.overwrite:
            raise FileExistsError(f"{model_path} exists; pass --overwrite")
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = AIModelAssetMetadata()
    metadata.author = "Alibaba Qwen"
    metadata.license = "Apache-2.0"
    metadata.model_description = (
        "Qwen3-Reranker-0.6B cross-encoder reranker (Qwen3-0.6B backbone; yes/no logit score). "
        "Output probs[1] = P(yes) = relevance. Source: "
        "https://huggingface.co/Qwen/Qwen3-Reranker-0.6B")
    metadata.creation_date = int(time.time())
    coreai_program.save_asset(model_path, metadata)
    print(f"[INFO] Saved {model_path}")

    ref_payload = {
        "model": MODEL_NAME,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "yes_id": yes_id,
        "no_id": no_id,
        "pad_token_id": int(tok.pad_token_id),
        "padding_side": "right",
        "prefix": PREFIX,
        "suffix": SUFFIX,
        "default_instruction": DEFAULT_INSTRUCTION,
        "output": "probs [1,2] = softmax([no, yes]); relevance = probs[1] = P(yes)",
        "pairs": {k: {"relevant": v[0], "query": v[1], "doc": v[2]} for k, v in PAIRS.items()},
        "scores": mine,
        "official_scores": oracle,
        "rank_groups": RANK_GROUPS,
    }
    (out_dir / "reference.json").write_text(json.dumps(ref_payload, indent=2, ensure_ascii=False))
    print(f"[INFO] Saved {out_dir / 'reference.json'}")

    tok_dir = out_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    tok.save_pretrained(tok_dir)
    print(f"[INFO] Saved tokenizer to {tok_dir}")
    print("[DONE] All torch-ladder gates passed; bundle written.")


if __name__ == "__main__":
    main()
