#!/usr/bin/env python3
"""Export mistralai/Shieldstral-1.0-3B as a single static Core AI classifier graph.

    (input_ids [1,S] int32, attention_mask [1,S] int32) -> probs [1,2] = softmax([no, yes])

Shieldstral is a *policy-conditioned* moderation model: the host writes the policy
in plain language (an Instruction and a Query) and the model answers "yes" or "no".
That answer is one next-token distribution restricted to two ids, so there is no
reason to ship a generative decoder for it. The whole tail is baked into the graph
— gather the last real token's hidden state (mask-based, right-pad safe under the
causal mask), apply *two rows* of the tied embedding as the head, softmax — and one
`.aimodel` forward is one verdict. No KV cache, no loop, no sampling.

Dropping the head to two rows is not just tidiness: the full head is 131 072 x 3072
= 805 MB of fp16 that a classifier never reads.

Two things this port rests on, both measured rather than assumed:

* `ministral3` is Mistral + YARN. The conversion venv is on transformers 4.57.6,
  which cannot load this checkpoint at all; 4.57.6's `MistralModel` carrying the
  checkpoint's `rope_parameters` as `rope_scaling` reproduces HF git main's native
  `ministral3` at cos 1.000000 on all nine oracle cases
  (`_smoke/test_shieldstral_torch_ladder.py`).
* The 4D mask is built here rather than left to HF. `create_causal_mask` returns a
  4D mask untouched, which keeps padding semantics explicit and export-stable.

Run:
    python conversion/export_shieldstral.py [fp16|int8lin|int4lin] [--seq-len 512]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import exports_dir, hf_snapshot  # noqa: E402
from _shieldstral_suite import SUITE  # the same nine cases the oracle scored  # noqa: E402

HF_ID = "mistralai/Shieldstral-1.0-3B"
DTYPE = torch.float16

# ---- host prompt contract (the chat template, flattened) -------------------------
# The template renders exactly: <s>[SYSTEM_PROMPT]{sys}[/SYSTEM_PROMPT][INST]{user}[/INST]
SYSTEM = ('Judge whether the Document meets the requirements based on the Query and the '
          'Instruction provided. Note that the answer can only be "yes" or "no".')
PREFIX = f"<s>[SYSTEM_PROMPT]{SYSTEM}[/SYSTEM_PROMPT][INST]"
SUFFIX = "[/INST]"
DEFAULT_INSTRUCTION = ("You are a strict safety moderator reviewing potentially harmful "
                       "content. Apply a low tolerance threshold.")
DEFAULT_QUERY = "Does this content promote physical violence?"


def format_body(instruction: str, query: str, document: str) -> str:
    return f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}"


def render_prompt(instruction: str, query: str, document: str) -> str:
    return PREFIX + format_body(instruction, query, document) + SUFFIX


def build_text_model(snap: Path, dtype=torch.float32):
    """4.57.6 `MistralModel` carrying Shieldstral's text_config, YARN included.

    Shared with `_smoke/test_shieldstral_torch_ladder.py`, which is what proves this
    construction equals HF git main's native `ministral3`.
    """
    from safetensors.torch import load_file
    from transformers import MistralConfig
    from transformers.models.mistral.modeling_mistral import MistralModel

    raw = json.loads((snap / "config.json").read_text())
    t = raw["text_config"]
    rope = dict(t["rope_parameters"])
    theta = rope.pop("rope_theta")
    rope.pop("type", None)                  # v5 duplicate of rope_type
    rope.pop("llama_4_scaling_beta", None)  # not consumed by the yarn init
    cfg = MistralConfig(
        vocab_size=t["vocab_size"], hidden_size=t["hidden_size"],
        intermediate_size=t["intermediate_size"],
        num_hidden_layers=t["num_hidden_layers"],
        num_attention_heads=t["num_attention_heads"],
        num_key_value_heads=t["num_key_value_heads"],
        head_dim=t["head_dim"], hidden_act=t["hidden_act"],
        max_position_embeddings=t["max_position_embeddings"],
        rms_norm_eps=t["rms_norm_eps"], rope_theta=theta, rope_scaling=rope,
        sliding_window=t["sliding_window"], tie_word_embeddings=True,
        attention_dropout=0.0, torch_dtype=dtype, attn_implementation="sdpa",
    )
    model = MistralModel(cfg).to(dtype).eval()

    sd = load_file(str(snap / "model.safetensors"))
    pref = "language_model.model."
    text_sd = {k[len(pref):]: v.to(dtype) for k, v in sd.items() if k.startswith(pref)}
    missing, unexpected = model.load_state_dict(text_sd, strict=False)
    missing = [m for m in missing if "rotary" not in m]
    if missing or unexpected:
        raise SystemExit(f"weight mismatch: missing={missing[:6]} unexpected={unexpected[:6]}")
    return model, cfg, text_sd["embed_tokens.weight"]


class ShieldstralClassifier(nn.Module):
    """Backbone -> last real token -> 2-row tied head -> softmax([no, yes])."""

    def __init__(self, backbone, embed_weight: torch.Tensor, no_id: int, yes_id: int):
        super().__init__()
        self.backbone = backbone
        # Two rows of the tied embedding ARE the head. Registered as a buffer so the
        # eager quantizer leaves them alone: 6 KB is not worth a scale factor.
        self.register_buffer("yn_head", embed_weight[[no_id, yes_id]].clone())

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        dtype = self.yn_head.dtype
        seq = input_ids.shape[1]
        keep = attention_mask.to(torch.bool)[:, None, None, :]            # [B,1,1,S] keys
        causal = torch.ones(seq, seq, dtype=torch.bool,
                            device=input_ids.device).tril()[None, None]
        mask4 = torch.where(keep & causal,
                            torch.zeros((), dtype=dtype, device=input_ids.device),
                            torch.full((), torch.finfo(dtype).min, dtype=dtype,
                                       device=input_ids.device))
        hidden = self.backbone(input_ids=input_ids.to(torch.long),
                               attention_mask=mask4,
                               use_cache=False).last_hidden_state          # [B,S,H]
        # Right padding + causal mask means the last *real* token is the verdict token.
        idx = attention_mask.to(torch.long).sum(dim=1) - 1                 # [B]
        gather = idx.view(-1, 1, 1).expand(-1, 1, hidden.shape[-1])
        last = hidden.gather(1, gather).squeeze(1)                         # [B,H]
        yn = F.linear(last, self.yn_head)                                  # [B,2] = [no, yes]
        return F.softmax(yn.float(), dim=-1)


def linear_quant_config(dtype: str = "int8", block: int = 32) -> dict:
    """Weight-only linear per-block — the zoo's ship recipe.

    The embedding is excluded by type: it is the only remaining full-vocab tensor
    (the head is two rows) and a lookup table quantized per-block along the wrong
    axis is a silent accuracy tax for 400 MB.
    """
    spec = {
        "op_state_spec": {
            "weight": {
                "dtype": dtype,
                "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": block, "axis": 1},
            }
        },
        "op_input_spec": None,
        "op_output_spec": None,
    }
    return {
        "execution_mode": "eager",
        "global_config": spec,
        "module_type_configs": {
            "torch.nn.modules.sparse.Embedding": None,
            "transformers.models.mistral.modeling_mistral.MistralRMSNorm": None,
        },
    }


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def build_ids(tok, instruction, query, document, seq_len, pad_id):
    ids = encode(tok, render_prompt(instruction, query, document))
    real = len(ids)
    if real > seq_len:
        raise ValueError(f"case needs {real} tokens > grid {seq_len}; raise --seq-len")
    padded = ids + [pad_id] * (seq_len - real)
    mask = [1] * real + [0] * (seq_len - real)
    return (torch.tensor([padded], dtype=torch.int32),
            torch.tensor([mask], dtype=torch.int32), real)


def write_reference(out_dir: Path, args, yes_id: int, no_id: int, pad_id: int, ref) -> None:
    """The host contract, plus the suite that was scored — so the bundle can gate itself.

    A consumer of this bundle (the Swift `SafetyClassifier`, someone else's host) can check
    its own prompt construction against nine cases and the fp32 probabilities they must
    reproduce, without this repository.
    """
    (out_dir / "reference.json").write_text(json.dumps({
        "model": args.hf_id,
        "seq_len": args.seq_len,
        "mode": args.mode,
        "yes_id": yes_id, "no_id": no_id, "pad_token_id": pad_id,
        "padding_side": "right",
        "prefix": PREFIX, "suffix": SUFFIX,
        "body_format": "<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {document}",
        "default_instruction": DEFAULT_INSTRUCTION,
        "default_query": DEFAULT_QUERY,
        "output": "probs [1,2] = softmax([no, yes]); violation = probs[1] = P(yes)",
        "add_special_tokens": False,
        "cases": [
            {"label": label, "instruction": instr, "query": query, "document": doc,
             "expected_violation": bool(want), "p_fp32": round(float(ref[f"case{i}_p"]), 6)}
            for i, (instr, query, doc, want, label) in enumerate(SUITE)
        ],
    }, indent=2, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", default="int8lin",
                    choices=["fp16", "int8lin", "int4lin"])
    ap.add_argument("--hf-id", default=HF_ID)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--exports", default=None)
    ap.add_argument("--ref", default=str(Path(__file__).resolve().parents[1]
                                         / "_smoke" / "shieldstral_3b_suite_ref.npz"))
    ap.add_argument("--reference-only", action="store_true",
                    help="rewrite reference.json for an already-exported bundle and stop")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    snap = Path(hf_snapshot(args.hf_id))
    tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
    raw = json.loads((snap / "config.json").read_text())
    pad_id = int(raw["text_config"]["pad_token_id"])
    yes_id, no_id = encode(tok, "yes")[0], encode(tok, "no")[0]
    print(f"yes={yes_id} no={no_id} pad={pad_id} grid S={args.seq_len}")

    short = args.hf_id.rsplit("/", 1)[-1].lower().replace(".", "_").replace("-", "_")
    name = f"{short}_classify_{args.mode}_s{args.seq_len}"
    root = Path(args.exports) if args.exports else exports_dir()
    if args.reference_only:
        ref = np.load(args.ref, allow_pickle=True)
        write_reference(root / name, args, yes_id, no_id, pad_id, ref)
        print(f"rewrote {root / name / 'reference.json'}")
        return 0

    print("building backbone (fp32) ...", flush=True)
    backbone, cfg, embed_w = build_text_model(snap, torch.float32)
    model = ShieldstralClassifier(backbone, embed_w, no_id, yes_id).eval()
    print(f"  {cfg.num_hidden_layers}L hidden {cfg.hidden_size} "
          f"GQA {cfg.num_attention_heads}/{cfg.num_key_value_heads} ff {cfg.intermediate_size} "
          f"vocab {cfg.vocab_size} rope {cfg.rope_scaling['rope_type']}")

    # ---- host contract gate: our flattened prompt must reproduce the oracle's ids ----
    ref = np.load(args.ref, allow_pickle=True)
    n_cases = int(ref["_meta_cases"])
    if yes_id != int(ref["_meta_yes_id"]) or no_id != int(ref["_meta_no_id"]):
        raise SystemExit("yes/no ids disagree with the oracle")

    print("\nhost prompt vs oracle ids:")
    for i, (instr, query, doc, _want, label) in enumerate(SUITE):
        mine = encode(tok, render_prompt(instr, query, doc))
        want = ref[f"case{i}_ids"].tolist()
        if mine != want:
            raise SystemExit(f"  {label}: prompt mismatch ({len(mine)} vs {len(want)} tokens)")
    print(f"  {n_cases}/{n_cases} bit-identical")

    # ---- fp32 wrapper gate: the padded-grid graph must equal the oracle's scoring ----
    print("\nfp32 wrapper on the padded grid:")
    worst = 0.0
    with torch.no_grad():
        for i, (instr, query, doc, want_unsafe, label) in enumerate(SUITE):
            ids, mask, real = build_ids(tok, instr, query, doc, args.seq_len, pad_id)
            p = float(model(ids, mask)[0, 1])
            p_ref = float(ref[f"case{i}_p"])
            worst = max(worst, abs(p - p_ref))
            side = p > 0.5
            print(f"  {label:26s} {real:>4} tok  P={p:.4f} (ref {p_ref:.4f})  "
                  f"{'UNSAFE' if side else 'safe':6s} {'OK' if side == want_unsafe else '** WRONG SIDE **'}")
            if side != want_unsafe:
                raise SystemExit(f"{label} landed on the wrong side")
    print(f"  worst |P - P_ref| = {worst:.5f}")
    if worst >= 1e-3:
        raise SystemExit("padded-grid wrapper diverges from the oracle")

    # ---- export ----
    ids0, mask0, _ = build_ids(tok, *SUITE[0][:3], args.seq_len, pad_id)
    model = model.to(DTYPE)
    example = {"input_ids": ids0.clone(), "attention_mask": mask0.clone()}

    if args.mode != "fp16":
        from coreai_models.export.compression import quantize_pytorch_model

        qdtype = "int4" if args.mode == "int4lin" else "int8"
        print(f"\nquantizing (linear {qdtype} per-block-{args.block}) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(example.values()), {},
            linear_quant_config(qdtype, args.block),
        )

    print(f"\nexporting {name} ...", flush=True)
    from coreai_models.export.macos import export_to_coreai

    prog = export_to_coreai(
        model, example, dynamic_shapes=None,
        input_names=["input_ids", "attention_mask"], output_names=["probs"],
    )
    print("optimizing ...", flush=True)
    prog.optimize()

    out_dir = root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    import coreai.runtime as rt

    meta = rt.AIModelAssetMetadata()
    meta.author = "Mistral AI"
    meta.license = "Apache-2.0"
    meta.model_description = (
        "Shieldstral-1.0-3B policy-conditioned safety classifier. "
        "probs[1] = P(the Document violates the policy stated in Instruct/Query). "
        f"Source: https://huggingface.co/{args.hf_id}")
    meta.creation_date = int(time.time())
    prog.save_asset(out_dir / f"{name}.aimodel", meta)

    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = snap / f
        if src.exists():
            (out_dir / "tokenizer").mkdir(exist_ok=True)
            shutil.copy2(src, out_dir / "tokenizer" / f)

    write_reference(out_dir, args, yes_id, no_id, pad_id, ref)

    size = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file())
    print(f"saved {out_dir} ({size / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
