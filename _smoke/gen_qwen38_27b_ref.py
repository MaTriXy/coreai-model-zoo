"""Dump the HF bf16 oracle for Qwen/Qwen3.8-27B (text path of the VLM checkpoint).

Runs in a transformers>=5.12 venv (~/.venvs/qwen38-oracle) — the export venv's 4.57
has no qwen3_5 modeling code. bf16, NOT fp32: fp32 for a 27.8B is ~111 GB RAM and the
margin>=0.1 gate rule absorbs bf16 noise (proven on the 3.5-35B / 3.6-27B oracles).

Greedy 16 continuation tokens off a chat-templated prompt, HF cache, eager attention.
Saves prompt ids, greedy ids, per-step top-2 margins AND the full logits row per step
(for per-position cos in the eager gate) to _smoke/qwen38_27b_ref.pt.

Usage: ~/.venvs/qwen38-oracle/bin/python _smoke/gen_qwen38_27b_ref.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

HF_ID = "Qwen/Qwen3.8-27B"
N_GEN = 16
PROMPT_MSGS = [{"role": "user", "content": "Explain in one sentence why the sky is blue."}]
OUT = Path(__file__).resolve().parent / "qwen38_27b_ref.pt"


def main() -> None:
    tok = AutoTokenizer.from_pretrained(HF_ID)
    enc = tok.apply_chat_template(
        PROMPT_MSGS, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    ids = enc["input_ids"]
    print(f"prompt ids ({ids.shape[1]}): {ids[0].tolist()}")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        HF_ID,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    model.eval()
    print("27B VLM loaded bf16 | text:", model.config.text_config.num_hidden_layers,
          "layers | untied head:", not model.config.tie_word_embeddings)

    gen, margins, logits_rows = [], [], []
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
        past = out.past_key_values
        row = out.logits[0, -1].float()
        for step in range(N_GEN):
            nxt = int(row.argmax())
            p = torch.softmax(row, dim=-1)
            top2 = torch.topk(p, 2).values
            margins.append(float(top2[0] - top2[1]))
            logits_rows.append(row.to(torch.float16).clone())
            gen.append(nxt)
            print(f"step {step}: top1={nxt} margin={margins[-1]:.3f}")
            if step == N_GEN - 1:
                break
            out = model(
                input_ids=torch.tensor([[nxt]], dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            row = out.logits[0, -1].float()

    torch.save(
        {
            "hf_id": HF_ID,
            "prompt_messages": PROMPT_MSGS,
            "prompt_ids": ids[0].to(torch.int32),
            "gen_ids": torch.tensor(gen, dtype=torch.int32),
            "margins": torch.tensor(margins),
            "logits_rows": torch.stack(logits_rows),  # [N_GEN, vocab] fp16
            "gen_text": tok.decode(gen, skip_special_tokens=False),
            "oracle_dtype": "bf16",
            "attn": "eager",
        },
        OUT,
    )
    print(f"greedy continuation: {gen}")
    print(f"text: {tok.decode(gen, skip_special_tokens=False)!r}")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
