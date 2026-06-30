# Community port — NOT an Apple model.
"""Tokenizer bridge for LLaDARunner (Swift host validation).

  write  <out_file> [prompt]   tokenize prompt (chat template) -> comma-separated ids in out_file
  decode <comma_ids>           decode generated token ids back to text (skip special tokens)
"""
import sys, os
from transformers import AutoTokenizer

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "d3LLM_LLaDA")
tok = AutoTokenizer.from_pretrained(W, trust_remote_code=True)

cmd = sys.argv[1]
if cmd == "write":
    out = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else (
        "Natalia sold clips to 48 friends in April, then half as many in May. "
        "How many clips did she sell altogether? Reason step by step.")
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(text)["input_ids"]
    open(out, "w").write(",".join(map(str, ids)))
    print(f"wrote {len(ids)} ids -> {out}")
elif cmd == "decode":
    ids = [int(x) for x in sys.argv[2].split(",") if x.strip()]
    print(repr(tok.decode(ids, skip_special_tokens=True)))
else:
    sys.exit("unknown cmd")
