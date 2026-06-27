#!/usr/bin/env python3
"""Export OpenBMB MiniCPM5-1B -> Core AI int8 (FM-format dynamic bundle).

MiniCPM5-1B is a plain `LlamaForCausalLM`, so unlike the other zoo LLMs (custom
`export_*_decode_pipelined.py` torch exports) it rides the STOCK `coreai.llm.export`
CLI. Two one-time patches to the local `coreai_models` package are required first
(the stock registry has no `llama` family and no metadata for this id):

  1. coreai_models/models/registry.py  — MODEL_TYPE_REMAPPING:
         "llama": "mistral",
     A plain Llama == the Mistral builder minus the sliding window (GQA + RoPE +
     RMSNorm + SiLU, no qkv bias, no qk-norm, honors config.head_dim).

  2. coreai_models/export/metadata.py  — AIModelMetadataFields:
         "openbmb/MiniCPM5-1B": AIModelMetadataFields(
             author="OpenBMB", license="Apache-2.0",
             model_description="MiniCPM5-1B is a 1.08B on-device causal LM from "
                 "OpenBMB ... Source: https://huggingface.co/openbmb/MiniCPM5-1B"),

The export itself is symmetric per-channel int8 (absmax, no clipping) via the
quantization_config in `minicpm5_int8sym.yaml` (sibling of this file). On iPhone
int8 is ~2.2x fp16 and lossless (24/24 token-exact vs HF fp32); on a compute-rich
Mac int8 is SLOWER than fp16 (dequant overhead), so the win is iPhone-side.

Ship the DYNAMIC bundle (this default macOS export) to the iPhone: it routes to the
pipelined engine. A `--platform iOS` static export routes to the staticShape engine
and fails at engine-create. See ../knowledge/minicpm5-1b.md for the full rationale.

    python conversion/export_minicpm5.py [OUT_DIR]

then verify greedy parity vs HF fp32 (token-exact) before shipping —
`conversion/verify_minicpm5.py` (the shipped int8 bundle scores 24/24 = lossless).
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
QCONFIG = HERE / "minicpm5_int8sym.yaml"
CHAT_EOS = "<|im_end|>"  # id 130073


def export(out_dir: Path) -> None:
    """Run the stock exporter (assumes the two registry/metadata patches above)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "run", "coreai.llm.export", "openbmb/MiniCPM5-1B",
         "--experimental", "--compute-precision", "float16",
         "--compression-config", str(QCONFIG)],
        cwd=out_dir, check=True)


def fix_chat_eos(tok_dir: Path) -> None:
    """Base eos is </s> (raw-text terminator); the chat template ends turns with
    <|im_end|>. The engine stops on the single eosTokenId, so a chat bundle must
    declare <|im_end|> as eos (exactly how Qwen ships) or it never halts."""
    cfg = tok_dir / "tokenizer_config.json"
    d = json.loads(cfg.read_text())
    d["eos_token"] = CHAT_EOS
    cfg.write_text(json.dumps(d, ensure_ascii=False, indent=2))
    stm = tok_dir / "special_tokens_map.json"
    if stm.exists():
        m = json.loads(stm.read_text())
        eos = m.get("eos_token")
        if isinstance(eos, dict):
            eos["content"] = CHAT_EOS
        else:
            m["eos_token"] = CHAT_EOS
        stm.write_text(json.dumps(m, ensure_ascii=False, indent=2))
    print(f"chat eos -> {CHAT_EOS} in {tok_dir}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("exports/minicpm5-1b")
    export(out)
    for tok in sorted(out.rglob("tokenizer")):
        fix_chat_eos(tok)
