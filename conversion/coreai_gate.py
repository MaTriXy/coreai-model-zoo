#!/usr/bin/env python3
"""Bundle gate: an exported .aimodel decode bundle vs its fp32 eager oracle, N/N greedy.

This is the bar the zoo cards publish ("16/16 oracle"). Unlike the eager numerics gates
(which quantize the PyTorch model and compare to fp32 — they check the *quant recipe*),
this gate drives the *exported bundle* through the Core AI engine and compares its greedy
decode to the overlay model run in fp32. It therefore checks the **conversion** end to end
— which is exactly what the OS-27-beta-2 / coreai-torch-0.4.0 IR-location incident broke and
what re-converting with 0.4.1 fixes. A bundle that loads is not enough; this proves it still
speaks.

Both sides see the SAME pre-tokenized ids (no chat template) and free-run greedy. A first
divergence is a PASS only if the fp32 oracle's top-2 margin there is < 0.1 (a knife-edge
tie, fp16 class) — otherwise FAIL. EOS ends both sides.

Usage:
    python3 coreai_gate.py <bundle-dir> <hf-id> [--arch KEY] [--prompt "..."] [-n 16]

`--arch` is auto-detected from the bundle/repo name for the known families; pass it
explicitly for a new model that reuses an existing family's overlay. Run from a checkout
whose sibling coreai-models has the zoo overlay applied (see zoo_convert.py doctor).

Findings baked in here because they are documented nowhere else (2026-07-18 recovery):
  - The engine needs COREAI_CHUNK_THRESHOLD=1 + variant coreai-pipelined. This gate disables
    warmup to isolate the checked generation; the runtime patch makes default warmup honor a
    static-S=1 input descriptor instead of submitting a synthetic 256-token prefill.
  - `llm-runner --inference-engine-variant` help text is stale; the real values are
    auto / coreai-sequential / coreai-pipelined / static-shape.
  - The oracle steps S=1 but position_ids carries the FULL 0..t range each step
    (dynamic full-length positions); passing a single position produces plausible garbage.
  - Each overlay builds its model differently — see ARCH below.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Per-arch overlay wiring, mirrored from each export_*.py. Each value is enough for the
# oracle subprocess (below) to rebuild the fp32 model and step it. `alias` maps extra
# model families onto an existing arch (same overlay, different weights).
ARCH = {
    "qwen3.5": {},
    "lfm2_5": {},
    "granite": {},
    "youtu": {},
    "nanbeige": {},
    "lfm2_moe": {},
    "qwen3_6_moe": {},
}
# Dense Qwen3.6-27B reuses the qwen3.5 overlay; the 35B-A3B is MoE (own overlay). Match the
# MoE substrings before the generic qwen3.6->qwen3.5 fallback.
ALIASES = {"ornith": "qwen3.5", "lfm2_moe": "lfm2_moe", "a1b": "lfm2_moe",
           "35b_a3b": "qwen3_6_moe", "35b-a3b": "qwen3_6_moe", "a3b": "qwen3_6_moe",
           "qwen3_6": "qwen3.5", "qwen3.6": "qwen3.5"}


def resolve_python(flag: str | None) -> str:
    if flag:
        return flag
    if env := os.environ.get("ZOO_CONVERT_PYTHON"):
        return env
    sibling = HERE.parent.parent / "coreai-models" / ".venv" / "bin" / "python"
    if sibling.exists():
        return str(sibling)
    return shutil.which("python3") or "python3"


def resolve_runner() -> str:
    base = HERE.parent.parent / "coreai-models"
    for c in (base / ".build" / "release" / "llm-runner",
              base / ".build" / "out" / "Products" / "Release" / "llm-runner"):
        if c.exists():
            return str(c)
    return "llm-runner"


def detect_arch(bundle: str, hf_id: str) -> str | None:
    name = Path(bundle).name.lower()
    hay = name + " " + hf_id.lower()
    # Aliases first: they carry the specific discriminators (a1b/a3b) that must beat the
    # generic family substring — e.g. "lfm2_5_8b_a1b" must route to lfm2_moe, not lfm2_5.
    a = next((v for sub, v in ALIASES.items() if sub in hay), None)
    if a:
        return a
    return next((k for k in ARCH if k in name or k.replace("_", "") in name or k in hf_id.lower()), None)


# The oracle runs in a child interpreter (the overlay venv). Kept as source text so the
# gate is a single file; each branch is a verbatim transcription of that model's export.
ORACLE_SRC = r'''
import json, sys, warnings
warnings.filterwarnings("ignore")
import torch
CTX = 4096
arch, hf_id, prompt, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
revision = (sys.argv[6] or None) if len(sys.argv) > 6 else None
# Oracle weight dtype: fp32 is the strict ceiling, but a 35B in fp32 is ~140 GB — past
# most machines. fp16 (the export's own trace dtype) fits and is a valid conversion check.
FP32 = {"fp32": torch.float32, "fp16": torch.float16}[sys.argv[5] if len(sys.argv) > 5 else "fp32"]

def build(arch, hf_id):
    from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
    if arch == "qwen3.5":
        from coreai_models.models.macos.qwen3_5 import Qwen3_5StatefulForCausalLM, build_decode_state
        try:
            m = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32, hf_config_attr="text_config")
        except Exception:
            m = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state","rec_state"]
    elif arch == "lfm2_5":
        from coreai_models.models.macos.lfm2 import lfm2_from_hf, build_decode_state
        m = lfm2_from_hf(hf_id, target_dtype=FP32, stateful=True)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state"]
    elif arch == "granite":
        from coreai_models.models.macos.granite4h import Granite4HForCausalLMStateful, build_decode_state
        m = Granite4HForCausalLMStateful.from_hf(hf_id, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state","rec_state"]
    elif arch == "youtu":
        from coreai_models.models.macos.youtu_absorbed import youtu_absorbed_from_hf, YoutuAbsorbedStatefulForCausalLM, build_absorbed_decode_state
        m = YoutuAbsorbedStatefulForCausalLM.from_causal_lm(youtu_absorbed_from_hf(hf_id, target_dtype=FP32))
        st = build_absorbed_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["kv_a","kv_b"]
    elif arch == "nanbeige":
        from transformers import AutoConfig
        from coreai_models.models.macos.llama import LlamaForCausalLM
        from coreai_models.models.macos.nanbeige import NanbeigeForCausalLM, create_cache_tensors
        from coreai_models.primitives.macos.cache import KVCache
        source_config = AutoConfig.from_pretrained(hf_id, revision=revision)
        model_classes = {"llama": LlamaForCausalLM, "nanbeige": NanbeigeForCausalLM}
        if source_config.model_type not in model_classes:
            raise ValueError(f"unsupported Nanbeige gate model_type: {source_config.model_type}")
        model_class = model_classes[source_config.model_type]
        m = model_class.from_hf_memory_efficient(
            hf_id, revision=revision, max_context_length=CTX, target_dtype=FP32
        )
        saved = m.config.max_position_embeddings; m.config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
        if source_config.model_type == "nanbeige":
            k, v = create_cache_tensors(m.config, dtype=FP32)
        else:
            k, v = KVCache.create_cache_tensors(m.config, dtype=FP32)
        m.config.max_position_embeddings = saved
        st = {"k_cache": k, "v_cache": v}; order = ["k_cache","v_cache"]
    elif arch == "lfm2_moe":
        from coreai_models.models.macos.lfm2_moe import lfm2_moe_from_hf, build_decode_state
        m = lfm2_moe_from_hf(hf_id, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state"]
    elif arch == "qwen3_6_moe":
        from coreai_models.models.macos.qwen3_5_moe import Qwen3_5MoeStatefulForCausalLM, build_decode_state
        try:
            m = Qwen3_5MoeStatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32, hf_config_attr="text_config")
        except Exception:
            m = Qwen3_5MoeStatefulForCausalLM.from_hf_memory_efficient(hf_id, max_context_length=CTX, target_dtype=FP32)
        st = build_decode_state(m.config, max_seq_len=CTX, dtype=FP32); order = ["k_cache","v_cache","conv_state","rec_state"]
    else:
        raise SystemExit("unknown arch: " + arch)
    m.eval()
    for layer in getattr(m.model, "layers", []):
        if getattr(layer, "is_full", True) is False and hasattr(layer, "linear_attn"):
            layer.linear_attn.use_loopfree_step = True
    return m, [st[k] for k in order]

from transformers import AutoTokenizer
try:
    tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
except Exception:
    # Some repos (LFM2.5) name a tokenizer_class this transformers build lacks
    # ("TokenizersBackend"); load the fast tokenizer straight from tokenizer.json,
    # bypassing class resolution. config.eos_token_id still drives EOS below.
    from huggingface_hub import hf_hub_download
    from transformers import PreTrainedTokenizerFast
    tok = PreTrainedTokenizerFast(
        tokenizer_file=hf_hub_download(hf_id, "tokenizer.json", revision=revision)
    )
ids = tok(prompt, return_tensors="pt").input_ids.to(torch.int32)
model, states = build(arch, hf_id)
eos = set()
for e in (getattr(tok, "eos_token_id", None), getattr(model.config, "eos_token_id", None)):
    if isinstance(e, int): eos.add(e)
    elif isinstance(e, (list, tuple)): eos.update(int(x) for x in e)
gen, margins, cur = [], [], ids
with torch.no_grad():
    for t in range(ids.shape[1] + n - 1):
        out = model(cur[:, t:t+1], torch.arange(t+1, dtype=torch.int32).unsqueeze(0), *states)
        # Feed the prompt one token at a time; only START collecting once its last token
        # is in (t == len-1 predicts token #1). Dropping this guard emits prompt-position
        # predictions as output and corrupts the sequence.
        if t < ids.shape[1] - 1: continue
        row = (out[0] if isinstance(out, (tuple, list)) else out)[0, 0].float()
        nxt = int(row.argmax())
        if nxt in eos: break
        p = torch.softmax(row, dim=-1); top2 = torch.topk(p, 2).values
        margins.append(float(top2[0]-top2[1])); gen.append(nxt)
        if len(gen) >= n: break
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.int32)], dim=1)
print(json.dumps({"input_ids": ids[0].tolist(), "gen_ids": gen, "margins": margins,
                  "gen_text": tok.decode(gen, skip_special_tokens=False)}))
'''


def run_oracle(
    python: str,
    arch: str,
    hf_id: str,
    prompt: str,
    n: int,
    dtype: str,
    revision: str | None,
) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(ORACLE_SRC)
        script = f.name
    r = subprocess.run([python, script, arch, hf_id, prompt, str(n), dtype, revision or ""],
                       capture_output=True, text=True, cwd=tempfile.gettempdir())
    line = next((line for line in r.stdout.splitlines() if line.startswith("{")), None)
    if not line:
        sys.exit("ORACLE FAILED:\n" + r.stdout[-1000:] + r.stderr[-1000:])
    return json.loads(line)


def run_engine(runner: str, bundle: str, input_ids: list[int], n: int) -> str | None:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"tokens": input_ids}, f)
        raw = f.name
    env = {"COREAI_CHUNK_THRESHOLD": "1", "PATH": "/usr/bin:/bin"}
    r = subprocess.run([runner, "--model", bundle, "--raw-tokens", raw, "--max-tokens", str(n),
                        "--temperature", "0.0", "--inference-engine-variant", "coreai-pipelined",
                        "--warmup", "off"], capture_output=True, text=True, env=env)
    out = r.stdout
    try:
        body = out.split("Generating...", 1)[1].split("⏱", 1)[0]  # between banner and the ⏱ summary
    except IndexError:
        return None
    if body.startswith("\n"):
        body = body[1:]
    if body.endswith("\n\n"):
        body = body[:-2]
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description="Gate an exported decode bundle against its fp32 oracle.")
    ap.add_argument("bundle")
    ap.add_argument("hf_id")
    ap.add_argument("--revision", help="immutable Hugging Face checkpoint revision")
    ap.add_argument("--arch", choices=list(ARCH))
    ap.add_argument("--prompt", default="The capital of France is",
                    help="deterministic prompt; open-ended ones hit ties and aren't good gates")
    ap.add_argument("-n", type=int, default=16)
    ap.add_argument("--oracle-dtype", choices=["fp32", "fp16"], default="fp32",
                    help="fp32 = strict ceiling; fp16 for models too big for fp32 (e.g. 35B needs ~140 GB)")
    ap.add_argument("--python", help="overlay interpreter (default: sibling coreai-models/.venv)")
    ap.add_argument("--runner", help="llm-runner executable (default: sibling coreai-models build)")
    args = ap.parse_args()

    arch = args.arch or detect_arch(args.bundle, args.hf_id)
    if not arch:
        sys.exit(f"no arch mapping for {args.bundle} — pass --arch")
    python = resolve_python(args.python)
    runner = args.runner or resolve_runner()

    oracle = run_oracle(
        python, arch, args.hf_id, args.prompt, args.n, args.oracle_dtype, args.revision
    )
    engine = run_engine(runner, args.bundle, oracle["input_ids"], args.n)

    print("=== GATE:", Path(args.bundle).name, f"(arch={arch})")
    print("  prompt :", repr(args.prompt), "->", oracle["input_ids"])
    print("  oracle :", repr(oracle["gen_text"]))
    print("  engine :", repr(engine))
    if engine is None:
        sys.exit("  RESULT: ERROR (engine produced no output)")
    if engine == oracle["gen_text"]:
        print(f"  RESULT: PASS — token-for-token == {args.oracle_dtype} oracle")
        return
    from transformers import AutoTokenizer
    try:
        tk = AutoTokenizer.from_pretrained(args.hf_id, revision=args.revision)
    except Exception:
        from huggingface_hub import hf_hub_download
        from transformers import PreTrainedTokenizerFast
        tk = PreTrainedTokenizerFast(
            tokenizer_file=hf_hub_download(
                args.hf_id, "tokenizer.json", revision=args.revision
            )
        )
    eng_ids = tk(engine, add_special_tokens=False).input_ids
    ref, margins = oracle["gen_ids"], oracle.get("margins", [])
    d = next((i for i in range(min(len(eng_ids), len(ref))) if eng_ids[i] != ref[i]),
             min(len(eng_ids), len(ref)))
    tie = d < len(margins) and margins[d] < 0.1
    print(f"  match  : {d}/{len(ref)} exact; first divergence at #{d}"
          + (f", margin {margins[d]:.4f}" if d < len(margins) else ""))
    if tie:
        print(f"  RESULT: PASS — diverges only at a top-2 tie (margin {margins[d]:.3f} < 0.1), fp16 class")
        return
    sys.exit("  RESULT: FAIL — bundle diverges from the fp32 oracle at a decisive position")


if __name__ == "__main__":
    main()
