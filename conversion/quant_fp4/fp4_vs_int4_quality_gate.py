#!/usr/bin/env python3
"""Stream D de-risk: FP4 (E2M1) vs int4-RTN vs int8 weight-only quality gate.

The Stream D thesis is that the zoo's int4 collapse is a property of int4 RTN,
and that FP4 (E2M1, near-FP8 quality) holds where int4 craters -- specifically on
*multi-token reasoning*, not single-token recall (the Nanbeige lesson: "Paris"
survives int4 but reasoning chains do not).

This harness fake-quantizes a model's Linear weights three ways and measures the
degradation each scheme causes, with NO device and NO TensorOps runtime needed
(weights are dequantized back to fp32, so the model runs as a normal fp32 graph).

Schemes (all weight-only, per-block along the input-channel axis, block_size=32 --
matching the shipped "4bit" preset granularity):
  * fp4   : E2M1 via torchao f32_to_f4_unpacked (the EXACT kernel coreai-opt's
            _fp4_forward calls), e8m0 power-of-2 block scale (OCP MX spec).
  * int4  : symmetric-with-clipping, levels [-7,7], MSE-optimal clip per block
            (matches coreai-opt symmetric_with_clipping + moe_metal _SYM_CLIPS).
  * int8  : symmetric-with-clipping, levels [-127,127].
  * fp16  : reference (no quant).

Metrics:
  1. Aggregate weight reconstruction relative error per scheme (deterministic).
  2. Perplexity on a fixed reasoning corpus (continuous quality signal).
  3. Greedy exact-match accuracy on arithmetic/logic word problems
     (the multi-token reasoning gate).

Usage:
  .venv/bin/python coreai-models-community/conversion/quant_fp4/fp4_vs_int4_quality_gate.py \
      [--model <hf_path_or_cache_snapshot>] [--max-new 200] [--ppl-tokens 1500]
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# torchao FP4 kernels -- identical to what coreai_opt.quantization.spec.fake_quantize._fp4_forward uses.
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, f4_unpacked_to_f32

_SYM_CLIPS = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)  # matches moe_metal int8sym / int4 clip search
BLOCK = 32
FP4_MAX = 6.0  # E2M1 max magnitude


# --------------------------------------------------------------------------- #
# Per-block fake-quant primitives.  Weight is [out, in]; we block along `in`.
# --------------------------------------------------------------------------- #
def _to_blocks(w: torch.Tensor, block: int = BLOCK):
    """Reshape [out, in] -> [out, nblk, block], right-padding `in` with zeros."""
    out, cin = w.shape
    pad = (-cin) % block
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    return w.reshape(out, -1, block), cin


def _from_blocks(wb: torch.Tensor, cin: int) -> torch.Tensor:
    out = wb.shape[0]
    return wb.reshape(out, -1)[:, :cin].contiguous()


def fake_quant_fp4(w: torch.Tensor) -> torch.Tensor:
    """E2M1 weight-only, e8m0 (power-of-2) per-block scale -- the coreai-opt fp4 path.

    Scale matches coreai_opt spec.py:155 (OCP MX):
        scale = 2^(floor(log2(max_abs)) - target_max_pow2),  target_max_pow2=2 for E2M1.
    The block max may then map to [4, 8) and clamp at the 6.0 grid edge -- standard MXFP4.
    """
    wb, cin = _to_blocks(w.float())
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-20)
    scale_exp = torch.floor(torch.log2(amax)) - 2.0  # target_max_pow2 = 2 (E2M1)
    scale = torch.exp2(scale_exp)
    scaled = (wb / scale).clamp(-FP4_MAX, FP4_MAX)
    codes = f32_to_f4_unpacked(scaled)          # f32 -> 4-bit codes
    deq = f4_unpacked_to_f32(codes)             # codes -> f32 grid values
    return _from_blocks(deq * scale, cin).to(w.dtype)


def _fake_quant_int_symclip(w: torch.Tensor, qmax: int) -> torch.Tensor:
    """Symmetric-with-clipping integer quant, MSE-optimal clip per block."""
    wb, cin = _to_blocks(w.float())
    amax = wb.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    best = None
    best_err = None
    for clip in _SYM_CLIPS:
        scale = (clip * amax) / qmax
        q = torch.round(wb / scale).clamp(-qmax, qmax)
        deq = q * scale
        err = (deq - wb).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best, best_err = deq, err
        else:
            take = err < best_err
            best = torch.where(take, deq, best)
            best_err = torch.where(take, err, best_err)
    return _from_blocks(best, cin).to(w.dtype)


def fake_quant_int4(w):
    return _fake_quant_int_symclip(w, qmax=7)


def fake_quant_int8(w):
    return _fake_quant_int_symclip(w, qmax=127)


SCHEMES = {
    "fp16": None,
    "int8": fake_quant_int8,
    "int4": fake_quant_int4,
    "fp4": fake_quant_fp4,
}


# --------------------------------------------------------------------------- #
# Eval corpus: reasoning-flavoured (so int4's reasoning cliff is exercised).
# --------------------------------------------------------------------------- #
PPL_TEXT = """To find the average speed we divide total distance by total time.
A train travels 240 kilometers in 3 hours, so its average speed is 240 / 3 = 80 km/h.
A baker has 5 trays with 12 cupcakes each, giving 5 times 12 = 60 cupcakes in total.
If 18 of them are sold, then 60 minus 18 = 42 cupcakes remain on the shelves.
A rectangle is 7 meters long and 4 meters wide, so its area is 7 times 4 = 28 square meters,
and its perimeter is 2 times (7 plus 4) = 22 meters.
Sarah saves 15 dollars every week. After 6 weeks she has saved 15 times 6 = 90 dollars.
She then spends 35 dollars, leaving her with 90 minus 35 = 55 dollars.
A car uses 8 liters of fuel per 100 kilometers. To drive 350 kilometers it needs
8 times 3.5 = 28 liters of fuel.
There are 3 boxes, and each box holds 24 bottles. The total number of bottles is
3 times 24 = 72. If they are shared equally among 8 friends, each friend gets 72 / 8 = 9 bottles.
A classroom has 28 students. If 3 of every 7 students wear glasses, then the number of
students wearing glasses is 28 divided by 7, times 3, which is 4 times 3 = 12 students.
A book has 320 pages. Tom reads 40 pages a day. He will finish the book in 320 / 40 = 8 days.
The temperature rose from minus 5 degrees to 12 degrees, a change of 12 minus (minus 5) = 17 degrees.
A square garden has a side of 9 meters, so its area is 9 times 9 = 81 square meters.
"""

REASONING_PROBLEMS = [
    ("A train travels 60 km in 1.5 hours. What is its average speed in km/h? Reply with the number only after thinking step by step.", 40),
    ("A baker makes 9 trays of 8 muffins. He sells 30 muffins. How many muffins are left? End with the number.", 42),
    ("Tom buys 4 notebooks at 3 dollars each and pays with a 20 dollar bill. How much change does he get? End with the number.", 8),
    ("A rectangle is 12 m long and 5 m wide. What is its area in square meters? End with the number.", 60),
    ("There are 5 boxes with 24 pencils each. They are shared equally among 8 students. How many pencils does each student get? End with the number.", 15),
    ("Sarah saves 12 dollars per week for 7 weeks, then spends 29 dollars. How much money does she have left? End with the number.", 55),
    ("A car uses 6 liters of fuel per 100 km. How many liters does it need for a 250 km trip? End with the number.", 15),
    ("A number is tripled and then 7 is added, giving 28. What was the original number? End with the number.", 7),
    ("A water tank holds 480 liters. It is drained at 40 liters per minute. How many minutes to empty it? End with the number.", 12),
    ("In a class of 30 students, 2 of every 5 play sports. How many students play sports? End with the number.", 12),
    ("A book has 270 pages. Mia reads 45 pages a day. How many days to finish it? End with the number.", 6),
    ("The temperature went from -8 degrees to 9 degrees. By how many degrees did it rise? End with the number.", 17),
]


def target_linears(model):
    """All decoder nn.Linear weights, excluding the lm_head (kept fp16, as shipped)."""
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and not name.endswith("lm_head"):
            out.append((name, mod))
    return out


@torch.no_grad()
def apply_scheme(linears, originals, fn):
    """Overwrite each linear weight with fn(original_weight). fn=None restores fp16."""
    dev = linears[0][1].weight.device
    rel_num = rel_den = 0.0
    for name, mod in linears:
        w0 = originals[name]  # on CPU
        if fn is None:
            mod.weight.copy_(w0.to(dev))
        else:
            wq = fn(w0)  # CPU fake-quant (torchao fp4 kernel is CPU-only)
            mod.weight.copy_(wq.to(dev))
            rel_num += (wq.float() - w0.float()).pow(2).sum().item()
            rel_den += w0.float().pow(2).sum().item()
    rel = math.sqrt(rel_num / rel_den) if rel_den else 0.0
    return rel


@torch.no_grad()
def perplexity(model, tok, text, max_tokens):
    ids = tok(text, return_tensors="pt").input_ids[:, : max_tokens + 1].to(model.device)
    out = model(ids)
    logits = out.logits[:, :-1, :].float()
    tgt = ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="mean"
    )
    return math.exp(nll.item()), ids.shape[1] - 1


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def extract_answer(text):
    nums = _NUM.findall(text.replace(",", ""))
    if not nums:
        return None
    try:
        return float(nums[-1])
    except ValueError:
        return None


@torch.no_grad()
def reasoning_accuracy(model, tok, max_new):
    correct = 0
    details = []
    for prompt, gold in REASONING_PROBLEMS:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to(model.device)
        gen = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(txt)
        ok = pred is not None and abs(pred - gold) < 1e-6
        correct += int(ok)
        details.append((gold, pred, ok))
    return correct, len(REASONING_PROBLEMS), details


def main():
    ap = argparse.ArgumentParser()
    default_model = sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/*")))
    ap.add_argument("--model", default=default_model[0] if default_model else "LiquidAI/LFM2.5-1.2B-Instruct")
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--ppl-tokens", type=int, default=1500)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    args = ap.parse_args()

    if args.device == "auto":
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        dev = args.device

    torch.manual_seed(0)
    print(f"[load] {args.model}  (device={dev})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval().to(dev)

    linears = target_linears(model)
    # Keep originals on CPU: the torchao fp4 kernel is not MPS-safe, so fake-quant
    # must run on CPU; only the dequantized result is copied back to the (mps) weight.
    originals = {name: mod.weight.detach().to("cpu").clone() for name, mod in linears}
    print(f"[info] quantizing {len(linears)} Linear weights (lm_head kept fp16), block={BLOCK}")

    rows = []
    for scheme, fn in SCHEMES.items():
        t0 = time.time()
        rel = apply_scheme(linears, originals, fn)
        ppl, ntok = perplexity(model, tok, PPL_TEXT, args.ppl_tokens)
        corr, total, details = reasoning_accuracy(model, tok, args.max_new)
        dt = time.time() - t0
        rows.append((scheme, rel, ppl, corr, total))
        print(f"[{scheme:4}] w_rel_err={rel:.4f}  ppl={ppl:7.3f}  "
              f"reasoning={corr}/{total}  ({dt:.0f}s, {ntok} ppl-tokens)")

    # restore fp16 to leave model clean
    apply_scheme(linears, originals, None)

    print("\n==================== Stream D quality gate ====================")
    print(f"model: {os.path.basename(args.model.rstrip('/'))}   block_size={BLOCK}")
    print(f"{'scheme':6} {'w_rel_err':>10} {'ppl':>9} {'reasoning':>11}")
    base_ppl = next(r[2] for r in rows if r[0] == "fp16")
    for scheme, rel, ppl, corr, total in rows:
        dppl = (ppl / base_ppl - 1.0) * 100.0
        print(f"{scheme:6} {rel:10.4f} {ppl:9.3f} {corr:>6}/{total:<4}   (ppl {dppl:+.1f}% vs fp16)")
    print("===============================================================")
    print("Gate: fp4 should track int8 (small ppl delta, reasoning preserved);")
    print("      int4 RTN is the cliff to beat. If fp4 ~ int8 >> int4, Stream D's")
    print("      FP4-via-TensorOps bet is validated (runtime still needs Stream B / OS27).")


if __name__ == "__main__":
    main()
