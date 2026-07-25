# Community port — NOT an Apple model.
"""S3 reference: dequant the BitCPM-CANN-8B TQ2_0 gguf, run an eager MiniCPM4-8B forward
(mup scalars baked from the gguf metadata) and greedily generate — eyeball coherence to
confirm arch + mup + ternary-dequant before any Metal kernel work.

The dequantized fp weights here ARE the oracle the ternary Metal kernel (S2) is gated against
(cos>=0.999): TQ2_0 -> {-1,0,1}*d per 256-block, embed Q4_K, head Q6_K. Run:

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitcpm/validate_ref.py --new 8
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import gguf
from gguf.quants import dequantize
from transformers import AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402

GGUF = str(work_path("_bitcpm_ckpt", "bitcpm4-8b-tq2_0.gguf"))
HF = str(work_path("_bitcpm_ckpt", "hf"))

# config (BitCPM-CANN-8B == MiniCPM4-8B); mup scalars verified against gguf metadata
H, FF, NL = 4096, 16384, 32
NH, NKV, HD = 32, 2, 128
VOCAB = 73448
EPS = 1e-6
ROPE_THETA = 10000.0
SCALE_EMB = 12.0
RESID = 0.2474873661994934   # scale_depth / sqrt(num_layers) = 1.4/sqrt(32)
LOGIT_DIV = 16.0             # hidden_size / dim_model_base = 4096/256


def load_weights(dtype=torch.bfloat16):
    r = gguf.GGUFReader(GGUF)
    ten = {t.name: t for t in r.tensors}

    def deq(name):
        t = ten[name]
        w = np.asarray(dequantize(t.data, t.tensor_type))   # fp32, torch [out, in] order
        return torch.from_numpy(w.copy()).to(dtype)

    W = {}
    W["embed"] = deq("token_embd.weight")          # [vocab, H]
    W["head"] = deq("output.weight")               # [vocab, H]
    W["final_norm"] = deq("output_norm.weight")    # [H]
    W["short_factor"] = torch.from_numpy(np.asarray(
        dequantize(ten["rope_factors_short.weight"].data, ten["rope_factors_short.weight"].tensor_type)
    ).copy()).float()                              # [HD/2]
    for i in range(NL):
        p = f"blk.{i}."
        W[f"l{i}.an"] = deq(p + "attn_norm.weight")
        W[f"l{i}.fn"] = deq(p + "ffn_norm.weight")
        W[f"l{i}.q"] = deq(p + "attn_q.weight")
        W[f"l{i}.k"] = deq(p + "attn_k.weight")
        W[f"l{i}.v"] = deq(p + "attn_v.weight")
        W[f"l{i}.o"] = deq(p + "attn_output.weight")
        W[f"l{i}.g"] = deq(p + "ffn_gate.weight")
        W[f"l{i}.u"] = deq(p + "ffn_up.weight")
        W[f"l{i}.d"] = deq(p + "ffn_down.weight")
    return W


def rms(x, w):
    f = x.float()
    f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
    return (f.to(x.dtype) * w)


def rope_tables(short_factor, seqlen, dtype=torch.float32):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2).float() / HD))   # [HD/2]
    eff = inv / short_factor                                            # longrope, scaling_factor=1
    t = torch.arange(seqlen).float()
    fr = torch.outer(t, eff)                                            # [T, HD/2]
    emb = torch.cat((fr, fr), dim=-1)                                   # [T, HD]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def forward(W, ids, cos, sin):
    x = F.embedding(ids, W["embed"]) * SCALE_EMB          # [1, T, H]
    T = ids.shape[1]
    cmask = torch.full((T, T), float("-inf")).triu(1)     # causal
    c = cos[:T].view(1, 1, T, HD)
    s = sin[:T].view(1, 1, T, HD)
    for i in range(NL):
        h = rms(x, W[f"l{i}.an"])
        q = (h @ W[f"l{i}.q"].T).view(1, T, NH, HD).transpose(1, 2)    # [1,NH,T,HD]
        k = (h @ W[f"l{i}.k"].T).view(1, T, NKV, HD).transpose(1, 2)
        v = (h @ W[f"l{i}.v"].T).view(1, T, NKV, HD).transpose(1, 2)
        qf, kf = q.float(), k.float()
        q = (qf * c + rotate_half(qf) * s).to(x.dtype)
        k = (kf * c + rotate_half(kf) * s).to(x.dtype)
        k = k.repeat_interleave(NH // NKV, dim=1)          # GQA
        v = v.repeat_interleave(NH // NKV, dim=1)
        sc = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(HD) + cmask
        ctx = (F.softmax(sc, dim=-1) @ v.float()).to(x.dtype)         # [1,NH,T,HD]
        ctx = ctx.transpose(1, 2).reshape(1, T, NH * HD)
        x = x + (ctx @ W[f"l{i}.o"].T) * RESID
        h2 = rms(x, W[f"l{i}.fn"])
        mlp = (F.silu(h2 @ W[f"l{i}.g"].T) * (h2 @ W[f"l{i}.u"].T)) @ W[f"l{i}.d"].T
        x = x + mlp * RESID
    x = rms(x, W["final_norm"])
    logits = (x / LOGIT_DIV) @ W["head"].T                 # [1, T, vocab]
    return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new", type=int, default=8)
    args = ap.parse_args()

    torch.manual_seed(0)
    print("loading + dequantizing gguf (8B, bf16)...", flush=True)
    W = load_weights()
    tok = AutoTokenizer.from_pretrained(HF, trust_remote_code=True)

    msgs = [{"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    cos, sin = rope_tables(W["short_factor"], ids.shape[1] + args.new + 4)
    print("prompt:", repr(args.prompt))
    out = []
    with torch.no_grad():
        for _ in range(args.new):
            logits = forward(W, ids, cos, sin)
            nxt = int(logits[0, -1].float().argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], dim=1)
            print("  ->", nxt, repr(tok.decode([nxt])), flush=True)
    print("\nGENERATION:", repr(tok.decode(out)))


if __name__ == "__main__":
    main()
