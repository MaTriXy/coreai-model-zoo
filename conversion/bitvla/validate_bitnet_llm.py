# Community port — NOT an Apple model.
"""S1 reference: BitNet b1.58 2B4T LLM arch (the BitVLA backbone) in plain torch, fake-quant
BitLinear matching transformers exactly (per-tensor absmean ternary weight + per-token int8
activation), loaded from the bf16 master. Greedy-generate to confirm the new arch (ReLU² FFN,
attn/ffn SubLN, RoPE theta 500k, tied head) before any Metal kernel.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitvla/validate_bitnet_llm.py --new 12
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402

CK = str(work_path("_bitvla_ckpt", "bitnet2b_bf16"))

# bitnet-b1.58-2B-4T config
NL, H, FF = 30, 2560, 6912
NH, NKV, HD = 20, 5, 128
VOCAB = 128256
EPS = 1e-5
ROPE_THETA = 500000.0


def weight_quant(w):  # per-tensor absmean ternary {-1,0,1}, dequantized (transformers WeightQuant)
    w = w.float()
    s = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * s).round().clamp(-1, 1) / s


def act_quant(x):  # per-token int8 absmax, dequantized (transformers ActQuant)
    x = x.float()
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    return (x * s).round().clamp(-128, 127) / s


class BitLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(o, i))

    def forward(self, x):
        return F.linear(act_quant(x), weight_quant(self.weight))


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        f = x.float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)
        return (f * self.weight).to(x.dtype)


def rope_tables(seqlen, dtype=torch.float32):
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2).float() / HD))
    t = torch.arange(seqlen).float()
    fr = torch.outer(t, inv)
    emb = torch.cat((fr, fr), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rot_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = BitLinear(H, NH * HD)
        self.k_proj = BitLinear(H, NKV * HD)
        self.v_proj = BitLinear(H, NKV * HD)
        self.o_proj = BitLinear(NH * HD, H)
        self.attn_sub_norm = RMSNorm(H)

    def forward(self, x, cos, sin, mask):
        b, T, _ = x.shape
        q = self.q_proj(x).view(b, T, NH, HD).transpose(1, 2)
        k = self.k_proj(x).view(b, T, NKV, HD).transpose(1, 2)
        v = self.v_proj(x).view(b, T, NKV, HD).transpose(1, 2)
        c = cos[:T].view(1, 1, T, HD); s = sin[:T].view(1, 1, T, HD)
        q = q * c + rot_half(q) * s
        k = k * c + rot_half(k) * s
        k = k.repeat_interleave(NH // NKV, 1); v = v.repeat_interleave(NH // NKV, 1)
        sc = (q @ k.transpose(-1, -2)) / math.sqrt(HD) + mask
        o = (F.softmax(sc, -1) @ v).transpose(1, 2).reshape(b, T, NH * HD)
        return self.o_proj(self.attn_sub_norm(o))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = BitLinear(H, FF)
        self.up_proj = BitLinear(H, FF)
        self.down_proj = BitLinear(FF, H)
        self.ffn_sub_norm = RMSNorm(FF)

    def forward(self, x):
        h = F.relu(self.gate_proj(x)) ** 2 * self.up_proj(x)   # relu2
        return self.down_proj(self.ffn_sub_norm(h))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = Attn(); self.mlp = MLP()
        self.input_layernorm = RMSNorm(H)
        self.post_attention_layernorm = RMSNorm(H)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class BitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, H)
        self.layers = nn.ModuleList([Layer() for _ in range(NL)])
        self.norm = RMSNorm(H)

    def forward(self, ids, cos, sin):
        x = self.embed_tokens(ids)
        T = ids.shape[1]
        mask = torch.full((T, T), float("-inf")).triu(1)
        for l in self.layers:
            x = l(x, cos, sin, mask)
        x = self.norm(x)
        return F.linear(x, self.embed_tokens.weight)   # tied head


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--new", type=int, default=12); args = ap.parse_args()
    print("loading bf16 master ...", flush=True)
    sd = load_file(f"{CK}/model.safetensors")
    m = BitNet().float().eval()
    own = m.state_dict()
    mapped = {}
    for k, v in sd.items():
        kk = k[len("model."):] if k.startswith("model.") else k
        if kk in own:
            mapped[kk] = v.float()
    miss = [k for k in own if k not in mapped]
    print(f"loaded {len(mapped)}/{len(own)} params; missing e.g. {miss[:3]}", flush=True)
    m.load_state_dict(mapped, strict=False, assign=True)
    tok = AutoTokenizer.from_pretrained(CK)
    msgs = [{"role": "user", "content": args.prompt}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    cos, sin = rope_tables(ids.shape[1] + args.new + 4)
    print("prompt:", repr(args.prompt), flush=True)
    out = []
    with torch.no_grad():
        for _ in range(args.new):
            logits = m(ids, cos, sin)
            nxt = int(logits[0, -1].argmax())
            out.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]])], 1)
            print("  ->", nxt, repr(tok.decode([nxt])), flush=True)
    print("GEN:", repr(tok.decode(out)), flush=True)


if __name__ == "__main__":
    main()
