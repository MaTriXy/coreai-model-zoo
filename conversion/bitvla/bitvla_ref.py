# Community port — NOT an Apple model.
"""Standalone torch REFERENCE (oracle) for BitVLA (lxsy/bitvla-bf16), built from the official
ustcwhy/BitVLA math verbatim: BitSigLIP-SO400M vision tower + 2-layer MLP projector + BitNet
b1.58-2B LLM + OpenVLA discrete-action detokenizer. Fake-quant BitLinear (per-tensor absmean
ternary weight + per-token int8 activation) exactly matches transformers `integrations/bitnet.py`
and `models/siglip/modeling_siglip.py`. The LLM half is the S1-validated arch (coherent text gen).

This is the parity oracle the Core AI export gates against, stage by stage:
  vision (cos), projected image embeds, action tokens / 7-DoF.

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitvla/bitvla_ref.py --vision   # S2 vision smoke
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file

CK = "/Users/majimadaisuke/code/coreai/_bitvla_ckpt/bitvla_bf16/model.safetensors"

# --- BitNet quant (verbatim official; per-tensor absmean W + per-token int8 A) ----------------- #


def weight_quant(w):
    w = w.float()
    s = 1.0 / w.abs().mean().clamp_(min=1e-5)
    return (w * s).round().clamp(-1, 1) / s


def act_quant(x):
    x = x.float()
    s = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_(min=1e-5)
    return (x * s).round().clamp(-128, 127) / s


class BitLinear(nn.Module):
    """W1.58-A8 fake-quant linear. bias optional (vision SigLIP linears have bias; LLM has none).

    prebaked=True: ``weight`` already holds the ternary-valued ``weight_quant(W)`` (so an int8
    exporter can compress it losslessly); forward then only does the A8 activation quant."""

    def __init__(self, i, o, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(o, i))
        self.bias = nn.Parameter(torch.zeros(o)) if bias else None
        self.prebaked = False
        self.skip_act = False        # fp16 activations (no in-graph int8 act_quant) for device GPU

    def prebake(self):
        with torch.no_grad():
            self.weight.copy_(weight_quant(self.weight))
        self.prebaked = True

    def forward(self, x):
        w = self.weight if self.prebaked else weight_quant(self.weight)
        xq = x if self.skip_act else act_quant(x)      # A8 (round/amax) stalls the h18p GPU in vision
        xq = xq.to(w.dtype)                            # quantizer may calibrate in fp32; match w
        b = self.bias.to(w.dtype) if self.bias is not None else None
        return F.linear(xq, w, b)


# =============================================================================================== #
# VISION: BitSigLIP-SO400M (hidden 1152, 26L, FFN 4304, 16hd, patch14/224 -> 256 tokens)
# =============================================================================================== #
V_H, V_L, V_FF, V_NH = 1152, 26, 4304, 16
V_HD = V_H // V_NH            # 72
V_EPS = 1e-6
PATCH, IMG = 14, 224
NPATCH = (IMG // PATCH) ** 2  # 256


class SiglipEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv2d(3, V_H, kernel_size=PATCH, stride=PATCH)  # valid
        self.position_embedding = nn.Embedding(NPATCH, V_H)
        self.register_buffer("position_ids", torch.arange(NPATCH).unsqueeze(0), persistent=False)

    def forward(self, pixel_values):
        patches = self.patch_embedding(pixel_values)             # [B, H, 16, 16]
        embeds = patches.flatten(2).transpose(1, 2)              # [B, 256, H]
        return embeds + self.position_embedding(self.position_ids)


class SiglipAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = BitLinear(V_H, V_H, bias=True)
        self.k_proj = BitLinear(V_H, V_H, bias=True)
        self.v_proj = BitLinear(V_H, V_H, bias=True)
        self.out_proj = BitLinear(V_H, V_H, bias=True)
        self.scale = V_HD ** -0.5

    def forward(self, x):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, V_NH, V_HD).transpose(1, 2)
        k = self.k_proj(x).view(b, t, V_NH, V_HD).transpose(1, 2)
        v = self.v_proj(x).view(b, t, V_NH, V_HD).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v)          # SDPA (matches official + export)
        o = o.transpose(1, 2).reshape(b, t, V_H)
        return self.out_proj(o)


class SiglipMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BitLinear(V_H, V_FF, bias=True)
        self.fc2 = BitLinear(V_FF, V_H, bias=True)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SiglipLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(V_H, eps=V_EPS)
        self.self_attn = SiglipAttention()
        self.layer_norm2 = nn.LayerNorm(V_H, eps=V_EPS)
        self.mlp = SiglipMLP()

    def forward(self, x):
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class BitSigLIP(nn.Module):
    """vision_feature_layer=-1: returns the LAST encoder-layer output (no post_layernorm)."""

    def __init__(self):
        super().__init__()
        self.embeddings = SiglipEmbeddings()
        self.layers = nn.ModuleList([SiglipLayer() for _ in range(V_L)])

    def forward(self, pixel_values):
        x = self.embeddings(pixel_values)
        for layer in self.layers:
            x = layer(x)
        return x                                                  # [B, 256, 1152]


class Projector(nn.Module):
    """multi_modal_projector: 1152 -> 2560 -gelu-> 2560 (fp, with bias)."""

    def __init__(self):
        super().__init__()
        self.linear_1 = nn.Linear(V_H, 2560)
        self.linear_2 = nn.Linear(2560, 2560)

    def forward(self, x):
        return self.linear_2(F.gelu(self.linear_1(x)))


# =============================================================================================== #
# LLM: BitNet b1.58 2B4T (30L, hidden 2560, FFN 6912, GQA 20/5 hd128, ReLU2, SubLN, RoPE 500k)
# =============================================================================================== #
L_H, L_L, L_FF = 2560, 30, 6912
L_NH, L_NKV, L_HD = 20, 5, 128
L_EPS = 1e-5
L_THETA = 500000.0
VOCAB = 128268


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        f = x.float()
        f = f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + L_EPS)
        return (f * self.weight).to(x.dtype)


def _rope(seqlen, dtype=torch.float32):
    inv = 1.0 / (L_THETA ** (torch.arange(0, L_HD, 2).float() / L_HD))
    fr = torch.outer(torch.arange(seqlen).float(), inv)
    emb = torch.cat((fr, fr), -1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _roth(x):
    a, b = x.chunk(2, -1)
    return torch.cat((-b, a), -1)


class LAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = BitLinear(L_H, L_NH * L_HD)
        self.k_proj = BitLinear(L_H, L_NKV * L_HD)
        self.v_proj = BitLinear(L_H, L_NKV * L_HD)
        self.o_proj = BitLinear(L_NH * L_HD, L_H)
        self.attn_sub_norm = RMSNorm(L_H)

    def forward(self, x, cos, sin, mask):
        b, T, _ = x.shape
        q = self.q_proj(x).view(b, T, L_NH, L_HD).transpose(1, 2)
        k = self.k_proj(x).view(b, T, L_NKV, L_HD).transpose(1, 2)
        v = self.v_proj(x).view(b, T, L_NKV, L_HD).transpose(1, 2)
        c = cos[:T].view(1, 1, T, L_HD); s = sin[:T].view(1, 1, T, L_HD)
        q = q * c + _roth(q) * s
        k = k * c + _roth(k) * s
        k = k.repeat_interleave(L_NH // L_NKV, 1); v = v.repeat_interleave(L_NH // L_NKV, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # SDPA causal
        o = o.transpose(1, 2).reshape(b, T, L_NH * L_HD)
        return self.o_proj(self.attn_sub_norm(o))


class LMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = BitLinear(L_H, L_FF)
        self.up_proj = BitLinear(L_H, L_FF)
        self.down_proj = BitLinear(L_FF, L_H)
        self.ffn_sub_norm = RMSNorm(L_FF)

    def forward(self, x):
        h = F.relu(self.gate_proj(x)) ** 2 * self.up_proj(x)
        return self.down_proj(self.ffn_sub_norm(h))


class LLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = LAttn(); self.mlp = LMLP()
        self.input_layernorm = RMSNorm(L_H)
        self.post_attention_layernorm = RMSNorm(L_H)

    def forward(self, x, cos, sin, mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
        return x + self.mlp(self.post_attention_layernorm(x))


class BitNetLLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB, L_H)
        self.layers = nn.ModuleList([LLayer() for _ in range(L_L)])
        self.norm = RMSNorm(L_H)
        self.lm_head = nn.Linear(L_H, VOCAB, bias=False)

    def forward(self, inputs_embeds):
        x = inputs_embeds
        T = x.shape[1]
        mask = torch.full((T, T), float("-inf")).triu(1)
        cos, sin = _rope(T)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.lm_head(self.norm(x))


def load_llm(llm: BitNetLLM):
    sd = load_file(CK)
    p = "language_model.model."
    own = {"embed_tokens.weight": sd[p + "embed_tokens.weight"],
           "norm.weight": sd[p + "norm.weight"],
           "lm_head.weight": sd["language_model.lm_head.weight"]}
    for i in range(L_L):
        s = f"{p}layers.{i}."; d = f"layers.{i}."
        own[d + "input_layernorm.weight"] = sd[s + "input_layernorm.weight"]
        own[d + "post_attention_layernorm.weight"] = sd[s + "post_attention_layernorm.weight"]
        own[d + "self_attn.attn_sub_norm.weight"] = sd[s + "self_attn.attn_sub_norm.weight"]
        own[d + "mlp.ffn_sub_norm.weight"] = sd[s + "mlp.ffn_sub_norm.weight"]
        for pr in ("q_proj", "k_proj", "v_proj", "o_proj"):
            own[d + "self_attn." + pr + ".weight"] = sd[s + "self_attn." + pr + ".weight"]
        for pr in ("gate_proj", "up_proj", "down_proj"):
            own[d + "mlp." + pr + ".weight"] = sd[s + "mlp." + pr + ".weight"]
    missing, _ = llm.load_state_dict({k: v.float() for k, v in own.items()}, strict=False)
    assert not missing, f"llm missing {missing}"
    return len(own)


SYS = ("System: A chat between a curious human and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the human's questions.<|eot_id|>")


def build_embeds(tok, llm, img_embeds, question):
    """LLaVA prompt: '<sys>Human: <image>\\n{q}<|eot_id|>Assistant: ' with 256 image embeds spliced."""
    pre = SYS + "Human: "
    post = "\n" + question + "<|eot_id|>Assistant: "
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=True).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    e_pre = llm.embed_tokens(pre_ids)
    e_post = llm.embed_tokens(post_ids)
    return torch.cat([e_pre, img_embeds, e_post], dim=1)


@torch.no_grad()
def generate(llm, embeds, new=20, stop_ids=(128009, 128001)):
    out = []
    cur = embeds
    for _ in range(new):
        logits = llm(cur)
        nxt = int(logits[0, -1].argmax())
        if nxt in stop_ids:
            break
        out.append(nxt)
        cur = torch.cat([cur, llm.embed_tokens(torch.tensor([[nxt]]))], dim=1)
    return out


# --- OpenVLA action detokenizer (verbatim bitnet_action_tokenizer.py + _unnormalize_actions) ---- #
TOTAL_VOCAB = 128268
N_BINS = 256


def detokenize_action(token_ids, norm_stats, unnorm_key):
    """7 action-token ids -> continuous 7-DoF via 256-bin centers + BOUNDS_Q99 unnormalization."""
    import numpy as np
    bins = np.linspace(-1, 1, N_BINS)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    ids = np.array(token_ids)
    disc = TOTAL_VOCAB - ids
    disc = np.clip(disc - 1, 0, bin_centers.shape[0] - 1)
    norm = bin_centers[disc]                                  # [-1,1] per dim
    st = norm_stats[unnorm_key]["action"]
    low, high = np.array(st["q01"]), np.array(st["q99"])
    mask = np.array(st.get("mask", np.ones_like(low, dtype=bool)))
    return np.where(mask, 0.5 * (norm + 1) * (high - low + 1e-8) + low, norm)


def action_prompt_embeds(tok, llm, img_embeds, instruction):
    q = f"What action should the robot take to {instruction}?"
    return build_embeds(tok, llm, img_embeds, q)


def load_vision(model: nn.Module, projector: nn.Module):
    sd = load_file(CK)
    vp = "vision_tower.vision_model."
    own = {}
    own["embeddings.patch_embedding.weight"] = sd[vp + "embeddings.patch_embedding.weight"]
    own["embeddings.patch_embedding.bias"] = sd[vp + "embeddings.patch_embedding.bias"]
    own["embeddings.position_embedding.weight"] = sd[vp + "embeddings.position_embedding.weight"]
    for i in range(V_L):
        s = f"{vp}encoder.layers.{i}."
        d = f"layers.{i}."
        for ln in ("layer_norm1", "layer_norm2"):
            own[d + ln + ".weight"] = sd[s + ln + ".weight"]
            own[d + ln + ".bias"] = sd[s + ln + ".bias"]
        for p in ("q_proj", "k_proj", "v_proj", "out_proj"):
            own[d + "self_attn." + p + ".weight"] = sd[s + "self_attn." + p + ".weight"]
            own[d + "self_attn." + p + ".bias"] = sd[s + "self_attn." + p + ".bias"]
        for p in ("fc1", "fc2"):
            own[d + "mlp." + p + ".weight"] = sd[s + "mlp." + p + ".weight"]
            own[d + "mlp." + p + ".bias"] = sd[s + "mlp." + p + ".bias"]
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in own.items()}, strict=False)
    assert not [m for m in missing if "position_ids" not in m], f"vision missing {missing}"
    pj = {"linear_1.weight": sd["multi_modal_projector.linear_1.weight"].float(),
          "linear_1.bias": sd["multi_modal_projector.linear_1.bias"].float(),
          "linear_2.weight": sd["multi_modal_projector.linear_2.weight"].float(),
          "linear_2.bias": sd["multi_modal_projector.linear_2.bias"].float()}
    projector.load_state_dict(pj)
    return len(own)


def preprocess(img) -> torch.Tensor:
    """SiglipImageProcessor: resize 224 bicubic, rescale 1/255, normalize mean/std 0.5 -> [1,3,224,224]."""
    from torchvision import transforms
    t = transforms.Compose([
        transforms.Resize((IMG, IMG), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),                                    # [0,1], CHW
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(img.convert("RGB")).unsqueeze(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vision", action="store_true")
    ap.add_argument("--caption", action="store_true", help="full VLM coherence check (image -> text)")
    ap.add_argument("--image", default=None)
    ap.add_argument("--question", default="What is in this image?")
    ap.add_argument("--new", type=int, default=24)
    args = ap.parse_args()

    vis = BitSigLIP().float().eval()
    proj = Projector().float().eval()
    n = load_vision(vis, proj)
    print(f"loaded {n} vision+proj tensors", flush=True)

    if args.image:
        from PIL import Image
        pv = preprocess(Image.open(args.image))
    else:
        torch.manual_seed(0)
        pv = torch.randn(1, 3, IMG, IMG)

    with torch.no_grad():
        feat = vis(pv)
        emb = proj(feat)
    print(f"vision feat {tuple(feat.shape)} mean {feat.mean():.4f} std {feat.std():.4f}", flush=True)
    print(f"proj  emb  {tuple(emb.shape)} mean {emb.mean():.4f} std {emb.std():.4f}", flush=True)

    if args.caption:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("/Users/majimadaisuke/code/coreai/_bitvla_ckpt/bitvla_bf16")
        llm = BitNetLLM().float().eval()
        m = load_llm(llm)
        print(f"loaded {m} llm tensors; generating ...", flush=True)
        embeds = build_embeds(tok, llm, emb, args.question)
        print(f"prompt embeds {tuple(embeds.shape)} (256 image + text)", flush=True)
        out = generate(llm, embeds, new=args.new)
        print("Q:", args.question, flush=True)
        print("A:", repr(tok.decode(out)), flush=True)


if __name__ == "__main__":
    main()
