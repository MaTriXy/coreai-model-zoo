"""Phase 2 — re-author the Nemotron 3.5 ASR cache-aware FastConformer encoder (OFFLINE mode),
export it to a static Core AI .aimodel, and gate it against the golden `enc_proj`
(run in the MAIN coreai-models venv).

Re-authored in plain torch from `model.safetensors`, mirroring transformers'
modeling_nemotron_asr_streaming.py exactly (offline path, no caches):

  mel [1,L,128] ---(causal Subsampling: conv_in + 2 dw-sep stages, 8x)---> [1,T,1024]
      -> 24 x ConformerBlock(1/2FF . chunked-limited relMHSA . causalConv(LayerNorm) . 1/2FF)
      -> prompt fusion: cat([h, one_hot 128]) -> MLP(1152->2048->1024)      (language conditioning)
      -> encoder_projector Linear(1024->640) -> enc_proj [1,T,640]

Differences vs ../parakeet/export_encoder.py (same skeleton otherwise):
  * convs are CAUSAL (left-pad k-1, right-pad s-1 on time; freq pad (2,1)) instead of SAME
  * attention is windowed with the `chunked_limited` mask: chunk = lookahead+1 frames,
    q attends kv iff 0 <= q_chunk - kv_chunk <= (sliding_window-1)//chunk. Baked as a
    [T,T] boolean constant for the fixed bucket (additive -inf on matrix_bd, exactly as HF).
  * conv module norm is LayerNorm (not BatchNorm), all linear/conv biases OFF
  * prompt one-hot [1,128] is a graph INPUT -> one .aimodel serves all 40 locales

Inputs:  mel [1,L,128] (processor layout), one_hot [1,128]
Output:  enc_proj [1,T,640]

Run:  coreai-models/.venv/bin/python export_encoder.py [--dtype float16] [--skip-export]
"""
from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"

HID, HEADS, HDIM, FF, NLAYERS = 1024, 8, 128, 4096, 24
MEL, SUB_CH, PROJ = 128, 256, 640
CONV_K = 9
NUM_PROMPTS, PROMPT_FF = 128, 2048
SLIDING_WINDOW = 57  # left context = 56


# --------------------------------------------------------------------------- causal convs
def causal_conv2d_pad(x: torch.Tensor, k: int = 3, s: int = 2) -> torch.Tensor:
    # freq: (k-1, s-1) = (2,1); time offline: (k-1, s-1) = (2,1)
    return nn.functional.pad(x, (k - 1, s - 1, k - 1, s - 1))


class Subsampling(nn.Module):
    """Causal stem conv + (log2(8)-1)=2 depthwise-separable stages + linear."""

    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(1, SUB_CH, 3, stride=2)
        self.layers = nn.ModuleList()
        for _ in range(2):
            stage = nn.Module()
            stage.depthwise_conv = nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, groups=SUB_CH)
            stage.pointwise_conv = nn.Conv2d(SUB_CH, SUB_CH, 1)
            self.layers.append(stage)
        # causal freq pad (k-1, s-1) adds +1 per stage: 128 -> 65 -> 33 -> 17 (NOT 128/8=16)
        f = MEL
        for _ in range(3):
            f = f // 2 + 1
        self.linear = nn.Linear(SUB_CH * f, HID, bias=True)

    def forward(self, mel):  # mel [1,L,128]
        x = mel.unsqueeze(1)                       # [1,1,L,128]
        x = torch.relu(self.conv_in(causal_conv2d_pad(x)))
        for stage in self.layers:
            x = stage.pointwise_conv(stage.depthwise_conv(causal_conv2d_pad(x)))
            x = torch.relu(x)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)  # [1,T,SUB_CH*16]
        return self.linear(x)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(HID, FF, bias=False)
        self.linear2 = nn.Linear(FF, HID, bias=False)

    def forward(self, x):
        return self.linear2(torch.nn.functional.silu(self.linear1(x)))


class ConvModule(nn.Module):
    """GLU pointwise -> CAUSAL depthwise (left pad k-1) -> LayerNorm -> silu -> pointwise."""

    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(HID, 2 * HID, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(HID, HID, CONV_K, groups=HID, bias=False)
        self.norm = nn.LayerNorm(HID)
        self.pointwise_conv2 = nn.Conv1d(HID, HID, 1, bias=False)

    def forward(self, x):  # [B,T,HID]
        x = x.transpose(1, 2)
        x = torch.nn.functional.glu(self.pointwise_conv1(x), dim=1)
        x = self.depthwise_conv(nn.functional.pad(x, (CONV_K - 1, 0)))
        x = self.norm(x.transpose(1, 2))
        x = torch.nn.functional.silu(x).transpose(1, 2)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class RelPosAttention(nn.Module):
    """Transformer-XL rel-pos attention with the chunked_limited additive mask on matrix_bd."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HID, HID, bias=False)
        self.k_proj = nn.Linear(HID, HID, bias=False)
        self.v_proj = nn.Linear(HID, HID, bias=False)
        self.o_proj = nn.Linear(HID, HID, bias=False)
        self.relative_k_proj = nn.Linear(HID, HID, bias=False)
        self.bias_u = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.bias_v = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.scaling = HDIM ** -0.5

    @staticmethod
    def _rel_shift(x):  # [B,H,T,2T-1]
        b, h, t, p = x.shape
        x = torch.nn.functional.pad(x, (1, 0))
        x = x.view(b, h, p + 1, t)
        return x[:, :, 1:].view(b, h, t, p)

    def forward(self, x, pos_emb, neg_mask):  # x [B,T,HID], pos_emb [1,2T-1,HID], neg_mask [1,1,T,T]
        B, T, _ = x.shape
        shape = (B, T, HEADS, HDIM)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q_u = q + self.bias_u.view(1, HEADS, 1, HDIM)
        q_v = q + self.bias_v.view(1, HEADS, 1, HDIM)
        rel_k = self.relative_k_proj(pos_emb).view(B, -1, HEADS, HDIM)
        matrix_bd = q_v @ rel_k.permute(0, 2, 3, 1)                 # [B,H,T,2T-1]
        matrix_bd = self._rel_shift(matrix_bd)[..., :T] * self.scaling
        matrix_bd = matrix_bd + neg_mask                            # -inf outside the window
        attn_weights = q_u @ k.transpose(-2, -1) * self.scaling + matrix_bd
        attn = torch.softmax(attn_weights, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, HID)
        return self.o_proj(out)


class ConformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward1 = FeedForward()
        self.self_attn = RelPosAttention()
        self.conv = ConvModule()
        self.feed_forward2 = FeedForward()
        self.norm_feed_forward1 = nn.LayerNorm(HID)
        self.norm_self_att = nn.LayerNorm(HID)
        self.norm_conv = nn.LayerNorm(HID)
        self.norm_feed_forward2 = nn.LayerNorm(HID)
        self.norm_out = nn.LayerNorm(HID)

    def forward(self, x, pos_emb, neg_mask):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb, neg_mask)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


def build_pos_emb(T: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HID, 2, dtype=torch.float) / HID))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float)
    freqs = (inv_freq[:, None] @ pos[None, :]).transpose(0, 1)      # [2T-1, HID/2]
    pe = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(2 * T - 1, HID)
    return pe[None]


def build_chunked_limited_neg_mask(T: int, lookahead: int) -> torch.Tensor:
    """[1,1,T,T] additive mask: 0 inside the window, -inf outside (HF chunked_limited)."""
    chunk = lookahead + 1
    left_chunks = (SLIDING_WINDOW - 1) // chunk
    idx = torch.arange(T)
    q_chunk = (idx // chunk)[:, None]
    kv_chunk = (idx // chunk)[None, :]
    diff = q_chunk - kv_chunk
    allowed = (diff >= 0) & (diff <= left_chunks)
    neg = torch.zeros(T, T)
    neg.masked_fill_(~allowed, float("-inf"))
    return neg[None, None]


class Encoder(nn.Module):
    def __init__(self, mel_len: int, lookahead: int):
        super().__init__()
        self.subsampling = Subsampling()
        self.layers = nn.ModuleList([ConformerBlock() for _ in range(NLAYERS)])
        # prompt fusion MLP + final projector
        self.prompt_linear_1 = nn.Linear(HID + NUM_PROMPTS, PROMPT_FF)
        self.prompt_linear_2 = nn.Linear(PROMPT_FF, HID)
        self.projector = nn.Linear(HID, PROJ, bias=True)
        T = mel_len
        for _ in range(3):  # causal pad (k-1,s-1): out = floor((L + 3 - 3)/2) + 1 = L//2 + 1
            T = T // 2 + 1
        self.T = T
        self.register_buffer("pos_emb", build_pos_emb(T), persistent=False)
        self.register_buffer("neg_mask", build_chunked_limited_neg_mask(T, lookahead), persistent=False)

    def forward(self, mel, one_hot):  # mel [1,L,128], one_hot [1,NUM_PROMPTS]
        x = self.subsampling(mel)
        for blk in self.layers:
            x = blk(x, self.pos_emb, self.neg_mask)
        oh = one_hot[:, None, :].expand(-1, x.shape[1], -1)
        fused = torch.cat([x, oh], dim=-1)
        fused = self.prompt_linear_2(torch.relu(self.prompt_linear_1(fused)))
        return self.projector(fused)


def load_weights(enc: Encoder):
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(MODEL, "model.safetensors")
    sd = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.subsampling.") or k.startswith("encoder.layers."):
                sd[k[len("encoder."):]] = f.get_tensor(k)
            elif k.startswith("prompt_projector.linear_1."):
                sd["prompt_linear_1." + k.split(".")[-1]] = f.get_tensor(k)
            elif k.startswith("prompt_projector.linear_2."):
                sd["prompt_linear_2." + k.split(".")[-1]] = f.get_tensor(k)
            elif k.startswith("encoder_projector."):
                sd["projector." + k[len("encoder_projector."):]] = f.get_tensor(k)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    miss = [m for m in missing if "pos_emb" not in m and "neg_mask" not in m]
    assert not miss and not unexpected, f"weight map mismatch: missing={miss[:8]} unexpected={unexpected[:8]}"
    print(f"[weights] loaded {len(sd)} tensors (strict map OK)")


# --------------------------------------------------------------------------- gate/export
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--oracle", default="oracle_en_US.npz")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    d = np.load(HERE / args.oracle)
    mel = torch.from_numpy(d["input_features"]).float()     # [1,L,128] (processor layout)
    golden = torch.from_numpy(d["enc_proj"]).float()        # [T,640]
    one_hot = torch.from_numpy(d["one_hot"]).float()[None]  # [1,128]
    lookahead = int(d["num_lookahead_tokens"])
    L = mel.shape[1]
    print(f"mel {tuple(mel.shape)} L={L} lookahead={lookahead} golden enc_proj {tuple(golden.shape)}")

    enc = Encoder(L, lookahead).eval()
    load_weights(enc)
    assert enc.T == golden.shape[0], f"T mismatch {enc.T} vs {golden.shape[0]}"

    with torch.no_grad():
        out = enc(mel, one_hot)[0]
    cos = torch.nn.functional.cosine_similarity(out.reshape(-1), golden.reshape(-1), dim=0).item()
    pertok = torch.nn.functional.cosine_similarity(out, golden, dim=-1)
    print(f"[eager fp32] global cos {cos:.6f}  per-token mean {pertok.mean():.6f} "
          f"min {pertok.min():.6f}  max|Δ| {(out - golden).abs().max():.4f}")
    if pertok.mean() < 0.999:
        print("❌ re-author DIVERGES — fix before export")
        raise SystemExit(1)
    print("✅ eager re-author matches golden")
    if args.skip_export:
        return

    # ---- export to Core AI + engine gate ----
    import asyncio
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    enc_d = enc.to(dtype)
    example = {"mel": torch.zeros(1, L, MEL, dtype=dtype),
               "one_hot": torch.zeros(1, NUM_PROMPTS, dtype=dtype)}
    print(f"[export] Nemotron FastConformer encoder ({args.dtype}) -> Core AI ...", flush=True)
    prog = export_to_coreai(
        enc_d, example, dynamic_shapes=None,
        input_names=("mel", "one_hot"), output_names=("enc_proj",),
        state_names=None, externalize_modules=[])
    prog.optimize()
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    aimodel = art / f"nemotron_asr_encoder_{args.dtype}_L{L}.aimodel"
    shutil.rmtree(aimodel, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "openmdw-1.1"
    prog.save_asset(aimodel, meta)
    sz = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {aimodel} ({sz:.1f} MB)")

    async def gate(unit):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        m = await rt.AIModel.load(str(aimodel), opts)
        fn = m.load_function("main")
        res = await fn({"mel": rt.NDArray(mel.to(dtype).numpy()),
                        "one_hot": rt.NDArray(one_hot.to(dtype).numpy())})
        eng = torch.from_numpy(res["enc_proj"].numpy().astype(np.float32))[0]
        pt = torch.nn.functional.cosine_similarity(eng, golden, dim=-1)
        ok = pt.mean() > 0.999 and pt.min() > 0.99
        print(f"[gate {unit}] per-token cos mean {pt.mean():.6f} min {pt.min():.6f} "
              f"max|Δ| {(eng - golden).abs().max():.3f} -> {'PASS' if ok else 'FAIL'}")
        return ok

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
