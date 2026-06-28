"""Phase 2 — re-author the Parakeet FastConformer encoder, export it to a static Core AI
.aimodel, and gate it against the golden `enc_proj` (run in the MAIN coreai-torch venv).

The encoder is re-authored in plain torch from `model.safetensors` (the isolated tf-5.x env
is NOT needed here — we gate against oracle.npz['enc_proj']). Architecture mirrors
transformers' modeling_parakeet.py exactly:

  mel [1,128,L] -> Subsampling(conv2d ×3, 8×) -> [1,T,1024]
               -> 24 × ConformerBlock(½FF · relMHSA · ConvModule · ½FF · LN_out)
               -> encoder_projector Linear(1024->640) -> enc_proj [1,T,640]

Rel-pos attention is Transformer-XL (bias_u/bias_v + relative_k_proj + rel_shift); the
positional table is a constant baked for the bucket length. Single full-length clip -> no
attention mask (the golden clip fills the bucket exactly, L=1485 -> T=186).

Run (MAIN venv; _GPU_LOCK held for the engine gate):
    coreai-models/.venv/bin/python export_encoder.py [--dtype float16] [--skip-export]
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
HID, HEADS, HDIM, FF, NLAYERS = 1024, 8, 128, 4096, 24
MEL, SUB_CH, VOCAB_PROJ = 128, 256, 640
CONV_K = 9


# --------------------------------------------------------------------------- modules
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(HID, FF, bias=False)
        self.linear2 = nn.Linear(FF, HID, bias=False)

    def forward(self, x):
        return self.linear2(torch.nn.functional.silu(self.linear1(x)))


class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(HID, 2 * HID, 1, bias=False)
        self.depthwise_conv = nn.Conv1d(HID, HID, CONV_K, padding=(CONV_K - 1) // 2, groups=HID, bias=False)
        self.norm = nn.BatchNorm1d(HID)
        self.pointwise_conv2 = nn.Conv1d(HID, HID, 1, bias=False)

    def forward(self, x):  # x [B,T,HID]
        x = x.transpose(1, 2)  # [B,HID,T]
        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.norm(x)
        x = torch.nn.functional.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class RelPosAttention(nn.Module):
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
    def _rel_shift(x):  # x [B,H,T,2T-1]
        b, h, t, p = x.shape
        x = torch.nn.functional.pad(x, (1, 0))
        x = x.view(b, h, p + 1, t)
        x = x[:, :, 1:].view(b, h, t, p)
        return x

    def forward(self, x, pos_emb):  # x [B,T,HID], pos_emb [1,2T-1,HID]
        B, T, _ = x.shape
        shape = (B, T, HEADS, HDIM)
        q = self.q_proj(x).view(shape).transpose(1, 2)  # [B,H,T,D]
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        q_u = q + self.bias_u.view(1, HEADS, 1, HDIM)
        q_v = q + self.bias_v.view(1, HEADS, 1, HDIM)
        rel_k = self.relative_k_proj(pos_emb).view(B, -1, HEADS, HDIM)  # [B,2T-1,H,D]
        matrix_ac = q_u @ k.transpose(-2, -1)                           # [B,H,T,T]
        matrix_bd = q_v @ rel_k.permute(0, 2, 3, 1)                     # [B,H,T,2T-1]
        matrix_bd = self._rel_shift(matrix_bd)[..., :T]
        attn = torch.softmax((matrix_ac + matrix_bd) * self.scaling, dim=-1)
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

    def forward(self, x, pos_emb):
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class Subsampling(nn.Module):
    """3 conv2d layers (8× on time AND freq) + linear. Indices match the HF ModuleList."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1, SUB_CH, 3, stride=2, padding=1),           # 0
            nn.ReLU(),                                              # 1
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),  # 2 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, 1),                           # 3 pointwise
            nn.ReLU(),                                              # 4
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),  # 5 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, 1),                           # 6 pointwise
            nn.ReLU(),                                              # 7
        ])
        out_freq = MEL // 8
        self.linear = nn.Linear(SUB_CH * out_freq, HID, bias=True)

    def forward(self, mel):  # mel [B,MEL,L]
        x = mel.transpose(1, 2).unsqueeze(1)  # [B,1,L,MEL]
        for layer in self.layers:
            x = layer(x)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)  # [B,T,SUB_CH*out_freq]
        return self.linear(x)


def build_pos_emb(T: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HID, 2, dtype=torch.float) / HID))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float)            # [2T-1]
    freqs = (inv_freq[:, None] @ pos[None, :]).transpose(0, 1)      # [2T-1, HID/2]
    pe = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(2 * T - 1, HID)
    return pe[None]                                                 # [1,2T-1,HID]


class Encoder(nn.Module):
    def __init__(self, mel_len: int):
        super().__init__()
        self.subsampling = Subsampling()
        self.layers = nn.ModuleList([ConformerBlock() for _ in range(NLAYERS)])
        self.projector = nn.Linear(HID, VOCAB_PROJ, bias=True)
        T = mel_len
        for _ in range(3):
            T = (T + 2 - 3) // 2 + 1
        self.T = T
        self.register_buffer("pos_emb", build_pos_emb(T), persistent=False)

    def forward(self, mel):  # mel [1,MEL,L]
        x = self.subsampling(mel)
        for blk in self.layers:
            x = blk(x, self.pos_emb)
        return self.projector(x)


def load_weights(enc: Encoder):
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("nvidia/parakeet-tdt-0.6b-v3", "model.safetensors")
    sd = {}
    with safe_open(p, framework="pt") as f:
        for k in f.keys():
            if k.startswith("encoder.subsampling."):
                sd[k[len("encoder."):]] = f.get_tensor(k)
            elif k.startswith("encoder.layers."):
                sd[k[len("encoder."):]] = f.get_tensor(k)
            elif k.startswith("encoder_projector."):
                sd["projector." + k[len("encoder_projector."):]] = f.get_tensor(k)
    missing, unexpected = enc.load_state_dict(sd, strict=False)
    # num_batches_tracked buffers are the only acceptable "missing" extras
    miss = [m for m in missing if not m.endswith("num_batches_tracked")]
    assert not miss and not unexpected, f"weight map mismatch: missing={miss[:5]} unexpected={unexpected[:5]}"
    print(f"[weights] loaded {len(sd)} tensors (strict map OK)")


# --------------------------------------------------------------------------- gate/export
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--oracle", default="oracle.npz")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    d = np.load(HERE / args.oracle)
    mel = torch.from_numpy(d["input_features"]).float()    # [1,L,128] (processor layout)
    golden = torch.from_numpy(d["enc_proj"]).float()       # [T,640]
    L = mel.shape[1]
    mel_in = mel.transpose(1, 2).contiguous()              # [1,128,L]
    print(f"mel {tuple(mel_in.shape)} L={L} golden enc_proj {tuple(golden.shape)}")

    enc = Encoder(L).eval()
    load_weights(enc)
    assert enc.T == golden.shape[0], f"T mismatch {enc.T} vs {golden.shape[0]}"

    with torch.no_grad():
        out = enc(mel_in)[0]                                # [T,640]
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
    example = {"mel": torch.zeros(1, MEL, L, dtype=dtype)}
    print(f"[export] FastConformer encoder ({args.dtype}) -> Core AI ...", flush=True)
    prog = export_to_coreai(
        enc_d, example, dynamic_shapes=None,
        input_names=("mel",), output_names=("enc_proj",),
        state_names=None, externalize_modules=[])
    prog.optimize()
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    aimodel = art / f"parakeet_encoder_{args.dtype}_L{L}.aimodel"
    shutil.rmtree(aimodel, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "cc-by-4.0"
    prog.save_asset(aimodel, meta)
    sz = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {aimodel} ({sz:.1f} MB)")

    async def gate(unit):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        m = await rt.AIModel.load(str(aimodel), opts)
        fn = m.load_function("main")
        res = await fn({"mel": rt.NDArray(mel_in.to(dtype).numpy())})
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
