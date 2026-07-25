"""LFM2.5-Audio — Milestone A: re-author the FastConformer audio encoder + MLP adapter in
plain torch from `model.safetensors`, gate eager fp32 vs the liquid_audio oracle
(`asr_pathref.npz['adapter_out']`), then export a static Core AI .aimodel and engine-gate it.

Graph:  mel [1,128,L] -> pre_encode(dw_striding conv2d ×3, 8×) -> [1,T,512]
        -> 17 × ConformerBlock(½FF · relMHSA(Transformer-XL) · ConvModule(BN) · ½FF · LN_out)
        -> audio_adapter( LN512 -> Lin512->2048 -> GELU -> Lin2048->2048 ) -> [1,T,2048]

Module names mirror the raw-NeMo checkpoint keys (pre_encode / layers.N.self_attn.linear_q /
conv.batch_norm / audio_adapter.model.N) so a stripped `conformer.` prefix loads strict.
Offline (att_context [-1,-1]) single clip -> full attention, no mask. bias ON everywhere.

Run (coreai-models/.venv, _GPU_LOCK held for gate):
    coreai-models/.venv/bin/python export_encoder_adapter.py [--dtype float16] [--skip-export]
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402
CKPT = hf_snapshot("LiquidAI/LFM2.5-Audio-1.5B", "model.safetensors")

HID, HEADS, HDIM, FF, NLAYERS = 512, 8, 64, 2048, 17
MEL, SUB_CH, LFM_DIM = 128, 256, 2048
CONV_K = 9


# --------------------------------------------------------------------------- conformer modules
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(HID, FF, bias=True)
        self.linear2 = nn.Linear(FF, HID, bias=True)

    def forward(self, x):
        return self.linear2(torch.nn.functional.silu(self.linear1(x)))


class ConvModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(HID, 2 * HID, 1, bias=True)
        self.depthwise_conv = nn.Conv1d(HID, HID, CONV_K, padding=(CONV_K - 1) // 2, groups=HID, bias=True)
        self.batch_norm = nn.BatchNorm1d(HID)
        self.pointwise_conv2 = nn.Conv1d(HID, HID, 1, bias=True)

    def forward(self, x):  # x [B,T,HID]
        x = x.transpose(1, 2)                       # [B,HID,T]
        x = self.pointwise_conv1(x)
        x = torch.nn.functional.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.silu(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class RelPosAttention(nn.Module):
    """NeMo Transformer-XL rel-pos MHA (names match the checkpoint)."""

    def __init__(self):
        super().__init__()
        self.linear_q = nn.Linear(HID, HID, bias=True)
        self.linear_k = nn.Linear(HID, HID, bias=True)
        self.linear_v = nn.Linear(HID, HID, bias=True)
        self.linear_out = nn.Linear(HID, HID, bias=True)
        self.linear_pos = nn.Linear(HID, HID, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(HEADS, HDIM))
        self.pos_bias_v = nn.Parameter(torch.zeros(HEADS, HDIM))
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
        q = self.linear_q(x).view(shape).transpose(1, 2)   # [B,H,T,D]
        k = self.linear_k(x).view(shape).transpose(1, 2)
        v = self.linear_v(x).view(shape).transpose(1, 2)
        q_u = q + self.pos_bias_u.view(1, HEADS, 1, HDIM)
        q_v = q + self.pos_bias_v.view(1, HEADS, 1, HDIM)
        rel_k = self.linear_pos(pos_emb).view(B, -1, HEADS, HDIM)  # [B,2T-1,H,D]
        matrix_ac = q_u @ k.transpose(-2, -1)                      # [B,H,T,T]
        matrix_bd = q_v @ rel_k.permute(0, 2, 3, 1)                # [B,H,T,2T-1]
        matrix_bd = self._rel_shift(matrix_bd)[..., :T]
        attn = torch.softmax((matrix_ac + matrix_bd) * self.scaling, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, HID)
        return self.linear_out(out)


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
    """dw_striding: conv2d ×(1 + 2×[dw+pw]) with stride2 pad1, then Linear. ModuleList idx match ckpt."""

    def __init__(self):
        super().__init__()
        self.conv = nn.ModuleList([
            nn.Conv2d(1, SUB_CH, 3, stride=2, padding=1),                       # 0
            nn.ReLU(True),                                                       # 1
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),   # 2 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, 1),                                        # 3 pointwise
            nn.ReLU(True),                                                       # 4
            nn.Conv2d(SUB_CH, SUB_CH, 3, stride=2, padding=1, groups=SUB_CH),   # 5 depthwise
            nn.Conv2d(SUB_CH, SUB_CH, 1),                                        # 6 pointwise
            nn.ReLU(True),                                                       # 7
        ])
        self.out = nn.Linear(SUB_CH * (MEL // 8), HID, bias=True)

    def forward(self, mel):  # mel [B,MEL,L]
        x = mel.transpose(1, 2).unsqueeze(1)                # [B,1,L,MEL]
        for layer in self.conv:
            x = layer(x)
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)  # [B,T,SUB_CH*(MEL//8)]
        return self.out(x)


class AudioAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(HID),           # 0
            nn.Linear(HID, LFM_DIM),     # 1
            nn.GELU(),                   # 2
            nn.Linear(LFM_DIM, LFM_DIM), # 3
        )

    def forward(self, x):
        return self.model(x)


def build_pos_emb(T: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, HID, 2, dtype=torch.float) / HID))
    pos = torch.arange(T - 1, -T, -1, dtype=torch.float)            # [2T-1]
    freqs = (inv_freq[:, None] @ pos[None, :]).transpose(0, 1)      # [2T-1, HID/2]
    pe = torch.stack([freqs.sin(), freqs.cos()], dim=-1).reshape(2 * T - 1, HID)
    return pe[None]                                                 # [1,2T-1,HID]


class EncoderAdapter(nn.Module):
    def __init__(self, mel_len: int):
        super().__init__()
        self.pre_encode = Subsampling()
        self.layers = nn.ModuleList([ConformerBlock() for _ in range(NLAYERS)])
        self.audio_adapter = AudioAdapter()
        T = mel_len
        for _ in range(3):
            T = (T + 2 - 3) // 2 + 1
        self.T = T
        self.register_buffer("pos_emb", build_pos_emb(T), persistent=False)

    def forward(self, mel):  # mel [1,MEL,L] -> [1,T,2048]
        x = self.pre_encode(mel)
        for blk in self.layers:
            x = blk(x, self.pos_emb)
        return self.audio_adapter(x)


def load_weights(m: EncoderAdapter):
    from safetensors import safe_open
    sd = {}
    with safe_open(CKPT, framework="pt") as f:
        for k in f.keys():
            if k.startswith("conformer.pre_encode.") or k.startswith("conformer.layers."):
                sd[k[len("conformer."):]] = f.get_tensor(k)
            elif k.startswith("audio_adapter."):
                sd[k] = f.get_tensor(k)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    miss = [x for x in missing if not x.endswith("num_batches_tracked") and "pos_emb" not in x]
    assert not miss and not unexpected, f"weight map mismatch: missing={miss[:6]} unexpected={unexpected[:6]}"
    print(f"[weights] loaded {len(sd)} tensors (strict map OK)")


# --------------------------------------------------------------------------- gate/export
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--ref", default=str(HERE / "asr_pathref.npz"))
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    d = np.load(args.ref)
    mel_in = torch.from_numpy(d["conf_in"]).float()        # [1,128,L]
    golden = torch.from_numpy(d["adapter_out"]).float()    # [T,2048]
    L = mel_in.shape[2]
    print(f"mel {tuple(mel_in.shape)} L={L} golden adapter_out {tuple(golden.shape)}")

    m = EncoderAdapter(L).eval()
    load_weights(m)
    assert m.T == golden.shape[0], f"T mismatch {m.T} vs {golden.shape[0]}"

    with torch.no_grad():
        out = m(mel_in)[0]                                  # [T,2048]
    cos = torch.nn.functional.cosine_similarity(out.reshape(-1), golden.reshape(-1), dim=0).item()
    pertok = torch.nn.functional.cosine_similarity(out, golden, dim=-1)
    print(f"[eager fp32] global cos {cos:.6f}  per-token mean {pertok.mean():.6f} "
          f"min {pertok.min():.6f}  max|Δ| {(out - golden).abs().max():.4f}")
    if pertok.mean() < 0.999:
        print("❌ re-author DIVERGES — fix before export")
        raise SystemExit(1)
    print("✅ eager re-author matches oracle")
    if args.skip_export:
        return

    import asyncio
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    m_d = m.to(dtype)
    example = {"mel": torch.zeros(1, MEL, L, dtype=dtype)}
    print(f"[export] LFM2-Audio encoder+adapter ({args.dtype}) -> Core AI ...", flush=True)
    prog = export_to_coreai(
        m_d, example, dynamic_shapes=None,
        input_names=("mel",), output_names=("audio_emb",),
        state_names=None, externalize_modules=[])
    prog.optimize()
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    aimodel = art / f"lfm_audio_encoder_adapter_{args.dtype}_L{L}.aimodel"
    shutil.rmtree(aimodel, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "lfm-open-license-v1.0"
    prog.save_asset(aimodel, meta)
    sz = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {aimodel} ({sz:.1f} MB)")

    async def gate(unit):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        mdl = await rt.AIModel.load(str(aimodel), opts)
        fn = mdl.load_function("main")
        res = await fn({"mel": rt.NDArray(mel_in.to(dtype).numpy())})
        eng = torch.from_numpy(res["audio_emb"].numpy().astype(np.float32))[0]
        pt = torch.nn.functional.cosine_similarity(eng, golden, dim=-1)
        ok = pt.mean() > 0.999 and pt.min() > 0.99
        print(f"[gate {unit}] per-token cos mean {pt.mean():.6f} min {pt.min():.6f} "
              f"max|Δ| {(eng - golden).abs().max():.3f} -> {'PASS' if ok else 'FAIL'}")
        return ok

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
