#!/usr/bin/env python
"""P0 parity gate: raw-MSL flash-decode SDPA (sliding + full) vs torch fp32 reference.

Part A — random tensors: hd 256 (sliding, window 512) and hd 512 (full/global),
positions spanning pre-window, window edge, and deep-context; a mild and a
harsh-magnitude variant (softmax stress).
Part B — real extract tensors: layer 0 (sliding) and layer 4 (full) q/k/v built
from the REAL mixed-bit weights + REAL prompt embeddings (sky prompt ids from
oracle_refs.json), fp32 host math faithful to the Core AI receiving end
(RMSNormImpl fp32 -> fp16*w, rotate-half RoPE with fp16 cos/sin, v scale-free).

Gate: cos >= 0.9999 and rel_l2 <= 3e-3 per case (fp16 I/O class; the dense
kernel precedent measured 3-7e-4 on real tensors).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import code_path, exports_dir  # noqa: E402
EXTRACT = Path(os.environ.get(
    "GEMMA4_EXTRACT", str(code_path("litertlm-convert", "out", "gemma4e2b_extract"))))
ORACLE_REFS = exports_dir() / "gemma4_e2b_mixedbit_decode_ffnfused" / "oracle_refs.json"

H = 8            # query heads
WINDOW = 512
HID = 1536
EPS = 1e-6

DEV = "mps"
LIB = torch.mps.compile_shader((HERE / "msl" / "gemma4_sdpa.metal").read_text())


# ---------- kernel + reference ------------------------------------------------------------------
def kernel_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: int,
                window: int | None, occ_g: int = 0) -> torch.Tensor:
    """q [H,hd] fp16 mps; k/v [Smax,hd] fp16 mps (rows 0..pos valid). -> ctx [H,hd] fp16.

    occ_g > 0 dispatches the G-way sequence-split occupancy kernel instead."""
    hd = q.shape[1]
    j0 = max(0, pos + 1 - window) if window else 0
    n = pos + 1 - j0
    ctx = torch.empty_like(q)
    if occ_g:
        LIB.flash_sdpa_decode_occ(q, k, v, ctx, hd, j0, n, occ_g,
                                  threads=[32, occ_g, H], group_size=[32, occ_g, 1])
    else:
        LIB.flash_sdpa_decode(q, k, v, ctx, hd, j0, n,
                              threads=[32, H], group_size=[32, H])
    torch.mps.synchronize()
    return ctx


def ref_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: int,
             window: int | None) -> torch.Tensor:
    """fp32 reference: scale-1.0 softmax over the (windowed) causal key range."""
    j0 = max(0, pos + 1 - window) if window else 0
    kk = k[j0:pos + 1].float()          # [n, hd]
    vv = v[j0:pos + 1].float()
    scores = q.float() @ kk.T           # [H, n], scale 1.0
    w = torch.softmax(scores, dim=-1)
    return (w @ vv)                     # [H, hd] fp32


def gate(name: str, q, k, v, pos: int, window: int | None, occ_g: int = 0) -> bool:
    ctx = kernel_sdpa(q.to(DEV), k.to(DEV), v.to(DEV), pos, window, occ_g).cpu().float()
    ref = ref_sdpa(q, k, v, pos, window)
    rel = (ctx - ref).norm() / ref.norm()
    cos = torch.nn.functional.cosine_similarity(
        ctx.reshape(1, -1), ref.reshape(1, -1)).item()
    am_k = ctx.reshape(-1).argmax().item()
    am_r = ref.reshape(-1).argmax().item()
    ok = cos >= 0.9999 and rel <= 3e-3 and am_k == am_r
    print(f"  {'PASS' if ok else 'FAIL'}  {name:44s} cos={cos:.6f} rel_l2={rel:.2e} "
          f"argmax {'==' if am_k == am_r else f'{am_k}!={am_r}'}")
    return ok


# ---------- Part A: random ----------------------------------------------------------------------
def part_a() -> bool:
    print("Part A — random tensors")
    ok = True
    torch.manual_seed(0)
    for hd, window, tag in ((256, WINDOW, "sliding"), (512, None, "full")):
        for pos in (0, 5, 100, 511, 512, 1000, 2047, 4095):
            smax = pos + 1
            for mag, mtag in ((1.0, "mild"), (6.0, "harsh")):
                q = (torch.randn(H, hd) * mag).to(torch.float16)
                k = (torch.randn(smax, hd) * mag).to(torch.float16)
                v = torch.randn(smax, hd).to(torch.float16)
                win = window if tag == "sliding" else None
                occ_g = 16 if hd == 256 else 8
                ok &= gate(f"{tag} hd{hd} pos={pos} {mtag}", q, k, v, pos, win)
                ok &= gate(f"{tag} hd{hd} pos={pos} {mtag} occ{occ_g}", q, k, v,
                           pos, win, occ_g)
    return ok


# ---------- Part B: real extract ----------------------------------------------------------------
def unpack_int2(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    p = packed_u8.reshape(rows, cols // 4).to(torch.int16)
    c = torch.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], dim=-1).reshape(rows, cols)
    return torch.where(c >= 2, c - 4, c).to(torch.int8)


def unpack_int4(packed_u8: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    p = packed_u8.reshape(rows, cols // 2).to(torch.int16)
    c = torch.stack([p & 0xF, p >> 4], dim=-1).reshape(rows, cols)
    return torch.where(c >= 8, c - 16, c).to(torch.int8)


class Extract:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.w = safe_open(str(root / "gemma4e2b_mixedbit_weights.safetensors"), framework="pt")
        self.n = safe_open(str(root / "gemma4e2b_fp32_norms.safetensors"), framework="pt")
        self.manifest = json.loads((root / "gemma4e2b_mixedbit_manifest.json").read_text())

    def dequant(self, key: str) -> torch.Tensor:
        m = self.manifest[key]
        rows, cols = m["shape"]
        packed = self.w.get_tensor(key)
        scale = self.w.get_tensor(key + ".scale").float()
        if m["bits"] == 8:
            c = packed.view(torch.int8).reshape(rows, cols).float()
        elif m["bits"] == 4:
            c = unpack_int4(packed, rows, cols).float()
        elif m["bits"] == 2:
            c = unpack_int2(packed, rows, cols).float()
        else:
            raise ValueError((key, m["bits"]))
        return c * scale.unsqueeze(1)

    def norm(self, key: str) -> torch.Tensor:
        return self.n.get_tensor(key).float()


def rmsnorm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Faithful RMSNormImpl: fp32 ms + rsqrt, fp16 downcast, then * fp16 weight."""
    xf = x.float()
    inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + EPS)
    return (x * inv).to(torch.float16) * w.to(torch.float16)


def rmsnorm_scalefree(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt((xf * xf).mean(-1, keepdim=True) + EPS)
    return (x * inv).to(torch.float16)


def rope_rotate_half(x: torch.Tensor, pos_ids: torch.Tensor,
                     inv_freq: torch.Tensor) -> torch.Tensor:
    """x [n, heads, hd] fp16, pos_ids [n] — composite RoPE (fp32 angle, fp16 cos/sin)."""
    angle = pos_ids.float().reshape(-1, 1) * inv_freq.reshape(1, -1)   # [n, hd/2]
    cos = angle.cos().to(x.dtype).unsqueeze(1)                          # [n, 1, hd/2]
    sin = angle.sin().to(x.dtype).unsqueeze(1)
    hd = x.shape[-1]
    x1, x2 = x[..., :hd // 2], x[..., hd // 2:]
    return torch.cat([cos * x1 - sin * x2, sin * x1 + cos * x2], dim=-1)


def sliding_inv_freq(hd: int, theta: float = 1e4) -> torch.Tensor:
    return 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))


def full_inv_freq(hd: int, theta: float = 1e6, partial: float = 0.25) -> torch.Tensor:
    rope_angles = int(partial * hd // 2)
    rotated = 1.0 / (theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / hd))
    return torch.cat([rotated, torch.zeros(hd // 2 - rope_angles)])


def part_b() -> bool:
    print("Part B — real extract tensors (sky prompt, layers 0 and 4)")
    ex = Extract(EXTRACT)
    refs = json.loads(ORACLE_REFS.read_text())
    sky = refs["prompts"]["Why is the sky blue?"]
    ids = torch.tensor(sky["prompt_ids"] + sky["bf16"], dtype=torch.long)  # ~47 real positions
    npos = ids.numel()

    # real embeddings: int2 row gather * scale * sqrt(hidden)
    m = ex.manifest["embed.composite"]
    rows, cols = m["shape"]
    packed = ex.w.get_tensor("embed.composite").reshape(rows, cols // 4)
    scale = ex.w.get_tensor("embed.composite.scale").float()
    emb = (unpack_int2(packed[ids], npos, cols).float()
           * scale[ids].unsqueeze(1) * (HID ** 0.5)).to(torch.float16)     # [n, 1536]

    ok = True
    for li, tag in ((0, "sliding"), (4, "full")):
        N, Q = f"layer_{li:02d}.", f"decode.layer_{li:02d}."
        hd = 256 if tag == "sliding" else 512
        x = rmsnorm(emb, ex.norm(N + "pre_attention_norm"))                # input_layernorm
        wq = ex.dequant(Q + "attn.q").to(torch.float16)                    # [H*hd, 1536]
        wk = ex.dequant(Q + "attn.k").to(torch.float16)                    # [hd, 1536]
        wv = ex.dequant(Q + "attn.v").to(torch.float16)
        q = (x @ wq.T).reshape(npos, H, hd)
        k = (x @ wk.T).reshape(npos, 1, hd)
        v = (x @ wv.T).reshape(npos, 1, hd)
        q = rmsnorm(q, ex.norm(N + "query_norm"))
        k = rmsnorm(k, ex.norm(N + "key_norm"))
        v = rmsnorm_scalefree(v)
        inv_freq = sliding_inv_freq(hd) if tag == "sliding" else full_inv_freq(hd)
        pos_ids = torch.arange(npos)
        q = rope_rotate_half(q, pos_ids, inv_freq)
        k = rope_rotate_half(k, pos_ids, inv_freq)
        kcache = k.reshape(npos, hd).contiguous()
        vcache = v.reshape(npos, hd).contiguous()
        window = WINDOW if tag == "sliding" else None
        for pos in (0, 3, npos // 2, npos - 1):
            ok &= gate(f"L{li} {tag} real pos={pos}", q[pos].contiguous(),
                       kcache, vcache, pos, window)
    return ok


if __name__ == "__main__":
    a = part_a()
    b = part_b()
    print(f"\nP0 GATE: {'PASS' if a and b else 'FAIL'}")
    sys.exit(0 if a and b else 1)
