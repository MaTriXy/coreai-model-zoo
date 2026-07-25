#!/usr/bin/env python
"""P1: full 35-layer raw-Metal decode chain (python-dispatched) + parity gates.

The RawLoop dispatches the EXACT kernel sequence the Swift runner will encode
(msl/gemma4_{sdpa,matvec,glue}.metal) on MPS buffers holding the transplant
weights in kernel layout. TorchRef is a faithful fp16 mirror (same op order,
fp16-dequantized weights — the DSL reference precedent) for per-layer triage.

Modes:
  --trace   per-layer cos RawLoop vs TorchRef over the sky prompt prefix
  --gate    greedy token gate: 3 prompts vs oracle_refs.json (fp16 + bf16 ids)

Softcap note: tanh(l/30)*30 is monotonic, so greedy argmax skips it (P5d
fp16-row-argmax precedent); logits comparisons in --trace apply it on host.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from p0_sdpa_parity import (
    EXTRACT, H, HID, LIB as SDPA_LIB, ORACLE_REFS, WINDOW,
    Extract, full_inv_freq, rmsnorm, rmsnorm_scalefree, rope_rotate_half,
    sliding_inv_freq, unpack_int2, unpack_int4,
)

HERE = Path(__file__).resolve().parent
DEV = "mps"
L = 35
FULL = {4, 9, 14, 19, 24, 29, 34}
FIRST_SHARED = 15
PROD_SLIDING, PROD_FULL = 13, 14
MAX_CTX = 4096
PLI_SCALE = 2.0 ** -0.5
MP_SCALE = HID ** -0.5

MATVEC_LIB = torch.mps.compile_shader((HERE / "msl" / "gemma4_matvec.metal").read_text())
GLUE_LIB = torch.mps.compile_shader((HERE / "msl" / "gemma4_glue.metal").read_text())


# ---------- weight prep (extract layout -> kernel layout, on MPS) --------------------------------
def pack_nibbles_i32(q_u: torch.Tensor) -> torch.Tensor:
    """[N, K] uint8 codes 0..15 -> [N, K/8] int32 (8 nibbles/word, low nibble first)."""
    N, K = q_u.shape
    q = q_u.numpy().astype(np.uint32).reshape(N, K // 8, 8)
    qp = np.zeros((N, K // 8), dtype=np.uint32)
    for j in range(8):
        qp |= q[..., j] << (j * 4)
    return torch.from_numpy(qp.view(np.int32))


class Int4W:
    def __init__(self, ex: Extract, key: str) -> None:
        m = ex.manifest[key]
        assert m["bits"] == 4, key
        self.N, self.K = m["shape"]
        packed = ex.w.get_tensor(key)
        scale = ex.w.get_tensor(key + ".scale").float()
        codes = (unpack_int4(packed, self.N, self.K).to(torch.int16) + 8).to(torch.uint8)
        self.qp = pack_nibbles_i32(codes).to(DEV)
        sc = scale.to(torch.float16).reshape(self.N, 1).expand(self.N, self.K // 64)
        self.sc = sc.contiguous().to(DEV)
        self.bi = (sc * -8.0).contiguous().to(DEV)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.N == 1536:   # occupancy: R=1 for the narrow outputs (down / attn o)
            MATVEC_LIB.matvec_int4aff_r1(x, self.qp, self.sc, self.bi, out, self.K,
                                         threads=[32, self.N], group_size=[32, 8])
        else:
            MATVEC_LIB.matvec_int4aff(x, self.qp, self.sc, self.bi, out, self.K,
                                      threads=[32, self.N // 4], group_size=[32, 8])
        return out

    def nx(self, x: torch.Tensor, nw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        MATVEC_LIB.matvec_int4aff_nx(x, nw, self.qp, self.sc, self.bi, out, self.K,
                                     threads=[32, self.N // 4], group_size=[32, 8])
        return out


class Int2W:
    def __init__(self, ex: Extract, key: str) -> None:
        m = ex.manifest[key]
        assert m["bits"] == 2, key
        self.N, self.K = m["shape"]
        packed = ex.w.get_tensor(key)
        self.qp = (packed.reshape(self.N, self.K // 4).contiguous()
                   .view(torch.int32).to(DEV))
        self.sc = ex.w.get_tensor(key + ".scale").float().contiguous().to(DEV)

    def __call__(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        if self.N == 1536:   # occupancy: R=1 for the narrow outputs (down)
            MATVEC_LIB.matvec_int2sym_r1(x, self.qp, self.sc, out, self.K,
                                         threads=[32, self.N], group_size=[32, 8])
        else:
            MATVEC_LIB.matvec_int2sym(x, self.qp, self.sc, out, self.K,
                                      threads=[32, self.N // 4], group_size=[32, 8])
        return out

    def nx(self, x: torch.Tensor, nw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        MATVEC_LIB.matvec_int2sym_nx(x, nw, self.qp, self.sc, out, self.K,
                                     threads=[32, self.N // 4], group_size=[32, 8])
        return out


class Int8W:
    def __init__(self, ex: Extract, key: str, scale_mul: float = 1.0) -> None:
        m = ex.manifest[key]
        assert m["bits"] == 8, key
        self.N, self.K = m["shape"]
        self.w8 = (ex.w.get_tensor(key).view(torch.int8)
                   .reshape(self.N, self.K).contiguous().to(DEV))
        self.sc = (ex.w.get_tensor(key + ".scale").float() * scale_mul).contiguous().to(DEV)

    def __call__(self, x: torch.Tensor, out: torch.Tensor, mode: int = 0,
                 mul: torch.Tensor | None = None, moff: int = 0) -> torch.Tensor:
        MATVEC_LIB.matvec_int8(x, self.w8, self.sc, mul if mul is not None else out,
                               out, self.K, mode, moff,
                               threads=[32, self.N // 4], group_size=[32, 8])
        return out


def _norm16(ex: Extract, key: str) -> torch.Tensor:
    return ex.norm(key).to(torch.float16).contiguous().to(DEV)


class Layer:
    def __init__(self, ex: Extract, li: int) -> None:
        N, Q = f"layer_{li:02d}.", f"decode.layer_{li:02d}."
        self.li = li
        self.full = li in FULL
        self.hd = 512 if self.full else 256
        self.write = li < FIRST_SHARED
        self.cache = li if self.write else (PROD_FULL if self.full else PROD_SLIDING)
        self.wq = Int4W(ex, Q + "attn.q")
        self.wo = Int4W(ex, Q + "attn.o")
        if self.write:
            self.wk = Int4W(ex, Q + "attn.k")
            self.wv = Int4W(ex, Q + "attn.v")
            self.key_norm = _norm16(ex, N + "key_norm")
        bits = ex.manifest[Q + "mlp.gating1"]["bits"]
        self.int2 = bits == 2
        WC = Int2W if self.int2 else Int4W
        self.wgate = WC(ex, Q + "mlp.gating1")
        self.wup = WC(ex, Q + "mlp.gating2")
        self.wdown = WC(ex, Q + "mlp.down")
        self.ple_gate = Int8W(ex, Q + "ple.gate")
        self.ple_proj = Int8W(ex, Q + "ple.proj")
        self.pre_attn = _norm16(ex, N + "pre_attention_norm")
        self.post_attn = _norm16(ex, N + "post_attention_norm")
        self.pre_ffw = _norm16(ex, N + "pre_ffw_norm")
        self.post_ffw = _norm16(ex, N + "post_ffw_norm")
        self.post_ple = _norm16(ex, N + "post_per_layer_input_norm")
        self.query_norm = _norm16(ex, N + "query_norm")
        self.layer_scalar = float(ex.norm(N + "skip_scale").to(torch.float16).item())

    def gateup(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        g, u = self.wgate, self.wup
        if self.int2:
            MATVEC_LIB.gateup_int2sym(x, g.qp, g.sc, u.qp, u.sc, out, g.K,
                                      threads=[32, g.N // 4], group_size=[32, 8])
        else:
            MATVEC_LIB.gateup_int4aff(x, g.qp, g.sc, g.bi, u.qp, u.sc, u.bi, out, g.K,
                                      threads=[32, g.N // 4], group_size=[32, 8])
        return out

    def gateup_nx(self, x: torch.Tensor, nw: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        g, u = self.wgate, self.wup
        if self.int2:
            MATVEC_LIB.gateup_int2sym_nx(x, nw, g.qp, g.sc, u.qp, u.sc, out, g.K,
                                         threads=[32, g.N // 4], group_size=[32, 8])
        else:
            MATVEC_LIB.gateup_int4aff_nx(x, nw, g.qp, g.sc, g.bi, u.qp, u.sc, u.bi,
                                         out, g.K,
                                         threads=[32, g.N // 4], group_size=[32, 8])
        return out


class RawLoop:
    def __init__(self, ex: Extract) -> None:
        t0 = time.time()
        m = ex.manifest["embed.composite"]
        V, _ = m["shape"]
        self.V = V
        self.emb_packed = ex.w.get_tensor("embed.composite").reshape(V, HID // 4).contiguous().to(DEV)
        self.emb_scale = ex.w.get_tensor("embed.composite.scale").float().contiguous().to(DEV)
        tables, scales = [], []
        for i in range(L):
            key = "ple_table.composite" + ("" if i == 0 else str(i))
            tables.append(ex.w.get_tensor(key).reshape(V, 128))
            scales.append(ex.w.get_tensor(key + ".scale").float())
        self.ple_packed = torch.cat(tables, dim=1).contiguous().to(DEV)
        self.ple_scale = torch.stack(scales, dim=1).contiguous().to(DEV)
        self.model_proj = Int8W(ex, "decode.ple.model_proj", scale_mul=MP_SCALE)
        self.proj_norm = _norm16(ex, "ple_projection_norm")
        self.final_norm = torch.from_numpy(
            np.load(ex.root / "final_norm.f32.npy")).to(torch.float16).contiguous().to(DEV)
        self.lm_head = Int2W(ex, "decode.lm_head")
        self.layers = [Layer(ex, li) for li in range(L)]
        self.invf_sliding = sliding_inv_freq(256).contiguous().to(DEV)
        self.invf_full = full_inv_freq(512).contiguous().to(DEV)
        # KV caches: one per non-shared layer, linear [MAX_CTX, hd].
        self.kcache, self.vcache = {}, {}
        for li in range(FIRST_SHARED):
            hd = 512 if li in FULL else 256
            self.kcache[li] = torch.zeros(MAX_CTX, hd, dtype=torch.float16, device=DEV)
            self.vcache[li] = torch.zeros(MAX_CTX, hd, dtype=torch.float16, device=DEV)
        # scratch
        f16 = lambda n: torch.empty(n, dtype=torch.float16, device=DEV)
        self.x = f16(HID)
        self.xn = f16(HID)
        self.pli = f16(L * 256).reshape(L, 256)
        self.ple_tok = f16(L * 256).reshape(L, 256)
        self.p8 = f16(L * 256)
        self.q = {256: f16(8 * 256), 512: f16(8 * 512)}
        self.qn = {256: f16(8 * 256), 512: f16(8 * 512)}
        self.kv = {256: f16(256), 512: f16(512)}
        self.kvn = {256: f16(256), 512: f16(512)}
        self.ctx = {256: f16(8 * 256), 512: f16(8 * 512)}
        self.attn = f16(HID)
        self.h_ffn = f16(12288)
        self.ffn = f16(HID)
        self.g256 = f16(256)
        self.p1536 = f16(HID)
        self.hidden = f16(HID)
        self.logits = f16(self.V)
        # GPU-resident token stream: prompt ids prefilled by host, generated ids
        # written by argmax_stage2 — no CPU sync inside the decode loop.
        self.toks = torch.zeros(MAX_CTX + 8, dtype=torch.int32, device=DEV)
        self.nparts = self.V // 512
        self.partv = torch.empty(self.nparts, dtype=torch.float32, device=DEV)
        self.parti = torch.empty(self.nparts, dtype=torch.int32, device=DEV)
        print(f"RawLoop weights loaded in {time.time() - t0:.1f}s")

    # per-layer trace hook (set in --trace mode)
    tap = None

    def head_argmax(self, pos: int) -> None:
        """final-norm-fused lm_head -> greedy argmax -> TOKS[pos+1], all on GPU."""
        self.lm_head.nx(self.x, self.final_norm, self.logits)
        GLUE_LIB.argmax_stage1(self.logits, self.partv, self.parti, self.V,
                               threads=[self.nparts * 256], group_size=[256])
        GLUE_LIB.argmax_stage2(self.partv, self.parti, self.toks, self.nparts, pos,
                               threads=[256], group_size=[256])

    def step(self, pos: int, want_logits: bool) -> torch.Tensor | None:
        GLUE_LIB.embed_gather(self.emb_packed, self.emb_scale, self.x, self.toks, pos,
                              threads=[384], group_size=[128])
        self.model_proj(self.x, self.p8)
        GLUE_LIB.ple_gather(self.ple_packed, self.ple_scale, self.ple_tok, self.toks, pos,
                            threads=[4480], group_size=[256])
        GLUE_LIB.rmsnorm_glue(self.p8, self.proj_norm, self.ple_tok, self.pli,
                              256, 1, PLI_SCALE, 0, threads=[32, L], group_size=[32, 8])
        if self.tap:
            self.tap("front", self.x, self.pli)

        for lay in self.layers:
            hd = lay.hd
            invf = self.invf_full if lay.full else self.invf_sliding
            kc, vc = self.kcache[lay.cache], self.vcache[lay.cache]
            # attention (pre_attn norm fused into the q/k/v matvec prologues)
            lay.wq.nx(self.x, lay.pre_attn, self.q[hd])
            GLUE_LIB.qknorm_rope(self.q[hd], lay.query_norm, invf, self.qn[hd],
                                 hd, pos, 0, threads=[32, 8], group_size=[32, 8])
            if lay.write:
                lay.wk.nx(self.x, lay.pre_attn, self.kv[hd])
                GLUE_LIB.qknorm_rope(self.kv[hd], lay.key_norm, invf, kc,
                                     hd, pos, pos, threads=[32, 1], group_size=[32, 1])
                lay.wv.nx(self.x, lay.pre_attn, self.kv[hd])
                GLUE_LIB.rmsnorm_glue(self.kv[hd], lay.key_norm, self.kv[hd], vc,
                                      hd, 2, 1.0, pos, threads=[32, 1], group_size=[32, 1])
            j0 = 0 if lay.full else max(0, pos + 1 - WINDOW)
            occ_g = 8 if lay.full else 16     # G*hd <= 4096 (16 KB tg partials)
            SDPA_LIB.flash_sdpa_decode_occ(self.qn[hd], kc, vc, self.ctx[hd], hd, j0,
                                           pos + 1 - j0, occ_g,
                                           threads=[32, occ_g, H],
                                           group_size=[32, occ_g, 1])
            lay.wo(self.ctx[hd], self.attn)
            GLUE_LIB.rmsnorm_glue(self.attn, lay.post_attn, self.x, self.x,
                                  HID, 1, 1.0, 0, threads=[32, 1], group_size=[32, 1])
            # FFN (pre_ffw norm fused into the gateup prologue)
            lay.gateup_nx(self.x, lay.pre_ffw, self.h_ffn)
            lay.wdown(self.h_ffn, self.ffn)
            GLUE_LIB.rmsnorm_glue(self.ffn, lay.post_ffw, self.x, self.x,
                                  HID, 1, 1.0, 0, threads=[32, 1], group_size=[32, 1])
            # PLE injection
            lay.ple_gate(self.x, self.g256, mode=1, mul=self.pli, moff=lay.li * 256)
            lay.ple_proj(self.g256, self.p1536)
            GLUE_LIB.rmsnorm_glue(self.p1536, lay.post_ple, self.x, self.x,
                                  HID, 1, lay.layer_scalar, 0,
                                  threads=[32, 1], group_size=[32, 1])
            if self.tap:
                self.tap(f"L{lay.li}", self.x)

        if not want_logits:
            return None
        GLUE_LIB.rmsnorm_glue(self.x, self.final_norm, self.x, self.hidden,
                              HID, 0, 1.0, 0, threads=[32, 1], group_size=[32, 1])
        self.lm_head(self.hidden, self.logits)
        return self.logits

    def generate(self, prompt_ids: list[int], n_gen: int) -> list[int]:
        """GPU-resident greedy generation (the Swift runner's exact schedule)."""
        for li in self.kcache:
            self.kcache[li].zero_()
            self.vcache[li].zero_()
        self.toks[:len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.int32)
        n_steps = len(prompt_ids) + n_gen - 1
        for pos in range(n_steps):
            self.step(pos, want_logits=False)
            if pos >= len(prompt_ids) - 1:
                self.head_argmax(pos)
        torch.mps.synchronize()
        return self.toks[len(prompt_ids):len(prompt_ids) + n_gen].cpu().tolist()


# ---------- faithful torch mirror (fp16 weights, kernel op order) --------------------------------
class RefLayer:
    def __init__(self, ex: Extract, li: int) -> None:
        N, Q = f"layer_{li:02d}.", f"decode.layer_{li:02d}."
        self.li = li
        self.full = li in FULL
        self.hd = 512 if self.full else 256
        self.write = li < FIRST_SHARED
        self.cache = li if self.write else (PROD_FULL if self.full else PROD_SLIDING)
        dq = lambda k: dequant16(ex, k)
        self.wq, self.wo = dq(Q + "attn.q"), dq(Q + "attn.o")
        if self.write:
            self.wk, self.wv = dq(Q + "attn.k"), dq(Q + "attn.v")
            self.key_norm = ex.norm(N + "key_norm").to(torch.float16)
        self.wgate, self.wup = dq(Q + "mlp.gating1"), dq(Q + "mlp.gating2")
        self.wdown = dq(Q + "mlp.down")
        self.ple_gate, self.ple_proj = dq(Q + "ple.gate"), dq(Q + "ple.proj")
        n16 = lambda k: ex.norm(k).to(torch.float16)
        self.pre_attn, self.post_attn = n16(N + "pre_attention_norm"), n16(N + "post_attention_norm")
        self.pre_ffw, self.post_ffw = n16(N + "pre_ffw_norm"), n16(N + "post_ffw_norm")
        self.post_ple = n16(N + "post_per_layer_input_norm")
        self.query_norm = n16(N + "query_norm")
        self.layer_scalar = ex.norm(N + "skip_scale").to(torch.float16)


def dequant16(ex: Extract, key: str) -> torch.Tensor:
    m = ex.manifest[key]
    rows, cols = m["shape"]
    packed = ex.w.get_tensor(key)
    scale = ex.w.get_tensor(key + ".scale").float()
    if m["bits"] == 8:
        c = packed.view(torch.int8).reshape(rows, cols).float()
    elif m["bits"] == 4:
        c = unpack_int4(packed, rows, cols).float()
    else:
        c = unpack_int2(packed, rows, cols).float()
    return (c * scale.unsqueeze(1)).to(torch.float16)


def mv(x: torch.Tensor, w16: torch.Tensor) -> torch.Tensor:
    """fp32-accumulate matvec on fp16-dequant weights, fp16 output (kernel class)."""
    return (x.float() @ w16.float().T).to(torch.float16)


class TorchRef:
    def __init__(self, ex: Extract) -> None:
        t0 = time.time()
        m = ex.manifest["embed.composite"]
        V, _ = m["shape"]
        self.V = V
        self.emb_packed = ex.w.get_tensor("embed.composite").reshape(V, HID // 4)
        self.emb_scale = ex.w.get_tensor("embed.composite.scale").float()
        tables, scales = [], []
        for i in range(L):
            key = "ple_table.composite" + ("" if i == 0 else str(i))
            tables.append(ex.w.get_tensor(key).reshape(V, 128))
            scales.append(ex.w.get_tensor(key + ".scale").float())
        self.ple_packed = torch.cat(tables, dim=1)
        self.ple_scale = torch.stack(scales, dim=1)
        self.model_proj = dequant16(ex, "decode.ple.model_proj")
        self.proj_norm = ex.norm("ple_projection_norm").to(torch.float16)
        self.final_norm = torch.from_numpy(
            np.load(ex.root / "final_norm.f32.npy")).to(torch.float16)
        self.lm_head = dequant16(ex, "decode.lm_head")
        self.layers = [RefLayer(ex, li) for li in range(L)]
        self.invf = {256: sliding_inv_freq(256), 512: full_inv_freq(512)}
        self.kcache = {li: [] for li in range(FIRST_SHARED)}
        self.vcache = {li: [] for li in range(FIRST_SHARED)}
        print(f"TorchRef weights loaded in {time.time() - t0:.1f}s")

    def step(self, tok: int, pos: int, want_logits: bool):
        taps = {}
        ids = torch.tensor([tok])
        x = (unpack_int2(self.emb_packed[ids], 1, HID).float()
             * self.emb_scale[ids].unsqueeze(1) * (HID ** 0.5)).to(torch.float16)[0]
        row = self.ple_packed[tok]
        c = torch.stack([row & 0xF, row >> 4], dim=-1).reshape(-1).to(torch.int16)
        c = torch.where(c >= 8, c - 16, c).reshape(L, 256).float()
        sc = (self.ple_scale[tok] * 16.0).to(torch.float16)
        ple_tok = (c.to(torch.float16) * sc.unsqueeze(1))
        p8 = mv(x, self.model_proj) * torch.tensor(MP_SCALE, dtype=torch.float16)
        proj = rmsnorm(p8.reshape(L, 256), self.proj_norm)
        pli = ((proj + ple_tok) * torch.tensor(PLI_SCALE, dtype=torch.float16))
        taps["front"] = (x.clone(), pli.clone())

        for lay in self.layers:
            hd = lay.hd
            xn = rmsnorm(x, lay.pre_attn)
            q = rmsnorm(mv(xn, lay.wq).reshape(8, hd), lay.query_norm)
            q = rope_rotate_half(q.unsqueeze(0), torch.tensor([pos]),
                                 self.invf[hd])[0].to(torch.float16)
            if lay.write:
                k = rmsnorm(mv(xn, lay.wk).reshape(1, hd), lay.key_norm)
                k = rope_rotate_half(k.unsqueeze(0), torch.tensor([pos]),
                                     self.invf[hd])[0].to(torch.float16)
                v = rmsnorm_scalefree(mv(xn, lay.wv).reshape(1, hd))
                self.kcache[lay.cache].append(k[0])
                self.vcache[lay.cache].append(v[0])
            kc = torch.stack(self.kcache[lay.cache])
            vc = torch.stack(self.vcache[lay.cache])
            j0 = 0 if lay.full else max(0, pos + 1 - WINDOW)
            scores = q.float() @ kc[j0:].float().T
            w = torch.softmax(scores, dim=-1)
            ctx = (w @ vc[j0:].float()).to(torch.float16)
            ao = mv(ctx.reshape(-1), lay.wo)
            x = x + rmsnorm(ao, lay.post_attn)
            xn = rmsnorm(x, lay.pre_ffw)
            tg = (xn.float() @ lay.wgate.float().T)
            tu = (xn.float() @ lay.wup.float().T)
            h = (torch.nn.functional.gelu(tg, approximate="tanh") * tu).to(torch.float16)
            x = x + rmsnorm(mv(h, lay.wdown), lay.post_ffw)
            g = mv(x, lay.ple_gate)
            g = torch.nn.functional.gelu(g.float(), approximate="tanh").to(torch.float16)
            g = g * pli[lay.li]
            p = mv(g, lay.ple_proj)
            x = (x + rmsnorm(p, lay.post_ple)) * lay.layer_scalar
            taps[f"L{lay.li}"] = x.clone()

        if not want_logits:
            return None, taps
        hidden = rmsnorm(x, self.final_norm)
        return mv(hidden, self.lm_head), taps


# ---------- modes --------------------------------------------------------------------------------
def run_trace(nsteps: int) -> None:
    ex = Extract(EXTRACT)
    refs = json.loads(ORACLE_REFS.read_text())
    ids = refs["prompts"]["Why is the sky blue?"]["prompt_ids"][:nsteps]
    raw, ref = RawLoop(ex), TorchRef(ex)
    worst: dict[str, float] = {}
    for pos, tok in enumerate(ids):
        taps_raw: dict[str, torch.Tensor] = {}
        raw.tap = lambda name, *bufs: taps_raw.__setitem__(
            name, tuple(b.cpu().clone() for b in bufs))
        raw.toks[pos] = tok
        lr = raw.step(pos, want_logits=True)
        raw.head_argmax(pos)                      # exercises the GPU argmax path too
        torch.mps.synchronize()
        gpu_am = int(raw.toks[pos + 1].item())
        lt, taps_ref = ref.step(tok, pos, want_logits=True)
        fx, fpli = taps_raw["front"]
        rx, rpli = taps_ref["front"]
        for name, a, b in (("front.x", fx, rx), ("front.pli", fpli.reshape(-1),
                                                 rpli.reshape(-1))):
            cosv = torch.nn.functional.cosine_similarity(
                a.float().reshape(1, -1), b.float().reshape(1, -1)).item()
            worst[name] = min(worst.get(name, 1.0), cosv)
        for li in range(L):
            a = taps_raw[f"L{li}"][0].float()
            b = taps_ref[f"L{li}"].float()
            cosv = torch.nn.functional.cosine_similarity(
                a.reshape(1, -1), b.reshape(1, -1)).item()
            worst[f"L{li:02d}"] = min(worst.get(f"L{li:02d}", 1.0), cosv)
        am_r = int(lr.cpu().float().argmax())
        am_t = int(lt.float().argmax())
        cosl = torch.nn.functional.cosine_similarity(
            lr.cpu().float().reshape(1, -1), lt.float().reshape(1, -1)).item()
        gm = "gpu==" if gpu_am == am_r else f"gpu!={gpu_am}"
        print(f"pos {pos:2d} tok {tok:6d}: logits cos={cosl:.6f} "
              f"argmax raw={am_r} ref={am_t} {'==' if am_r == am_t else '!!'} {gm}")
        raw.toks[pos + 1] = 0                     # trace feeds prompt ids, not argmax
    print("\nworst per-tap cos over positions:")
    for name in sorted(worst):
        flag = "" if worst[name] > 0.999 else "   <-- LOW"
        print(f"  {name:10s} {worst[name]:.6f}{flag}")


def run_gate() -> bool:
    ex = Extract(EXTRACT)
    refs = json.loads(ORACLE_REFS.read_text())
    raw = RawLoop(ex)
    all_ok = True
    for prompt, d in refs["prompts"].items():
        ids = list(d["prompt_ids"])
        n_gen = len(d["fp16"])
        t0 = time.time()
        out = raw.generate(ids, n_gen)
        dt = time.time() - t0
        m16 = sum(1 for a, b in zip(out, d["fp16"]) if a == b)
        exact16 = out == d["fp16"]
        exactbf = out == d["bf16"]
        fork = next((i for i, (a, b) in enumerate(zip(out, d["fp16"])) if a != b), None)
        status = "EXACT==fp16" if exact16 else ("EXACT==bf16" if exactbf
                 else f"fork@{fork} ({m16}/{n_gen} match fp16)")
        ok = exact16 or exactbf
        all_ok &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {prompt!r}: {status}  "
              f"[{(len(ids) + n_gen) / dt:.1f} steps/s]")
        if not ok:
            print(f"    raw : {out}")
            print(f"    fp16: {d['fp16']}")
            print(f"    bf16: {d['bf16']}")
    print(f"\nP1 TOKEN GATE: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--nsteps", type=int, default=6)
    args = ap.parse_args()
    if args.trace:
        run_trace(args.nsteps)
    if args.gate:
        ok = run_gate()
        raise SystemExit(0 if ok else 1)
