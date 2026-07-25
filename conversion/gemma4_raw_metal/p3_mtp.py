#!/usr/bin/env python
"""P3b/P3c: MTP drafter on the raw loop + the chained round (draft x3 -> verify).

Drafter = Section-11 transplant (4 tiny layers, cross-attending the main L13/L14
caches, 18 static int8 act-quant points). Drafts chain ON GPU (argmax -> TOKS ->
next embed_gather; post_proj writes the next chained hidden directly). Verify is
the P3a bit-exact S=4 forward, so the emitted stream provably equals the S=1 loop.

Gates:
  --parity   kernel drafter vs the torch fp32 module (gemma4_mtp_drafter) on live
             state at several anchors: argmax agreement (alpha-preserving standard)
  --gate     MTP generation == S=1 generation byte-for-byte, 3 prompts, GEN=64
             (+ tokens/round per prompt)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import p1_chain as p1
from p1_chain import EXTRACT, GLUE_LIB, HID, Extract, Int4W, Int8W, _norm16
from p3_verify import RawLoopV, VLIB, S

HERE = Path(__file__).resolve().parent
DEV = "mps"
DLIB = torch.mps.compile_shader((HERE / "msl" / "gemma4_drafter.metal").read_text())

DDIM = 256
DHEADS = 4
G_DRAFTS = 3
SLIDING_CACHE, FULL_CACHE = 13, 14   # main-layer caches the drafter cross-attends


def _aq_scales(root: Path) -> dict[str, float]:
    aq = json.loads((root / "gemma4e2b_drafter_act_quant.json").read_text())

    def one(substr: str) -> float:
        hits = [v["scale"] for k, v in aq.items() if substr in k]
        assert len(hits) == 1, (substr, hits)
        return float(hits[0])

    out = {"pre_proj": one("mtp_pre_proj"), "post_proj": one("mtp_post_proj")}
    for i in range(4):
        out[f"l{i}.q"] = one(f"layer_{i}/layer_{i}.pre_q/attn.pre_q")
        out[f"l{i}.attn_vec"] = one(f"layer_{i}/layer_{i}.post_qkv/attn.post_qkv")
        out[f"l{i}.gating"] = one(f"layer_{i}/layer_{i}.post_qkv/mlp/gating_einsum1")
        out[f"l{i}.down"] = one(f"layer_{i}/layer_{i}.post_qkv/mlp/linear")
    return out


class DrafterLayerW:
    def __init__(self, ex: Extract, fx, i: int, aq: dict[str, float]) -> None:
        P = f"drafter.layer_{i:02d}."
        X = f"layer_{i}."
        self.hd = 512 if i == 3 else 256
        self.cache = FULL_CACHE if i == 3 else SLIDING_CACHE
        self.wq = Int8W(ex, P + "attn.q")
        self.wo = Int8W(ex, P + "attn.o")
        self.wgate = Int4W(ex, P + "mlp.gating1")
        self.wup = Int4W(ex, P + "mlp.gating2")
        self.wdown = Int4W(ex, P + "mlp.down")
        n16 = lambda k: fx.get_tensor(X + k).float().to(torch.float16).contiguous().to(DEV)
        self.pre_attn = n16("pre_attention_norm")
        self.post_attn = n16("post_attention_norm")
        self.pre_ffw = n16("pre_ffw_norm")
        self.post_ffw = n16("post_ffw_norm")
        self.q_norm = n16("q_norm")
        self.aq_q = aq[f"l{i}.q"]
        self.aq_attn_vec = aq[f"l{i}.attn_vec"]
        self.aq_gating = aq[f"l{i}.gating"]
        self.aq_down = aq[f"l{i}.down"]


class RawLoopMTP(RawLoopV):
    def __init__(self, ex: Extract) -> None:
        super().__init__(ex)
        from safetensors import safe_open
        fx = safe_open(str(ex.root / "gemma4e2b_drafter_extras.safetensors"),
                       framework="pt")
        sc = json.loads((ex.root / "gemma4e2b_drafter_extras.json").read_text())
        aq = _aq_scales(ex.root)
        self.d_pre_proj = Int8W(ex, "drafter.dot_general")
        pp_codes = fx.get_tensor("post_proj.weight_int8").flatten().view(torch.uint8)
        self.d_post_proj_w8 = pp_codes.view(torch.int8).reshape(HID, DDIM).contiguous().to(DEV)
        self.d_post_proj_sc = fx.get_tensor("post_proj.scale").float().contiguous().to(DEV)
        self.d_head = Int4W(ex, "drafter.lm_head")
        self.d_final_norm = fx.get_tensor("final_norm").float().to(torch.float16).contiguous().to(DEV)
        self.d_invf = {256: fx.get_tensor("rope_freq_128").float().contiguous().to(DEV),
                       512: fx.get_tensor("rope_freq_256").float().contiguous().to(DEV)}
        self.d_layers = [DrafterLayerW(ex, fx, i, aq) for i in range(4)]
        self.d_skip = [float(sc[f"layer_{i}.skip_scale"]) for i in range(4)]
        self.aq_pre_proj = aq["pre_proj"]
        self.aq_post_proj = aq["post_proj"]
        f16 = lambda n: torch.empty(n, dtype=torch.float16, device=DEV)
        self.act = f16(2 * HID)          # concat(emb, hidden) — pre_proj input
        self.xd = f16(DDIM)
        self.hnorm = f16(DDIM)
        self.qd = f16(DHEADS * 512)
        self.qdn = f16(DHEADS * 512)
        self.ctxd = f16(DHEADS * 512)
        self.od = f16(DDIM)
        self.hffn_d = f16(2048)
        self.ffnd = f16(DDIM)
        self.fin = f16(DDIM)
        self.dlogits = f16(self.V)
        self.dhidden = f16(HID)          # h-tilde[a-1] — the drafter kickoff hidden

    # ---- one draft forward: reads act (emb ‖ hidden), writes TOKS[pos_tok+1] + chained hidden
    def draft_forward(self, qpos: int, tok_pos: int) -> None:
        pp = self.d_pre_proj
        DLIB.matvec_int8_aq(self.act, pp.w8, pp.sc, self.xd, pp.K, self.aq_pre_proj,
                            threads=[32, DDIM // 4], group_size=[32, 8])
        for i, dl in enumerate(self.d_layers):
            hd = dl.hd
            kc, vc = self.kcache[dl.cache], self.vcache[dl.cache]
            occ_g = 8 if hd == 512 else 16
            win = 0 if hd == 512 else p1.WINDOW
            j0 = max(0, qpos + 1 - win) if win else 0
            n = qpos + 1 - j0
            GLUE_LIB.rmsnorm_glue(self.xd, dl.pre_attn, self.xd, self.hnorm,
                                  DDIM, 0, 1.0, 0, threads=[32, 1], group_size=[32, 1])
            DLIB.matvec_int8_aq(self.hnorm, dl.wq.w8, dl.wq.sc, self.qd, DDIM, dl.aq_q,
                                threads=[32, DHEADS * hd // 4], group_size=[32, 8])
            GLUE_LIB.qknorm_rope(self.qd, dl.q_norm, self.d_invf[hd], self.qdn,
                                 hd, qpos, 0, threads=[32, DHEADS], group_size=[32, DHEADS])
            p1.SDPA_LIB.flash_sdpa_decode_occ(self.qdn, kc, vc, self.ctxd, hd, j0, n,
                                              occ_g, threads=[32, occ_g, DHEADS],
                                              group_size=[32, occ_g, 1])
            DLIB.matvec_int8_aq(self.ctxd, dl.wo.w8, dl.wo.sc, self.od, DHEADS * hd,
                                dl.aq_attn_vec, threads=[32, DDIM // 4], group_size=[32, 8])
            GLUE_LIB.rmsnorm_glue(self.od, dl.post_attn, self.xd, self.xd,
                                  DDIM, 1, 1.0, 0, threads=[32, 1], group_size=[32, 1])
            GLUE_LIB.rmsnorm_glue(self.xd, dl.pre_ffw, self.xd, self.hnorm,
                                  DDIM, 0, 1.0, 0, threads=[32, 1], group_size=[32, 1])
            g, u, d = dl.wgate, dl.wup, dl.wdown
            DLIB.gateup_int4aff_aq(self.hnorm, g.qp, g.sc, g.bi, u.qp, u.sc, u.bi,
                                   self.hffn_d, DDIM, dl.aq_gating, dl.aq_down,
                                   threads=[32, 2048 // 4], group_size=[32, 8])
            p1.MATVEC_LIB.matvec_int4aff(self.hffn_d, d.qp, d.sc, d.bi, self.ffnd, 2048,
                                         threads=[32, DDIM // 4], group_size=[32, 8])
            GLUE_LIB.rmsnorm_glue(self.ffnd, dl.post_ffw, self.xd, self.xd,
                                  DDIM, 1, self.d_skip[i], 0,
                                  threads=[32, 1], group_size=[32, 1])
        GLUE_LIB.rmsnorm_glue(self.xd, self.d_final_norm, self.xd, self.fin,
                              DDIM, 0, 1.0, 0, threads=[32, 1], group_size=[32, 1])
        h = self.d_head
        p1.MATVEC_LIB.matvec_int4aff(self.fin, h.qp, h.sc, h.bi, self.dlogits, DDIM,
                                     threads=[32, h.N // 4], group_size=[32, 8])
        GLUE_LIB.argmax_stage1(self.dlogits, self.partv, self.parti, self.V,
                               threads=[self.nparts * 256], group_size=[256])
        GLUE_LIB.argmax_stage2(self.partv, self.parti, self.toks, self.nparts, tok_pos,
                               threads=[256], group_size=[256])
        # chained hidden for the next draft (post_proj of fin) straight into act[1536:]
        DLIB.matvec_int8_aq(self.fin, self.d_post_proj_w8, self.d_post_proj_sc,
                            self.act[HID:], DDIM, self.aq_post_proj,
                            threads=[32, HID // 4], group_size=[32, 8])

    def draft_chain(self, a: int) -> None:
        """3 chained drafts: emb(TOKS[a+i]) ‖ hidden -> TOKS[a+i+1]. qpos = a-1 FIXED."""
        self.act[HID:] = self.dhidden          # draft 1 hidden = h-tilde[a-1]
        for i in range(G_DRAFTS):
            GLUE_LIB.embed_gather(self.emb_packed, self.emb_scale, self.act[:HID],
                                  self.toks, a + i, threads=[384], group_size=[128])
            self.draft_forward(a - 1, a + i)

    # ---- MTP generation (lossless vs the S=1 loop by verify bit-exactness) ----------------
    def mtp_generate(self, prompt_ids: list[int], n_gen: int):
        for li in self.kcache:
            self.kcache[li].zero_()
            self.vcache[li].zero_()
        plen = len(prompt_ids)
        self.toks[:plen] = torch.tensor(prompt_ids, dtype=torch.int32)
        # prompt: S=1 steps; at the last one materialize hidden + argmax -> TOKS[plen]
        for pos in range(plen - 1):
            self.step(pos, want_logits=False)
        self.step(plen - 1, want_logits=False)
        GLUE_LIB.rmsnorm_glue(self.x, self.final_norm, self.x, self.dhidden,
                              HID, 0, 1.0, 0, threads=[32, 1], group_size=[32, 1])
        self.lm_head(self.dhidden, self.logits)
        GLUE_LIB.argmax_stage1(self.logits, self.partv, self.parti, self.V,
                               threads=[self.nparts * 256], group_size=[256])
        GLUE_LIB.argmax_stage2(self.partv, self.parti, self.toks, self.nparts, plen - 1,
                               threads=[256], group_size=[256])
        a = plen                                # anchor: TOKS[a] filled, unprocessed
        emitted = 0
        rounds = 0
        while emitted < n_gen:
            self.draft_chain(a)                 # TOKS[a+1..a+3] = drafts (GPU-chained)
            self.verify_step(a)                 # bit-exact S=4; targets[1..4]
            torch.mps.synchronize()             # ONE sync per round
            tgt = [int(self.targets[m + 1].item()) for m in range(S)]
            drafts = [int(self.toks[a + 1 + i].item()) for i in range(G_DRAFTS)]
            n_acc = 0
            while n_acc < G_DRAFTS and drafts[n_acc] == tgt[n_acc]:
                n_acc += 1
            import os
            if os.environ.get("G4_MTP_DEBUG") == "1":
                print(f"round a={a} drafts={drafts} targets={tgt} nAcc={n_acc}")
            self.toks[a + n_acc + 1] = tgt[n_acc]          # bonus/correction token
            self.dhidden.copy_(self.hidden4[n_acc])        # h-tilde of the new anchor-1
            emitted += n_acc + 1
            a += n_acc + 1
            rounds += 1
        torch.mps.synchronize()
        out = self.toks[plen + 1:plen + 1 + n_gen].cpu().tolist()
        # NOTE: TOKS[plen] is the first generated token (from the bootstrap argmax);
        # emitted tokens per round land at TOKS[a+1..]; total stream = TOKS[plen:].
        out = [int(self.toks[plen].item())] + out[:n_gen - 1]
        return out, rounds, emitted


# ---------- torch fp32 reference drafter (parity) -------------------------------------------
def build_torch_drafter():
    from coreai_models.models.macos.gemma4_mtp_drafter import (
        Gemma4MtpDrafter, load_transplant)

    class FP32Embed(torch.nn.Module):
        def __init__(self, ex: Extract) -> None:
            super().__init__()
            m = ex.manifest["embed.composite"]
            V, _ = m["shape"]
            self.packed = ex.w.get_tensor("embed.composite").reshape(V, HID // 4)
            self.scale = ex.w.get_tensor("embed.composite.scale").float()

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            b, s = ids.shape
            flat = ids.reshape(-1)
            codes = p1.unpack_int2(self.packed[flat], flat.numel(), HID).float()
            return (codes * self.scale[flat].unsqueeze(1) * (HID ** 0.5)).reshape(b, s, HID)

    ex = Extract(EXTRACT)
    tm = Gemma4MtpDrafter()
    load_transplant(tm, str(EXTRACT), dtype=torch.float32, fp_head=True)
    tm.embed_tokens = FP32Embed(ex)
    return tm.eval()


def run_parity() -> bool:
    refs = json.loads(p1.ORACLE_REFS.read_text())
    sky = refs["prompts"]["Why is the sky blue?"]["prompt_ids"]
    ex = Extract(EXTRACT)
    raw = RawLoopMTP(ex)
    tm = build_torch_drafter()
    ok_all = True
    for warm in (6, 20, 40):
        # S=1 to anchor a = plen + warm, hidden materialized
        for li in raw.kcache:
            raw.kcache[li].zero_()
            raw.vcache[li].zero_()
        plen = len(sky)
        raw.toks[:plen] = torch.tensor(sky, dtype=torch.int32)
        a = plen + warm
        for pos in range(a):
            raw.step(pos, want_logits=False)
            if pos >= plen - 1:
                raw.head_argmax(pos)
        GLUE_LIB.rmsnorm_glue(raw.x, raw.final_norm, raw.x, raw.dhidden,
                              HID, 0, 1.0, 0, threads=[32, 1], group_size=[32, 1])
        raw.draft_chain(a)
        torch.mps.synchronize()
        kernel_drafts = [int(raw.toks[a + 1 + i].item()) for i in range(G_DRAFTS)]
        # torch reference chain from the same state
        p = a
        k11 = raw.kcache[13].cpu().float().reshape(1, 1, -1, 256)
        v11 = raw.vcache[13].cpu().float().reshape(1, 1, -1, 256)
        k14 = raw.kcache[14].cpu().float().reshape(1, 1, -1, 512)
        v14 = raw.vcache[14].cpu().float().reshape(1, 1, -1, 512)
        MAX = k11.shape[2]
        ar = torch.arange(MAX)
        msl = ((ar >= max(0, p - p1.WINDOW)) & (ar < p)).reshape(1, 1, 1, MAX).float()
        msf = (ar < p).reshape(1, 1, 1, MAX).float()
        hidden = raw.dhidden.cpu().float().reshape(1, 1, HID)
        tok = int(raw.toks[a].item())
        ref_drafts = []
        with torch.no_grad():
            for _ in range(G_DRAFTS):
                ids = torch.tensor([[tok]], dtype=torch.int64)
                posid = torch.tensor([[p - 1]], dtype=torch.int32)
                _, proj, amax = tm(ids, hidden, posid, msl, msf, k11, v11, k14, v14)
                tok = int(amax.item())
                ref_drafts.append(tok)
                hidden = proj
        match = sum(1 for x, y in zip(kernel_drafts, ref_drafts) if x == y)
        # fp16 kernel vs fp32 reference: occasional near-tie argmax flips are the
        # expected class (verified: cos 0.9999+, kernel pick = torch rank-2, margin
        # ~0.25). Drafts are alpha-preserving, never correctness-bearing — gate on
        # majority agreement; tokens/round in --gate is the real judge.
        ok = match >= 2
        ok_all &= ok
        print(f"  anchor +{warm}: kernel {kernel_drafts} vs torch {ref_drafts} "
              f"-> {match}/3 {'PASS' if ok else 'FAIL'}")
    return ok_all


def run_gate() -> bool:
    refs = json.loads(p1.ORACLE_REFS.read_text())
    ex = Extract(EXTRACT)
    raw = RawLoopMTP(ex)
    n_gen = 64
    ok_all = True
    for prompt, d in refs["prompts"].items():
        ids = list(d["prompt_ids"])
        t0 = time.time()
        s1 = raw.generate(ids, n_gen)
        t1 = time.time() - t0
        t0 = time.time()
        mtp, rounds, emitted = raw.mtp_generate(ids, n_gen)
        t2 = time.time() - t0
        ok = mtp == s1
        ok_all &= ok
        tpr = emitted / rounds if rounds else 0.0
        print(f"  {'PASS' if ok else 'FAIL'}  {prompt!r}: "
              f"{'LOSSLESS==S1' if ok else 'MISMATCH'}  tokens/round {tpr:.2f} "
              f"[S1 {n_gen / t1:.1f} tok/s -> MTP {n_gen / t2:.1f} tok/s python]")
        if not ok:
            fork = next((i for i, (x, y) in enumerate(zip(mtp, s1)) if x != y), None)
            print(f"    fork@{fork}\n    mtp: {mtp}\n    s1 : {s1}")
    return ok_all


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parity", action="store_true")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    ok = True
    if args.parity:
        print("drafter parity (kernel vs torch fp32 module):")
        ok &= run_parity()
    if args.gate:
        print("MTP lossless gate:")
        ok &= run_gate()
    print(f"\nP3 {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)
