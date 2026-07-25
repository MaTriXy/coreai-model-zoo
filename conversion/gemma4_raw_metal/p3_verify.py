#!/usr/bin/env python
"""P3a: S=4 VERIFY forward on the raw loop + bit-identical gate vs sequential S=1.

The M=4 kernels keep every per-row dot in the SAME block/code order as the M=1
kernels, verify K/V writes reproduce the S=1 writes, and the S=4 attention reads
the same rows — so verify(4 tokens) must equal 4 sequential S=1 steps BYTE-FOR-BYTE
(fp16 logits torch.equal). That makes MTP losslessness exact, not a near-tie claim.

Gate: sky prompt -> S=1 greedy to an anchor, record 4 S=1 step logits, then one
verify_step over the same 4 tokens -> compare logits + argmax targets. Repeated at
a deep anchor (pos > 512) so the sliding-window clamp path is exercised per query.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

import p1_chain as p1
from p1_chain import (EXTRACT, FULL, GLUE_LIB, HID, L, WINDOW, Extract, RawLoop)

HERE = Path(__file__).resolve().parent
DEV = "mps"
VLIB = torch.mps.compile_shader((HERE / "msl" / "gemma4_verify.metal").read_text())
S = 4


class RawLoopV(RawLoop):
    """RawLoop + S=4 verify forward (the MTP verify path)."""

    def __init__(self, ex: Extract) -> None:
        super().__init__(ex)
        f16 = lambda *shape: torch.empty(*shape, dtype=torch.float16, device=DEV)
        self.x4 = f16(S, HID)
        self.xn4 = f16(S, HID)
        self.pli4 = f16(S, L, 256)
        self.ple4 = f16(S, L, 256)
        self.p84 = f16(S, L * 256)
        self.q4 = {256: f16(S * 8 * 256), 512: f16(S * 8 * 512)}
        self.qn4 = {256: f16(S * 8 * 256), 512: f16(S * 8 * 512)}
        self.kv4 = {256: f16(S, 256), 512: f16(S, 512)}
        self.ctx4 = {256: f16(S * 8 * 256), 512: f16(S * 8 * 512)}
        self.attn4 = f16(S, HID)
        self.hffn4 = f16(S, 12288)
        self.ffn4 = f16(S, HID)
        self.g4 = f16(S, 256)
        self.p4 = f16(S, HID)
        self.hidden4 = f16(S, HID)
        self.logits4 = f16(S, self.V)
        # targets[1..4] = greedy argmax of the 4 verify positions (slot 0 unused)
        self.targets = torch.zeros(S + 1, dtype=torch.int32, device=DEV)

    def verify_step(self, pos0: int) -> None:
        """One S=4 forward over TOKS[pos0..pos0+3]; K/V rows pos0..pos0+3 written;
        logits4 + targets[1..4] + hidden4 (drafter tap) produced on GPU."""
        M = p1.MATVEC_LIB  # noqa: F841 (M=1 lib not used here, but kept for clarity)
        VLIB.embed_gather4(self.emb_packed, self.emb_scale, self.x4, self.toks, pos0,
                           threads=[384, S], group_size=[128, 1])
        mp = self.model_proj
        VLIB.matvec_int8_m4(self.x4, mp.w8, mp.sc, self.p84, self.p84, mp.K, mp.N,
                            0, 0, 0, threads=[32, mp.N // 2], group_size=[32, 8])
        VLIB.ple_gather4(self.ple_packed, self.ple_scale, self.ple4, self.toks, pos0,
                         threads=[4480, S], group_size=[256, 1])
        GLUE_LIB.rmsnorm_glue(self.p84, self.proj_norm, self.ple4, self.pli4,
                              256, 1, p1.PLI_SCALE, 0,
                              threads=[32, S * L], group_size=[32, 8])

        for lay in self.layers:
            hd = lay.hd
            invf = self.invf_full if lay.full else self.invf_sliding
            kc, vc = self.kcache[lay.cache], self.vcache[lay.cache]
            win = 0 if lay.full else WINDOW
            occ_g = 8 if lay.full else 16
            # attention
            GLUE_LIB.rmsnorm_glue(self.x4, lay.pre_attn, self.x4, self.xn4,
                                  HID, 0, 1.0, 0, threads=[32, S], group_size=[32, 4])
            VLIB.matvec_int4aff_m4(self.xn4, lay.wq.qp, lay.wq.sc, lay.wq.bi,
                                   self.q4[hd], HID, lay.wq.N,
                                   threads=[32, lay.wq.N // 2], group_size=[32, 8])
            VLIB.qknorm_rope4(self.q4[hd], lay.query_norm, invf, self.qn4[hd],
                              hd, pos0, 8, 0, threads=[32, S * 8], group_size=[32, 8])
            if lay.write:
                VLIB.matvec_int4aff_m4(self.xn4, lay.wk.qp, lay.wk.sc, lay.wk.bi,
                                       self.kv4[hd], HID, hd,
                                       threads=[32, hd // 2], group_size=[32, 8])
                VLIB.qknorm_rope4(self.kv4[hd], lay.key_norm, invf, kc,
                                  hd, pos0, 1, pos0, threads=[32, S], group_size=[32, 4])
                VLIB.matvec_int4aff_m4(self.xn4, lay.wv.qp, lay.wv.sc, lay.wv.bi,
                                       self.kv4[hd], HID, hd,
                                       threads=[32, hd // 2], group_size=[32, 8])
                GLUE_LIB.rmsnorm_glue(self.kv4[hd], lay.key_norm, self.kv4[hd], vc,
                                      hd, 2, 1.0, pos0, threads=[32, S], group_size=[32, 4])
            VLIB.flash_sdpa_verify(self.qn4[hd], kc, vc, self.ctx4[hd],
                                   hd, pos0, win, occ_g, 8,
                                   threads=[32, occ_g, S * 8], group_size=[32, occ_g, 1])
            VLIB.matvec_int4aff_m4(self.ctx4[hd], lay.wo.qp, lay.wo.sc, lay.wo.bi,
                                   self.attn4, 8 * hd, HID,
                                   threads=[32, HID // 2], group_size=[32, 8])
            GLUE_LIB.rmsnorm_glue(self.attn4, lay.post_attn, self.x4, self.x4,
                                  HID, 1, 1.0, 0, threads=[32, S], group_size=[32, 4])
            # FFN
            GLUE_LIB.rmsnorm_glue(self.x4, lay.pre_ffw, self.x4, self.xn4,
                                  HID, 0, 1.0, 0, threads=[32, S], group_size=[32, 4])
            g, u, d = lay.wgate, lay.wup, lay.wdown
            if lay.int2:
                VLIB.gateup_int2sym_m4(self.xn4, g.qp, g.sc, u.qp, u.sc,
                                       self.hffn4, HID, g.N,
                                       threads=[32, g.N // 2], group_size=[32, 8])
                VLIB.matvec_int2sym_m4(self.hffn4, d.qp, d.sc, self.ffn4, d.K, HID,
                                       threads=[32, HID // 2], group_size=[32, 8])
            else:
                VLIB.gateup_int4aff_m4(self.xn4, g.qp, g.sc, g.bi, u.qp, u.sc, u.bi,
                                       self.hffn4, HID, g.N,
                                       threads=[32, g.N // 2], group_size=[32, 8])
                VLIB.matvec_int4aff_m4(self.hffn4, d.qp, d.sc, d.bi, self.ffn4, d.K, HID,
                                       threads=[32, HID // 2], group_size=[32, 8])
            GLUE_LIB.rmsnorm_glue(self.ffn4, lay.post_ffw, self.x4, self.x4,
                                  HID, 1, 1.0, 0, threads=[32, S], group_size=[32, 4])
            # PLE injection
            pg, pp = lay.ple_gate, lay.ple_proj
            VLIB.matvec_int8_m4(self.x4, pg.w8, pg.sc, self.pli4, self.g4,
                                HID, 256, 1, lay.li * 256, L * 256,
                                threads=[32, 256 // 2], group_size=[32, 8])
            VLIB.matvec_int8_m4(self.g4, pp.w8, pp.sc, self.p4, self.p4,
                                256, HID, 0, 0, 0,
                                threads=[32, HID // 2], group_size=[32, 8])
            GLUE_LIB.rmsnorm_glue(self.p4, lay.post_ple, self.x4, self.x4,
                                  HID, 1, lay.layer_scalar, 0,
                                  threads=[32, S], group_size=[32, 4])

        # head: final norm (materialized — also the drafter kickoff tap), logits, targets
        GLUE_LIB.rmsnorm_glue(self.x4, self.final_norm, self.x4, self.hidden4,
                              HID, 0, 1.0, 0, threads=[32, S], group_size=[32, 4])
        VLIB.matvec_int2sym_m4(self.hidden4, self.lm_head.qp, self.lm_head.sc,
                               self.logits4, HID, self.V,
                               threads=[32, self.V // 2], group_size=[32, 8])
        for m in range(S):
            GLUE_LIB.argmax_stage1(self.logits4[m], self.partv, self.parti, self.V,
                                   threads=[self.nparts * 256], group_size=[256])
            GLUE_LIB.argmax_stage2(self.partv, self.parti, self.targets, self.nparts, m,
                                   threads=[256], group_size=[256])


def gate_at_anchor(raw: RawLoopV, prompt_ids: list[int], warm_gen: int) -> bool:
    """S=1 to anchor P=len(prompt)+warm_gen, record 4 S=1 logits, verify, compare."""
    for li in raw.kcache:
        raw.kcache[li].zero_()
        raw.vcache[li].zero_()
    raw.toks[:len(prompt_ids)] = torch.tensor(prompt_ids, dtype=torch.int32)
    plen = len(prompt_ids)
    anchor = plen + warm_gen
    # S=1 to the anchor: TOKS[plen..anchor] filled by GPU argmax
    for pos in range(anchor):
        raw.step(pos, want_logits=False)
        if pos >= plen - 1:
            raw.head_argmax(pos)
    # 4 sequential S=1 steps at pos0..pos0+3 (want_logits materializes raw.logits)
    pos0 = anchor
    ref_logits, ref_next = [], []
    for i in range(S):
        lg = raw.step(pos0 + i, want_logits=True)
        raw.head_argmax(pos0 + i)   # fills TOKS[pos0+i+1] (greedy chain)
        torch.mps.synchronize()
        ref_logits.append(lg.cpu().clone())
        ref_next.append(int(raw.toks[pos0 + i + 1].item()))
    # verify over the SAME 4 tokens (TOKS[pos0..pos0+3] already hold them)
    raw.verify_step(pos0)
    torch.mps.synchronize()
    ok = True
    for m in range(S):
        got = raw.logits4[m].cpu()
        exact = torch.equal(got, ref_logits[m])
        md = (got.float() - ref_logits[m].float()).abs().max().item()
        tgt = int(raw.targets[m + 1].item())
        tok_ok = tgt == ref_next[m]
        ok &= exact and tok_ok
        print(f"  m={m} pos={pos0 + m}: logits {'BYTE-EXACT' if exact else f'maxdiff {md:.3e}'}"
              f"  target {tgt} {'==' if tok_ok else f'!= {ref_next[m]}'} s1")
    return ok


def main() -> None:
    refs = json.loads(p1.ORACLE_REFS.read_text())
    sky = refs["prompts"]["Why is the sky blue?"]["prompt_ids"]
    ex = Extract(EXTRACT)
    raw = RawLoopV(ex)
    print("anchor: shallow (prompt+8)")
    ok = gate_at_anchor(raw, sky, 8)
    print("anchor: deep (prompt+600, sliding window clamp active)")
    t0 = time.time()
    ok &= gate_at_anchor(raw, sky, 600)
    print(f"  (deep run {time.time() - t0:.1f}s)")
    print(f"\nP3a VERIFY GATE: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
