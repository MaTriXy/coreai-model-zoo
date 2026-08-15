#!/usr/bin/env python3
"""Gate: the embeddings-input decoder + interleaved mRoPE, eager fp16 CPU,
teacher-forced on MIXED text+image sequences vs the bf16 HF oracle.

This is the wiring gate for the vision path's TEXT half (the tower has its own
gate): host mRoPE planes (qwen38vl_host.mrope_positions asserted against the
oracle's CAPTURED positions), embed splice at ``<|image_pad|>``, the
``Qwen3_5VLStatefulEmbeds`` graph with the chunked GDN path (the export
semantics), and the decode-step position-resume rule.

Two suite cases carry full logits rows: case 0 (image-first) and case 3
(text-BEFORE-image — exercises a nonzero image start). Criterion is the family
rule: per-position argmax vs the oracle with the margin>=0.1 knife-edge waiver,
plus logits cos.

Run (coreai-models/.venv, CPU, ~55 GB RAM):
    ../coreai-models/.venv/bin/python _smoke/test_qwen38vl_mixed_eager_gate.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

DEFAULT_SUITE = Path(__file__).parent / "qwen38vl_suite_512.npz"
HF_ID = "Qwen/Qwen3.8-27B"
MARGIN = 0.1
KV_SEQ = 2048


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--cases", default=None, help="comma list; default = rows-carrying")
    args = ap.parse_args()

    import torch

    from qwen38vl_host import mrope_positions, splice_embeds
    from coreai_models.models.macos.qwen3_5 import Qwen3_5VLStatefulEmbeds, build_decode_state
    from coreai_models.primitives.macos.cache import KVCache, SSMState

    suite = np.load(args.suite)
    n_cases = int(suite["_meta_cases"])
    cases = ([int(c) for c in args.cases.split(",")] if args.cases else
             [c for c in range(n_cases) if f"case{c}_logits_rows" in suite])
    print(f"gating cases {cases} (of {n_cases})")

    print(f"loading {HF_ID} text decoder fp16 (embeds variant) ...")
    model = Qwen3_5VLStatefulEmbeds.from_hf_memory_efficient(
        HF_ID, max_context_length=4096, target_dtype=torch.float16,
        hf_config_attr="text_config")
    model.eval()
    cfg = model.config
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_chunk = True  # export semantics
    embed_table = model.model.embed_tokens.weight.detach().numpy()

    ok_all = True
    for case in cases:
        ids = suite[f"case{case}_ids"].astype(np.int64)
        grid = tuple(int(v) for v in suite[f"case{case}_grid_thw"][0])
        want_pos = suite[f"case{case}_pos_prefill"]
        want_steps = suite[f"case{case}_pos_steps"]
        want_delta = int(suite[f"case{case}_rope_delta"])
        rows = suite[f"case{case}_logits_rows"]
        gen = suite[f"case{case}_gen"].astype(np.int64)
        margins = suite[f"case{case}_margins"]
        S = ids.size

        # 1. host mRoPE contract vs the oracle's captured positions
        pos, delta = mrope_positions(ids, [grid])
        if not (np.array_equal(pos, want_pos) and delta == want_delta):
            bad = np.nonzero((pos != want_pos).any(0))[0][:5]
            print(f"case {case}: FAIL host mrope planes differ at {bad} "
                  f"(delta {delta} vs {want_delta})")
            ok_all = False
            continue
        step0 = S + np.arange(want_steps.shape[1]) + delta
        if want_steps.size and not np.array_equal(
                want_steps, np.broadcast_to(step0, (3, step0.size))):
            print(f"case {case}: FAIL decode-step resume rule vs captured positions")
            ok_all = False
            continue
        print(f"case {case}: host mrope planes + delta {delta} match the oracle")

        # 2. teacher-forced eager decoder
        embeds = splice_embeds(
            ids, embed_table, suite[f"case{case}_image_embeds"].astype(np.float16))
        state = build_decode_state(cfg, max_seq_len=KV_SEQ, dtype=torch.float16)
        kw = {k: state[k] for k in ("k_cache", "v_cache", "conv_state", "rec_state")}

        def step(x_embeds, ramp_len, p3, kw=kw):
            with torch.no_grad():
                logits = model(
                    inputs_embeds=torch.from_numpy(x_embeds[None]).to(torch.float16),
                    position_ids=torch.arange(ramp_len, dtype=torch.int32)[None],
                    pos_t=torch.from_numpy(p3[0:1].astype(np.int32)),
                    pos_h=torch.from_numpy(p3[1:2].astype(np.int32)),
                    pos_w=torch.from_numpy(p3[2:3].astype(np.int32)),
                    **kw,
                )
            return logits[0, -1].float().numpy()

        # Prefill in PF=32 chunks + S=1 remainder — the BUNDLE's exact contract.
        # Never one-shot: the loop-free chunk scan's doubling-inverse is only
        # stable for short chunks; at S~300 with the weak decays real prompts
        # produce (g ~ 0 on image spans) it overflows even in fp32 (isolated
        # 2026-08-15: layer-0 GDN NaN on real embeds, finite at S<=64).
        PF = 16  # mirror the ship bundle's chunk
        o = 0
        while o + PF <= S:
            row = step(embeds[o:o + PF], o + PF, pos[:, o:o + PF])
            o += PF
        while o < S:
            row = step(embeds[o:o + 1], o + 1, pos[:, o:o + 1])
            o += 1
        n_match = n_soft = 0
        for k in range(rows.shape[0]):
            want_row = rows[k].astype(np.float32)
            got_top = int(row.argmax())
            want_top = int(want_row.argmax())
            c = float(row @ want_row / (np.linalg.norm(row) * np.linalg.norm(want_row)))
            if got_top == want_top:
                n_match += 1
                tag = "ok"
            elif margins[k] < MARGIN:
                n_soft += 1
                tag = f"tie (margin {margins[k]:.3f})"
            else:
                tag = f"CONFIDENT FLIP (margin {margins[k]:.3f}, got {got_top})"
                ok_all = False
            print(f"  step {k:2d}: top1 {want_top} cos {c:.4f} {tag}")
            if k == rows.shape[0] - 1:
                break
            nxt = int(gen[k])  # teacher-force the ORACLE's token
            p3 = np.full((3, 1), S + k + delta, dtype=np.int32)
            row = step(embed_table[nxt][None].copy(), S + k + 1, p3)
        print(f"case {case}: {n_match}/{rows.shape[0]} exact, {n_soft} knife-edge ties")

    print("GATE PASS" if ok_all else "GATE FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
