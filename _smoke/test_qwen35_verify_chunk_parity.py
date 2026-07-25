"""Gate: spec-decode verify-forward numerics on the qwen3_5 GDN hybrid.

Spec-decode needs a STATIC S=K+1 verify graph (dynamic-query while_loop doesn't
lower; JIT dynamic-query dies ANECompile/MTL4). The GDN path for S>1 is
`_gated_delta_chunk` (loop-free chunked scan). Known fp16 hazard: the in-graph
doubling-inverse NaNs at chunk>=64 — verify chunks are <=17, so this gates the
small-S regime before any export.

  A. Unit: _gated_delta_chunk at S in {2,9,17} vs _gated_delta_step looped S
     times (state threaded), fp16 in/out, realistic magnitudes, GVA shapes.
  B. Model (Qwen3.5-0.8B, eager CPU): teacher-forced logits of the last 9
     positions from ONE query_len=9 chunked forward vs 9 sequential
     query_len=1 loop-free-step forwards (shipped decode semantics).
     Gate: per-position argmax 9/9 + logits cos + end-state parity
     (continuing generation after a verify forward must be seamless).

Run: coreai-models/.venv/bin/python test_qwen35_verify_chunk_parity.py
"""
from __future__ import annotations

import sys

import torch

from coreai_models.models.macos.qwen3_5 import (
    Qwen3_5StatefulForCausalLM,
    _gated_delta_chunk,
    _gated_delta_step,
    build_decode_state,
)

torch.manual_seed(0)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


def gate_a_unit() -> bool:
    """Chunk vs step-looped parity on raw GDN tensors (fp16 in/out)."""
    ok = True
    b, h, dk, dv = 1, 48, 128, 128  # 27B GVA value-head count (post q/k repeat)
    for s in (2, 9, 17):
        q = torch.randn(b, h, s, dk, dtype=torch.float16)
        k = torch.randn(b, h, s, dk, dtype=torch.float16)
        v = torch.randn(b, h, s, dv, dtype=torch.float16)
        # realistic decay/beta: g negative (decay logits), beta in (0,1)
        g = (-torch.rand(b, h, s) * 3.0).to(torch.float16)
        beta = torch.rand(b, h, s).to(torch.float16)
        s0 = (torch.randn(b, h, dk, dv) * 0.1).to(torch.float16)

        out_c, st_c = _gated_delta_chunk(q, k, v, g, beta, s0)

        st = s0
        outs = []
        for t in range(s):
            o, st = _gated_delta_step(
                q[:, :, t : t + 1], k[:, :, t : t + 1], v[:, :, t : t + 1],
                g[:, :, t : t + 1], beta[:, :, t : t + 1], st,
            )
            outs.append(o)
        out_s = torch.cat(outs, dim=1)  # [b,s,h,dv]

        c_out, c_st = cos(out_c, out_s), cos(st_c, st)
        md_out = float((out_c.float() - out_s.float()).abs().max())
        md_st = float((st_c.float() - st.float()).abs().max())
        nan = bool(out_c.isnan().any() or st_c.isnan().any())
        line_ok = c_out > 0.9999 and c_st > 0.9999 and not nan
        ok &= line_ok
        print(f"A S={s:2d}: out cos={c_out:.6f} maxdiff={md_out:.2e} | "
              f"state cos={c_st:.6f} maxdiff={md_st:.2e} | nan={nan} "
              f"{'PASS' if line_ok else 'FAIL'}")
    return ok


def set_gdn_mode(model, *, chunk: bool) -> None:
    for layer in model.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_chunk = chunk
            layer.linear_attn.use_loopfree_step = not chunk


@torch.no_grad()
def gate_b_model(hf_id: str = "Qwen/Qwen3.5-0.8B", n_verify: int = 9) -> bool:
    from transformers import AutoTokenizer

    print(f"loading {hf_id} fp16 (eager CPU) ...")
    model = Qwen3_5StatefulForCausalLM.from_hf_memory_efficient(
        hf_id, max_context_length=4096, target_dtype=torch.float16,
        hf_config_attr="text_config")
    model.eval()
    cfg = model.config

    tok = AutoTokenizer.from_pretrained(hf_id)
    text = ("def fib(n):\n    if n < 2:\n        return n\n"
            "    return fib(n - 1) + fib(n - 2)\n\nprint(fib(10))")
    ids = tok(text, return_tensors="pt").input_ids.to(torch.int32)
    n = ids.shape[1]
    assert n > n_verify + 4, f"prompt too short ({n})"
    print(f"prompt tokens: {n} (prefix {n - n_verify} + verify block {n_verify})")

    def seq_path(upto: int, state: dict) -> list[torch.Tensor]:
        """Sequential loop-free-step decode semantics; returns per-pos logits."""
        set_gdn_mode(model, chunk=False)
        logits = []
        for t in range(upto):
            pos = torch.arange(t + 1, dtype=torch.int32).unsqueeze(0)
            out = model(ids[:, t : t + 1], pos, state["k_cache"], state["v_cache"],
                        state["conv_state"], state["rec_state"])
            logits.append(out[:, 0].float())
        return logits

    # Path A: full sequential reference
    st_a = build_decode_state(cfg, max_seq_len=n + 8, dtype=torch.float16)
    ref = seq_path(n, st_a)

    # Path B: sequential prefix, then ONE chunked verify forward over last 9
    st_b = build_decode_state(cfg, max_seq_len=n + 8, dtype=torch.float16)
    _ = seq_path(n - n_verify, st_b)
    set_gdn_mode(model, chunk=True)
    pos = torch.arange(n, dtype=torch.int32).unsqueeze(0)
    ver = model(ids[:, n - n_verify:], pos, st_b["k_cache"], st_b["v_cache"],
                st_b["conv_state"], st_b["rec_state"])  # [1, 9, vocab]

    ok = True
    match = 0
    for i in range(n_verify):
        p = n - n_verify + i
        a_ref = int(ref[p].argmax())
        a_ver = int(ver[0, i].float().argmax())
        c = cos(ref[p], ver[0, i].float())
        m = a_ref == a_ver
        match += m
        ok &= m and c > 0.999
        print(f"B pos {p:3d}: argmax ref={a_ref:6d} ver={a_ver:6d} "
              f"cos={c:.6f} {'OK' if m else 'FLIP'}")
    print(f"B argmax match: {match}/{n_verify}")

    # end-state parity (verify forward must leave the same state as sequential)
    for k in ("k_cache", "v_cache", "conv_state", "rec_state"):
        c = cos(st_a[k], st_b[k])
        md = float((st_a[k].float() - st_b[k].float()).abs().max())
        line_ok = c > 0.999
        ok &= line_ok
        print(f"B state {k:10s}: cos={c:.6f} maxdiff={md:.2e} "
              f"{'OK' if line_ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok_a = gate_a_unit()
    ok_b = gate_b_model()
    print(f"\nGATE A (unit chunk-vs-step): {'PASS' if ok_a else 'FAIL'}")
    print(f"GATE B (0.8B model verify-forward): {'PASS' if ok_b else 'FAIL'}")
    sys.exit(0 if (ok_a and ok_b) else 1)
