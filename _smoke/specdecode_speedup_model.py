#!/usr/bin/env python3
"""
Speculative-decoding speedup model for the Core AI pipelined engine (Stream C).

Decision question: given our SHIPPED on-device decode tok/s, what per-token
acceptance rate (alpha) and draft length (K) does each draft source need to
beat the baseline autoregressive decode? Greedy, linear-chain (n-gram / vanilla
draft), lossless verify.

Math (Leviathan et al. 2023):
  expected new tokens per verify round  E(alpha,K) = (1 - alpha**(K+1)) / (1 - alpha)
    -- accepted drafts (geometric) + the 1 bonus/correction token.

Cost per round (in units of one TARGET decode step t_dec = 1/decode_tps):
  - verify forward over K+1 tokens ~= c_v target-steps.
      c_v ~ 1.0 when decode is purely weight-bandwidth-bound and K is small
      (the weights are streamed once and serve all K+1 positions). We also show
      a conservative c_v=1.3 to account for the K+1x LM-head (large vocab) and
      attention over K query rows.
  - draft cost:
      n-gram / prompt-lookup: ~0 (host hashtable lookup, no model forward).
      vanilla draft model:     K * r target-steps, r = t_draft / t_dec
                               = decode_tps_target / decode_tps_draft.

  speedup = E(alpha,K) / (draft_cost + c_v)
"""

def E(alpha, K):
    if alpha >= 1.0:
        return K + 1
    return (1.0 - alpha ** (K + 1)) / (1.0 - alpha)

def speedup_ngram(alpha, K, c_v):
    return E(alpha, K) / c_v

def speedup_vanilla(alpha, K, c_v, r):
    return E(alpha, K) / (K * r + c_v)

def best_K(fn, alpha, Ks):
    return max(Ks, key=lambda K: fn(alpha, K))

ALPHAS = [0.5, 0.6, 0.7, 0.8, 0.9]
KS = [2, 3, 4, 5, 6, 8]

print("=" * 74)
print("SPEC-DECODE SPEEDUP MODEL — Core AI pipelined engine, greedy linear chain")
print("=" * 74)

# ---- n-gram / prompt-lookup (draft cost ~ 0) -------------------------------
for c_v in (1.0, 1.3):
    print(f"\n[A] n-gram / prompt-lookup  (draft~0, verify cost c_v={c_v})")
    print("     alpha |  " + "  ".join(f"K={K:<2d}" for K in KS) + "   | best K (speedup)")
    print("    -------+-" + "-" * (6 * len(KS)) + "--+----------------")
    for a in ALPHAS:
        row = "  ".join(f"{speedup_ngram(a,K,c_v):4.2f}" for K in KS)
        bk = best_K(lambda al,K: speedup_ngram(al,K,c_v), a, KS)
        print(f"      {a:.1f}  |  {row}   |  K={bk} ({speedup_ngram(a,bk,c_v):.2f}x)")

# ---- vanilla draft model (qwen3.5-0.8B drafting Qwen3.6-27B) ----------------
# r = decode_tps_target / decode_tps_draft.
# Shipped numbers: 27B dense ~15.9 t/s (plan). 0.8B device ~50 t/s; but on the
# 27B's host the small model runs faster -> show a range of r.
print("\n[B] vanilla draft model  (verify cost c_v=1.0)")
print("    r = t_draft/t_dec = target_tps/draft_tps  (smaller r = faster draft)")
for r in (0.10, 0.20, 0.32):
    print(f"\n    r={r}  (draft ~{1/r:.1f}x faster than target decode)")
    print("     alpha |  " + "  ".join(f"K={K:<2d}" for K in KS) + "   | best K (speedup)")
    print("    -------+-" + "-" * (6 * len(KS)) + "--+----------------")
    for a in ALPHAS:
        row = "  ".join(f"{speedup_vanilla(a,K,1.0,r):4.2f}" for K in KS)
        bk = best_K(lambda al,K: speedup_vanilla(al,K,1.0,r), a, KS)
        print(f"      {a:.1f}  |  {row}   |  K={bk} ({speedup_vanilla(a,bk,1.0,r):.2f}x)")

# ---- break-even acceptance --------------------------------------------------
print("\n" + "=" * 74)
print("BREAK-EVEN: minimum alpha for >1.0x (net win) at the best K")
print("=" * 74)
def breakeven(fn, Ks):
    lo = 0.0
    for a100 in range(1, 100):
        a = a100 / 100.0
        if max(fn(a, K) for K in Ks) > 1.0:
            return a
    return 1.0
print(f"  n-gram (c_v=1.0):  alpha >= {breakeven(lambda a,K: speedup_ngram(a,K,1.0), KS):.2f}")
print(f"  n-gram (c_v=1.3):  alpha >= {breakeven(lambda a,K: speedup_ngram(a,K,1.3), KS):.2f}")
for r in (0.10, 0.20, 0.32):
    be = breakeven(lambda a,K: speedup_vanilla(a,K,1.0,r), KS)
    print(f"  vanilla (r={r}):   alpha >= {be:.2f}")

print("\nNotes:")
print(" - n-gram needs ANY positive acceptance to win because its draft is free;")
print("   real n-gram alpha is high on code/RAG/structured, ~0 on free-form chat.")
print(" - vanilla draft pays K*r per round, so it needs a genuinely fast draft AND")
print("   decent alpha; r=0.32 (0.8B vs 27B-on-iPhone) is marginal, r<=0.2 is the")
print("   regime where vanilla clearly wins.")
print(" - c_v is the key device unknown to MEASURE first: the cost of a K+1-token")
print("   verify forward relative to one decode step. If c_v drifts >1.5 (large-vocab")
print("   LM head dominates), shrink K or compute the head only on needed positions.")
