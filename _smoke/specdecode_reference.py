#!/usr/bin/env python3
"""
Reference implementation + losslessness proof for greedy speculative decoding
(Stream C). Model-free and device-free: a deterministic oracle stands in for the
target model's argmax so we can lock down the accept/reject + bonus + rollback
INDEXING — the #1 bug source — before the Swift engine mirrors it.

What this proves:
  1. LOSSLESSNESS: greedy spec-decode output is token-identical to plain greedy
     for any oracle, any draft length K, any n-gram order — across many random
     oracles. (If this passes, the control logic has no off-by-one.)
  2. ROLLBACK SEMANTICS: modeled as "commit only accepted tokens" — the exact
     analog of rewinding processedTokenCount in CoreAIPipelinedEngine.
  3. ACCEPTANCE (alpha): realized accepted-per-round on repetitive vs random
     streams, the empirical prior the speedup model assumes.

Engine mapping (CoreAIPipelinedEngine):
  - argmax_fn(seq)                  -> argmax of the logits at position len(seq).
       i=0 term  = the CACHED last-step logits (already on device).
       i>=1 term = position i-1 of the ONE verify-forward over the K draft tokens
                   (logits buffer [1,K,vocab]).
  - commit accepted draft tokens    -> KV rows [n..n+accepted-1] are already correct.
  - processedTokenCount = n+accepted -> "rollback": stale KV rows >= that are
       overwritten by the next forward, never attended (causal mask + positions).
"""

import random


# ----------------------------------------------------------------------------
# Drafters (training-free)
# ----------------------------------------------------------------------------
class PromptLookupDrafter:
    """n-gram / prompt-lookup: last `ng` committed tokens -> the continuation that
    most recently followed that same n-gram earlier in the stream."""

    def __init__(self, ngram=3, max_draft=8):
        self.ngram = ngram
        self.max_draft = max_draft

    def draft(self, seq, K):
        K = min(K, self.max_draft)
        n = len(seq)
        for ng in range(self.ngram, 0, -1):          # back off to shorter n-grams
            if n < ng:
                continue
            key = tuple(seq[n - ng:])
            # search most-recent earlier occurrence of `key`
            for i in range(n - ng - 1, -1, -1):
                if tuple(seq[i:i + ng]) == key:
                    cont = seq[i + ng:i + ng + K]
                    if cont:
                        return list(cont)
                    break
        return []


# ----------------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------------
def greedy_decode(argmax_fn, prompt, n):
    seq = list(prompt)
    out = []
    for _ in range(n):
        t = argmax_fn(seq)
        seq.append(t)
        out.append(t)
    return out


def greedy_spec_decode(argmax_fn, prompt, n, K, drafter, stats=None):
    """Greedy speculative decoding. Provably lossless: every committed token is
    the model's greedy argmax given the committed prefix."""
    seq = list(prompt)
    out = []
    rounds = 0
    accepted_total = 0
    verify_forwards = 0   # = number of multi-token verify rounds (the costed op)

    while len(out) < n:
        draft = drafter.draft(seq, K)

        if not draft:
            # no draft available -> one ordinary autoregressive step
            t = argmax_fn(seq)
            seq.append(t); out.append(t)
            rounds += 1
            continue

        verify_forwards += 1
        rounds += 1

        # verify draft[i] against the model's greedy token a_i = argmax(prefix+draft[:i])
        accepted = 0
        correction = None
        for i in range(len(draft)):
            a_i = argmax_fn(seq + draft[:i])
            if a_i == draft[i]:
                accepted += 1
            else:
                correction = a_i
                break

        # commit accepted draft tokens (KV rows already correct)
        emitted_this_round = 0
        for i in range(accepted):
            seq.append(draft[i]); out.append(draft[i]); emitted_this_round += 1
            if len(out) >= n:
                break
        accepted_total += accepted

        if len(out) >= n:
            break

        # emit correction (first mismatch) or bonus (all accepted) — "rollback" is
        # implicit: we simply did not commit draft tokens past `accepted`.
        if correction is not None:
            seq.append(correction); out.append(correction)
        else:
            bonus = argmax_fn(seq)          # all accepted -> seq == prefix+draft
            seq.append(bonus); out.append(bonus)

    if stats is not None:
        stats["rounds"] = rounds
        stats["verify_forwards"] = verify_forwards
        stats["accepted_total"] = accepted_total
        stats["generated"] = len(out)
        # mean accepted draft tokens per verify round (excl. the always-1 bonus/correction)
        stats["alpha_realized"] = (accepted_total / verify_forwards) if verify_forwards else 0.0
    return out[:n]


# ----------------------------------------------------------------------------
# Deterministic oracles (stand-ins for the target model's argmax)
# ----------------------------------------------------------------------------
def make_hash_oracle(vocab, seed):
    """Pure-random next-token: argmax depends on a hash of the WHOLE prefix.
    Worst case for n-gram (alpha ~ 0). Stresses correctness, not acceptance."""
    def f(seq):
        h = seed
        for t in seq[-16:]:                  # bounded context; deterministic
            h = (h * 1000003 + t + 1) & 0xFFFFFFFF
        return h % vocab
    return f


def make_markov_oracle(vocab, order, seed, repeat_bias):
    """Next token depends only on the last `order` tokens (a fixed table), with a
    knob `repeat_bias` that makes continuations repetitive (structured/code-like)
    vs varied (chat-like). Higher bias -> higher n-gram acceptance."""
    table = {}
    rng = random.Random(seed)

    def f(seq):
        key = tuple(seq[-order:])
        if key not in table:
            if rng.random() < repeat_bias and len(seq) >= order:
                # echo a token that recently followed a similar context -> repetition
                table[key] = seq[-order]
            else:
                table[key] = rng.randrange(vocab)
        return table[key]
    return f


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def test_losslessness():
    print("[1] LOSSLESSNESS — spec-decode output must equal plain greedy")
    fails = 0
    cases = 0
    for seed in range(200):
        rng = random.Random(seed)
        vocab = rng.choice([16, 64, 256])
        prompt = [rng.randrange(vocab) for _ in range(rng.randint(1, 8))]
        n = rng.randint(1, 60)
        # mix of oracle kinds
        if seed % 3 == 0:
            oracle = make_hash_oracle(vocab, seed)
        else:
            oracle = make_markov_oracle(vocab, rng.choice([1, 2, 3]), seed,
                                        rng.choice([0.0, 0.5, 0.9]))
        base = greedy_decode(oracle, prompt, n)
        for K in (1, 2, 3, 4, 8):
            for ng in (1, 2, 3):
                spec = greedy_spec_decode(oracle, prompt, n, K,
                                          PromptLookupDrafter(ngram=ng, max_draft=8))
                cases += 1
                if spec != base:
                    fails += 1
                    if fails <= 3:
                        print(f"    MISMATCH seed={seed} K={K} ng={ng}")
                        print(f"      base={base[:20]}")
                        print(f"      spec={spec[:20]}")
    print(f"    {cases - fails}/{cases} cases token-identical "
          f"{'PASS' if fails == 0 else 'FAIL'}")
    return fails == 0


def test_acceptance():
    print("\n[2] ACCEPTANCE (alpha_realized) by stream type, K=4, ngram=3")
    for label, bias in [("random (chat-like)", 0.0),
                        ("mixed", 0.5),
                        ("repetitive (code/structured-like)", 0.9)]:
        accs = []
        for seed in range(30):
            rng = random.Random(1000 + seed)
            vocab = 128
            oracle = make_markov_oracle(vocab, 2, 1000 + seed, bias)
            prompt = [rng.randrange(vocab) for _ in range(4)]
            stats = {}
            greedy_spec_decode(oracle, prompt, 200, 4,
                               PromptLookupDrafter(ngram=3, max_draft=8), stats=stats)
            # per-token acceptance estimate: accepted / (verify positions tried)
            accs.append(stats["alpha_realized"])
        mean = sum(accs) / len(accs)
        print(f"    {label:38s}  mean accepted/round = {mean:.2f}")


def test_realized_speedup():
    print("\n[3] REALIZED tokens-per-verify vs the analytic model E(alpha,K)")
    vocab = 128
    for bias, name in [(0.9, "repetitive"), (0.5, "mixed")]:
        tot_tokens = tot_forwards = 0
        for seed in range(30):
            oracle = make_markov_oracle(vocab, 2, 2000 + seed, bias)
            rng = random.Random(2000 + seed)
            prompt = [rng.randrange(vocab) for _ in range(4)]
            stats = {}
            greedy_spec_decode(oracle, prompt, 300, 6,
                               PromptLookupDrafter(ngram=3, max_draft=8), stats=stats)
            # total decode "steps" with spec = verify_forwards + plain steps;
            # tokens generated per costed forward is the headline.
            tot_tokens += stats["generated"]
            tot_forwards += stats["rounds"]   # rounds ~ forwards (each round = 1 forward)
        print(f"    {name:12s} K=6: {tot_tokens/tot_forwards:.2f} tokens / forward "
              f"(baseline = 1.00; >1 = speedup before c_v)")


if __name__ == "__main__":
    ok = test_losslessness()
    test_acceptance()
    test_realized_speedup()
    print("\nVERDICT:", "control logic LOSSLESS ✓" if ok else "BUG — fix before engine work")
