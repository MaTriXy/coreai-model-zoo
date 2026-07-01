# Speculative decoding on the pipelined decode bundles — design + feasibility (2026-07-01)

> Axis-2 of the flagship stack ([[flagship-full-tuning-stack]]): the ONLY lever that beats the decode
> bandwidth wall — verify K tokens per forward. **Multiplies on top of the Axis-1 byte cut** (Axis-1 cuts
> per-forward cost; spec-decode cuts forward count). Industry: n-gram 2–4× (training-free), EAGLE-3 3–5×.

## Feasibility — MOSTLY ALREADY THERE (verified this session, no export change needed)

1. **Verify-forward output is native.** The decode exports (`export_{lfm2_moe,qwen3_6_moe}_*`) do NOT set
   `last_token_only`, so the model default (`last_token_only = False`) holds → the head runs on ALL
   positions → the graph outputs `[1, S, vocab]`. Feeding `[1, K]` returns K per-position logit vectors
   in ONE forward. The bundle is causal, so position i's logits are conditioned on the prefix ≤ i — exactly
   what verification needs. **No re-export required** (the LFM #2 bundle on the A19 already qualifies).
2. **The S=K forward machinery exists.** `PipelinedBench.chunkStep` already feeds `[1, q]` input_ids +
   `[1, processed+q]` position_ids via `InferenceFunction.run` with the 4 states threaded as MutableViews.
   It currently resolves the output to `[1, 1, vocab]`; for verify, resolve to `[1, q, vocab]` and read all q.
3. **The only genuinely new work** = the host verify loop + state rollback (below). Not a kernel, not an
   export change.

## The loop (n-gram draft — the training-free first win)

State per step: accepted context `ctx` (token ids), `processed` = KV length, the 4 device states.
```
loop:
  # 1. DRAFT (n-gram / prompt-lookup, no model): find the longest suffix of ctx that occurred earlier;
  #    propose the K tokens that followed it. (∅ → fall back to 1 normal decode step.)
  draft = ngram_lookup(ctx, K)                        # e.g. K=4
  # 2. SNAPSHOT the recurrent SSM state (conv_state, rec_state) — small, cheap to copy.
  snap = (conv.copy(), rec.copy());  base = processed
  # 3. VERIFY-FORWARD: one forward over [last_accepted, *draft] = [1, 1+len(draft)].
  logits = chunkStep(tokens=[ctx[-1], *draft], processed=base)   # -> [1, 1+len(draft), vocab]
  argmax = [argmaxF(logits[:, i]) for i in range(1+len(draft))]
  # 4. ACCEPT the longest prefix where draft[i] == argmax[i]; argmax at first mismatch = the free "bonus".
  j = longest_prefix_match(draft, argmax[1:])
  accepted = draft[:j] + [argmax[j]]                  # j drafts + 1 bonus token (always ≥1 token/forward)
  ctx += accepted
  # 5. ROLLBACK to the accepted length:
  #    - KV cache: append-only per position → just set processed = base + len(accepted) (stale slots
  #      beyond it are overwritten next forward). Trivial.
  #    - SSM (conv/rec): the verify-forward advanced them by 1+len(draft) tokens in-place. Restore `snap`
  #      then re-apply the `len(accepted)` accepted tokens as one small [1,len(accepted)] forward
  #      (cheap: len(accepted) ≤ K+1 ≪ the K weight-reads spec-decode saved). OR keep per-token SSM
  #      snapshots and truncate. Restore-and-replay is simplest and correct.
  processed = base + len(accepted)
```
**Lossless**: every emitted token equals greedy argmax of the target model (verify), so the output
distribution is identical to plain greedy — spec-decode only changes SPEED, never quality.

## Speedup model
- tokens/forward = `len(accepted)` = j+1 (1 ≤ · ≤ K+1). Avg acceptance `ā` → speedup ≈ `ā` (minus the
  small SSM restore-replay overhead). n-gram: `ā` ~2–4 on code/RAG/structured, ~1 on free chat.
- **Multiplies with Axis-1**: e.g. Qwen3.6 dense+experts-int4 ≈ 1.97× per-forward × n-gram ~2.5× ≈ **~5×**.
- Ladder: n-gram (0-cost) → vanilla draft = shipped qwen3.5-0.8B / Qwen3-0.6B (~2×, no training) →
  **EAGLE-3 head** (train via Red Hat Speculators, Qwen3 supported, 3–5×, accept 0.80–0.88).

## Integration path (on-device, PipelinedBench)
1. Add a `PB_SPEC` mode: extend `chunkStep` to resolve the output as `[1, q, vocab]` and return all q rows;
   implement `ngram_lookup` + the accept/rollback loop above; report tok/s + a token-match vs plain greedy
   (must be EXACT — lossless gate). Reuse the LFM #2 bundle already on the A19 (no re-export).
2. Prove n-gram on a structured prompt (code/JSON) where `ā` is high; then wire the vanilla-draft variant
   (second engine feeding a small model's argmax as the draft); then EAGLE-3.
3. Port the same loop into the real inference engine (CoreAIChat / runtime) once validated.

## Risks / open items
- **SSM rollback cost**: restore-and-replay adds a small `[1,len(accepted)]` forward per step. If it erodes
  the win, keep per-token conv/rec snapshots (K small states) and truncate instead. Measure first.
- **Pure-attention models** (if any flagship is GQA-only) have trivial rollback (KV counter only) — cheapest.
  Qwen3.6/LFM are HYBRID (conv+rec) → need the snapshot path. GLM-4.7 (MLA, no SSM) → KV-only rollback.
- **Draft quality drives `ā`**: n-gram wins only on input-grounded tasks; EAGLE-3 is the general 3–5×.
- Verify is a `[1,K]` forward = the same weight read as one decode step (weights dominate; the extra K−1
  positions add ~0 weight bytes), so a rejected round costs ≈ 1 normal step — spec-decode is ~never a loss.

## Why this is the max-speed lever
Axis-1 kernels cap at ~2× (byte floor). Spec-decode is the only lever that cuts the FORWARD COUNT, so it
stacks multiplicatively — the ~3× that turns "~2× flagship" into "~5–6× flagship". Feasibility is high
(verify-forward native; machinery exists); the build is host-loop + rollback, not a kernel or re-export.
```
