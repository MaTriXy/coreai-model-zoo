# X post drafts (user posts; understated, technical, link is incidental)

## A — the port

Nemotron-3-Nano 4B (NVIDIA's Mamba2 + attention + MLP hybrid) running on Core AI,
on an iPhone 17 Pro.

42 blocks: 21 Mamba2 mixers, 17 MLPs, 4 NoPE attention layers. int8, 4.3 GB,
AOT-compiled. 16.0 tok/s decode, 85.2 on an M4 Max. Greedy output is
token-identical to the fp32 transformers rollout.

No custom kernel. At S=1 the selective scan is a single recurrence step, so the
decode graph is loop-free and lowers as-is.

https://huggingface.co/mlboydaisuke/Nemotron-3-Nano-4B-CoreAI

---

## B — the 4-bit trilemma (probably the most useful thing here)

Wanted this 4B under 3 GB. Tried every 4-bit scheme I had. All three fail, and
each one fails for a *different* reason:

  int4 symmetric      27-29/33 oracle top-1   dequants at 2.20 ms/step (int8: 2.31)
  int4 asymmetric     33/33                   3.0 tok/s on device (int8: 16.0)
  int4 k-means (LUT)  22/33                   — worst, and it's the one the fast kernel reads

Symmetric is fast but wrong. Asymmetric is right — token-identical greedy to int8
over 48 tokens — but the zero-point costs ~0.4 ms per layer, so 42 layers of it
lose 4.6x while reading 1.45x fewer bytes. k-means shares 16 centroids across
32 rows x K columns, no scale along K; that was exact on another model, worst here.

The fast path and the correct path are different schemes. So the lever isn't a
better quantizer, it's a fused *asymmetric* int4 matvec kernel. Nobody has one,
because the LUT format exists precisely to avoid the affine path.

---

## C — the bandwidth model correction (small, and I got it wrong first)

I predicted 10-11 tok/s from the bundle size. It did 16.0.

The embedding table is a one-row gather, not a matmul. It doesn't get read per
token. Drop it and the per-token read is 3.78 GB — at ~60 GB/s that's a 15.9 tok/s
ceiling. Measured 16.0. Saturated.

The trap: on a **tied**-embedding model the lm_head *is* the embedding table, so
it does get read every token, and "bundle size / bandwidth" happens to be right.
That's why the estimate worked on the last model and was 40% low on this one.
Check whether the head is tied before you trust it.

---

## D — the honest one

I wrote a fused Metal kernel for the Mamba2 decode step. It measured 1.3x slower
per layer than the graph the compiler already emits, so I wrote that up: "the
optimizer beats the hand kernel."

It doesn't. I had two hypotheses for the loss and both were wrong. Occupancy? The
kernel ran 1536 threads, each serializing a 128-wide reduction; spreading that
across a simdgroup gave 32x the threads and bought nothing. Dispatch overhead? Two
kernels per layer are *cheaper* than one — which no launch-cost model can produce.

It was the boundary. A custom-kernel op is a fusion barrier, and my call site did
`state.float()` on the way in and `.half()` on the way out: a fp32 blow-up of the
recurrent state, ~3 MB per layer that the stock graph never pays. Hand it across in
fp16 and the kernel wins by 5%.

Two adjacent kernels are cheap because they share one boundary. That's the whole
lesson, and I had to measure three things to find it.

---

Recommendation: **B**, then **A** a day or two later.

B is the only one that changes what someone else does tomorrow. "The 4-bit scheme
your kernel can run and the 4-bit scheme that keeps your accuracy are not the same
scheme" is counterintuitive, self-contained, and needs no context about the zoo.
A is the artifact announcement and lands better once B has framed why int8.
C is a good reply-thread addendum to A (it explains the 16 tok/s). D is inward-facing
— true, useful, but it's a process lesson, not a result.

Notes:
- No zoo pitch. Links are incidental.
- All numbers measured: iPhone 17 Pro (A19 Pro) cooled, PipelinedBench, AOT h18p;
  M4 Max via raw AIModel calls. Quality = teacher-forced top-1 vs the fp32 oracle at
  the 33 margin-clean prompt positions.
- D's figures are granite-4.0-h-350m: 0.161 ms/mamba-layer stock, 0.209 with the fp32
  boundary, 0.180 with the fp16 one; full model 1.01–1.07x over three runs.
- Card: zoo/nemotron-3-nano.md
