# Security

This repository distributes **converted model weights** and the code that produced them. That
makes the interesting question less "is there a vulnerability in the code" and more "can you
tell whether the bytes you loaded are the bytes that were checked." This page answers that
honestly, including where the answer is *no*.

## Reporting

Open a [security advisory](../../security/advisories/new) for anything you would rather not
say in public — a bundle that behaves unlike its source model, a repository that appears to
have been tampered with, a link that resolves somewhere unexpected. Ordinary bugs belong in
[issues](../../issues); use the advisory path when disclosure itself is the risk.

There is no bounty, and this is a community project maintained by one person: expect a reply
in days, not hours.

## What the integrity story actually is

**Pinned revisions, not signatures.** A model is identified by a Hugging Face repository *and
an immutable commit revision*. The revision is what was gated; a later push to that repository
cannot change what a pinned consumer receives. [CoreAIKit](https://github.com/john-rocky/coreai-kit)
resolves every download through the pin, including the sibling subtrees of multi-part models.

**What is not done:** bundles are not code-signed, and there is no per-file checksum manifest
beyond what Hugging Face itself stores. Conversion is not byte-deterministic here — the same
recipe run twice produces bundles that differ — so a checksum of *your* rebuild will not match
the published one by design. Integrity therefore rests on the revision pin and on Hugging
Face's own storage, not on a signature you can verify offline.

**A `.aimodel` bundle is data that a runtime executes.** Treat one from any source the way you
would treat a binary dependency. The scripts in `conversion/` are the provenance: they show
exactly what was built and from which checkpoint.

## Checking a bundle yourself

Two commands, neither of which requires trusting this repository:

```bash
python3 conversion/zoo_verify.py <hf-repo>   # tokenizer, chat template, ctx, precision
                                             # vs the source model the bundle names
python3 conversion/coreai_gate.py <bundle> <hf-id> --revision <sha> --transcript out.json
                                             # rebuild the fp32 reference and compare a
                                             # greedy decode token for token
```

The first is metadata conformance and runs in minutes over the whole catalog. The second is
the numerical check and needs a Mac plus the source checkpoint. Where a gate transcript is
published, re-running only the engine side against it needs neither the oracle nor the fp32
weights — just the bundle and `llm-runner`.

## If you are shipping something you have to support

The maintainer runs the gates; nobody independently re-runs them. That is a real limit, and
the honest advice follows from it:

- **Mirror the bundles you depend on** into storage you control, rather than fetching a
  personal Hugging Face namespace at runtime.
- **Pin the package**, not a branch, and re-verify after any OS or toolchain bump — the
  `coreai-core` wheel is OS-coupled and a beta bump has already invalidated previously
  exported bundles once (FB23666783).
- **Re-run the conversion yourself** from the recipe if the artifact's provenance has to be
  yours. Every recipe exists so that this is possible.

## Scope

In scope: the conversion and verification scripts here, the published bundles and their
recipes, and this repository's own metadata. Model *content* — what a model says, what it was
trained on, its licence and its biases — is upstream's, and each card links the source model.
Compromise of a third-party contributor's Hugging Face namespace is out of our control; report
it and the affected entries will be delisted from the catalog pending resolution.
