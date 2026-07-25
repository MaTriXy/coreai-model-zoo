# Catalog plan — making the published catalog reproducible and checkable

> **Subordinate to [`../ZOO_BLUEPRINT.md`](../ZOO_BLUEPRINT.md).** The blueprint decides *what
> the zoo ships and why*; this plan only makes what is already shipped reproducible and
> verifiable. Where the two disagree, the blueprint wins.
>
> Status: **C0, C1, C2 and the layout half of C4 are done** (2026-07-25). C3 (oracles)
> and C4.1 (device tier) are not started.
> No step here publishes anything — see [Guardrails](#guardrails).

## Why this is not a new pillar

The blueprint's pillars need this work but do not describe it:

| Blueprint pillar | What it assumes | What this plan supplies |
| --- | --- | --- |
| **P1 Model coverage** — keep shipping models Apple doesn't | that a shipped port stays shipped across SDK betas | a check that runs over the whole catalog in minutes after each beta |
| **P5 Knowledge** — the porting playbook | that "how it was made" is written down | the machine-readable half of it: `models/<model>/recipe.toml` |
| **P6 Community ops** — monthly drops, contributions | that a contributor can reproduce a bundle without asking | `zoo_convert.py run <name>` plus the prerequisites it needs |

**Numbering:** the blueprint owns `P1`–`P6` for pillars. This plan's phases are `C0`–`C4` so
that "P2" means one thing in both documents. (The earlier draft of this file reused P0–P4 and
collided with the blueprint's pillar numbers.)

## What the catalog actually looks like

Measured 2026-07-25 by `scripts/gen_inventory.py`; the full table is
[`models/_INVENTORY.md`](models/_INVENTORY.md).

| Layer | Count |
| --- | --- |
| Published Hugging Face repos | **123** (122 owned + 1 contributor-owned) |
| Of those, Core AI repos | **70** (the rest: pre-Core-AI Core ML ports, LiteRT collaboration repos) |
| Bundles inside them | **238** |
| Core AI repos with a card in `models/<model>/` | **52** |
| Repos with a recipe | **52** (was 6) |
| Bundles with an automated tier-1 check | **222** (was ~0) |
| Core AI repos with no downloads in 30 days | **55** |

Ten of the uncarded repos are the `-CoreAI-official` bench exports of Apple's own recipes —
blueprint P2, not undocumented ports. Eight are zoo ports published without a card; those are
listed for the owner.

## The three things that blocked reproduction

1. **Scripts hardcoded one machine's home directory.** 47 files, 69 occurrences — not the 15
   the earlier draft counted. Fixed in C0.1.
2. **Prerequisites were prose.** `notes = ["Runtime needs apps/…patch + COREAI_CHUNK_THRESHOLD=1"]`
   is invisible to a runner. Fixed in C2 by splitting them into fields, and by distinguishing
   *export-time* prerequisites from *run-time* ones — a bundle rebuilt without the runtime
   patch looks correct and then misbehaves in the app.
3. **The shipped configuration was often unrecorded.** Fixed where it could be derived; where it
   could not, the entry says so instead of guessing.

## Phases

### C0 — make the existing tree honest ✅ done

- **C0.1 Remove hardcoded paths.** 47 files now resolve through `conversion/_paths.py`
  (`ZOO_WORK_ROOT` / `ZOO_EXPORTS` / `ZOO_CODE_ROOT` / `HF_HUB_CACHE`, with defaults that
  reproduce the layout the published bundles were built in).
  *Accepted*: `grep -rln "/Users/<name>" conversion/` returns nothing.
- **C0.2 Resolve working-tree drift.** Far more than the 6 modified files and 2 untracked
  directories the draft listed: 75 uncommitted source files, including the exporters for
  several shipped bundles. Committed as source; oracle dumps, HF staging directories and the
  reassembled upstream VoxCPM reference are covered by per-port `.gitignore` files. Nothing was
  deleted.
  *Accepted*: `git status --porcelain conversion` is empty.
- **C0.3 Inventory.** `models/_INVENTORY.md`, generated — do not hand-edit.
  *Accepted*: every published repo has a row with downloads, bundle count, card, recipe, kit
  enrollment and tier-1 result.

### C1 — tier-1 verification ✅ done

`conversion/zoo_verify.py` checks four things per bundle with no oracle, no device and no
weights: eos/bos, chat template, context length, declared precision.

**Change from the draft:** expectations are **read from the source repository at run time**
rather than transcribed into 50 hand-written `verify.toml` files. A transcription can be wrong
and goes stale; the source repo cannot. `models/<model>/verify.toml` is now only for recording a
*deliberate* deviation — and once recorded, the recorded value becomes the bar.

First full run over 222 bundles: **162 PASS, 8 DIFF, 10 FAIL, 42 SKIPPED**.

*Accepted*: the defect list exists and Gemma-4-12B and 31B are on it with eos mismatches.

**But the known-answer test's premise was wrong, and the result is more useful than the test.**
The draft expected 12B/31B to be shipping `eos_token: "<eos>"`. They are not: they ship
`<turn|>`, Gemma 4's turn terminator — they are the two that were *fixed*. E2B and E4B still
ship `<eos>`, which a host loop stops on only at end-of-sequence, never at end-of-turn. Both
sets are flagged (12B/31B as an undeclared deviation from source, E2B/E4B as an `eos vs eot`
finding), so the acceptance test passes either way — but the defect is in the models the draft
called clean. Verification earns its keep by contradicting the plan that asked for it.

The 10 FAILs are all Gemma 4 E2B/E4B bundles shipping **no chat template at all** while their
source ships one; E2B is the most-downloaded text model in the catalog.

### C2 — recipes for the carded catalog ✅ done

6 → **56 recipes**, 38 `verified` and 18 `unverified`, one `recipe.toml` per model
directory beside its card.

Recipes are **derived, not remembered**: the exporters build their bundle name from their
arguments, so a published name inverts back to a command, and where a card also documents a
command the two are checked against each other. `verified` means both agree (or the naming rule
determines the arguments outright). `unverified` means a flag changes the artifact but not the
name — `--head-sym` on the MoE gather exports, the mode behind a bare `decoder/` directory —
and the entry records the known part plus the exact question.

*Accepted*: `zoo_convert.py run <name> --dry-run` prints a complete command plus prerequisites
for all 38 verified recipes; the unverified ones refuse to run without `--force`.

### C3 — oracles and tiers 2–3 (not started)

Order strictly by 30-day downloads from `_INVENTORY.md`. The 55 zero-download Core AI repos are
last, and some of them are candidates for unpublishing rather than verifying.

- **C3.1** Generalise the per-model scripts in `_smoke/` into one oracle generator and one
  comparator.
- **C3.2** Generate oracles and publish them beside each bundle under `oracle/`.
  **Each upload is user-gated** (blueprint: pushes and external posts are user-gated).
- **C3.3** Fill `[numeric]` and `[generation]` expectations. Tolerances come from measurement,
  per architecture — never a global default.

Mac-GPU work in this phase takes the `_GPU_LOCK` at the work root
(`conversion/_paths.py: gpu_lock()`), per the blueprint's parallel-session rule.

*Accept*: the top 20 models by downloads pass tiers 1–3 or have a recorded reason they cannot.

### C4 — device tier and layout (C4.2/C4.3 done; C4.1 not started)

- **C4.1** Wire the nightly device gate to a `[device]` expectation. `require_backend` must fail
  on silent CPU fallback, not warn.
- **C4.2 Layout ✅ done.** Cards moved to `models/<model>/README.md` beside their
  `recipe.toml`, mirroring `apple/coreai-models` (which is `models/<family>/README.md` +
  optional `export.py` / `*.yaml`). Two deliberate differences: the exporters stay in
  `conversion/` because several families share one (the Qwen3.5 script also drives Ornith and
  Qwen3.6-27B), and `recipe.toml` / `verify.toml` are our additions — Apple has no equivalent.
  The gen-cards conflict was resolved rather than avoided: `cards.json` now points at the new
  paths, so the "Use it" block still round-trips byte-identically to the Hugging Face README.
  Every old `zoo/<model>.md` path stays as a redirect stub, because ~50 published Hugging Face
  READMEs link to it. **`scripts/gen-cards` has not been re-run** — it builds Swift and needs
  the kit checkout; the owner should run it once to confirm the round-trip.
- **C4.3 ✅ done.** `skills/` mirrors Apple's agent-plugin layout (`.claude-plugin`,
  `.codex-plugin`, `gemini-extension.json`, `skills/<name>/SKILL.md`) with two skills:
  `reproduce-a-zoo-model` and `port-a-model-to-the-zoo`. `models/index.json` is the
  machine-readable catalog they read first.

*Accept*: an agent given only this repo's README can reproduce and verify one model end to end.

## Open questions for the owner

Ordered by cost of being wrong. The full lists live in `models/_INVENTORY.md`.

1. **10 FAIL bundles** — Gemma 4 E2B/E4B ship no chat template. Republish with one, or is the
   host expected to supply it?
2. **`eos` on E2B/E4B** — should they carry `<turn|>` like 12B/31B, or is `<eos>` deliberate?
3. **18 unverified recipes** — one flag answer each (mostly "was `--head-sym` passed?").
4. **8 zoo ports with no card** — write the card, or unpublish the repo?
5. **A published metadata leak** — the MinerU layout decoder's `hf_model_id` on Hugging Face is
   a local absolute path from this machine.
6. **Two cards point at the wrong exporter** — the qwen3.6-35B and GLM-4.7-Flash cards name the
   pre-gather-kernel script, which produces a bundle that was not the one published.
7. **`README.md` claims** every model "ships with the recipe that produced it". True for 52 of
   70 Core AI repos now, with 18 of those recipes marked unverified. The sentence should either
   be qualified or the gap closed.

## Instructions for the agent

- Work one phase at a time and run its acceptance test before moving on. If it fails, fix it;
  do not redefine the test. If the test's *premise* turns out to be wrong (as C1's did), report
  that rather than adjusting the result to match.
- **Never guess a shipped configuration.** `status = "unverified"` plus a precise question is
  the correct output when the repo does not record it.
- Read before writing. `PORTING.md` is the path, `knowledge/*.md` the depth,
  `conversion/overlay/README.md` the model-code patch mechanism, and the model's own card is
  usually the record of what shipped.
- One-off scripts (`_*_hf_upload.py`) are often the only copy of a publishing step. Confirm
  with the owner before deleting one.
- Code comments and documentation in English.

## Guardrails

- **No publishing.** No Hugging Face pushes, no posts, no PRs against `apple/*`. Uploading
  oracles in C3.2 requires explicit approval each time.
- **No deletion of artifacts.** Never `rm` bundles, oracles or export outputs; several are the
  only copy in existence. Ignore them in git instead.
- **Keep the repo small.** No models, checkpoints, build output, `.venv` or caches.
- **Do not change export hyperparameters while migrating.** A recipe must reproduce the
  published bundle, not improve it. Improvements are a separate, owner-approved change.
- Verification tiers that cannot run report `skipped`. Never `pass` for a tier that did not
  execute.
