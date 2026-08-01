# `cli/` — coreai export / doctor / verify

Three commands for the part of a Core AI port that is knowledge rather than code: which
route a model has, which known trap an artifact is standing on, and whether the bundle still
speaks.

```
python3 cli/coreai_export.py <hf-id | short-name | checkpoint-dir> [--device mac|iphone] [--run]
python3 cli/coreai_export.py --list           # the whole support matrix

python3 cli/coreai_doctor.py <bundle-dir | *.aimodel | *.aimodelc | checkpoint-dir | hf-id | *.py>
python3 cli/coreai_doctor.py --rules          # every rule, machine-readable

python3 cli/coreai_verify.py <bundle-dir> [-n 16] [--prompt "..."] [--transcript out.json]
python3 cli/coreai_verify.py <bundle-dir> --plan     # what it would do, and what blocks it

python3 cli/selftest.py                       # decision-rule fixtures
```

The three compose in the obvious order and share their readers — `verify` gets the graph
facts it routes on from `doctor`, and `export` runs `doctor`'s checkpoint rules as a
pre-flight and refuses `--run` on a fatal or silent finding.

Stdlib only for local targets. `huggingface_hub` is needed for an `org/name` target
(config files and safetensors *headers* only — no weights are downloaded).
`xcrun coreai-build inspect` is used for the graph-level rules when the Xcode 27 toolchain
is present, and skipped with a note when it is not. `export` reads Apple's tables out of the
installed `coreai_models` and falls back to a dated snapshot, loudly, if it cannot.

---

# `export` — the router

**It does not convert anything.** It answers, before you spend an hour finding out the hard
way: does this model have a route, through which backend, does that route have an iOS path,
and what exactly is unvalidated about it.

| backend | meaning |
|---|---|
| `preset` | Apple's stock exporter with a named preset for this exact checkpoint. Precision, compression and context length are all resolved, and Apple has run the combination. |
| `generic` | Apple's stock exporter routing by HF `model_type` only. It runs. Nothing about the recipe is validated for *these* weights. |
| `zoo` | A recorded community recipe. Reproduces a bundle that shipped and gated. |
| `none` | The `model_type` does not route. Not a CLI problem — a new architecture needs a re-authored model class. Saying so plainly is the output. |

Default is print-and-stop; `--run` executes, and refuses if the route is blocked or if
doctor's checkpoint pre-flight found a fatal or silent-corruption pattern.

### What it can route today

| | |
|---|---:|
| Apple named presets (validated combinations) | 12 checkpoints |
| Apple `model_type` — generic, unvalidated | 19 values |
| **zoo recorded recipes** | **55 source checkpoints** |
| zoo ports whose upstream is not on the Hub (RF-DETR, YOLOX, AdcSR, …) | 5 |
| unresolved | 0 |

The zoo number was **13** until this pass. Not because 42 models were unroutable — because
`recipe.toml` records `hf_repo`, which is the *output* repo, and only 13 recipes happened to
name the model they convert *from*. The other 42 record it somewhere else: the export
script's argparse default, or the model card's upstream link. Reading those three places
instead of one closes it. The durable fix is a `source_hf_id` field in the zoo's recipe
schema; this reads around its absence rather than editing that repo.

### The number that motivates the generic tier

Apple's stock exporter accepts **14 `model_type`s, and only 6 have an iOS path**
(mistral, olmo2, phi3, qwen2, qwen3, smollm3 — plus `llama`→`mistral` and `qwen2_5`→`qwen2`,
which is why most plain Llama checkpoints route). Gemma-3, Gemma-4, gpt-oss, Mixtral,
Qwen3-MoE, Qwen3-VL and Qwen3.5 are macOS-only.

Nothing tells you that up front. The exporter's own `--dry-run` resolves
`gemma-3-4b-it --platform iOS` without a murmur; the failure is
`raise ValueError("Model 'gemma3' does not support iOS variant")` at
`export/pipeline.py:150`, reached only after `AutoConfig` has read the checkpoint. `export`
turns that into a `BLOCKED` line before anything downloads.

### Validation

| log | what it shows |
|---|---|
| `export-routing-cases.txt` | seven targets covering every branch: preset (exit 0), generic-with-iOS (0), zoo ×2 (0), iOS-cliff (2), silent-preflight (1), no-route (2) |
| `export-commands-resolve.txt` | the emitted commands run through the exporter's own `--dry-run` and resolve; plus the `pipeline.py:150` raise site behind the `BLOCKED` claim |

`export --run` has since produced a real bundle end to end (`end-to-end-qwen3-0.6b.txt`),
which `doctor` then found clean and `verify` gated 16/16 against the fp32 oracle. Export is
convert-only — no `AIModel.load`, no `SpecializationOptions` anywhere in
`export/pipeline.py` — so it does not contend for the exclusive GPU. Running the *bundle*
does, which is why `verify` checks the lock and `export` does not.

`--verify-tables` diffs the vendored fallback snapshot against the installed
`coreai_models` and exits non-zero on drift, so a stale table is loud rather than silent.

---

# `verify` — the gate

**A bundle that loads is not a port.** This drives the bundle and the reference over the
same ids and compares them, with the two rules the notes insist on:

- **Validate the prompt before the bundle.** Every oracle position must clear a top-2
  margin floor (0.1) in fp32. A near-tie is a coin flip that healthy int8 noise flips and
  fp16 passes by luck — a 14/16 there gates nothing, in either direction. This *refuses*
  such a prompt rather than scoring against it. It is computable from the oracle alone,
  before a bundle exists.
- **Judge a divergence by the margin, not by the divergence.** A first mismatch below the
  floor is an fp16 knife-edge tie; above it, a real disagreement.

Two backends, chosen automatically. `zoo` — the family has a hand-transcribed fp32 oracle
in `conversion/coreai_gate.py`, which is the authority for those models, so this prints the
delegated command instead of keeping a second copy to drift. `stock` — everything else,
whose reference is plain `transformers` and needs no overlay. That second case is the gap:
`coreai_gate.py` covers 7 zoo families and cannot gate a stock-recipe bundle at all.

Which driver can run the bundle is a property of the **graph**: a dynamic-shaped logits
output cannot be executed by the Python runtime, so it must go through `llm-runner` — which
means the GPU, and therefore the exclusive-GPU convention. Both are checked before anything
long-running starts, and `_GPU_LOCK` being held stops the run rather than contending with it.

### It immediately caught something

The canonical gate prompt — `"The capital of France is"`, the one this repo recommended and
shipped as the default — **fails its own margin rule at n=16** on Qwen3-0.6B: positions 1
and 5 sit at 0.0885 and 0.0041. It is deterministic at the *first* token and not over a
16-token continuation, because after "Paris." the model free-runs into a list where the next
country is a near-tie.

The recommendation and the margin rule were both in the notes and had conflicted at n=16 for
as long as both existed. Nothing surfaced it until a tool checked the prompt instead of
trusting it.

**Fixed 2026-08-01.** The default in `coreai_verify.py` and `conversion/coreai_gate.py` is
now `"The alphabet begins A, B, C, D, E, F,"`. Measured across two model families at n=16,
fp32, before changing it — a counting sequence was also tried and rejected, which is why the
obvious-looking alternative is not the one that shipped:

| prompt | Qwen3-0.6B | SmolLM2-360M |
| --- | --- | --- |
| `"The capital of France is"` (old default) | ✗ min 0.0041 | ✗ min 0.0172 |
| `"Counting up: 1, 2, 3, 4, 5, 6,"` | ✓ 0.6500 | ✗ min 0.0289 |
| `"The alphabet begins A, B, C, D, E, F,"` | ✓ **0.9585** | ✓ **0.9351** |

Reciting a fixed sequence holds because there is nothing to free-run into. Counting drifts
once the numbers get long enough to admit a second plausible formatting.

### Validation

| log | what it shows |
|---|---|
| `end-to-end-qwen3-0.6b.txt` | export → doctor → verify on one model: real bundle produced, lint clean, **16/16 token-exact against the fp32 oracle** |
| `verify-validation.txt` | the rejected canonical prompt (exit 3), the full passing run (exit 0), and the zoo delegation |
| `verify-qwen3-0.6b-transcript.json` | the transcript: input ids, both sides' output, per-step margins, verdict, environment |
| `selftest.txt` | 18 source-rule checks + 8 over the verdict rule, including both sides of the margin floor |

---

# `doctor` — the lint

Reads an artifact, reports the known failure patterns it matches, and cites where each one
is written down.

## Why

A conversion that errors is cheap. The expensive class is the one where `torch.export`
succeeds, the bundle loads, the model generates fluent text — and the numbers are wrong, or
the app never stops generating, or it works on your Mac and produces garbage on the phone.
Nothing in the log says so.

The rules are the accumulated bodies. `DOCTOR_RULES.md` is the table: 64 patterns, each with
the symptom as you actually experience it, how to detect it mechanically, and a citation.
45 of them run. It also explains what doctor is *not* — `conversion/zoo_verify.py` checks a
bundle against its source repo, which is a different question and catches different things.

## What it reads

| scope | target | catches |
|---|---|---|
| asset | `.aimodel` / `.aimodelc` directory | IR provenance, AOT staleness, symlink traps |
| graph | via `coreai-build inspect --ops --json` | state count, IO shapes, op distribution, vocab agreement |
| bundle | LanguageBundle directory | runtime contract, tokenizer class, chat surface, eos |
| checkpoint | HF checkpoint directory or repo id | quant recipe, activation scales, eos, block divisibility |
| source | PyTorch modelling code | the converter and delegate op traps |
| env | the working directory | the one env defect whose output is a bad asset |

## Output shape

Findings split into **DEFECTS** (something is wrong with the artifact) and **NOTES AND SHIP
REQUIREMENTS** (the artifact is fine and its host must do something specific, or it breaks).
Only defects affect the exit status: `2` for fatal/silent, `1` for runaway/perf, `0`
otherwise. That split matters — a healthy, device-gated 4.6 GB bundle legitimately comes
back with four requirements and zero defects, and a tool that called that a failure would
get muted.

## Validation

`logs/` holds the runs that back the claims:

| log | what it shows |
|---|---|
| `case-a-known-broken.txt` | a 0.4.0-era bundle: 1 fatal, 1 runaway, exit 2 |
| `case-a-ground-truth-load-abort.txt` | the same asset actually aborting at `AIModel.load`, so the fatal is not an assertion |
| `case-b-known-good.txt` | the device-gated nanbeige4.2-3B ship bundle: 0 defects, 4 ship requirements, exit 0 |
| `case-c-source-lint-rf-detr.txt` | the source lint over stock `transformers` RF-DETR, independently re-finding the patterns that port hit |
| `case-d-checkpoint-wna8o8.txt` | the Gemma-4 mobile QAT checkpoint, flagged from its safetensors headers before any export |
| `case-e-published-gemma3-eos.txt` | the eos rule swept across all 18 published Gemma tokenizer configs, **after** the fix below — none fire |
| `selftest.txt` | 18 fixture checks over 16 source rules |

A sweep over all 90 local bundles reported findings on 41. Every finding class in that
sweep was hand-verified against the artifact before this was written; the false positives
found on the way (a vision encoder held to the LanguageBundle contract, a ship manifest
that merely shares the name `metadata.json`, a Jinja template that renders `eos_token`
mid-expression, and the *working* `div(x, 2, rounding_mode="floor")` form) are fixed and
covered by fixtures. A lint that flags the documented workaround is worse than no lint.

### First real catch: three published repos

The sweep found `EOS-NOT-EMITTED-BY-TEMPLATE` live on Hugging Face —
`gemma-3-4b-it-CoreAI-official`, `gemma-3-12b-it-CoreAI-official` and
`functiongemma-270m-coreml` all declared `eos_token: "<eos>"` (id 1, document end) while
their chat template ends a turn with `<end_of_turn>` (id 106, and upstream
`generation_config.eos_token_id` is `[1, 106]`). Any runtime that derives its stop token
from `eos_token` alone — swift-transformers does — generated to the token cap.

Fixed 2026-07-31 by `logs/fix_gemma_eos.py` (one field, byte-range replacement, verified on
re-read). `logs/audit_gemma_eos.py` re-swept all 18 published Gemma tokenizer configs
afterwards: none fire.

Worth noting what the audit *also* corrected. `GEMMA4_12B_STATE.md` had warned since July
that the published Gemma-4 12B/31B bundles still carried the old `<eos>`; they did not — the
note was stale, and acting on it would have been wasted work. Reading the artifacts beat
reading the note about the artifacts.

The same sweep flagged `CHAT-TEMPLATE-MISSING` on the four legacy
`gemma-4-E{2,4}B*-coreml` ports, which shipped a tokenizer and no template — a runtime
applying one had nothing to apply and fell back to raw completion without a word. They also
carried the `<eos>` defect. Fixed 2026-07-31 by `logs/fix_gemma4_coreml_chat.py`, which
ships `google/gemma-4-{E2B,E4B}-it`'s own template verbatim; both sizes serve the same file
and it is byte-identical to the one the Core AI Gemma-4 bundles already carry, so this
adopts a decision already made rather than making a new one.

All 18 published Gemma tokenizer configs now come back clean. The two that still report no
chat template are the embeddinggemma repos — an embedding model has no chat surface, which
is the rule reporting correctly, not a gap.

## Where this lives

`cli/` inside the zoo, alongside the `conversion/` scripts and the `knowledge/` notes the
rules are transcribed from. That is deliberate and reversible:

- **The rules are `knowledge/` transcribed.** In one repo a knowledge update and the rule it
  implies are one commit. Across two repos the table silently falls behind.
- **`conversion/zoo_convert.py doctor` was already here** and checks the *environment*. Two
  commands named `doctor` in one workflow is a defect, so they were reconciled rather than
  left to coexist: `coreai_doctor.py --env` runs the same overlay probe, making the artifact
  lint a superset, and `zoo_convert.py doctor` now points at it.
- **The discoverability payoff is on `export`, not `doctor`.** The zoo already carries
  traffic, `llms.txt` and the AIO surface.

Path resolution is location-independent — `find_zoo_root()` walks up for the directory
holding both `models/` and `conversion/`, and falls back to the sibling layout — so lifting
`cli/` out into a standalone `coreai-cli` repo later is a directory move and nothing else.
Do that when `export` routes beyond the set Apple and the zoo already cover; until then a
standalone repo would be a thinner front door than this one.

## Status

All three commands run, and the chain has been exercised end to end on one model: Qwen3-0.6B
exported through the router, linted clean, gated 16/16 against its fp32 reference. Nothing
here is published.

The honest boundary on `export`: it routes over the set Apple already supports plus the
zoo's recorded recipes. It does **not** widen that set, and the kickoff's framing of
answering coreai-models#56 ("model-by-model support does not seem sustainable") is only
half-answered by it — the other half is "how do you make a new architecture's re-authoring
cheap", which is not a CLI feature. The README should keep saying so rather than letting the
command's name imply otherwise.
