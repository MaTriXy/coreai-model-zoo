#!/usr/bin/env python3
"""The parts of an export script that are not the recipe — written once.

Every export driver ends the same way: write `metadata.json` next to the `.aimodel`, and
copy the tokenizer in beside it. That tail was copy-pasted into 28 drivers, and the copies
drifted, because nothing made them agree. What the drift actually cost:

  * `write_bundle_metadata` existed in ten variants differing by one key each — a pinned
    revision, a `weights` provenance string, a second entry point. Each new need forked the
    whole function instead of adding an argument, and one fork broke its own caller:
    `export_gemma4_mixedbit_verify_pipelined.py` loads the decode driver as a module and
    calls its `write_bundle_metadata` with five arguments, but that copy grew a sixth
    required parameter (`weights_source`). It raises `TypeError` — *after* `save_asset`, so
    the cost is the whole export. It is called once, at the end, which is why it survived.
  * `save_tokenizer` existed in **seven** variants differing only in the `allow_patterns`
    handed to `snapshot_download` and the filter applied to the result. Three of them are
    internally contradictory: the filter accepts `special_tokens_map.json` while the
    patterns never request it, so the file cannot arrive. That is latent rather than
    realized — the one published bundle built this way (`models/qwen3.6`) comes from a
    checkpoint that ships no `special_tokens_map.json`, so nothing was lost — but the next
    model that ships one would lose it silently. Two other variants copy *everything*
    downloaded, which puts a stray `README.txt` in the bundle's tokenizer directory.
    None of the seven loses the chat template: `chat_template*` matches `.jinja` in all of
    them. Worth stating, because it is the failure you would assume.
  * `head_quant_spec` existed in three variants that were the *same function*: two were the
    `block32` case of the third with `sym` fixed, spelled out by hand.

`linear_quant_config` is deliberately **not** here, even though 32 drivers define one. What
a driver quantizes, what it leaves in fp16, and why, *is* the recipe — the thing a reader
opens the driver to learn. Twenty of the 32 are genuinely different, and hiding the twelve
that agree behind an import would move the interesting half of the file out of view to save
a dozen lines. The rule this module follows: extract the tail, keep the recipe.

Known inconsistency, preserved on purpose: most drivers write `"compression": null` even
when the bundle is int8, because they never passed a mode. Published bundles carry that
null, so the default here reproduces it rather than quietly correcting the record. Pass
`mode=` to declare the scheme (two drivers do).
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# What a tokenizer directory may consist of, across every model this repository converts.
# The union of the seven hand-written variants. `snapshot_download` needs the globs; the
# copy step needs the name test as well, because `*.txt` also matches a repository's
# README, which does not belong in a bundle.
_TOKENIZER_PATTERNS = [
    "tokenizer*", "*.txt", "chat_template*", "*.jinja",
    "special_tokens*", "vocab*", "merges*",
]
_TOKENIZER_FILES = {
    "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json",
}


def is_tokenizer_file(name: str) -> bool:
    """Whether a file downloaded by `_TOKENIZER_PATTERNS` belongs in the bundle."""
    return (name.startswith("tokenizer") or name.startswith("chat_template")
            or name.endswith(".jinja") or name in _TOKENIZER_FILES)


def write_bundle_metadata(
    out_dir: Path,
    name: str,
    hf_id: str,
    vocab_size: int,
    max_ctx: int,
    *,
    functions: Sequence[str] = ("main",),
    embedded_tokenizer: bool = True,
    revision: str | None = None,
    weights: str | None = None,
    mode: str | None = None,
    language_extra: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write the `metadata.json` that sits beside `<name>.aimodel` in a bundle.

    `functions` lists the entry points the asset exposes — `("main", "prefill")` for a
    bundle carrying a chunked-prefill function alongside decode. `revision` and `weights`
    record provenance under `source` when the checkpoint is pinned or the weights came
    from somewhere other than the config's own repository. `mode` names the quantization
    scheme; `fp16` and `None` both mean uncompressed. `language_extra` and `extra` add
    keys to the `language` block and the top level for the rare bundle that needs one
    (an `inputs_embeds` entry shape, a verify query length).
    """
    language: dict[str, Any] = {
        "tokenizer": hf_id,
        "vocab_size": vocab_size,
        "max_context_length": max_ctx,
        "embedded_tokenizer": embedded_tokenizer,
    }
    language.update(language_extra or {})
    language["function_map"] = {"main": list(functions)}

    source: dict[str, Any] = {"model_definition": "torch", "hf_model_id": hf_id}
    if revision:
        source["hf_revision"] = revision
    if weights:
        source["weights"] = weights

    meta: dict[str, Any] = {
        "metadata_version": "0.2",
        "kind": "llm",
        "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": language,
        "source": source,
        "compression": None if mode in (None, "fp16") else {"scheme": mode},
    }
    meta.update(extra or {})
    meta["compilation"] = {"date": datetime.now(timezone.utc).isoformat(), "targets": []}

    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def head_quant_spec(gran: str = "block32", sym: bool = False) -> dict:
    """Quantization spec for an untied `lm_head`. Ship shape: `block32`, `sym=True`.

    `sym` picks plain `symmetric` (absmax) over `symmetric_with_clipping` — the
    big-vocab-head rule. Large vocabulary heads are fat-tailed, and clipping craters their
    outlier rows: the signature is a single position dropping to cos 0.62 while its
    neighbours sit at 0.999x, which reads like a random glitch rather than a quantization
    choice. It cost a 6/16 oracle flip on qwen-2B before it was understood; absmax gates
    16/16 on the same weights.

    ⚠️ `gran="perchan"` (per_channel, axis 0) is **broken on the macOS-27-beta MPSGraph GPU
    delegate** — garbage logits, cos ~0 against torch, at any vocab shape and under either
    qscheme, with a minimal head-only reproduction from 2026-06-11. It is a delegate
    lowering bug, not quantization damage: the same graph at per-block-32 is cos 0.9999x.
    The option stays only so a future OS build can be re-tested.
    See `knowledge/compression-reference.md` and `knowledge/pipelined-engine.md`.
    """
    if gran == "perchan":
        g: dict = {"type": "per_channel", "axis": 0}
    else:
        g = {"type": "per_block", "block_size": int(gran[len("block"):]), "axis": 1}
    return {
        "op_state_spec": {
            "weight": {
                "dtype": "int8",
                "qscheme": "symmetric" if sym else "symmetric_with_clipping",
                "granularity": g,
            }
        },
        "op_input_spec": None,
        "op_output_spec": None,
    }


def save_tokenizer(hf_id: str, out_dir: Path, *, via_transformers: bool = True) -> None:
    """Copy `hf_id`'s tokenizer into `<out_dir>/tokenizer`.

    `via_transformers` tries `AutoTokenizer.save_pretrained` first and falls back to
    copying the raw files. Pass `False` for a model whose `model_type` the installed
    transformers does not know, where the fallback is the only path that ever runs — and
    for any model whose tokenizer files must reach the bundle *verbatim*, since
    `save_pretrained` re-serializes them.
    """
    if via_transformers:
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained(hf_id).save_pretrained(out_dir / "tokenizer")
            return
        except Exception as e:  # noqa: BLE001 — any failure means: copy the files instead
            print(f"AutoTokenizer failed ({e}); copying raw tokenizer files")

    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(hf_id, allow_patterns=_TOKENIZER_PATTERNS))
    (out_dir / "tokenizer").mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and is_tokenizer_file(f.name):
            shutil.copy2(f, out_dir / "tokenizer" / f.name)
