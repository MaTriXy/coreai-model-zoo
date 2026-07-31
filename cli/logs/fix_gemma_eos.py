"""Fix EOS-NOT-EMITTED-BY-TEMPLATE on the published Gemma bundles.

The bundles declare `eos_token: "<eos>"` (id 1, document end) while their chat template
ends a turn with `<end_of_turn>` (id 106). swift-transformers derives its stop token from
`tokenizer_config.eos_token` alone, so a generic app never sees the turn terminator and
runs to maxTokens. Upstream `generation_config.eos_token_id` is `[1, 106]`; 106 is the one
that ends a turn.

Same fix `export_gemma4_12b_decode_pipelined.py::save_tokenizer` already applies at source
for Gemma-4 (which uses `<turn|>` for the same role).

The edit is a single byte-range replacement so the rest of the file — including a 1 MB
`added_tokens_decoder` — stays byte-identical.

    python3 fix_gemma_eos.py            # dry run: report what would change
    python3 fix_gemma_eos.py --apply    # download, patch, upload, verify
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OLD = '"eos_token": "<eos>"'
NEW = '"eos_token": "<end_of_turn>"'
TERMINATOR = "<end_of_turn>"

TARGETS = [
    ("mlboydaisuke/gemma-3-4b-it-CoreAI-official", "macos/tokenizer/tokenizer_config.json"),
    ("mlboydaisuke/gemma-3-12b-it-CoreAI-official", "macos/tokenizer/tokenizer_config.json"),
    ("mlboydaisuke/functiongemma-270m-coreml", "hf_model/tokenizer_config.json"),
]

COMMIT = (
    "Stop chat turns on <end_of_turn>, not <eos>\n\n"
    "tokenizer_config declared eos_token = \"<eos>\" (id 1, document end), but the chat "
    "template ends a turn with <end_of_turn> (id 106) and upstream generation_config lists "
    "eos_token_id = [1, 106]. A runtime that derives its stop token from eos_token alone "
    "(swift-transformers does) never saw the turn terminator and generated to the token cap.\n\n"
    "Render-safe: the template emits the terminator literally, not via {{ eos_token }}, so "
    "rendering is unchanged. Only this one field differs; the rest of the file is byte-identical."
)

STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
EOS_VAR = re.compile(r"\beos_token\b")


def preflight(api: HfApi, repo: str, path: str) -> tuple[str, str]:
    """Return (current text, chat template). Raises if any assumption does not hold."""
    text = Path(hf_hub_download(repo, path)).read_text()
    n = text.count(OLD)
    if n != 1:
        raise SystemExit(f"{repo}: found {n} occurrences of {OLD} — refusing to guess")

    files = api.list_repo_files(repo)
    prefix = path.rsplit("tokenizer_config.json", 1)[0]
    tj = next(f for f in files if f.startswith(prefix) and "chat_template" in f)
    template = Path(hf_hub_download(repo, tj)).read_text()

    if TERMINATOR not in template:
        raise SystemExit(f"{repo}: template does not emit {TERMINATOR}")
    if EOS_VAR.search(STRING.sub("", template)):
        raise SystemExit(f"{repo}: template renders the eos_token VARIABLE — not render-safe")

    vocab = Path(hf_hub_download(repo, prefix + "tokenizer.json")).read_text()
    if f'"content": "{TERMINATOR}"' not in vocab and f'"{TERMINATOR}"' not in vocab:
        raise SystemExit(f"{repo}: {TERMINATOR} is not in the tokenizer vocabulary")
    return text, template


def main() -> None:
    apply = "--apply" in sys.argv
    api = HfApi()
    for repo, path in TARGETS:
        text, _ = preflight(api, repo, path)
        patched = text.replace(OLD, NEW)
        assert len(patched) - len(text) == len(NEW) - len(OLD)
        print(f"== {repo}")
        print(f"   {path}: {OLD}  ->  {NEW}")
        print(f"   {len(text)} bytes, 1 field changed, {TERMINATOR} verified in template + vocab")
        if not apply:
            print("   (dry run — pass --apply to upload)")
            continue

        api.upload_file(
            path_or_fileobj=patched.encode(),
            path_in_repo=path,
            repo_id=repo,
            commit_message=COMMIT.split("\n", 1)[0],
            commit_description=COMMIT.split("\n\n", 1)[1],
        )
        check = Path(hf_hub_download(repo, path, force_download=True)).read_text()
        ok = NEW in check and OLD not in check and len(check) == len(patched)
        print(f"   uploaded and re-read: {'VERIFIED' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{repo}: verification failed")


if __name__ == "__main__":
    main()
