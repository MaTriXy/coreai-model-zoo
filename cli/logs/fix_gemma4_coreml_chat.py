"""Give the legacy Gemma-4 Core ML ports a chat template, and a stop token that works.

Four repos ship `hf_model/` with a tokenizer but no chat template, and declare
`eos_token: "<eos>"` (id 1, document end) while the model ends a chat turn with `<turn|>`
(id 106; the E4B repos' own generation_config lists eos_token_id = [1, 106, 50]).

Both defects are silent. A runtime with no template falls back to raw completion without
warning, so the model never sees turn markers; a runtime that stops on eos_token alone
never sees the turn end and generates to the cap. `cli/coreai_doctor.py` reports them as
CHAT-TEMPLATE-MISSING and EOS-NOT-EMITTED-BY-TEMPLATE.

The template is `google/gemma-4-{E2B,E4B}-it`'s own, verbatim. Both sizes serve the same
file (18569 bytes, sha256 0a2c8073…) and it is byte-identical to the one the shipped
CoreAI Gemma-4 bundles already carry — the 2026-07-09 canonical revision this repo adopted
on 2026-07-25. So this is not a new editorial choice, it is the one already made.

    python3 fix_gemma4_coreml_chat.py            # dry run
    python3 fix_gemma4_coreml_chat.py --apply
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OLD_EOS = '"eos_token": "<eos>"'
NEW_EOS = '"eos_token": "<turn|>"'
TERMINATOR = "<turn|>"
STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
EOS_VAR = re.compile(r"\beos_token\b")

TARGETS = {
    "mlboydaisuke/gemma-4-E2B-coreml": "google/gemma-4-E2B-it",
    "mlboydaisuke/gemma-4-E2B-stateful-coreml": "google/gemma-4-E2B-it",
    "mlboydaisuke/gemma-4-E4B-coreml": "google/gemma-4-E4B-it",
    "mlboydaisuke/gemma-4-E4B-multimodal-coreml": "google/gemma-4-E4B-it",
}

COMMIT_TITLE = "Ship the chat template, and stop turns on <turn|>"
COMMIT_BODY = (
    "hf_model/ carried a tokenizer but no chat template, so a runtime applying one had "
    "nothing to apply and silently fell back to raw completion — the model never saw a turn "
    "marker. And tokenizer_config declared eos_token = \"<eos>\" (id 1, document end) while "
    "this model ends a chat turn with <turn|> (id 106); generation_config lists "
    "eos_token_id = [1, 106, 50]. A runtime that stops on eos_token alone therefore never saw "
    "the turn end and generated to its cap.\n\n"
    "The template is the source model's own file, verbatim, and is byte-identical to the one "
    "the Core AI Gemma-4 bundles already ship. It emits <turn|> literally rather than through "
    "{{ eos_token }}, so retagging eos_token does not change rendering."
)


def canonical_template(source: str) -> bytes:
    return Path(hf_hub_download(source, "chat_template.jinja", force_download=True)).read_bytes()


def main() -> None:
    apply = "--apply" in sys.argv
    api = HfApi()

    for repo, source in TARGETS.items():
        template = canonical_template(source)
        digest = hashlib.sha256(template).hexdigest()[:16]

        # The template must emit the terminator literally, or retagging eos_token would
        # change what gets rendered.
        assert TERMINATOR.encode() in template, f"{source}: template does not emit {TERMINATOR}"
        assert not EOS_VAR.search(STRING.sub("", template.decode())), \
            f"{source}: template renders the eos_token variable — retagging is not render-safe"

        cfg_path = Path(hf_hub_download(repo, "hf_model/tokenizer_config.json"))
        cfg_text = cfg_path.read_text()
        n = cfg_text.count(OLD_EOS)

        # The terminator must exist in THIS repo's vocabulary, or the id will not resolve.
        vocab = json.load(open(hf_hub_download(repo, "hf_model/tokenizer.json")))
        ids = {t["content"]: t["id"] for t in vocab.get("added_tokens") or []}
        assert TERMINATOR in ids, f"{repo}: {TERMINATOR} is not in the tokenizer"

        files = {s.rfilename for s in api.model_info(repo).siblings}
        has_template = "hf_model/chat_template.jinja" in files

        print(f"== {repo}")
        print(f"   template  <- {source}  {len(template)} bytes  sha256 {digest}"
              f"{'  (already present — skipping)' if has_template else ''}")
        print(f"   eos_token {OLD_EOS} -> {NEW_EOS}  ({n} occurrence(s), {TERMINATOR}={ids[TERMINATOR]})")
        if n != 1:
            raise SystemExit(f"{repo}: expected exactly 1 eos_token line, found {n}")
        if not apply:
            continue

        if not has_template:
            api.upload_file(path_or_fileobj=template,
                            path_in_repo="hf_model/chat_template.jinja", repo_id=repo,
                            commit_message=COMMIT_TITLE, commit_description=COMMIT_BODY)
        api.upload_file(path_or_fileobj=cfg_text.replace(OLD_EOS, NEW_EOS).encode(),
                        path_in_repo="hf_model/tokenizer_config.json", repo_id=repo,
                        commit_message=COMMIT_TITLE, commit_description=COMMIT_BODY)

        back_cfg = Path(hf_hub_download(repo, "hf_model/tokenizer_config.json",
                                        force_download=True)).read_text()
        back_tpl = Path(hf_hub_download(repo, "hf_model/chat_template.jinja",
                                        force_download=True)).read_bytes()
        ok = NEW_EOS in back_cfg and OLD_EOS not in back_cfg and back_tpl == template
        print(f"   re-read: {'VERIFIED' if ok else 'MISMATCH'}")
        if not ok:
            raise SystemExit(f"{repo}: verification failed")

    if not apply:
        print("\ndry run — pass --apply to upload")


if __name__ == "__main__":
    main()
