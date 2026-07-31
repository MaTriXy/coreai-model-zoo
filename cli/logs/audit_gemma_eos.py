"""Audit every published Gemma CoreAI repo for EOS-NOT-EMITTED-BY-TEMPLATE.

Read-only: fetches tokenizer_config.json and the chat template only.
"""

import json
import re
import sys

from huggingface_hub import HfApi, hf_hub_download

STRING = re.compile(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"")
EOS_VAR = re.compile(r"\beos_token\b")

api = HfApi()
repos = sorted(
    m.id for m in api.list_models(author="mlboydaisuke", limit=200)
    if "gemma" in m.id.lower()
)

for repo in repos:
    try:
        files = api.list_repo_files(repo)
    except Exception as exc:
        print(f"{repo}: cannot list ({type(exc).__name__})")
        continue
    configs = [f for f in files if f.endswith("tokenizer_config.json")]
    if not configs:
        print(f"{repo}: no tokenizer_config.json (not a chat bundle)")
        continue
    for tc in configs:
        prefix = tc.rsplit("tokenizer_config.json", 1)[0]
        cfg = json.load(open(hf_hub_download(repo, tc)))
        eos = cfg.get("eos_token")
        eos = eos.get("content") if isinstance(eos, dict) else eos
        template = cfg.get("chat_template")
        tj = next((f for f in files
                   if f.startswith(prefix) and "chat_template" in f), None)
        if not isinstance(template, str) and tj:
            template = open(hf_hub_download(repo, tj)).read()
        if not isinstance(template, str):
            print(f"{repo}  {tc}: eos={eos!r}  NO CHAT TEMPLATE")
            continue
        literal = eos in template
        variable = bool(EOS_VAR.search(STRING.sub("", template)))
        markers = [m for m in ("<end_of_turn>", "<turn|>", "<|im_end|>")
                   if m in template]
        verdict = "clean" if (literal or variable) else "FIRES"
        print(f"{repo}  {tc}")
        print(f"    eos={eos!r} literal={literal} var={variable} "
              f"turn-markers={markers}  => {verdict}")
