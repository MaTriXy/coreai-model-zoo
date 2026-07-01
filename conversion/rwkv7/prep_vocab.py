# Community port — NOT an Apple model.
"""Emit a Swift-friendly RWKV World vocab: one line per token `<idx>\\t<base64(bytes)>`.

The upstream `rwkv_vocab_v20230424.txt` stores each token as a Python repr (`'...'` utf-8
or `b'...'` raw bytes) that needs `eval()` to parse — fiddly and unsafe to reimplement in
Swift. This bakes the raw token bytes to base64 so `RWKVWorldTokenizer.swift` just
base64-decodes each line and builds the byte trie directly (greedy longest-match).

Self-verifies: rebuilds a trie from the base64 output and checks encode/decode parity vs
the reference RWKV_TOKENIZER on sample strings.

  cd ~/code/coreai && .venv/bin/python \
    coreai-models-community/conversion/rwkv7/prep_vocab.py [--out rwkv_vocab.tsv]
"""
from __future__ import annotations

import argparse
import base64
import glob
import importlib.util
import os


def find_snapshot() -> str:
    root = os.path.expanduser(
        "~/.cache/huggingface/hub/models--RWKV--RWKV7-Goose-World3-1.5B-HF/snapshots")
    return sorted(glob.glob(os.path.join(root, "*")))[-1]


def parse_vocab(vocab_file: str) -> dict[int, bytes]:
    """idx -> token bytes, mirroring RWKV_TOKENIZER.__init__ parsing."""
    idx2tok: dict[int, bytes] = {}
    for line in open(vocab_file, "r", encoding="utf-8"):
        idx = int(line[: line.index(" ")])
        x = eval(line[line.index(" "): line.rindex(" ")])
        x = x.encode("utf-8") if isinstance(x, str) else x
        assert isinstance(x, bytes) and len(x) == int(line[line.rindex(" "):])
        idx2tok[idx] = x
    return idx2tok


# Minimal byte-trie greedy longest-match (the exact Swift port target).
class _Trie:
    def __init__(self):
        self.to: dict[int, _Trie] = {}
        self.value: tuple[bytes, int] | None = None

    def add(self, key: bytes, idx: int):
        u = self
        for ch in key:
            u = u.to.setdefault(ch, _Trie())
        u.value = (key, idx)

    def encode(self, src: bytes) -> list[int]:
        out, i, n = [], 0, len(src)
        while i < n:
            u, j, last = self, i, None
            while j < n and src[j] in u.to:
                u = u.to[src[j]]
                j += 1
                if u.value is not None:
                    last = (j, u.value[1])
            assert last is not None, f"no token at byte {i}"
            out.append(last[1])
            i = last[0]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rwkv_vocab.tsv")
    args = ap.parse_args()
    snap = find_snapshot()
    vocab_file = os.path.join(snap, "rwkv_vocab_v20230424.txt")
    idx2tok = parse_vocab(vocab_file)
    print(f"parsed {len(idx2tok)} tokens (idx {min(idx2tok)}..{max(idx2tok)})")

    with open(args.out, "w") as f:
        for idx in sorted(idx2tok):
            f.write(f"{idx}\t{base64.b64encode(idx2tok[idx]).decode()}\n")
    print(f"wrote {args.out}")

    # --- verify: rebuild trie from base64 tsv, compare to reference tokenizer ---
    rebuilt: dict[int, bytes] = {}
    for line in open(args.out):
        i, b64 = line.rstrip("\n").split("\t")
        rebuilt[int(i)] = base64.b64decode(b64)
    assert rebuilt == idx2tok, "base64 round-trip mismatch"
    trie = _Trie()
    for idx, tok in rebuilt.items():
        trie.add(tok, idx)

    spec = importlib.util.spec_from_file_location(
        "hf_rwkv_tokenizer", os.path.join(snap, "hf_rwkv_tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ref = mod.RWKV_TOKENIZER(vocab_file)

    samples = ["User: Tell me about the moon.\n\nAssistant:", "The three primary colors are",
               "17 + 25 = 42", "日本語のテスト", "emoji 🌙 and symbols ±∞"]
    ok = True
    for s in samples:
        b = s.encode("utf-8")
        mine = trie.encode(b)
        gold = ref.encodeBytes(b)
        dec = b"".join(rebuilt[i] for i in mine).decode("utf-8")
        match = (mine == gold and dec == s)
        ok &= match
        print(f"  {'OK' if match else '!!'} {s!r} -> {len(mine)} toks")
    print("\nVERIFY:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
