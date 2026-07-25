#!/usr/bin/env python
"""P2b: interleave-4 repack for the A19 wide-load kernels.

Rewrites pack/ -> pack_il/ with every main-model QP tensor (L*.{wq,wk,wv,wo,
gate,up,down}.qp + lm_head.qp) permuted from row-major [N, kw] words to
4-row-interleaved [N/4, kw, 4]: word w of rows 4g..4g+3 sits at one uint4.
The R=4 kernels then fetch all four rows' words with a single 16 B load —
per-row word VALUES (and therefore every dot-product order) are unchanged,
so the gates' bit-exactness argument carries over verbatim.

Drafter (dft.*) tensors, scales/biases and everything else stay flat; tensor
offsets are preserved (in-place permutation), so the manifest only gains
per-tensor "il4": true + meta "interleave4": true.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "pack"
DST = HERE / "pack_il"

IL_RE = re.compile(r"^(L\d\d\.(wq|wk|wv|wo|gate|up|down)\.qp|lm_head\.qp)$")


def main() -> None:
    DST.mkdir(exist_ok=True)
    manifest = json.loads((SRC / "gemma4_pack.json").read_text())
    blob = bytearray((SRC / "gemma4_pack.bin").read_bytes())

    n_il = 0
    bytes_il = 0
    for name, info in manifest["tensors"].items():
        if not IL_RE.match(name):
            continue
        assert info["dtype"] == "i32", name
        n, kw = info["shape"]
        assert n % 4 == 0, name
        off, nb = info["offset"], info["nbytes"]
        arr = np.frombuffer(bytes(blob[off:off + nb]), dtype=np.uint32).reshape(n, kw)
        il = np.ascontiguousarray(arr.reshape(n // 4, 4, kw).transpose(0, 2, 1))
        blob[off:off + nb] = il.tobytes()
        info["il4"] = True
        n_il += 1
        bytes_il += nb

    manifest["meta"]["interleave4"] = True
    (DST / "gemma4_pack.bin").write_bytes(bytes(blob))
    (DST / "gemma4_pack.json").write_text(json.dumps(manifest))
    print(f"interleaved {n_il} tensors ({bytes_il / 1e6:.0f} MB of "
          f"{len(blob) / 1e9:.2f} GB) -> {DST}")


if __name__ == "__main__":
    main()
