"""End-to-end ENGINE generation check: feed the oracle's assembled prefix embeddings
into the unified bundle's `prefill` function, then run REAL autoregressive greedy decode
(not teacher-forced) via the `decode` function until EOS, detokenize, and print the
markdown. Confirms the shipped Core AI bundles actually generate correct OCR text.

Run:  cd ~/code/coreai/coreai-models && PYTHONPATH=../_unlimited_ocr \
          .venv/bin/python ../_unlimited_ocr/generate.py [--bundle DIR] [--max N]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from model import DECODE_STATE_NAMES, build_decode_state, ocr_decoder_from_safetensors

HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpt"
REF = HERE / "out/_oracle/oracle_tensors.npz"
CACHE_LEN = 2048
DTYPE = torch.float16
EOS = 1


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=str(HERE / "out/_dec_unified"))
    ap.add_argument("--max", type=int, default=600)
    ap.add_argument("--no-repeat-ngram", type=int, default=0, help="0=greedy; else block repeated n-grams")
    args = ap.parse_args()

    import coreai.runtime as rt
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    model = ocr_decoder_from_safetensors(str(CKPT), target_dtype=DTYPE)  # embed table for decode tokens
    Lm = int(np.load(REF)["dec_embeds"].shape[1])
    dec = torch.from_numpy(np.load(REF)["dec_embeds"]).to(DTYPE)  # [1,Lm,1280]

    aimodel = Path(args.bundle) / f"{Path(args.bundle).name}.aimodel"
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = await rt.AIModel.load(str(aimodel), gpu)
    fn_prefill, fn_decode = m.load_function("prefill"), m.load_function("decode")

    state = {DECODE_STATE_NAMES[0]: rt.NDArray(np.ascontiguousarray(
                 build_decode_state(model.config, CACHE_LEN, DTYPE)["k_cache"].numpy())),
             DECODE_STATE_NAMES[1]: rt.NDArray(np.ascontiguousarray(
                 build_decode_state(model.config, CACHE_LEN, DTYPE)["v_cache"].numpy()))}

    def argmax_blocked(logits: np.ndarray, gen: list[int], n: int) -> int:
        if n <= 0 or len(gen) < n - 1:
            return int(np.argmax(logits))
        order = np.argsort(logits)[::-1]
        prefix = tuple(gen[-(n - 1):])
        banned = {gen[i + n - 1] for i in range(len(gen) - n + 1) if tuple(gen[i:i + n - 1]) == prefix}
        for t in order:
            if int(t) not in banned:
                return int(t)
        return int(order[0])

    t0 = time.time()
    resp = await fn_prefill(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(dec.numpy()))}, state=state)
    logits = resp["logits"].numpy().astype(np.float32).reshape(-1)
    gen: list[int] = []
    tok_id = argmax_blocked(logits, gen, args.no_repeat_ngram)
    with torch.no_grad():
        for p in range(Lm, Lm + args.max):
            gen.append(tok_id)
            if tok_id == EOS:
                break
            emb = model.model.embed_tokens(torch.tensor([[tok_id]])).to(DTYPE)  # [1,1,1280]
            pos = torch.tensor([p], dtype=torch.int32)
            res = await fn_decode(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(emb.numpy())),
                                          "pos": rt.NDArray(np.ascontiguousarray(pos.numpy()))}, state=state)
            tok_id = argmax_blocked(res["logits"].numpy().astype(np.float32).reshape(-1), gen, args.no_repeat_ngram)
    dt = time.time() - t0

    body = [t for t in gen if t != EOS]
    text = tok.decode(body)  # skip_special_tokens=True (markdown view)
    raw = tok.decode(body, skip_special_tokens=False)  # shows <|det|>/<tr>/<td> structure
    (HERE / "out/_engine_ocr.md").write_text(raw)
    print(f"[generate] {len(gen)} tokens in {dt:.1f}s ({len(gen)/dt:.0f} tok/s), EOS={'yes' if gen[-1]==EOS else 'NO(hit max)'}", flush=True)

    oracle_ids = np.load(REF)["token_ids"][0][Lm:].tolist()
    oracle_ids = [t for t in oracle_ids if t != EOS]
    match = sum(a == b for a, b in zip(gen[:len(oracle_ids)], oracle_ids))
    print(f"[vs oracle] greedy-prefix token match: {match}/{min(len(body),len(oracle_ids))} "
          f"(oracle used no_repeat_ngram=35, so divergence after a repeat is expected)", flush=True)
    print("\n" + "=" * 70 + "\n[ENGINE OCR OUTPUT — raw (structure tokens shown)]\n" + "=" * 70)
    print(raw)

    rep = HERE / "out/_oracle/equiv_report.json"
    if rep.exists():
        oracle_text = json.load(open(rep)).get("oracle_text", "")
        print("\n" + "=" * 70 + f"\n[ORACLE TEXT] ({len(oracle_text)} chars)\n" + "=" * 70)
        print(oracle_text[:1500])


if __name__ == "__main__":
    asyncio.run(main())
