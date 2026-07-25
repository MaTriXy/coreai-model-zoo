"""LFM2.5-Audio Milestone A — on-engine gate via the AOT-compiled bundle.

The plain .aimodel JIT-specializes on the Mac GPU by re-compiling per position
length, which sends the ANE compiler into a multi-GB thrash on this OS. The AOT
`.aimodelc` (coreai-build compile --platform macOS --preferred-compute gpu
--expect-frequent-reshapes --architecture h16c) is pre-specialized; load it with
`.default` options (NOT .gpu, which re-JITs and wedges — GLM-Image lesson) and it
runs the s=1 prefill + greedy decode directly.

Run:  coreai-models/.venv/bin/python aot_gate.py <bundle.h16c.aimodelc> \
          [--source oracle,coreai]
"""
import argparse
import asyncio
from pathlib import Path

import numpy as np

import coreai.runtime as rt
import export_lfm2_embeds_decode as L
from coreai_models.models.macos.lfm2 import DECODE_STATE_NAMES, build_decode_state
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN

DTYPE = __import__("torch").float16


def _nd(a):
    return rt.NDArray(np.ascontiguousarray(a))


async def aot_greedy(aimodelc, cfg, pe, embed_w, n_gen):
    # AOT bundle: load pre-specialized graph with default options (no re-JIT).
    m = await rt.AIModel.load(str(Path(aimodelc).resolve()), rt.SpecializationOptions.default())
    fn = m.load_function("main")
    st = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    state = {n: _nd(t.numpy()) for n, t in
             zip(DECODE_STATE_NAMES, [st["k_cache"], st["v_cache"], st["conv_state"]])}
    S = pe.shape[0]
    pe16 = pe.astype(np.float16)
    ew16 = embed_w.to(DTYPE).numpy()

    async def step(emb_row, total_positions):
        ie = _nd(emb_row.reshape(1, 1, -1).astype(np.float16))
        pos = _nd(np.arange(total_positions, dtype=np.int32)[None])
        out = await fn(inputs={"inputs_embeds": ie, "position_ids": pos}, state=state)
        return out["logits"].numpy()

    logits = None
    for i in range(S):
        logits = await step(pe16[i], i + 1)
    t = int(logits[0, -1].argmax()); ids = [t]
    for i in range(n_gen - 1):
        logits = await step(ew16[t], S + i + 1)
        t = int(logits[0, -1].argmax()); ids.append(t)
    return ids


ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("bundle")
ap.add_argument("--source", default="oracle,coreai")
args = ap.parse_args()
sources = [s.strip() for s in args.source.split(",") if s.strip()]

asr_ref = np.load(L.ORACLE / "asr_ref.npz")["ids"].tolist()
n_gen = len(asr_ref)
print(f"oracle asr_ref ({n_gen}): {asr_ref}", flush=True)
cfg = L.load_lfm_config()
embed_w = L.embed_table()
allok = True
for src in sources:
    pe = L.prefill_embeds(src)
    ids = asyncio.run(aot_greedy(args.bundle, cfg, pe, embed_w, n_gen))
    ok = ids == asr_ref
    nm = sum(a == b for a, b in zip(ids, asr_ref))
    allok = allok and ok
    print(f"[aot-gpu {src:6s}] {nm}/{n_gen} {'MATCH OK' if ok else 'DIVERGE'}  {ids}", flush=True)
print("\nVERDICT:", "LFM2-Audio Milestone A on-engine ASR EXACT vs oracle" if allok else "NEEDS REVIEW",
      flush=True)
