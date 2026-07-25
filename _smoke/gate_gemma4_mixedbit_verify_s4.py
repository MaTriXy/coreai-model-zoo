# Community port — NOT an Apple model.
"""Gate the S=4 verify bundle (python GPU runtime) against the P3 fp16 oracle ids.

France anchor ("What is the capital of France?", 16 prompt ids = 4 exact chunks;
fp16 oracle continuation 8/8 id-exact vs the delegate in P3). The verify graph
must reproduce the greedy continuation with PERFECT drafts: prefill via 4-token
chunks, then MTP-style rounds where the drafts are the known next tokens — every
verify row argmax must equal the oracle id. Also sanity-checks the activations
and k/v-row outputs (finite, expected shapes).

Keep it SHORT (each new position_ids length costs a ~10 s GPU respecialization;
long python-runtime runs can wedge the MTL4 queue — P3 lesson).

Run: cd coreai-models && ../coreai-models-community/_GPU_LOCK etiquette, then
  .venv/bin/python ../coreai-models-community/_smoke/gate_gemma4_mixedbit_verify_s4.py
"""
import asyncio
import json
import sys
from pathlib import Path

import numpy as np

BUNDLE = "exports/gemma4_e2b_mixedbit_verify_s4"
REFS = "exports/gemma4_e2b_mixedbit_decode/oracle_refs.json"
N_SLOTS, HD_MAX, MAX_SEQ = 15, 512, 2048


async def run() -> bool:
    import coreai.runtime as rt

    refs = json.loads(Path(REFS).read_text())["prompts"]["What is the capital of France?"]
    prompt = refs["prompt_ids"]          # 16 ids
    cont = refs["fp16"]                  # 8 ids, starts 818
    assert len(prompt) % 4 == 0

    m = await rt.AIModel.load(
        f"{BUNDLE}/gemma4_e2b_mixedbit_verify_s4.aimodel",
        rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()),
    )
    fn = m.load_function("main")
    state = {
        "keyCache": rt.NDArray(np.zeros((N_SLOTS, 1, 1, MAX_SEQ, HD_MAX), np.float16)),
        "valueCache": rt.NDArray(np.zeros((N_SLOTS, 1, 1, MAX_SEQ, HD_MAX), np.float16)),
    }
    print("engine loaded", flush=True)

    async def verify(tokens: list[int], processed: int):
        pos = np.arange(processed + 4, dtype=np.int32).reshape(1, -1)
        res = await fn(inputs={
            "input_ids": rt.NDArray(np.array(tokens, np.int32).reshape(1, 4)),
            "position_ids": rt.NDArray(pos),
        }, state=state)
        logits = res["logits"].numpy()          # [1,4,V]
        acts = res["activations"].numpy()       # [1,4,1536]
        rows = {k: res[k].numpy() for k in ("k11_rows", "v11_rows", "k14_rows", "v14_rows")}
        return logits, acts, rows

    # prefill: exact 4-chunks
    ok = True
    for c in range(len(prompt) // 4):
        logits, acts, rows = await verify(prompt[4 * c:4 * c + 4], processed=4 * c)
        assert np.isfinite(acts).all() and all(np.isfinite(r).all() for r in rows.values())
    a0 = int(logits[0, -1].argmax())
    print(f"prefill kickoff a0={a0} (ref {cont[0]})", flush=True)
    ok &= a0 == cont[0]

    # rounds with perfect drafts from the oracle continuation
    processed = len(prompt)
    idx = 0                                    # cont[idx] == next expected token (== a0)
    for r in range(2):
        feed = [cont[idx + i] if idx + i < len(cont) else 0 for i in range(4)]
        n_check = min(4, len(cont) - idx - 1)
        logits, acts, rows = await verify(feed, processed=processed)
        got = [int(logits[0, i].argmax()) for i in range(4)]
        exp = [cont[idx + 1 + i] for i in range(n_check)]
        match = got[:n_check] == exp
        print(f"round {r}: fed={feed} got={got[:n_check]} exp={exp} "
              f"{'OK' if match else 'MISMATCH'}", flush=True)
        ok &= match
        # activations sanity: kickoff row norm in the plausible range
        an = float(np.linalg.norm(acts[0, -1]))
        print(f"  activations[-1] norm={an:.1f}  k11_rows shape={rows['k11_rows'].shape}",
              flush=True)
        ok &= 10.0 < an < 2000.0
        processed += 4
        idx += 4
        if idx + 1 >= len(cont):
            break
    return ok


def main():
    ok = asyncio.run(run())
    print("VERIFY-S4 GATE", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
