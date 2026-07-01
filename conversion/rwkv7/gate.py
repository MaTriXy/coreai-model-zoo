# Community port — NOT an Apple model.
"""S2 oracle gate: the Core AI engine (decode .aimodel) vs the validate_ref pure-torch
decode for RWKV-7. TEACHER-FORCED top-1 — feeds the SAME fixed token stream (prompt +
the ref's greedy continuation) to both, S=1 step by step, and compares argmax per step +
max|Δlogits|. recState/shiftState NDArrays mutate in place (the O(1) fixed-size recurrent
state — NO KV cache). Teacher-forcing isolates per-step numerical fidelity from free-running
path divergence (a borderline EOS near-tie flips one argmax and forks the whole tail).

GPU compute unit (the CPU delegate rejects the graph: CoreAICompiler error 3); the engine
runs under ~/code/coreai/_GPU_LOCK (macOS-27 beta GPU panic under parallel load).

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/rwkv7/gate.py <aimodel_path> [n]
"""
import asyncio
import importlib.util
import os
import shutil
import sys
import time

import numpy as np
import torch

import coreai.runtime as rt
from coreai_models.models.macos.rwkv7 import build_decode_state, rwkv7_from_hf

GPU_LOCK = os.path.expanduser("~/code/coreai/_GPU_LOCK")

REF = "/Users/majimadaisuke/code/coreai/coreai-models-community/conversion/rwkv7/validate_ref.py"
_spec = importlib.util.spec_from_file_location("validate_ref", REF)
ref_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ref_mod)

PROMPTS = [
    "Tell me about the moon.",
    "What is 17 + 25?",
    "The three primary colors are",
    "Write a short poem about the sea.",
]


def acquire_gpu_lock(timeout=240):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            os.mkdir(GPU_LOCK)
            return
        except FileExistsError:
            time.sleep(2)
    raise TimeoutError("could not acquire _GPU_LOCK")


@torch.no_grad()
def ref_stream(ref, emb, ids, n):
    """Greedy ref continuation, then the per-step argmax + logits over the full
    teacher-forced stream (prompt + continuation)."""
    ref.reset()
    logits = None
    for t in ids:
        logits = ref.step(emb[t])
    cont = []
    for _ in range(n):
        nxt = int(logits.argmax())
        cont.append(nxt)
        if nxt == 0:
            break
        logits = ref.step(emb[nxt])
    seq = ids + cont[:-1]
    ref.reset()
    args, lgs = [], []
    for t in seq:
        lg = ref.step(emb[t])
        args.append(int(lg.argmax()))
        lgs.append(lg)
    return seq, args, lgs


async def engine_stream(fn, cfg, seq, ref_args, ref_lgs, state_dtype=torch.float32):
    st = build_decode_state(cfg, dtype=state_dtype)
    rec = rt.NDArray(np.ascontiguousarray(st["rec_state"].numpy()))
    shift = rt.NDArray(np.ascontiguousarray(st["shift_state"].numpy()))
    agree, pmax = 0, 0.0
    for i, t in enumerate(seq):
        res = await asyncio.wait_for(fn(
            inputs={"input_ids": rt.NDArray(np.array([[t]], dtype=np.int32)),
                    "position_ids": rt.NDArray(np.array([[0]], dtype=np.int32))},
            state={"recState": rec, "shiftState": shift}), timeout=120)
        elg = torch.from_numpy(np.asarray(res["logits"].numpy())[0, -1].copy())
        agree += int(elg.argmax() == ref_args[i])
        pmax = max(pmax, (elg - ref_lgs[i]).abs().max().item())
    return agree, pmax


async def main():
    aimodel = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    # State dtype must match the exported graph's state inputs: only the fp32 export
    # uses fp32 states; fp16 / int8 / int4 exports all carry fp16 states.
    state_dtype = torch.float32 if "fp32" in aimodel else torch.float16

    snap = ref_mod.find_snapshot()
    cfg_d, sd = ref_mod.load_weights(snap)
    ref = ref_mod.RWKV7Ref(cfg_d, sd)
    emb = sd["model.embeddings.weight"]
    tok = ref_mod.load_world_tokenizer(snap)
    cfg = rwkv7_from_hf(snap, torch.float32).config

    acquire_gpu_lock()
    try:
        m = await rt.AIModel.load(
            aimodel,
            rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        fn = m.load_function("main")

        tot_agree = tot_steps = 0
        gmax = 0.0
        for p in PROMPTS:
            ids = [0] + tok(f"User: {p}\n\nAssistant:").input_ids
            seq, r_args, r_lgs = ref_stream(ref, emb, ids, n)
            agree, pmax = await engine_stream(fn, cfg, seq, r_args, r_lgs, state_dtype)
            tot_agree += agree
            tot_steps += len(seq)
            gmax = max(gmax, pmax)
            print(f"{'OK ' if agree == len(seq) else '~~ '}{p!r:42s} top1 {agree}/{len(seq)}  "
                  f"max|Δ|={pmax:.3e}", flush=True)
        print(f"\n=== engine vs torch teacher-forced top1: {tot_agree}/{tot_steps}  "
              f"global max|Δlogits|={gmax:.3e} ===")
    finally:
        shutil.rmtree(GPU_LOCK, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
