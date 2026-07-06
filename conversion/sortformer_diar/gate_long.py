"""Gate the host loop over the LONG clip (stream_ref_long.npz) — this actually exercises AOSC
compress_spkcache across multiple chunks. Runs BOTH the eager torch graph and the shipped fp16
.aimodel (Mac GPU) and compares total_preds to NeMo forward_streaming.

Run (MAIN coreai-models venv):  coreai-models/.venv/bin/python gate_long.py
"""
from __future__ import annotations
import asyncio, os
import numpy as np, torch, torch.nn.functional as F
import coreai.runtime as rt
from host_loop import run, eager_forward
from sortformer_model import Sortformer, load_ckpt
from gate_e2e_engine import engine_forward, AIMODEL
from export_sortformer import CKPT

HERE = os.path.dirname(os.path.abspath(__file__))


def score(tag, total, ref):
    n = min(total.shape[1], ref.shape[1])
    got, exp = total[:, :n], ref[:, :n]
    cos = F.cosine_similarity(got.reshape(-1), exp.reshape(-1), dim=0).item()
    maxd = (got - exp).abs().max().item()
    agree = ((got > 0.5) == (exp > 0.5)).float().mean().item()
    ok = agree > 0.99 and total.shape[1] == ref.shape[1]
    print(f"[{tag}] {tuple(total.shape)} vs {tuple(ref.shape)}  cos {cos:.6f}  max|Δ| {maxd:.5f}  "
          f"activity-agree {agree*100:.2f}%  -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    d = np.load(os.path.join(HERE, "stream_ref_long.npz"))
    mel = torch.from_numpy(d["mel"]).float()
    mel_len = torch.from_numpy(d["mel_len"]).long()
    ref = torch.from_numpy(d["total_preds"]).float()

    model = Sortformer().eval(); load_ckpt(model, CKPT)
    ok1 = score("eager", run(eager_forward(model), mel, mel_len), ref)

    loop = asyncio.new_event_loop()
    opts = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    m = loop.run_until_complete(rt.AIModel.load(AIMODEL, opts))
    fn = m.load_function("main")
    ok2 = score("engine", run(engine_forward(fn, loop), mel, mel_len), ref)
    raise SystemExit(0 if (ok1 and ok2) else 1)


if __name__ == "__main__":
    main()
