"""Build the STATIC Sortformer forward_for_export graph (fixed shapes), validate it in eager
against the NeMo oracle for both captured chunks, then export to Core AI and gate CPU + GPU.

Static layout (fixed-buffer / async convention):
  chunk_mel [1, TF_MAX=1520, 128]   host zero-pads each mel chunk to 1520  -> pre_encode -> 190 embs
  spkcache  [1, SPK=188,   512]     host-maintained speaker cache (zero-padded to 188)
  valid     [1, T=378]              1 for real frames, 0 for pad. Two blocks:
                                      spkcache: [0 : spkcache_len]
                                      chunk:    [188 : 188 + chunk_pe_len]
  -> preds  [1, 378, 4]             speaker activity; host reads spkcache-preds [0:spkcache_len]
                                    and chunk-preds [188 : 188+chunk_pe_len].

Run (MAIN coreai-models venv; _GPU_LOCK for the engine gate):
    coreai-models/.venv/bin/python export_sortformer.py [--dtype float16] [--skip-export]
"""
from __future__ import annotations
import argparse
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sortformer_model import Sortformer, load_ckpt, build_pos_emb

HERE = Path(__file__).resolve().parent
CKPT = str(HERE / "_nemo" / "model_weights.ckpt")
TF_MAX, SPK, PE_MAX, T = 1520, 188, 190, 378
MEL = 128


class ExportWrapper(nn.Module):
    """Static graph with pos_emb baked; inputs = (chunk_mel, spkcache, valid)."""

    def __init__(self, core: Sortformer):
        super().__init__()
        self.core = core
        self.register_buffer("pos_emb", build_pos_emb(T), persistent=False)

    def forward(self, chunk_mel, spkcache, valid):
        return self.core(chunk_mel, spkcache, valid, self.pos_emb)


def cos_pt(a, b):
    return F.cosine_similarity(a.reshape(-1).float(), b.reshape(-1).float(), dim=0).item()


def build_inputs(d, c, dtype):
    """Static inputs + the oracle preds slice + graph-valid-position slice for chunk `c`."""
    mel_chunk = torch.from_numpy(d[f"{c}_mel_chunk"]).float()      # [1,Tf,128]
    spk_len = int(d[f"{c}_spkcache"].shape[1])                     # 0 or 188
    spk = torch.from_numpy(d[f"{c}_spkcache"]).float()            # [1,spk_len,512]
    chunk_pe_len = int(d[f"{c}_chunk_pe"].shape[1])               # 189 or 82
    preds_g = torch.from_numpy(d[f"{c}_preds"]).float()          # [1, spk_len+chunk_pe_len, 4]

    chunk_mel = torch.zeros(1, TF_MAX, MEL)
    chunk_mel[:, : mel_chunk.shape[1]] = mel_chunk
    spkcache = torch.zeros(1, SPK, 512)
    spkcache[:, :spk_len] = spk

    valid = torch.zeros(1, T)
    valid[:, :spk_len] = 1.0
    valid[:, SPK : SPK + chunk_pe_len] = 1.0

    # graph output positions holding the oracle preds (concat_embs order = [spkcache | chunk]):
    gpos = list(range(spk_len)) + list(range(SPK, SPK + chunk_pe_len))
    return (chunk_mel.to(dtype), spkcache.to(dtype), valid.to(dtype)), preds_g, torch.tensor(gpos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    d = np.load(os.path.join(HERE, "chunk_io.npz"))
    core = Sortformer().eval()
    load_ckpt(core, CKPT)
    wrap = ExportWrapper(core).eval()
    print("[weights] strict load OK")

    # ---- eager fp32 static gate ----
    worst = 1.0
    for c in ("c0", "c1"):
        (chunk_mel, spkcache, valid), preds_g, gpos = build_inputs(d, c, torch.float32)
        with torch.no_grad():
            preds, _ = wrap(chunk_mel, spkcache, valid)               # [1,378,4], chunk_pe
        got = preds[0][gpos]                                          # [N,4] at valid positions
        cst = cos_pt(got, preds_g[0])
        maxd = (got - preds_g[0]).abs().max().item()
        print(f"[eager fp32] {c}: valid={gpos.numel()} cos {cst:.6f} max|Δ| {maxd:.5f}")
        worst = min(worst, cst)
    print(f"[eager fp32] worst cos {worst:.6f} -> {'OK' if worst > 0.999 else 'DIVERGES'}")
    if worst <= 0.999:
        raise SystemExit(1)
    if args.skip_export:
        return

    # ---- export to Core AI ----
    import asyncio
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    wrap_d = ExportWrapper(Sortformer().eval())
    load_ckpt(wrap_d.core, CKPT)
    wrap_d = wrap_d.to(dtype).eval()
    example = {
        "chunk_mel": torch.zeros(1, TF_MAX, MEL, dtype=dtype),
        "spkcache": torch.zeros(1, SPK, 512, dtype=dtype),
        "valid": torch.ones(1, T, dtype=dtype),
    }
    print(f"[export] Sortformer forward_for_export ({args.dtype}) -> Core AI ...", flush=True)
    prog = export_to_coreai(
        wrap_d, example, dynamic_shapes=None,
        input_names=("chunk_mel", "spkcache", "valid"), output_names=("preds", "chunk_pe"),
        state_names=None, externalize_modules=[])
    prog.optimize()
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    aimodel = art / f"sortformer_{args.dtype}.aimodel"
    shutil.rmtree(aimodel, ignore_errors=True)
    meta = rt.AIModelAssetMetadata()
    meta.license = "cc-by-4.0"
    prog.save_asset(aimodel, meta)
    sz = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {aimodel} ({sz:.1f} MB)")

    async def gate(unit):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        m = await rt.AIModel.load(str(aimodel), opts)
        fn = m.load_function("main")
        ok_all = True
        for c in ("c0", "c1"):
            (chunk_mel, spkcache, valid), preds_g, gpos = build_inputs(d, c, dtype)
            res = await fn({
                "chunk_mel": rt.NDArray(chunk_mel.numpy()),
                "spkcache": rt.NDArray(spkcache.numpy()),
                "valid": rt.NDArray(valid.numpy()),
            })
            preds = torch.from_numpy(res["preds"].numpy().astype(np.float32))[0]
            _ = res["chunk_pe"]  # second output present; validated end-to-end in gate_e2e_engine.py
            got = preds[gpos]
            cst = cos_pt(got, preds_g[0])
            maxd = (got - preds_g[0]).abs().max().item()
            ok = cst > 0.999 and maxd < 0.05
            ok_all &= ok
            print(f"[gate {unit}] {c}: cos {cst:.6f} max|Δ| {maxd:.4f} -> {'PASS' if ok else 'FAIL'}")
        return ok_all

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
