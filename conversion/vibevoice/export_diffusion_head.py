# Community port — NOT an Apple model.
"""Export the VibeVoice diffusion head (prediction_head) + acoustic_connector to Core AI .aimodel
and engine-gate vs the golden oracle.

  diffusion head: (noisy[2,64], timesteps[2], condition[2,896]) -> eps[2,64]   (CFG batch-2)
  connector:      (latent[1,1,64]) -> embed[1,1,896]

Both are tiny continuous-feedback FF graphs -> fp16 (quant-sensitive, VoxCPM/dots lesson).

  PYTHONPATH=. <coreai-venv>/bin/python export_diffusion_head.py
"""
from __future__ import annotations
import argparse, asyncio, shutil, subprocess, sys, glob
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from torch_overlays import DiffusionHeadOverlay, ConnectorOverlay  # noqa: E402

ART = HERE / "artifacts"
REVISION = "6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"  # the revision this port was gated against
SNAP = hf_snapshot("microsoft/VibeVoice-Realtime-0.5B", revision=REVISION)


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


async def _engine(aim, ref, names, out_name):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    inp = {n: rt.NDArray(np.ascontiguousarray(ref[n].numpy())) for n in names}
    r = await fn(inputs=inp)
    return r[out_name].numpy()


async def run(DT):
    z = np.load(ART / "oracle_ref.npz")
    sd = load_file(glob.glob(SNAP + "/*.safetensors")[0])
    results = {}

    # ---------- diffusion head ----------
    head = DiffusionHeadOverlay().to(DT).eval().load_upstream(sd)
    names = ("noisy_images", "timesteps", "condition")
    ref = {"noisy_images": torch.from_numpy(z["dhead.in_noisy"]).to(DT),
           "timesteps": torch.from_numpy(z["dhead.in_t"]).to(DT),
           "condition": torch.from_numpy(z["dhead.in_cond"]).to(DT)}
    with torch.inference_mode():
        t_out = head(*[ref[n] for n in names]).float().numpy()
    print(f"[dhead] torch overlay vs oracle: cos={cos(t_out, z['dhead.out_eps']):.6f}")
    prog = export_to_coreai(head, ref, dynamic_shapes=None, input_names=names,
                            output_names=("eps",), state_names=None)
    aim = _save(prog, ART / "vibevoice_diffusion_head_fp16")
    eng = await _engine(aim, ref, names, "eps")
    c = cos(eng, z["dhead.out_eps"])
    print(f"[dhead] engine vs oracle: cos={c:.6f}  ({_du(aim)})")
    results["dhead"] = c

    # ---------- connector ----------
    conn = ConnectorOverlay().to(DT).eval().load_upstream(sd)
    cnames = ("features",)
    cref = {"features": torch.from_numpy(z["conn.in0"]).to(DT)}
    with torch.inference_mode():
        ct = conn(cref["features"]).float().numpy()
    print(f"[conn]  torch overlay vs oracle: cos={cos(ct, z['conn.out']):.6f}")
    cprog = export_to_coreai(conn, cref, dynamic_shapes=None, input_names=cnames,
                             output_names=("embed",), state_names=None)
    caim = _save(cprog, ART / "vibevoice_connector_fp16")
    ceng = await _engine(caim, cref, cnames, "embed")
    cc = cos(ceng, z["conn.out"])
    print(f"[conn]  engine vs oracle: cos={cc:.6f}  ({_du(caim)})")
    results["conn"] = cc

    return results


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    DT = torch.float16 if a.dtype == "fp16" else torch.float32
    res = await run(DT)
    ok = all(c >= 0.999 for c in res.values())
    print(f"\n>>> diffusion_head+connector {a.dtype}: " +
          "  ".join(f"{k}={v:.6f}" for k, v in res.items()) +
          f" -> {'GATE PASS' if ok else 'GATE FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
