# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b2",
#     "coreai-torch>=0.4.1",
#     "torch==2.9.0",
#     "numpy",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
"""Ladder 3: export TimesFmCore -> Core AI .aimodel, gate engine vs HF oracle projections.

Graph = pure feed-forward transformer over patch tokens (fixed N=16 == ctx 512).
Inputs:  tok_in (1,N,64), cos (1,N,80), sin (1,N,80), attn_bias (1,1,N,N)
Outputs: proj_point (1,N,1280), proj_q (1,N,10240)
"""
import argparse, asyncio, shutil, time
from pathlib import Path
import numpy as np
import torch

from timesfm_core import TimesFmCore, load_core_from_safetensors, rope_cos_sin

CFG = dict(patch=32, horizon=128, hidden=1280, layers=20, heads=16, head_dim=80,
           inter=1280, q=9, oql=1024, eps=1e-6)
N = 16  # patches; overridden by --ctx (N = ctx // patch)


def build_inputs_from_oracle(z, dtype):
    """Return list of (name->tensor) dicts for each series' PRIMARY (non-flipped) pass.
    Uses oracle-captured tok_in; pos=arange (no padding), pure causal mask."""
    tok = torch.tensor(z["tok_in"])                       # (3,16,64)
    pos = torch.arange(N, dtype=torch.float32)[None]      # (1,16)
    cos, sin = rope_cos_sin(pos, CFG["head_dim"])          # (1,16,80)
    neg = torch.finfo(torch.float32).min
    causal = torch.triu(torch.full((N, N), neg), 1)[None, None]  # (1,1,16,16)
    items = []
    for i in range(tok.shape[0]):
        items.append(dict(
            tok_in=tok[i:i+1].to(dtype), cos=cos.to(dtype), sin=sin.to(dtype),
            attn_bias=causal.to(dtype)))
    return items


def export(core, dtype, out_path):
    import coreai.runtime as rt
    from coreai_torch import TorchConverter, get_decomp_table
    import copy
    if dtype != torch.float32:
        core = copy.deepcopy(core).to(dtype)
    ex = (torch.zeros(1, N, 64, dtype=dtype), torch.zeros(1, N, 80, dtype=dtype),
          torch.zeros(1, N, 80, dtype=dtype), torch.zeros(1, 1, N, N, dtype=dtype))
    t0 = time.time()
    with torch.no_grad():
        ep = torch.export.export(core, ex)
    ep = ep.run_decompositions(get_decomp_table())
    print(f"[export] torch.export+decomp {time.time()-t0:.1f}s")
    prog = TorchConverter().add_exported_program(
        exported_program=ep,
        input_names=["tok_in", "cos", "sin", "attn_bias"],
        output_names=["proj_point", "proj_q"],
    ).to_coreai()
    prog.optimize()
    shutil.rmtree(out_path, ignore_errors=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = rt.AIModelAssetMetadata()
    meta.author = "Google (TimesFM 2.5); Core AI export: coreai-model-zoo"
    meta.license = "Apache-2.0"
    meta.model_description = (
        "TimesFM 2.5 200M decoder-only time-series forecasting transformer (graph core). "
        "Inputs: patch tokens + RoPE cos/sin + causal mask; outputs: point/quantile projections. "
        "Host does RevIN/flip/quantile-head. https://huggingface.co/google/timesfm-2.5-200m-transformers")
    meta.creation_date = int(time.time())
    prog.save_asset(out_path, meta)
    mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"[convert] saved {out_path} ({mb:.1f} MB)")


def cos_sim(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


async def verify(core, z, dtype, out_path, unit):
    import coreai.runtime as rt
    opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
    model = await rt.AIModel.load(out_path, opts)
    fn = model.load_function("main")
    items = build_inputs_from_oracle(z, dtype)
    cmin = 0.9999 if dtype == torch.float32 else 0.997
    names = z["series_names"]
    worst = 1.0
    for i, it in enumerate(items):
        with torch.no_grad():
            tpp, tpq = core.to(dtype)(it["tok_in"], it["cos"], it["sin"], it["attn_bias"])
        out = await fn({k: rt.NDArray(v.numpy()) for k, v in it.items()})
        epp = out["proj_point"].numpy().astype(np.float32)
        epq = out["proj_q"].numpy().astype(np.float32)
        # engine vs oracle (HF), and engine vs torch-core
        c_pp = cos_sim(epp, z["proj_point_out"][i])
        c_pq = cos_sim(epq, z["proj_q_out"][i])
        c_tp = cos_sim(epp, tpp.float().numpy())
        worst = min(worst, c_pp, c_pq)
        print(f"  [{str(names[i]):8s}] engine-vs-HF point cos={c_pp:.7f} q cos={c_pq:.7f} | engine-vs-torch cos={c_tp:.7f}")
    ok = worst > cmin
    print(f"[verify:{unit}:{dtype}] {'PASS' if ok else 'FAIL'} (min cos={worst:.7f}, thr={cmin})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--st", required=True, help="model.safetensors path")
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float16")
    ap.add_argument("--out-dir", default="exports")
    ap.add_argument("--unit", default="cpu")
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--oracle", default=None)
    ap.add_argument("--skip-convert", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the no-padding projection verify (use e2e_engine_gate for padded ctx)")
    args = ap.parse_args()
    global N
    N = args.ctx // CFG["patch"]
    dtype = torch.float32 if args.dtype == "float32" else torch.float16
    core = load_core_from_safetensors(args.st, CFG)
    tag = "fp32" if dtype == torch.float32 else "fp16"
    out_path = Path(args.out_dir) / f"timesfm_2p5_200m_ctx{args.ctx}_{tag}.aimodel"
    if not args.skip_convert:
        export(core, dtype, out_path)
    if not args.no_verify:
        z = np.load(args.oracle or f"oracle_{args.ctx}.npz", allow_pickle=True)
        ok = asyncio.run(verify(core, z, dtype, out_path, args.unit))
        print("RESULT:", "PASS" if ok else "FAIL")
    else:
        print("[export] done (verify skipped; run e2e_engine_gate)")


if __name__ == "__main__":
    main()
