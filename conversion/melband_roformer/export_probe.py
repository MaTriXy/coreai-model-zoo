"""Export SepCore (the Mel-Band RoFormer neural core) to Core AI + Mac GPU engine-gate.
Graph I/O: stft_real [1, 2050, 801, 2] -> masked_real [1, 2050, 801, 2].
Mirrors the Stable Audio DiT probe idiom (drop SDPA from externalize -> native lower).
"""
import os, sys, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))

# --- avoid the CUDA sdp_kernel context manager during export (CPU/export safe) ---
import models.mel_band_roformer.attend as _att
import torch.nn.functional as _F
def _flash(self, q, k, v):
    return _F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
_att.Attend.flash_attn = _flash

import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer
from export_core import SepCore, HostDSP

CFG = os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")
CKPT = os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt")
with open(CFG) as f:
    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
model = MelBandRoformer(**dict(config.model)).eval()
model.load_state_dict(torch.load(CKPT, map_location="cpu"), strict=False)

core = SepCore(model).eval()
host = HostDSP(model)

oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
raw = oracle["raw_audio"].unsqueeze(0)                      # [1,2,352800]
with torch.no_grad():
    stft_real = host.stft(raw).float().contiguous()        # [1,2050,801,2]
    ref_out = core(stft_real).float()                      # [1,2050,801,2]
print("core I/O:", tuple(stft_real.shape), "->", tuple(ref_out.shape),
      "F_sel", core.freq_indices.numel())

ref = {"stft_real": stft_real}
print("[probe] export_to_coreai ...", flush=True)
prog = export_to_coreai(core, ref, dynamic_shapes=None,
                        input_names=("stft_real",), output_names=("masked",), state_names=None)
print("[probe] EXPORT OK ✅", flush=True)
prog.optimize()

import shutil, asyncio, coreai.runtime as rt
out_dir = Path(HERE) / "artifacts" / "mbr_core_fp32_probe"
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
aim = out_dir / "mbr_core.aimodel"
prog.save_asset(aim, rt.AIModelAssetMetadata())
print("[probe] saved", aim, flush=True)

async def gate():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    def nd(a): return rt.NDArray(np.ascontiguousarray(a.numpy().astype(np.float32)))
    r = await fn(inputs={"stft_real": nd(stft_real)})
    eng = torch.as_tensor(np.asarray(r["masked"].numpy()).astype(np.float32))
    a, b = eng.reshape(-1).double(), ref_out.reshape(-1).double()
    cc = (torch.dot(a, b) / (a.norm() * b.norm() + 1e-12)).item()
    md = (a - b).abs().max().item()
    print(f"[probe] ENGINE(core) vs torch cos={cc:.7f} max|d|={md:.2e}  {'PASS' if cc>=0.999 else 'CHECK'}")
    # also gate end-to-end audio through host istft
    vocals_eng = host.istft(eng.reshape(ref_out.shape), length=raw.shape[-1])
    vocals_ref = host.istft(ref_out, length=raw.shape[-1])
    va, vb = vocals_eng.reshape(-1).double(), vocals_ref.reshape(-1).double()
    ca = (torch.dot(va, vb) / (va.norm() * vb.norm() + 1e-12)).item()
    print(f"[probe] ENGINE(audio) vs torch cos={ca:.7f}  {'PASS' if ca>=0.999 else 'CHECK'}")
asyncio.run(gate())
