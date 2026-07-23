"""fp16 ship export of SepCore + Mac GPU engine-gate (fp16 NDArrays).
Gate vs the fp32 torch core output; masks are quant-sensitive so watch cos.
"""
import os, sys, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))

import models.mel_band_roformer.attend as _att
import torch.nn.functional as _F
_att.Attend.flash_attn = lambda self, q, k, v: _F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]

import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer
from export_core import SepCore, HostDSP

CFG = os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")
with open(CFG) as f:
    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
model = MelBandRoformer(**dict(config.model)).eval()
model.load_state_dict(torch.load(os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt"), map_location="cpu"), strict=False)

host = HostDSP(model)
oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
raw = oracle["raw_audio"].unsqueeze(0)
with torch.no_grad():
    stft_f32 = host.stft(raw).float().contiguous()
    core_f32 = SepCore(model).eval()
    ref_out = core_f32(stft_f32).float()                     # fp32 reference

# fp16 core + fp16 input
core = SepCore(model).eval().half()
stft_f16 = stft_f32.half().contiguous()
with torch.no_grad():
    torch_f16 = core(stft_f16).float()
a, b = torch_f16.reshape(-1).double(), ref_out.reshape(-1).double()
print(f"torch fp16 vs fp32 core: cos={(torch.dot(a,b)/(a.norm()*b.norm())).item():.7f}")

ref = {"stft_real": stft_f16}
prog = export_to_coreai(core, ref, dynamic_shapes=None,
                        input_names=("stft_real",), output_names=("masked",), state_names=None)
prog.optimize()
import shutil, asyncio, coreai.runtime as rt
out_dir = Path(HERE) / "artifacts" / "mbr_core_fp16"
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
aim = out_dir / "mbr_core_fp16.aimodel"
prog.save_asset(aim, rt.AIModelAssetMetadata())
sz = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
print(f"saved {aim.name}  {sz:.0f} MB")

async def gate():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"stft_real": rt.NDArray(np.ascontiguousarray(stft_f16.numpy().astype(np.float16)))})
    eng = torch.as_tensor(np.asarray(r["masked"].numpy()).astype(np.float32))
    a, b = eng.reshape(-1).double(), ref_out.reshape(-1).double()
    cc = (torch.dot(a, b) / (a.norm() * b.norm())).item()
    print(f"ENGINE fp16(core) vs fp32 torch: cos={cc:.7f}  {'PASS' if cc>=0.999 else 'CHECK'}")
    vocals_eng = host.istft(eng.reshape(ref_out.shape), length=raw.shape[-1])
    import soundfile as sf
    sf.write(os.path.join(HERE, "_precheck", "vocals_engine_fp16.wav"),
             vocals_eng[0].T.numpy(), oracle["sr"], subtype="FLOAT")
    vr = host.istft(ref_out, length=raw.shape[-1])
    va, vb = vocals_eng.reshape(-1).double(), vr.reshape(-1).double()
    print(f"ENGINE fp16(audio) vs fp32 torch: cos={(torch.dot(va,vb)/(va.norm()*vb.norm())).item():.7f}")
asyncio.run(gate())
