"""Export SepFull2 (frames->recon, in-graph STFT/iSTFT) fp16 + Mac GPU gate +
regenerate ship bundle + goldens. AOT (h18p) is a separate xcrun step after this.
"""
import os, sys, json, shutil, numpy as np, torch, asyncio
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
from export_core2 import SepFull2, frame_host, overlap_add_host, N_FFT, HOP, PAD

with open(os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")) as fp:
    config = ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))
C = config.inference.chunk_size; NOV = config.inference.num_overlap
model = MelBandRoformer(**dict(config.model)).eval()
model.load_state_dict(torch.load(os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt"), map_location="cpu"), strict=False)
oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
raw = oracle["raw_audio"].numpy(); sr = int(oracle["sr"])
nfr = 1 + (C + 2 * PAD - N_FFT) // HOP

# reference vocals = the precheck oracle (model() output on raw_audio). Take it BEFORE
# .half(), since SepCore shares model's submodules by reference and .half() mutates them.
voc_ref = oracle["vocals"].numpy()
full = SepFull2(model).eval().half()
frames = torch.tensor(frame_host(raw, nfr), dtype=torch.float16).unsqueeze(0)   # [1,2,nfr,N]
with torch.no_grad():
    ref_recon = full(frames).float()

prog = export_to_coreai(full, {"frames": frames}, dynamic_shapes=None,
                        input_names=("frames",), output_names=("recon",), state_names=None)
prog.optimize()
import coreai.runtime as rt
out_dir = Path(HERE) / "artifacts" / "mbr_full_fp16"
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
aim = out_dir / "mbr_full_fp16.aimodel"
prog.save_asset(aim, rt.AIModelAssetMetadata())
sz = sum(f.stat().st_size for f in out_dir.rglob("*") if f.is_file()) / 1e6
print(f"saved {aim.name} {sz:.0f} MB")

def cos(a, b):
    a, b = np.asarray(a).ravel().astype(np.float64), np.asarray(b).ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

async def gate_and_ship():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"frames": rt.NDArray(np.ascontiguousarray(frames.numpy().astype(np.float16)))})
    recon = np.asarray(r["recon"].numpy()).astype(np.float32)[0]         # [2,nfr,N]
    voc = overlap_add_host(recon, C)
    print(f"ENGINE fp16 (frames->recon->overlap-add) vs reference vocals: cos {cos(voc, voc_ref):.7f}  "
          f"{'PASS' if cos(voc, voc_ref) >= 0.999 else 'CHECK'}")

    # ship: bundle + metadata + golden (engine vocals)
    macos = Path(HERE) / "ship_macos"; ios = Path(HERE) / "ship_ios"
    for d in (macos, ios):
        d.mkdir(exist_ok=True)
    dst = macos / "mbr_full_fp16.aimodel"
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(aim, dst)
    # drop the old stft_real bundle name if present
    for stale in ["mbr_core_fp16.aimodel"]:
        p = macos / stale
        if p.exists(): shutil.rmtree(p)
    meta = {"model": "MelBandRoformer-Vocal (Kim Vocal, MIT)", "license": "MIT", "sample_rate": sr,
            "chunk_size": C, "n_fft": N_FFT, "hop_length": HOP, "pad": PAD, "n_frames": int(nfr),
            "num_overlap": NOV, "graph_io": "frames[1,2,nfr,N]->recon[1,2,nfr,N]", "target": "vocals",
            "note": "host: reflect-pad N//2, frame stride hop, feed graph, overlap-add, /sum(win^2), trim pad; instrumental=mix-vocals"}
    json.dump(meta, open(macos / "metadata.json", "w"), indent=2)
    json.dump(meta, open(ios / "metadata.json", "w"), indent=2)
    np.concatenate([raw[0, :C], raw[1, :C]]).astype(np.float32).tofile(macos / "golden_raw.f32")
    np.concatenate([voc[0], voc[1]]).astype(np.float32).tofile(macos / "golden_vocals.f32")
    print("ship_macos:", sorted(os.listdir(macos)))
asyncio.run(gate_and_ship())
