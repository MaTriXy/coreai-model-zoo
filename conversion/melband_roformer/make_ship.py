"""Assemble the ship bundles (macOS .aimodel + iOS .h18p.aimodelc + metadata.json)
and dump a golden (raw chunk -> numpy-DSP + engine -> vocals) so the Swift/vDSP
host has an on-device self-test target. golden_*.f32 layout = ch0 then ch1.
"""
import os, sys, json, shutil, numpy as np, torch, asyncio
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))
import yaml
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer
from export_core import SepCore
import coreai.runtime as rt
# reuse the verified numpy DSP recipe
from gate_dsp_numpy import np_stft, np_istft, N_FFT, HOP

with open(os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")) as f:
    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
C = config.inference.chunk_size
NOV = config.inference.num_overlap
model = MelBandRoformer(**dict(config.model)).eval()
model.load_state_dict(torch.load(os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt"), map_location="cpu"), strict=False)
oracle = torch.load(os.path.join(HERE, "_precheck", "ref_oracle.pt"))
raw = oracle["raw_audio"].numpy()                # [2, C]
sr = int(oracle["sr"])

macos = os.path.join(HERE, "ship_macos"); ios = os.path.join(HERE, "ship_ios")
os.makedirs(macos, exist_ok=True); os.makedirs(ios, exist_ok=True)
# copy bundles
src_mac = os.path.join(HERE, "artifacts", "mbr_core_fp16", "mbr_core_fp16.aimodel")
dst_mac = os.path.join(macos, "mbr_core_fp16.aimodel")
if os.path.exists(dst_mac): shutil.rmtree(dst_mac) if os.path.isdir(dst_mac) else os.remove(dst_mac)
(shutil.copytree if os.path.isdir(src_mac) else shutil.copy)(src_mac, dst_mac)
src_ios = os.path.join(HERE, "artifacts", "ios_h18p", "mbr_core_fp16.h18p.aimodelc")
dst_ios = os.path.join(ios, "mbr_core_fp16.h18p.aimodelc")
if os.path.exists(dst_ios): shutil.rmtree(dst_ios)
shutil.copytree(src_ios, dst_ios)

# stft frame count for a full chunk
nfr = np_stft(raw[:, :C]).shape[1]
meta = {"model": "MelBandRoformer-Vocal (Kim Vocal, MIT)", "license": "MIT",
        "sample_rate": sr, "chunk_size": C, "n_fft": N_FFT, "hop_length": HOP, "win_length": N_FFT,
        "num_overlap": NOV, "F2": 2 * (N_FFT // 2 + 1), "n_frames": int(nfr),
        "target": "vocals", "note": "instrumental = mix - vocals; stft_real (f s) layout idx=f*2+s"}
for d in (macos, ios):
    json.dump(meta, open(os.path.join(d, "metadata.json"), "w"), indent=2)
print("metadata:", meta)

# golden: raw 8s chunk -> numpy stft -> fp16 engine core -> numpy istft -> vocals
async def golden():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(dst_mac, gpu)).load_function("main")
    sr_in = np_stft(raw[:, :C])[None].astype(np.float16)   # add batch dim -> [1,2050,801,2]
    r = await fn(inputs={"stft_real": rt.NDArray(np.ascontiguousarray(sr_in))})
    masked = np.asarray(r["masked"].numpy()).astype(np.float32)[0]   # drop batch -> [2050,801,2]
    voc = np_istft(masked, 2, C)                   # [2, C]
    raw_flat = np.concatenate([raw[0, :C], raw[1, :C]]).astype(np.float32)      # ch0||ch1
    voc_flat = np.concatenate([voc[0], voc[1]]).astype(np.float32)
    raw_flat.tofile(os.path.join(HERE, "ship_macos", "golden_raw.f32"))
    voc_flat.tofile(os.path.join(HERE, "ship_macos", "golden_vocals.f32"))
    print("golden vocals rms %.4f, saved golden_raw/golden_vocals.f32 (2*%d floats each)" % (np.sqrt((voc**2).mean()), C))
asyncio.run(golden())
print("ship_macos:", os.listdir(macos)); print("ship_ios:", os.listdir(ios))
