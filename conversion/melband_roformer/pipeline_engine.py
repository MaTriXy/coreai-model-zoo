"""Full-song separation driven by the fp16 Core AI engine core (Mac GPU).
Reproduces the reference chunk overlap-add host loop (num_overlap=2, reflect
border pad, fade windowing) but each 8s chunk runs host_stft -> engine core ->
host_istft. This IS the host loop the Swift app will implement. Outputs full
vocals + instrumental for a real listen.
"""
import os, sys, time, numpy as np, torch, asyncio
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "kim"))
import yaml, soundfile as sf, librosa
from ml_collections import ConfigDict
from models.mel_band_roformer import MelBandRoformer
from export_core import HostDSP
import coreai.runtime as rt

CFG = os.path.join(HERE, "_ref", "kim", "configs", "config_vocals_mel_band_roformer.yaml")
with open(CFG) as f:
    config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
model = MelBandRoformer(**dict(config.model)).eval()   # only for HostDSP stft kwargs + no weights needed
model.load_state_dict(torch.load(os.path.join(HERE, "_ckpt", "MelBandRoformer.ckpt"), map_location="cpu"), strict=False)
host = HostDSP(model)

C = config.inference.chunk_size        # 352800
N = config.inference.num_overlap       # 2
step = C // N
fade = C // 10
AIM = os.path.join(HERE, "artifacts", "mbr_core_fp16", "mbr_core_fp16.aimodel")

def windowing(C, fade):
    w = torch.ones(C); w[-fade:] *= torch.linspace(1, 0, fade); w[:fade] *= torch.linspace(0, 1, fade)
    return w

async def main():
    song = librosa.example("fishin")
    wav, sr = librosa.load(song, sr=44100, mono=False)
    if wav.ndim == 1: wav = np.stack([wav, wav], 0)
    mix = torch.tensor(wav, dtype=torch.float32)          # [2, T]
    T = mix.shape[1]
    border = C - step
    mixp = torch.nn.functional.pad(mix, (border, border), mode="reflect")

    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(AIM, gpu)).load_function("main")

    result = torch.zeros(2, mixp.shape[1]); counter = torch.zeros(2, mixp.shape[1])
    win = windowing(C, fade)
    i = 0; nchunk = 0; t0 = time.time()
    total = mixp.shape[1]
    while i < total:
        part = mixp[:, i:i + C]
        L = part.shape[-1]
        if L < C:
            pad_mode = "reflect" if L > C // 2 + 1 else "constant"
            part = torch.nn.functional.pad(part, (0, C - L), mode=pad_mode)
        sr_in = host.stft(part.unsqueeze(0)).half().numpy().astype(np.float16)
        r = await fn(inputs={"stft_real": rt.NDArray(np.ascontiguousarray(sr_in))})
        masked = torch.as_tensor(np.asarray(r["masked"].numpy()).astype(np.float32))
        voc = host.istft(masked, length=C)[0]              # [2, C]
        w = win.clone()
        if i == 0: w[:fade] = 1
        elif i + C >= total: w[-fade:] = 1
        result[:, i:i + L] += voc[:, :L] * w[:L]
        counter[:, i:i + L] += w[:L]
        i += step; nchunk += 1
    dt = time.time() - t0
    vocals = (result / counter.clamp(min=1e-8))
    vocals = vocals[:, border:border + T]                  # unpad
    inst = mix - vocals
    sf.write(os.path.join(HERE, "_precheck", "song_vocals.wav"), vocals.T.numpy(), sr, subtype="FLOAT")
    sf.write(os.path.join(HERE, "_precheck", "song_instrumental.wav"), inst.T.numpy(), sr, subtype="FLOAT")
    print(f"{nchunk} chunks, {dt:.1f}s for {T/sr:.0f}s audio = {T/sr/dt:.1f}x realtime (Mac GPU fp16)")
    print("vocals rms %.4f  instrumental rms %.4f" % (vocals.pow(2).mean().sqrt(), inst.pow(2).mean().sqrt()))
    print("saved song_vocals.wav + song_instrumental.wav")
asyncio.run(main())
