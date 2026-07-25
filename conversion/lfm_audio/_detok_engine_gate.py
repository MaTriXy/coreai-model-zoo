import sys, asyncio, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402
import coreai.runtime as rt
import detokenizer as DT
torch.set_grad_enabled(False)
O = work_path("_lfmaudio_oracle")
NF = 64
aim = sys.argv[1]; unit = sys.argv[2] if len(sys.argv)>2 else "gpu"
codes = torch.from_numpy(np.load(O/"tts_dump.npz")["codes"]).long()[None][:,:, :NF]
detok = DT.load_detokenizer(torch.float32)
emb = DT.codes_to_embeds(detok, codes)
spec_eager = DT.DetokSpec(detok)(emb).float().numpy()

async def run():
    opts = (rt.SpecializationOptions.cpu_only() if unit=="cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    m = await rt.AIModel.load(str(Path(aim).resolve()), opts)
    fn = m.load_function("main")
    r = await fn({"inputs_embeds": rt.NDArray(np.ascontiguousarray(emb.to(torch.float16).numpy()))})
    return r["spec"].numpy().astype(np.float32)

spec_eng = asyncio.run(run())
a,b = spec_eager.reshape(-1), spec_eng.reshape(-1)
cos = float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
snr = 10*np.log10((a**2).sum()/(((a-b)**2).sum()+1e-12))
print(f"[detok {unit}] spec cos={cos:.6f} SNR={snr:.2f}dB max|d|={np.abs(a-b).max():.4f}")
we = DT.spec_to_wav(detok, torch.from_numpy(spec_eager)).numpy()
wg = DT.spec_to_wav(detok, torch.from_numpy(spec_eng)).numpy()
n=min(len(we),len(wg)); wsnr=10*np.log10((we[:n]**2).sum()/(((we[:n]-wg[:n])**2).sum()+1e-12))
print(f"[detok {unit}] wav(engine vs eager) SNR={wsnr:.2f}dB")
print("VERDICT:", "DETOK ENGINE OK" if cos>0.999 else "REVIEW")
