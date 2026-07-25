import numpy as np, torch, soundfile as sf
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402
import detokenizer as DT
O = work_path("_lfmaudio_oracle")
d = np.load(O/"tts_dump.npz")
codes = torch.from_numpy(d["codes"]).long()[None]      # [1,8,599]
ref = d["wav"].astype(np.float32)                       # oracle detok(greedy codes)
m = DT.load_detokenizer(torch.float32)
with torch.no_grad():
    wav = m(codes).float().numpy()
n = min(len(wav), len(ref)); wav=wav[:n]; refn=ref[:n]
mad = np.abs(wav-refn).max()
snr = 10*np.log10((refn**2).sum()/(((wav-refn)**2).sum()+1e-12))
print(f"detok wav len mine={len(wav)} oracle={len(ref)}  max|d|={mad:.6e}  SNR={snr:.2f} dB")
sf.write("/tmp/_detok_mine.wav", wav, 24000)
print("VERDICT:", "DETOK MATCH" if snr>40 else "DIVERGE")
