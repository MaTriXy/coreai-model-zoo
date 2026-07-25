import numpy as np, torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402
import depthformer as D
O = work_path("_lfmaudio_oracle")
d = np.load(O/"tts_dump.npz")
hidden = torch.from_numpy(d["hidden"]).float()      # [T,2048]
codes  = d["codes"]                                  # [8,T]
f0log  = d["frame0_logits"]                          # [8,2049]
m = D.load_depthformer(torch.float32)
T = hidden.shape[0]
# frame0 logits parity
din0 = m.depth_linear(hidden[0]).view(D.CODEBOOKS, D.DIM)
dtok = torch.zeros(D.DIM); kc=[None]*6; vc=[None]*6; mylog=[]
for i in range(D.CODEBOOKS):
    x=(din0[i]+dtok)[None,None,:]
    for li,l in enumerate(m.layers): x,kc[li],vc[li]=l(x,kc[li],vc[li])
    lg=m.heads[i].logits(x.squeeze(0).squeeze(0)); mylog.append(lg.detach().numpy())
    tk=int(lg.argmax()); dtok=m.heads[i].embed(torch.tensor(tk))
mylog=np.stack(mylog)
cos=np.sum(mylog*f0log)/(np.linalg.norm(mylog)*np.linalg.norm(f0log))
print(f"frame0 per-CB logits cos={cos:.6f} max|d|={np.abs(mylog-f0log).max():.4f}")
# full token match (all frames)
nmatch=0; firstbad=None
for t in range(T):
    toks=m.sample_frame(hidden[t])
    ok=toks==codes[:,t].tolist()
    nmatch+=ok
    if not ok and firstbad is None: firstbad=(t, toks, codes[:,t].tolist())
print(f"depthformer frame token-match: {nmatch}/{T} frames exact")
if firstbad: print("first mismatch frame", firstbad[0], "mine", firstbad[1], "oracle", firstbad[2])
print("VERDICT:", "DEPTHFORMER EXACT" if nmatch==T else "DIVERGE")
