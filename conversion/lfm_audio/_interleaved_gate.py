"""Milestone C: replicate liquid_audio generate_interleaved (6 text : 12 audio) with the
re-authored LFM2 (overlay) + host depthformer + audio_embedding, vs the oracle stream."""
import numpy as np, torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402
torch.set_grad_enabled(False)
import export_lfm2_embeds_decode as L
import depthformer as DF
from coreai_models.models.macos.lfm2 import build_decode_state
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN as CAP

O = work_path("_lfmaudio_oracle")
d = np.load(O/"interleaved_dump.npz")
prefill = torch.from_numpy(d["prefill_emb"]).float()
tmark, tids, acodes = d["tmark"], d["tids"], d["acodes"]      # oracle stream
N_TEXT, N_AUDIO = 6, 12

lfm = L.load_backbone(L.load_lfm_config(), target_dtype=torch.float32)
dfm = DF.load_depthformer(torch.float32)
aemb = DF.load_audio_embedding(torch.float32)
st = build_decode_state(lfm.config, CAP, dtype=torch.float32)
k, v, conv = st["k_cache"], st["v_cache"], st["conv_state"]

in_emb = prefill[None]; seqlen = 0
mode = "text"; left = N_TEXT; text_done = False
mine = []
for _ in range(len(tmark) + 4):
    left -= 1
    q = in_emb.shape[1]; seqlen += q
    pos = torch.arange(seqlen, dtype=torch.int32)[None]
    h = lfm.model.forward_stateful_embeds(in_emb, pos, KVCache(k, v), conv)[0, -1]
    if mode == "text":
        tok = int(lfm.lm_head(h).argmax())
        if tok == 7: break
        mine.append(("T", tok))
        if tok == 130: text_done = True
        if not left or text_done: mode = "audio"; left = N_AUDIO
        in_emb = lfm.model.embed_tokens(torch.tensor(tok))[None, None, :]
    else:
        codes = dfm.sample_frame(h)
        if not left and not text_done: mode = "text"; left = N_TEXT
        if codes[0] == 2048: mode = "text"
        mine.append(("A", codes))
        in_emb = aemb(torch.tensor(codes))[None, None, :]

# compare to oracle stream
n = min(len(mine), len(tmark)); ok = True
for i in range(n):
    kind, val = mine[i]
    if kind == "T":
        if tmark[i] != 1 or tids[i] != val: ok = False; print(f"  step {i} T mismatch mine={val} oracle=({tmark[i]},{tids[i]})"); break
    else:
        if tmark[i] != 0 or acodes[i].tolist() != val: ok = False; print(f"  step {i} A mismatch"); break
nT = sum(1 for k,_ in mine if k=="T"); nA = len(mine)-nT
print(f"my interleaved stream: {nT} text + {nA} audio (len {len(mine)}); oracle len {len(tmark)}")
print("VERDICT:", "INTERLEAVED EXACT" if (ok and len(mine)==len(tmark)) else "REVIEW")
