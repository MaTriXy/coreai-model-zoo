"""Eager end-to-end TTS: re-authored LFM2 (overlay embeds path) generation loop +
host depthformer + audio_embedding feedback -> 8-CB codes, vs oracle greedy codes.
Then detok(generated codes) -> wav. Proves the full re-authored pipeline + the audio
feedback loop stay aligned. Run: coreai-models/.venv/bin/python _tts_e2e_eager.py [Nframes]
"""
import sys, numpy as np, torch, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402
torch.set_grad_enabled(False)
import export_lfm2_embeds_decode as L
import depthformer as DF
import detokenizer as DT
from coreai_models.models.macos.lfm2 import build_decode_state
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN as CAP

O = work_path("_lfmaudio_oracle")
d = np.load(O/"tts_dump.npz")
prefill = torch.from_numpy(d["prefill_emb"]).float()      # [37,2048]
codes_ref = d["codes"]                                     # [8,599]
NF = int(sys.argv[1]) if len(sys.argv) > 1 else 96
DT_ = torch.float32

lfm = L.load_backbone(L.load_lfm_config(), target_dtype=torch.float32)                # Lfm2EmbedsForCausalLMStateful (eager)
dfm = DF.load_depthformer(DT_)
aemb = DF.load_audio_embedding(DT_)
cfg = lfm.config
st = build_decode_state(cfg, CAP, dtype=DT_)
k, v, conv = st["k_cache"], st["v_cache"], st["conv_state"]

in_emb = prefill[None]                                     # [1,37,2048]
mode = "text"; seqlen = 0; codes_out = []; feedback0 = None
for step in range(37 + NF + 4):
    q = in_emb.shape[1]; seqlen += q
    pos = torch.arange(seqlen, dtype=torch.int32)[None]
    h = lfm.model.forward_stateful_embeds(in_emb, pos, KVCache(k, v), conv)   # [1,q,2048]
    last = h[0, -1]
    if mode == "text":
        tok = int(lfm.lm_head(last).argmax())
        if tok == 7:
            break
        if tok == 128:
            mode = "audio"
        in_emb = lfm.model.embed_tokens(torch.tensor(tok))[None, None, :]
    else:
        toks = dfm.sample_frame(last)                     # host depthformer greedy [8]
        if toks[0] == 2048:
            break
        codes_out.append(toks)
        fb = aemb(torch.tensor(toks))                     # [2048] feedback
        if feedback0 is None:
            feedback0 = fb.numpy()
        in_emb = fb[None, None, :]
        if len(codes_out) >= NF:
            break

gen = np.array(codes_out).T                                # [8, F]
F = gen.shape[1]
ref = codes_ref[:, :F]
nmatch = int((gen == ref).all(0).sum())
print(f"generated {F} audio frames (after text_ids up to audio_start=128)")
print(f"codes exact-frame match: {nmatch}/{F}")
# feedback0 parity
fb_ref = d["feedback0"]
print(f"feedback0 max|d| vs oracle: {np.abs(feedback0-fb_ref).max():.5e}")
# detok generated codes -> wav
wav = DT.load_detokenizer(DT_)(torch.from_numpy(gen).long()[None]).numpy()
sf.write("/tmp/_tts_e2e.wav", wav, 24000)
print(f"detok(generated) wav {wav.shape} {wav.shape[0]/24000:.2f}s -> /tmp/_tts_e2e.wav")
print("VERDICT:", "E2E CODES EXACT" if nmatch == F else f"DIVERGE at frame {int((gen!=ref).any(0).argmax())}")
