"""On-engine end-to-end TTS: Core AI LFM2 (AOT .aimodelc, hidden output) generation loop
+ host depthformer + audio_embedding feedback -> codes; detok -> wav. vs oracle greedy."""
import sys, asyncio, numpy as np, torch, soundfile as sf
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402
torch.set_grad_enabled(False)
import coreai.runtime as rt
import export_lfm2_embeds_decode as L
import depthformer as DF
import detokenizer as DT
from coreai_models.models.macos.lfm2 import build_decode_state, DECODE_STATE_NAMES
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN as CAP

O = work_path("_lfmaudio_oracle")
d = np.load(O/"tts_dump.npz")
prefill = d["prefill_emb"].astype(np.float16)          # [37,2048]
codes_ref = d["codes"]
AIMC = sys.argv[1]; NF = int(sys.argv[2]) if len(sys.argv) > 2 else 48

cfg = L.load_lfm_config()
dfm = DF.load_depthformer(torch.float32)
aemb = DF.load_audio_embedding(torch.float32)
embT = L.embed_table()                                  # [65536,2048] fp32 tied table

def nd(a): return rt.NDArray(np.ascontiguousarray(a))

async def main():
    m = await rt.AIModel.load(str(Path(AIMC).resolve()), rt.SpecializationOptions.default())
    fn = m.load_function("main")
    st = build_decode_state(cfg, CAP, dtype=torch.float16)
    state = {n: nd(t.numpy()) for n, t in zip(DECODE_STATE_NAMES,
             [st["k_cache"], st["v_cache"], st["conv_state"]])}
    seqlen = 0
    async def step(emb):                                 # emb [2048] fp16 -> hidden [2048] fp32
        nonlocal seqlen; seqlen += 1
        pos = nd(np.arange(seqlen, dtype=np.int32)[None])
        r = await fn(inputs={"inputs_embeds": nd(emb.reshape(1,1,-1).astype(np.float16)),
                             "position_ids": pos}, state=state)
        return torch.from_numpy(r["hidden"].numpy().astype(np.float32)).reshape(-1)
    # prefill (37 prompt embeds as s=1 steps)
    h = None
    for i in range(prefill.shape[0]):
        h = await step(prefill[i])
    mode = "text"; codes_out = []
    for _ in range(NF + 8):
        if mode == "text":
            tok = int((h @ embT.T).argmax())
            if tok == 7: break
            if tok == 128: mode = "audio"
            h = await step(embT[tok].numpy().astype(np.float16))
        else:
            toks = dfm.sample_frame(h)                   # host depthformer
            if toks[0] == 2048: break
            codes_out.append(toks)
            fb = aemb(torch.tensor(toks)).numpy().astype(np.float16)
            h = await step(fb)
            if len(codes_out) >= NF: break
    return np.array(codes_out).T

gen = asyncio.run(main())
F = gen.shape[1]; ref = codes_ref[:, :F]
nmatch = int((gen == ref).all(0).sum())
print(f"[engine e2e] {F} audio frames; codes exact-frame match {nmatch}/{F}")
if nmatch < F:
    fb = int((gen != ref).any(0).argmax()); print(f"  first diverge frame {fb}: mine {gen[:,fb].tolist()} oracle {ref[:,fb].tolist()}")

DETOK_AIM = sys.argv[3] if len(sys.argv) > 3 else None
detok = DT.load_detokenizer(torch.float32)
if DETOK_AIM:  # fully on-engine: codes -> Core AI detok backbone -> host iSTFT
    emb = DT.codes_to_embeds(detok, torch.from_numpy(gen).long()[None]).to(torch.float16)
    async def dt_run():
        m = await rt.AIModel.load(str(Path(DETOK_AIM).resolve()),
            rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        r = await m.load_function("main")({"inputs_embeds": nd(emb.numpy())})
        return torch.from_numpy(r["spec"].numpy().astype(np.float32))
    wav = DT.spec_to_wav(detok, asyncio.run(dt_run())).numpy()
    tag = "engine-LFM+engine-detok"
else:
    wav = detok(torch.from_numpy(gen).long()[None]).numpy()
    tag = "engine-LFM+host-detok"
sf.write("/tmp/_tts_engine.wav", wav, 24000)
print(f"[{tag}] detok wav {wav.shape} {wav.shape[0]/24000:.2f}s -> /tmp/_tts_engine.wav")
print("VERDICT:", "ENGINE E2E EXACT" if nmatch==F else "ENGINE E2E CLOSE (fp16 drift)")
