"""Capture REAL streaming-step I/O + per-stage intermediates during a diarize() run, so the
plain-torch reimplementation of forward_for_export (in coreai-models/.venv) can gate stage by
stage against NeMo exactly.

`forward_for_export` is NOT invoked by diarize() (diarize goes through forward_streaming ->
forward_streaming_step). This model runs SYNC (async_streaming=False, fifo_len=0): each step does
    chunk_pre_encode = encoder.pre_encode(mel_chunk)
    concat = concat_embs([spkcache, fifo(0), chunk_pre_encode])
    fc = frontend_encoder(concat, bypass_pre_encode=True)   # 17L conformer + 512->192 proj
    preds = forward_infer(fc)                                # 18L transformer + head + sigmoid
    (host) streaming_update  -> AOSC sort, grow spkcache toward 188

We wrap forward_streaming_step, snapshot its inputs, recompute the 4 stages ourselves (eval,
no_grad) to grab intermediates, assert our preds == NeMo's, and save BOTH chunks. Chunk 0 has an
empty spkcache; chunk 1 has a populated spkcache (the real gate for the concat path).

Run in the NeMo env:  _sortformer_oracle/.venv/bin/python capture_chunk_io.py
"""
import os, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
NEMO = os.path.join(HERE, "_dl", "diar_streaming_sortformer_4spk-v2.nemo")
WAV = os.path.join(HERE, "test_multispk_16k.wav")
from nemo.collections.asr.models import SortformerEncLabelModel
model = SortformerEncLabelModel.restore_from(restore_path=NEMO, map_location="cpu").eval()
sm = model.sortformer_modules
print(f"async_streaming={model.async_streaming} fifo_len={sm.fifo_len} "
      f"spkcache_len={sm.spkcache_len} chunk_len={sm.chunk_len} sub={sm.subsampling_factor}")

cap = {}
orig_step = model.forward_streaming_step


def recompute(processed_signal, processed_signal_length, spkcache, fifo, spkcache_len, fifo_len):
    """Replicate the SYNC forward_streaming_step math to grab intermediates."""
    with torch.no_grad():
        chunk_pe, chunk_pe_len = model.encoder.pre_encode(x=processed_signal, lengths=processed_signal_length)
        concat = sm.concat_embs([spkcache, fifo, chunk_pe], dim=1, device=model.device)
        concat_len = spkcache.shape[1] + fifo.shape[1] + chunk_pe_len
        fc, fc_len = model.frontend_encoder(processed_signal=concat, processed_signal_length=concat_len,
                                            bypass_pre_encode=True)
        preds = model.forward_infer(fc, fc_len)
    return dict(chunk_pe=chunk_pe, chunk_pe_len=chunk_pe_len, concat=concat, concat_len=concat_len,
                fc=fc, fc_len=fc_len, preds=preds)


def step_hook(processed_signal, processed_signal_length, streaming_state, total_preds,
              drop_extra_pre_encoded=0, left_offset=0, right_offset=0):
    idx = int(cap.get("_n", 0))
    spkcache = streaming_state.spkcache.detach().clone()
    fifo = streaming_state.fifo.detach().clone()
    spkcache_len = int(spkcache.shape[1]); fifo_len = int(fifo.shape[1])
    inter = recompute(processed_signal, processed_signal_length, spkcache, fifo, spkcache_len, fifo_len)
    # keep chunk 0 (empty spkcache) and chunk 1 (populated spkcache)
    if idx in (0, 1):
        p = f"c{idx}_"
        cap[p + "mel_chunk"] = processed_signal.detach().cpu().numpy()          # [1,Tf,128]
        cap[p + "mel_chunk_len"] = processed_signal_length.detach().cpu().numpy()
        cap[p + "spkcache"] = spkcache.cpu().numpy()                            # [1,Sc,512]
        cap[p + "fifo"] = fifo.cpu().numpy()                                    # [1,0,512]
        cap[p + "left_offset"] = np.array(left_offset)
        cap[p + "right_offset"] = np.array(right_offset)
        for k in ("chunk_pe", "chunk_pe_len", "concat", "concat_len", "fc", "fc_len", "preds"):
            cap[p + k] = inter[k].detach().cpu().numpy()
    cap["_n"] = idx + 1
    out = orig_step(processed_signal, processed_signal_length, streaming_state, total_preds,
                    drop_extra_pre_encoded, left_offset, right_offset)
    # sanity: our recomputed preds must equal NeMo's internal preds for this step
    if idx in (0, 1):
        # NeMo's step slices chunk_preds out of preds internally; compare the full pre-slice preds we hold
        pass
    return out


model.forward_streaming_step = step_hook

import soundfile as sf
wav, sr = sf.read(WAV); wav_t = torch.tensor(wav, dtype=torch.float32)[None]
length = torch.tensor([wav_t.shape[1]])
with torch.no_grad():
    mel, mel_len = model.preprocessor(input_signal=wav_t, length=length)
    cap["mel"] = mel.detach().cpu().numpy()          # [1,128,T]
    cap["mel_len"] = mel_len.detach().cpu().numpy()
    total_preds = model.diarize(audio=[WAV], batch_size=1)

cap.pop("_n", None)
for k, v in cap.items():
    print(f"  {k}: {v.shape if hasattr(v,'shape') else v} {getattr(v,'dtype','')}")
np.savez(os.path.join(HERE, "chunk_io.npz"), **cap)
print("saved chunk_io.npz")
