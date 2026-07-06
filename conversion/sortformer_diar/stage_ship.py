"""Assemble ship_macos/ for the coreai-audio Diarize path: the fp16 .aimodel + the mel filterbank +
golden mel / golden streaming preds (for the Swift DiarizeSelfTest) + the demo wav + metadata.
Mirrors conversion/melband_roformer/ship_macos. Run in the MAIN coreai-models venv:
    coreai-models/.venv/bin/python stage_ship.py
"""
import json, os, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SHIP = os.path.join(HERE, "ship_macos")
os.makedirs(SHIP, exist_ok=True)

# 1. the exported fp16 graph bundle
src_aimodel = os.path.join(HERE, "artifacts", "sortformer_float16.aimodel")
dst_aimodel = os.path.join(SHIP, "sortformer_float16.aimodel")
shutil.rmtree(dst_aimodel, ignore_errors=True)
shutil.copytree(src_aimodel, dst_aimodel)

# 2. mel filterbank (librosa-slaney [128,257] row-major, straight from the NeMo preprocessor)
shutil.copy(os.path.join(HERE, "_mel_tables", "sortformer_mel_filters_128x257.f32"),
            os.path.join(SHIP, "sortformer_mel_filters_128x257.f32"))

# 3. golden mel + golden streaming preds (from the NeMo capture) for the headless self-test
d = np.load(os.path.join(HERE, "stream_ref.npz"))
mel = d["mel"][0].astype(np.float32)              # [128,T]  mel-major, row-major
mel_len = int(d["mel_len"][0])
total_preds = d["total_preds"][0].astype(np.float32)   # [n_out,4]  frame-major
mel.tofile(os.path.join(SHIP, "golden_mel_128xT.f32"))
total_preds.tofile(os.path.join(SHIP, "golden_total_preds.f32"))

# 4. demo wav
shutil.copy(os.path.join(HERE, "test_multispk_16k.wav"),
            os.path.join(SHIP, "test_multispk_16k.wav"))

# 5. metadata
meta = {
    "model": "Streaming Sortformer 4-spk v2 (NVIDIA)",
    "license": "cc-by-4.0",
    "sample_rate": 16000, "n_mels": 128, "n_fft": 512, "win_length": 400, "hop": 160,
    "preemph": 0.97, "normalize": "NA", "mag_power": 2.0, "log_guard": 2 ** -24, "pad_to": 16,
    "subsampling_factor": 8, "frame_hop_ms": 80, "n_spk": 4,
    "spkcache_len": 188, "fifo_len": 0, "chunk_len": 188,
    "chunk_left_context": 1, "chunk_right_context": 1, "spkcache_update_period": 188,
    "spkcache_sil_frames_per_spk": 3, "scores_boost_latest": 0.05,
    "sil_threshold": 0.2, "pred_score_threshold": 0.25,
    "strong_rate": 0.75, "weak_rate": 1.5, "min_pos_rate": 0.5, "max_index": 99999,
    "tf_max": 1520, "spk": 188, "pe_max": 190, "t": 378,
    "graph_io": "chunk_mel[1,1520,128],spkcache[1,188,512],valid[1,378] -> preds[1,378,4],chunk_pe[1,190,512]",
    "golden": {"mel_shape": list(mel.shape), "mel_len": mel_len,
               "total_preds_shape": list(total_preds.shape)},
    "note": "host: NeMo 128-mel (preemph->stft512/win400/hop160->slaney mel->log, normalize=NA) -> "
            "feat_chunks (188*8, L/R ctx 1) -> per chunk zero-pad mel to 1520, build spkcache(188)+valid, "
            "run graph, slice chunk preds, streaming_update + AOSC compress_spkcache; "
            "threshold 0.5/frame/spk (frame=80ms) -> speaker turns.",
}
with open(os.path.join(SHIP, "metadata.json"), "w") as f:
    json.dump(meta, f, indent=2)

sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(SHIP) for f in fs) / 1e6
print(f"staged {SHIP} ({sz:.1f} MB)")
for f in sorted(os.listdir(SHIP)):
    print("  ", f)
