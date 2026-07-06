"""Python reference of the Sortformer streaming HOST loop: chunk extraction + concat + (graph) +
sync streaming_update with the AOSC speaker-cache compression, all re-implemented from NeMo's
sortformer_modules.py (inference path only: permute_spk=False, training=False). Drives the
byte-exact eager graph (sortformer_model.Sortformer) over the full clip and gates the accumulated
total_preds against NeMo's forward_streaming output (stream_ref.npz).

This validates the AOSC algorithm end-to-end before the Swift port. Run in coreai-models venv:
    coreai-models/.venv/bin/python host_loop.py
"""
from __future__ import annotations
import math, os
import numpy as np
import torch
import torch.nn.functional as F
from sortformer_model import Sortformer, load_ckpt, build_pos_emb

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_nemo", "model_weights.ckpt")

# ---- streaming params (from _nemo/model_config.yaml) ----
SPKCACHE_LEN = 188
FIFO_LEN = 0
CHUNK_LEN = 188
LEFT_CTX, RIGHT_CTX = 1, 1
SUB = 8
UPDATE_PERIOD = 188
SIL_PER_SPK = 3
N_SPK = 4
SCORES_BOOST_LATEST = 0.05
SIL_THRESHOLD = 0.2
PRED_SCORE_THRESHOLD = 0.25
STRONG_RATE, WEAK_RATE, MINPOS_RATE = 0.75, 1.5, 0.5
MAX_INDEX = 99999


# --------------------------------------------------------------------------- AOSC helpers
def get_log_pred_scores(preds):
    lp = torch.log(torch.clamp(preds, min=PRED_SCORE_THRESHOLD))
    l1 = torch.log(torch.clamp(1.0 - preds, min=PRED_SCORE_THRESHOLD))
    l1_sum = l1.sum(dim=2, keepdim=True).expand(-1, -1, N_SPK)
    return lp - l1 + l1_sum - math.log(0.5)


def disable_low_scores(preds, scores, min_pos_per_spk):
    is_speech = preds > 0.5
    scores = torch.where(is_speech, scores, torch.tensor(float("-inf")))
    is_pos = scores > 0
    is_nonpos_replace = (~is_pos) & is_speech & (is_pos.sum(dim=1, keepdim=True) >= min_pos_per_spk)
    return torch.where(is_nonpos_replace, torch.tensor(float("-inf")), scores)


def boost_topk_scores(scores, n_boost, scale_factor=1.0, offset=0.5):
    _, topk = torch.topk(scores, n_boost, dim=1, largest=True, sorted=False)
    b = torch.arange(scores.shape[0]).view(-1, 1, 1)
    s = torch.arange(scores.shape[2]).view(1, 1, -1)
    scores[b, topk, s] -= scale_factor * math.log(offset)
    return scores


def get_topk_indices(scores):
    # NeMo _get_topk_indices: n_frames is the *padded* frame count (scores already has the
    # SIL_PER_SPK inf rows appended), so remainder wraps by (n_frames+SIL) and the extra silence
    # rows land at index >= n_frames_no_sil -> disabled -> mean_sil emb. (Deriving n_frames from a
    # pre-pad count would shift real-frame indices by SIL per speaker; matches sortformer_modules.py.)
    B, n_frames, _ = scores.shape
    n_no_sil = n_frames - SIL_PER_SPK
    flat = scores.permute(0, 2, 1).reshape(B, -1)
    topk_vals, topk_idx = torch.topk(flat, SPKCACHE_LEN, dim=1, sorted=False)
    topk_idx = torch.where(topk_vals != float("-inf"), topk_idx, torch.tensor(MAX_INDEX))
    topk_sorted, _ = torch.sort(topk_idx, dim=1)
    is_disabled = topk_sorted == MAX_INDEX
    topk_sorted = torch.remainder(topk_sorted, n_frames)
    is_disabled = is_disabled | (topk_sorted >= n_no_sil)
    topk_sorted = topk_sorted.masked_fill(is_disabled, 0)
    return topk_sorted, is_disabled


def gather_spkcache(emb_seq, preds, topk_idx, is_disabled, mean_sil_emb):
    emb_dim, n_spk = emb_seq.shape[2], preds.shape[2]
    ie = topk_idx.unsqueeze(-1).expand(-1, -1, emb_dim)
    emb_g = torch.gather(emb_seq, 1, ie)
    msil = mean_sil_emb.unsqueeze(1).expand(-1, SPKCACHE_LEN, -1)
    emb_g = torch.where(is_disabled.unsqueeze(-1), msil, emb_g)
    ip = topk_idx.unsqueeze(-1).expand(-1, -1, n_spk)
    preds_g = torch.gather(preds, 1, ip)
    preds_g = torch.where(is_disabled.unsqueeze(-1), torch.tensor(0.0), preds_g)
    return emb_g, preds_g


def compress_spkcache(emb_seq, preds, mean_sil_emb):
    """Inference AOSC: keep SPKCACHE_LEN most-important frames (permute_spk=False, training=False)."""
    B, n_frames, n_spk = preds.shape
    per_spk = SPKCACHE_LEN // n_spk - SIL_PER_SPK
    strong = math.floor(per_spk * STRONG_RATE)
    weak = math.floor(per_spk * WEAK_RATE)
    min_pos = math.floor(per_spk * MINPOS_RATE)

    scores = get_log_pred_scores(preds)
    scores = disable_low_scores(preds, scores, min_pos)
    if SCORES_BOOST_LATEST > 0:
        scores[:, SPKCACHE_LEN:, :] += SCORES_BOOST_LATEST
    scores = boost_topk_scores(scores, strong, scale_factor=2)
    scores = boost_topk_scores(scores, weak, scale_factor=1)
    if SIL_PER_SPK > 0:
        pad = torch.full((B, SIL_PER_SPK, n_spk), float("inf"))
        scores = torch.cat([scores, pad], dim=1)
    topk_idx, is_disabled = get_topk_indices(scores)
    return gather_spkcache(emb_seq, preds, topk_idx, is_disabled, mean_sil_emb)


def get_silence_profile(mean_sil_emb, n_sil_frames, emb_seq, preds):
    is_sil = preds.sum(dim=2) < SIL_THRESHOLD
    sil_count = is_sil.sum(dim=1)
    if sil_count.sum() == 0:
        return mean_sil_emb, n_sil_frames
    sil_sum = (emb_seq * is_sil.unsqueeze(-1)).sum(dim=1)
    upd_n = n_sil_frames + sil_count
    total = mean_sil_emb * n_sil_frames.unsqueeze(1) + sil_sum
    return total / torch.clamp(upd_n.unsqueeze(1), min=1), upd_n


# --------------------------------------------------------------------------- streaming loop
def feat_chunks(mel, mel_len):
    """Replicates SortformerModules.streaming_feat_loader. mel [1,128,T]."""
    feat_len = mel.shape[2]
    stt = 0
    while stt < feat_len:
        left = min(LEFT_CTX * SUB, stt)
        end = min(stt + CHUNK_LEN * SUB, feat_len)
        right = min(RIGHT_CTX * SUB, feat_len - end)
        chunk = mel[:, :, stt - left : end + right]
        flen = torch.clamp(mel_len - stt + left, 0, chunk.shape[2])
        yield chunk.transpose(1, 2), flen, left, right     # [1,Tf,128]
        stt = end


def eager_forward(model):
    """chunk_forward for the plain-torch graph: returns (preds [1,Sc+Pe,4], chunk_pe [1,Pe,512])."""
    def f(chunk_feat, spkcache):
        with torch.no_grad():
            chunk_pe = model.pre_encode(chunk_feat)                 # [1,Pe,512]
            concat = torch.cat([spkcache, chunk_pe], dim=1)         # [1,Sc+Pe,512]
            T = concat.shape[1]
            valid = torch.ones(1, T)
            cab, cm, tfb, om = model.masks_from_valid(valid)
            fc = model.conformer_proj(concat, build_pos_emb(T), cab, cm)
            preds = model.infer(fc, tfb, om)                        # [1,Sc+Pe,4]
        return preds, chunk_pe
    return f


def run(chunk_forward, mel, mel_len):
    """Drive the streaming diarization loop with a pluggable `chunk_forward(chunk_feat, spkcache)`
    that returns (preds in [spkcache|chunk] concat layout, chunk_pe embeddings)."""
    B = 1
    spkcache = torch.zeros(B, 0, 512)
    spkcache_preds = None
    fifo = torch.zeros(B, 0, 512)
    mean_sil = torch.zeros(B, 512)
    n_sil = torch.zeros(B, dtype=torch.long)
    total = torch.zeros(B, 0, N_SPK)

    for chunk_feat, flen, left, right in feat_chunks(mel, mel_len):
        preds, chunk_pe = chunk_forward(chunk_feat, spkcache)       # preds [1,Sc+Pe,4]
        Sc = spkcache.shape[1]
        lc = round(left / SUB)
        rc = math.ceil(right / SUB)
        chunk_len = chunk_pe.shape[1] - lc - rc
        chunk_emb = chunk_pe[:, lc : chunk_len + lc]
        chunk_preds = preds[:, Sc + lc : Sc + chunk_len + lc]

        fifo = torch.cat([fifo, chunk_emb], dim=1)
        fifo_preds = torch.cat([preds[:, Sc : Sc + FIFO_LEN], chunk_preds], dim=1)

        if fifo.shape[1] > FIFO_LEN:  # always true (FIFO_LEN=0)
            pop = UPDATE_PERIOD
            pop = max(pop, chunk_len - FIFO_LEN + 0)
            pop = min(pop, 0 + chunk_len)
            pop_embs = fifo[:, :pop]
            pop_preds = fifo_preds[:, :pop]
            mean_sil, n_sil = get_silence_profile(mean_sil, n_sil, pop_embs, pop_preds)
            fifo = fifo[:, pop:]
            spkcache = torch.cat([spkcache, pop_embs], dim=1)
            if spkcache_preds is not None:
                spkcache_preds = torch.cat([spkcache_preds, pop_preds], dim=1)
            if spkcache.shape[1] > SPKCACHE_LEN:
                if spkcache_preds is None:
                    spkcache_preds = torch.cat([preds[:, :Sc], pop_preds], dim=1)
                spkcache, spkcache_preds = compress_spkcache(spkcache, spkcache_preds, mean_sil)

        total = torch.cat([total, chunk_preds], dim=1)
    return total


def main():
    d = np.load(os.path.join(HERE, "stream_ref.npz"))
    mel = torch.from_numpy(d["mel"]).float()
    mel_len = torch.from_numpy(d["mel_len"]).long()
    ref = torch.from_numpy(d["total_preds"]).float()

    model = Sortformer().eval()
    load_ckpt(model, CKPT)
    total = run(eager_forward(model), mel, mel_len)
    n = min(total.shape[1], ref.shape[1])
    got, exp = total[:, :n], ref[:, :n]
    cos = F.cosine_similarity(got.reshape(-1), exp.reshape(-1), dim=0).item()
    maxd = (got - exp).abs().max().item()
    # activity agreement at 0.5 threshold (the actual diarization decision)
    agree = ((got > 0.5) == (exp > 0.5)).float().mean().item()
    print(f"total_preds host {tuple(total.shape)} vs ref {tuple(ref.shape)}")
    print(f"cos {cos:.6f}  max|Δ| {maxd:.5f}  activity-agree {agree*100:.2f}%")
    ok = cos > 0.999 and total.shape[1] == ref.shape[1]
    print("->", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
