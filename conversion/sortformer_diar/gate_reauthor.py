"""Gate the plain-torch re-authoring (sortformer_model.py) against NeMo intermediates in
chunk_io.npz, stage by stage, for both captured chunks. Run in coreai-models venv (torch 2.9):

    coreai-models/.venv/bin/python gate_reauthor.py
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F
from sortformer_model import Sortformer, load_ckpt, build_pos_emb, NEG_INF

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(HERE, "_nemo", "model_weights.ckpt")


def cos(a, b):
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return F.cosine_similarity(a, b, dim=0).item()


def main():
    d = np.load(os.path.join(HERE, "chunk_io.npz"))
    model = Sortformer().eval()
    load_ckpt(model, CKPT)
    print("[weights] strict load OK")

    worst = 1.0
    for c in ("c0", "c1"):
        mel_chunk = torch.from_numpy(d[f"{c}_mel_chunk"]).float()          # [1,Tf,128]
        concat = torch.from_numpy(d[f"{c}_concat"]).float()               # [1,T,512]
        concat_len = int(d[f"{c}_concat_len"].reshape(-1)[0])
        chunk_pe_g = torch.from_numpy(d[f"{c}_chunk_pe"]).float()
        fc_g = torch.from_numpy(d[f"{c}_fc"]).float()
        preds_g = torch.from_numpy(d[f"{c}_preds"]).float()
        T = concat.shape[1]
        print(f"\n=== {c}: mel_chunk{tuple(mel_chunk.shape)} concat{tuple(concat.shape)} "
              f"len={concat_len} ===")

        with torch.no_grad():
            # stage 1: pre_encode
            pe = model.pre_encode(mel_chunk)
            c1_cos = cos(pe, chunk_pe_g)
            print(f"  [1] pre_encode      cos {c1_cos:.6f}  shape {tuple(pe.shape)} vs {tuple(chunk_pe_g.shape)}")

            # captured concat is fully valid + unpadded -> valid = all ones
            valid = torch.ones(1, T)
            conf_att_bias, conv_mask, tf_bias, out_mask = model.masks_from_valid(valid)
            pos_emb = build_pos_emb(T)

            # stage 3: conformer + proj  (feed captured concat)
            fc = model.conformer_proj(concat, pos_emb, conf_att_bias, conv_mask)
            c3_cos = cos(fc, fc_g)
            print(f"  [3] conformer+proj  cos {c3_cos:.6f}")

            # stage 5+6: transformer + head (feed captured fc)
            preds = model.infer(fc_g, tf_bias, out_mask)
            c6_cos = cos(preds, preds_g)
            maxd = (preds - preds_g).abs().max().item()
            print(f"  [6] transf+head     cos {c6_cos:.6f}  max|Δ| {maxd:.5f}")

            # end-to-end on captured concat (pre_encode fed separately since concat already includes it)
            preds_e2e = model.infer(fc, tf_bias, out_mask)
            e2e_cos = cos(preds_e2e, preds_g)
            print(f"  [*] concat->preds   cos {e2e_cos:.6f}  max|Δ| {(preds_e2e-preds_g).abs().max():.5f}")

        worst = min(worst, c1_cos, c3_cos, c6_cos, e2e_cos)

    print(f"\nworst stage cos = {worst:.6f}  ->  {'PASS' if worst > 0.999 else 'FAIL'}")
    raise SystemExit(0 if worst > 0.999 else 1)


if __name__ == "__main__":
    main()
