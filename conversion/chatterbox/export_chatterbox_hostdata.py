"""Export the host-side data the Swift ChatterboxTTS pipeline needs alongside the 4 Core AI
graphs (t3 / s3gen_encoder / s3gen_estimator / s3gen_hift_trunk). The graphs are embeds-in /
feature-in; the host assembles embeds, samples, runs the CFM Euler loop, and does the vocoder
source + STFT/iSTFT. This dumps the weights/constants + default-voice conditioning as raw
float32 .bin (row-major) + a config.json, mirroring the Kokoro host-data pattern.

Run in the coreai venv (chatterbox installed): .venv/bin/python .../export_chatterbox_hostdata.py --out-dir exports/chatterbox_hostdata
"""
import argparse
import json
import os

import numpy as np
import torch


def dump(arr: torch.Tensor, path: str):
    np.ascontiguousarray(arr.detach().cpu().float().numpy()).astype(np.float32).tofile(path)


def main():
    import perth
    class _NW:
        def apply_watermark(self, w, sample_rate=None, **k):
            return w
    perth.PerthImplicitWatermarker = _NW
    import chatterbox.models.s3gen.decoder as dec
    dec.add_optional_chunk_mask = lambda xs, m, a, b, c, scs, ndlc: m
    from chatterbox.tts import ChatterboxTTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exports/chatterbox_hostdata")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    m = ChatterboxTTS.from_pretrained(device="cpu")
    t3, s3 = m.t3, m.s3gen
    hp = t3.hp
    o = args.out_dir

    # --- T3 embed tables (host assembles inputs_embeds) ---
    dump(t3.text_emb.weight, f"{o}/t3_text_emb.bin")          # [704, 1024]
    dump(t3.speech_emb.weight, f"{o}/t3_speech_emb.bin")      # [8194, 1024]
    dump(t3.text_pos_emb.emb.weight, f"{o}/t3_text_pos.bin")  # [2050, 1024]
    dump(t3.speech_pos_emb.emb.weight, f"{o}/t3_speech_pos.bin")  # [4100, 1024]

    # --- default-voice conditioning: precompute the T3 cond prefix (perceiver over speaker_emb
    #     + emotion) so the host needs NO cond_enc/perceiver ---
    cond = m.conds.t3
    with torch.no_grad():
        cond_prefix = t3.prepare_conditioning(cond)  # [1, len_cond, 1024]
    dump(cond_prefix[0], f"{o}/t3_cond_prefix.bin")
    len_cond = cond_prefix.shape[1]

    # --- S3Gen host inputs ---
    dump(s3.flow.input_embedding.weight, f"{o}/s3_input_emb.bin")         # [6561, 512] token->feat
    gen = m.conds.gen
    with torch.no_grad():
        spk = s3.flow.spk_embed_affine_layer(torch.nn.functional.normalize(gen["embedding"], dim=1))
    dump(spk, f"{o}/s3_spk.bin")                                           # [1, 80] (affined)
    dump(gen["prompt_token"].float(), f"{o}/s3_prompt_token.bin")          # [1, 157]
    dump(gen["prompt_feat"], f"{o}/s3_prompt_feat.bin")                    # [1, 314, 80]

    # --- HiFT host DSP constants (STFT window, istft params, source params) ---
    hift = s3.mel2wav
    dump(hift.stft_window, f"{o}/hift_stft_window.bin")

    cfg = {
        "t3": {
            "start_text_token": hp.start_text_token, "stop_text_token": hp.stop_text_token,
            "start_speech_token": hp.start_speech_token, "stop_speech_token": hp.stop_speech_token,
            "n_channels": hp.n_channels, "text_vocab": hp.text_tokens_dict_size,
            "speech_vocab": hp.speech_tokens_dict_size, "len_cond": int(len_cond),
            "layers": 30, "heads": 16, "head_dim": 64,
        },
        "s3gen": {
            "n_mels": 80, "cfm_steps": 10, "t_scheduler": "cosine",
            "input_emb_dim": 512, "speech_codebook": 6561,
        },
        "hift": {
            "n_fft": hift.istft_params["n_fft"], "hop_len": hift.istft_params["hop_len"],
            "sr": m.sr,
        },
        "chat_template": "<|begin_of_text|><|User|>%@<|Assistant|>",
    }
    with open(f"{o}/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # tokenizer (the chatterbox EnTokenizer -> ship the vocab file it wraps)
    try:
        m.tokenizer.tokenizer.save(f"{o}/tokenizer.json")
    except Exception as e:
        print("tokenizer dump note:", e)

    print("host-data written to", o)
    for fn in sorted(os.listdir(o)):
        print("  ", fn, os.path.getsize(os.path.join(o, fn)), "B")


if __name__ == "__main__":
    main()
