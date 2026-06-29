"""Convert the T5-XXL text encoder (PixArt subfolder) to Core AI.

T5 -> last_hidden_state (1, 256, 4096). int32 input_ids (cast to long inside the
graph; int64 graph inputs can error on the runtime). Gate on a real tokenized prompt.

Usage: python _conv_t5.py [seq_len]
"""
import sys
import numpy as np
import torch
import torch.nn as nn

import _common as C
import coreai_kit as K

PROMPT = ("A clear glass of water on a wooden table, slow motion droplet falling "
          "into it creating ripples, cinematic, soft natural light")
T5_DIR = "ckpts/pixart"


class T5Wrap(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, input_ids, attention_mask):
        ids = input_ids.to(torch.long)  # int32 in -> long for embedding gather
        return self.enc(input_ids=ids, attention_mask=attention_mask)[0]


def main():
    seq = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    from transformers import T5EncoderModel, T5Tokenizer

    tok = T5Tokenizer.from_pretrained(T5_DIR, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(T5_DIR, subfolder="text_encoder").to(
        torch.float32).eval()
    print(f"[t5] loaded encoder, {sum(p.numel() for p in enc.parameters())/1e9:.2f}B params")

    ti = tok(PROMPT, padding="max_length", max_length=seq, truncation=True,
             add_special_tokens=True, return_tensors="pt")
    input_ids = ti.input_ids.to(torch.int32)
    attn = ti.attention_mask.to(torch.float32)
    print(f"[t5] input_ids {tuple(input_ids.shape)} nonpad={int(attn.sum())}")

    model = T5Wrap(enc).eval()
    ex = (input_ids, attn)
    names_in = ["input_ids", "attention_mask"]
    names_out = ["text_embeds"]

    with torch.no_grad():
        ref = model(*ex).numpy()
    print(f"[t5] eager out {ref.shape} mean={ref.mean():.4f} std={ref.std():.4f}")

    out = "coreai_out/t5_fp32.aimodel"
    K.convert(model, ex, names_in, names_out, out, optimize=False)
    print("[t5] converted (optimize=False)")

    feed = {"input_ids": input_ids.numpy(), "attention_mask": attn.numpy()}
    got = K.run(out, feed, compute="cpu")["text_embeds"]
    print(f"[t5] coreai out {got.shape} mean={got.mean():.4f} std={got.std():.4f}")
    print(f"[t5] COS = {C.cos(got, ref):.6f}  maxdiff={np.abs(got-ref).max():.3e}")


if __name__ == "__main__":
    main()
