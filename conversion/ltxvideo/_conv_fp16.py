"""fp16 bundles: fp16 weights, fp32 IO (FP16IO casts float inputs->fp16, output->fp32;
int inputs pass through). Halves bundle size. Gate on GPU/visual, NOT CPU cos
(forcing fp16 compute on CPU overflows attention/norm; see TripoSplat).

Usage: python _conv_fp16.py {dit|vae|t5} [H W F seq]
"""
import sys
import numpy as np
import torch
import torch.nn as nn

import _common as C
import coreai_kit as K


class FP16IO(nn.Module):
    def __init__(self, model, wdtype=torch.float16):
        super().__init__()
        self.wdtype = wdtype
        self.model = model.to(wdtype)

    def forward(self, *args):
        cast = [a.to(self.wdtype) if a.is_floating_point() else a for a in args]
        out = self.model(*cast)
        return out.float()


def main():
    net = sys.argv[1]
    H = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 256
    F = int(sys.argv[4]) if len(sys.argv) > 4 else 25
    TS = int(sys.argv[5]) if len(sys.argv) > 5 else 256
    bf16 = "--bf16" in sys.argv
    wdtype = torch.bfloat16 if bf16 else torch.float16
    suffix = "bf16" if bf16 else "fp16"
    lf, lh, lw = C.latent_dims(H, W, F)
    n = lf * lh * lw
    torch.manual_seed(0)
    FP16IO_ = lambda m: FP16IO(m, wdtype)

    if net == "dit":
        from _conv_dit import DiTWrap
        model = FP16IO_(DiTWrap(C.load_dit()))
        ex = (torch.randn(1, n, C.LATENT_CH), C.build_indices_grid(F, H, W),
              torch.randn(1, TS, C.CAPTION_CH), torch.ones(1, TS), torch.full((1, 1), 0.7))
        names_in = ["hidden_states", "indices_grid", "encoder_hidden_states",
                    "encoder_attention_mask", "timestep"]
        names_out = ["sample"]
    elif net == "vae":
        from _conv_vae import VAEDecWrap
        ts = (1, 3, lf * C.TEMPORAL, lh * C.SPATIAL, lw * C.SPATIAL)
        model = FP16IO_(VAEDecWrap(C.load_vae(), ts))
        ex = (torch.randn(1, C.LATENT_CH, lf, lh, lw), torch.tensor([0.05]))
        names_in, names_out = ["latent", "timestep"], ["pixels"]
    elif net == "t5":
        from transformers import T5EncoderModel, T5Tokenizer
        from _conv_t5 import T5Wrap, PROMPT, T5_DIR
        tok = T5Tokenizer.from_pretrained(T5_DIR, subfolder="tokenizer")
        enc = T5EncoderModel.from_pretrained(T5_DIR, subfolder="text_encoder").eval()
        ti = tok(PROMPT, padding="max_length", max_length=TS, truncation=True,
                 add_special_tokens=True, return_tensors="pt")
        # T5Wrap casts ids->long; FP16IO would skip int. Wrap enc only (mask is float).
        model = FP16IO_(T5Wrap(enc))
        ex = (ti.input_ids.to(torch.int32), ti.attention_mask.to(torch.float32))
        names_in, names_out = ["input_ids", "attention_mask"], ["text_embeds"]
    else:
        raise SystemExit("net must be dit|vae|t5")

    model = model.eval()
    out = f"coreai_out/{net}_{suffix}.aimodel"
    K.convert(model, ex, names_in, names_out, out, optimize=False)
    print(f"[{net} {suffix}] converted -> {out}")


if __name__ == "__main__":
    main()
