"""End-to-end Z-Image generation driven by the Core AI DiT bundle (bf16) + a VAE.

Reproduces the reference sampler loop on the exported DiT graph:
  latent = noise0
  for s in 0..STEPS-1:
     pos = DiT(patch(latent), cap_cond, adaln[s], rope, pad)   # cond
     neg = DiT(patch(latent), cap_uncond, adaln[s], rope, pad)  # uncond
     noise_pred = -(pos + g*(pos-neg))          # Z-Image CFG (negated)
     latent = latent + dsigma[s] * noise_pred   # FlowMatchEuler
  image = VAE(latent)

Gates vs the fp32 oracle: per-step latent trajectory (vs latent_1..7) and the
decoded image (vs image_ref.png). VAE is torch fp32 here (--vae torch) or the
Core AI VAE bundle (--vae <bundle.aimodel>).

  python pipeline_engine.py <dit_bundle.aimodel> --dtype bf16 [--vae <vae.aimodel>]
"""
import argparse
import asyncio
import json
import os

import numpy as np
import torch

import coreai.runtime as rt
from zimage_host import build_native_inputs, unpatchify_velocity

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")
DTYPE = torch.float16


def load(name, shape):
    return torch.from_numpy(np.fromfile(os.path.join(OUT, f"{name}.f32"), "<f4")).reshape(shape).float()


def corr(a, b):
    return float(np.corrcoef(a.flatten().numpy(), b.flatten().numpy())[0, 1])


def nd(a):
    return rt.NDArray(a.detach().cpu().to(DTYPE).contiguous())


ORDER = ("img_tokens", "cap_feats", "adaln", "x_cos", "x_sin", "cap_cos", "cap_sin",
         "x_pad_mask", "cap_pad_mask")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dit")
    ap.add_argument("--dtype", default="bf16", choices=["fp16", "bf16", "fp32"],
                    help="the graph's IO dtype — fp32 for the shipped --io-fp32 bundles")
    ap.add_argument("--vae", default="torch", help="'torch' or a Core AI VAE bundle path")
    ap.add_argument("--encoder", default="oracle", help="'oracle' or a Core AI encoder bundle path")
    ap.add_argument("--enc-L", type=int, default=64)
    ap.add_argument("--enc-dtype", default=None, choices=["fp16", "bf16", "fp32"],
                    help="encoder graph dtype if it differs from the DiT's")
    ap.add_argument("--prompt", default=None, help="override prompt (default: oracle prompt)")
    ap.add_argument("--tag", default=None,
                    help="use ref_image.py artifacts (noise/sigmas/ref/meta) for this tag — "
                         "required for a size or prompt other than the oracle's")
    args = ap.parse_args()
    global DTYPE
    DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    build_inputs = build_native_inputs

    if args.tag:
        meta = json.load(open(os.path.join(OUT, f"meta_{args.tag}.json")))
        lat, steps, guid = meta["lat"], meta["steps"], meta["guidance"]
        branches = 2 if guid > 0 else 1
        sigmas = load(f"sigmas_{args.tag}", (steps + 1,))
        ref_png = f"ref_{args.tag}.png"
        noise_name = f"noise_{args.tag}"
    else:
        meta = json.load(open(os.path.join(OUT, "meta.json")))
        lat, steps, branches, guid = meta["lat"], meta["steps"], meta["branches"], meta["guidance"]
        sigmas = load("sigmas", (steps + 1,))
        ref_png = "image_ref.png"
        noise_name = "noise0"
    Lc, Lu = meta.get("cap_cond_L", 0), meta.get("cap_uncond_L", 0)
    dsigma = torch.tensor([sigmas[i + 1] - sigmas[i] for i in range(steps)])

    # Host prep needs only the transformer (patchify / RoPE / unpatchify / t_embedder).
    # The tokenizer + embed_tokens feed the encoder graph; embed_tokens is a lookup, so
    # bf16 is plenty. The fp32 VAE is loaded only for --vae torch. Loading the whole
    # fp32 ZImagePipeline here would read ~40 GB for nothing.
    from diffusers import ZImageTransformer2DModel
    print("[engine] loading host-prep components ...", flush=True)
    MODEL = "Tongyi-MAI/Z-Image-Turbo"
    rm = ZImageTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=torch.float32).eval()
    prompt = args.prompt or meta["prompt"]

    if args.encoder == "oracle":
        assert args.tag is None, "--tag needs a real encoder (oracle caps are prompt-specific)"
        cap_cond = load("cap_cond", (1, Lc, 2560))[0]
        cap_uncond = load("cap_uncond", (1, Lu, 2560))[0]
    else:
        import time as _t
        print("[engine] running Core AI encoder for cond + uncond caps ...", flush=True)
        _e0 = _t.time()
        enc = (await rt.AIModel.load(args.encoder, rt.SpecializationOptions.default())).load_function("main")

        enc_dt = ({"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
                  .get(args.enc_dtype or "", DTYPE))

        # The shipped encoder graph takes input_ids (embed_tokens is inside it), so the host
        # needs a tokenizer and nothing else — no 7.5 GB text_encoder just for a lookup table.
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL, subfolder="tokenizer")
        L = args.enc_L

        async def encode(text):
            templated = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                add_generation_prompt=True, enable_thinking=True)
            valid_ids = tok(templated, return_tensors="pt").input_ids
            Lv = valid_ids.shape[1]
            assert Lv <= L, f"prompt is {Lv} tokens; the encoder graph is fixed at {L}"
            ids = torch.full((1, L), tok.pad_token_id or 0, dtype=torch.int32)
            ids[0, :Lv] = valid_ids[0, :Lv].to(torch.int32)
            neg = torch.finfo(enc_dt).min
            m = torch.triu(torch.full((L, L), neg), 1)
            m[:, Lv:] = neg                                    # padding keys are unattendable
            r = await enc(inputs={"input_ids": rt.NDArray(ids.contiguous()),
                                  "mask": rt.NDArray(m[None, None].to(enc_dt).contiguous())})
            return torch.as_tensor(r["penultimate"].numpy().astype(np.float32))[0, :Lv]  # [Lv,2560]

        _load_s = _t.time() - _e0
        _e1 = _t.time()
        cap_cond = await encode(prompt)
        cap_uncond = await encode("")
        print(f"[engine] encoder: load {_load_s:.1f}s + 2 calls {_t.time()-_e1:.2f}s", flush=True)

    dit = (await rt.AIModel.load(args.dit, rt.SpecializationOptions.default())).load_function("main")

    async def velocity(latent, cap, adaln):
        ins = build_inputs(rm, latent, cap)
        ins["adaln"] = adaln
        r = await dit(inputs={k: nd(ins[k]) for k in ORDER})
        u = torch.as_tensor(r["velocity"].numpy().astype(np.float32))
        return unpatchify_velocity(rm, u, ins["x_size"], ins["n_img"])  # [C,1,H,W]

    latent = load(noise_name, (1, 16, lat, lat))
    print(f"[engine] denoising {steps} steps (g={guid}, lat={lat}, "
          f"n_cap cond/uncond {((cap_cond.shape[0]+31)//32)*32}/{((cap_uncond.shape[0]+31)//32)*32}) ...", flush=True)
    import time
    t0 = time.time()
    with torch.no_grad():
        for s in range(steps):
            t_norm = float(1.0 - sigmas[s])
            adaln = rm.t_embedder(torch.tensor([t_norm]) * rm.t_scale)
            lat_in = latent[0]  # [C,H,W] -> need [C,1,H,W]
            lat_in = lat_in.unsqueeze(1)
            pos = await velocity(lat_in, cap_cond, adaln)
            if branches == 2 and guid > 0:
                neg = await velocity(lat_in, cap_uncond, adaln)
                pred = pos + guid * (pos - neg)
            else:
                pred = pos
            if torch.isnan(pos).any():
                print(f"  !! step {s}: DiT cond output NaN (latent|max|={float(latent.abs().max()):.1f}, "
                      f"adaln|max|={float(adaln.abs().max()):.2f})", flush=True)
            noise_pred = -pred                       # [C,1,H,W]
            latent = latent + dsigma[s] * noise_pred.squeeze(1)[None]
            if args.prompt is None and args.tag is None and s + 1 < steps:
                ref = load(f"latent_{s+1}", (1, 16, 1, lat, lat)).squeeze(2)
                print(f"  step {s}: latent vs oracle corr {corr(latent, ref):.6f}")

    dt = time.time() - t0
    n_fwd = steps * (2 if (branches == 2 and guid > 0) else 1)
    print(f"[engine] denoise: {dt:.1f}s for {steps} steps ({n_fwd} DiT forwards, {dt/n_fwd:.2f}s each)", flush=True)

    # decode
    tv = time.time()
    unscale = latent / 0.3611 + 0.1159
    if args.vae == "torch":
        from diffusers import AutoencoderKL
        _vae = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32).eval()
        img = _vae.decode(unscale).sample
    else:
        vae = (await rt.AIModel.load(args.vae, rt.SpecializationOptions.default())).load_function("main")
        r = await vae(inputs={"z": rt.NDArray(latent.to(torch.float32).contiguous())})
        img = torch.as_tensor(r["image"].numpy().astype(np.float32))
    print(f"[engine] vae: {time.time()-tv:.1f}s (incl. load)", flush=True)
    im = ((img[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().byte().numpy()
    from PIL import Image
    stem = f"engine_{args.tag}" if args.tag else ("engine_image" if args.prompt is None else "engine_prompt")
    out_png = os.path.join(OUT, f"{stem}.png")
    Image.fromarray(im).save(out_png)

    # only a tagged run (own reference) or the untouched oracle run has a valid reference
    compare = args.tag is not None or args.prompt is None
    ref_path = os.path.join(OUT, ref_png)
    if compare and os.path.exists(ref_path):
        ref_img = np.asarray(Image.open(ref_path).convert("RGB")).astype(np.float32)
        if ref_img.shape == im.shape:
            mse = float(((im.astype(np.float32) - ref_img) ** 2).mean())
            psnr = 10 * np.log10(255 ** 2 / mse) if mse > 0 else 99.0
            print(f"[engine] saved {out_png}  |  vs {ref_png} PSNR {psnr:.2f} dB", flush=True)
            return
    print(f"[engine] saved {out_png}  |  prompt: {prompt!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
