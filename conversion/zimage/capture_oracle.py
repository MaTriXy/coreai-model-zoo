"""Capture the Z-Image-Turbo ground-truth oracle for the Core AI port.

Runs the fp32 diffusers ZImagePipeline once and records, at the transformer
boundary, everything the Core AI graphs must reproduce:

  - image_ref.png            reference decoded image
  - cap_cond / cap_uncond    encoder penultimate hidden [1, L, 2560] per branch
                             (variable L, the true token count; padded on host)
  - per step s (teacher forcing):
      latent_s   [1,16,lat,lat]  latent fed to the transformer at step s
      tnorm_s    scalar          normalized timestep the transformer sees
      adaln_s    [1,256]         t_embedder(t*t_scale) global adaLN input
      vel_pos_s  [1,16,lat,lat]  transformer velocity, cond branch (unpatchified)
      vel_neg_s  [1,16,lat,lat]  transformer velocity, uncond branch
  - noise0, sigmas, dsigma, final_latents, guidance_scale

Gate targets built on this: DiT teacher-forced parity (vel), encoder parity (cap),
VAE parity (final_latents -> image), full-loop e2e (final_latents + image).

Run (coreai-models venv, from conversion/zimage/):
  python capture_oracle.py --size 512 --steps 8 --prompt "..."
"""
import argparse
import json
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")


def savef(t, name):
    a = t.detach().cpu().float().numpy() if hasattr(t, "detach") else np.asarray(t, np.float32)
    np.ascontiguousarray(a, "<f4").tofile(os.path.join(OUT, f"{name}.f32"))
    return a.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--prompt", default="a red apple on a wooden table, studio lighting")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    from diffusers import ZImagePipeline
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    print(f"[oracle] loading pipeline (fp32) ...", flush=True)
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu")
    rm = pipe.transformer
    lat = args.size // 8

    rec = {"latent": [], "tnorm": [], "adaln": [], "vel": []}
    real_fwd = rm.forward

    def fwd_hook(x, t, cap_feats, return_dict=True, **kw):
        # x: list of [C,1,H,W]; t: normalized timestep(s) broadcast to batch; cap_feats: list.
        out = real_fwd(x, t, cap_feats, return_dict=False, **kw)[0]
        # record per-branch velocity (unpatchified [C,1,H,W]); adaln + latent from branch 0.
        rec["latent"].append(x[0].detach().clone())          # [C,1,H,W]
        rec["tnorm"].append(float(t.reshape(-1)[0]))
        rec["adaln"].append(rm.t_embedder(t[:1] * rm.t_scale).detach().clone())  # [1,256]
        rec["vel"].append([o.detach().clone() for o in out])  # list length = branches
        return Transformer2DModelOutput(sample=out) if return_dict else (out,)

    rm.forward = fwd_hook

    sig_holder = {}
    real_step = pipe.scheduler.step

    def step_hook(model_output, timestep, sample, *a, **k):
        sig_holder.setdefault("sample0", sample.detach().clone())
        return real_step(model_output, timestep, sample, *a, **k)

    pipe.scheduler.step = step_hook

    # capture the encoder penultimate hidden (cap_feats) per branch
    cap_cond, cap_uncond = pipe.encode_prompt(
        args.prompt, do_classifier_free_guidance=True, negative_prompt="",
        max_sequence_length=512)
    print(f"[oracle] cap cond L={cap_cond[0].shape[0]} uncond L={cap_uncond[0].shape[0]}", flush=True)
    savef(cap_cond[0][None], "cap_cond")       # [1,L,2560]
    savef(cap_uncond[0][None], "cap_uncond")

    g = torch.Generator("cpu").manual_seed(args.seed)
    print(f"[oracle] generating (size={args.size}, steps={args.steps}, g={args.guidance}) ...", flush=True)
    img = pipe(
        args.prompt, height=args.size, width=args.size,
        num_inference_steps=args.steps, guidance_scale=args.guidance,
        negative_prompt="", generator=g,
    ).images[0]
    img.save(os.path.join(OUT, "image_ref.png"))

    branches = 2 if len(rec["vel"][0]) == 2 else 1
    steps = len(rec["latent"])
    print(f"[oracle] recorded {steps} transformer calls, branches={branches}", flush=True)

    # per-step teacher-forcing tensors
    for s in range(steps):
        savef(rec["latent"][s][None], f"latent_{s}")               # [1,C,1,H,W]
        savef(rec["adaln"][s], f"adaln_{s}")                       # [1,256]
        savef(rec["vel"][s][0][None], f"vel_pos_{s}")             # [1,C,1,H,W]
        if branches == 2:
            savef(rec["vel"][s][1][None], f"vel_neg_{s}")
    savef(sig_holder["sample0"], "noise0")                         # [1,16,lat,lat]

    sched_sig = pipe.scheduler.sigmas.detach().cpu().numpy()
    dsigma = np.array([sched_sig[i + 1] - sched_sig[i] for i in range(steps)], "<f4")
    savef(sched_sig, "sigmas")
    savef(dsigma, "dsigma")

    meta = dict(size=args.size, lat=lat, steps=steps, branches=branches,
                guidance=args.guidance, seed=args.seed, prompt=args.prompt,
                cap_cond_L=int(cap_cond[0].shape[0]), cap_uncond_L=int(cap_uncond[0].shape[0]),
                tnorm=[round(t, 6) for t in rec["tnorm"][::branches]],
                sigmas=[round(float(s), 6) for s in sched_sig])
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=2)
    print(f"[oracle] meta: {json.dumps(meta)}", flush=True)
    print(f"[oracle] saved to {OUT}/  (image_ref.png + teacher-forcing bins)", flush=True)


if __name__ == "__main__":
    main()
