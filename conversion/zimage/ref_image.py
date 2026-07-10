"""Render a fp32 diffusers reference image (and save its initial noise) for an
arbitrary prompt / size / seed, so engine runs at new resolutions or prompt
lengths have a real PSNR gate rather than only a visual check.

  python ref_image.py --prompt "..." --size 1024 --steps 8 --seed 1234 --tag long
Writes oracle/ref_<tag>.png and oracle/noise_<tag>.f32
"""
import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "oracle")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    from diffusers import ZImagePipeline
    print("[ref] loading pipeline (fp32) ...", flush=True)
    pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.float32).to("cpu")

    grab = {}
    real_step = pipe.scheduler.step

    def hook(model_output, timestep, sample, *a, **k):
        grab.setdefault("noise0", sample.detach().clone())
        return real_step(model_output, timestep, sample, *a, **k)

    pipe.scheduler.step = hook

    g = torch.Generator("cpu").manual_seed(args.seed)
    img = pipe(args.prompt, height=args.size, width=args.size,
               num_inference_steps=args.steps, guidance_scale=args.guidance,
               negative_prompt="", generator=g).images[0]
    img.save(os.path.join(OUT, f"ref_{args.tag}.png"))
    np.ascontiguousarray(grab["noise0"].numpy(), "<f4").tofile(os.path.join(OUT, f"noise_{args.tag}.f32"))

    # sigmas are RESOLUTION-DEPENDENT (mu = calculate_shift(image_seq_len)) — save them.
    sig = pipe.scheduler.sigmas.detach().cpu().numpy().astype("<f4")
    sig.tofile(os.path.join(OUT, f"sigmas_{args.tag}.f32"))
    import json
    json.dump(dict(prompt=args.prompt, size=args.size, steps=args.steps, seed=args.seed,
                   guidance=args.guidance, lat=args.size // 8,
                   sigmas=[round(float(s), 6) for s in sig]),
              open(os.path.join(OUT, f"meta_{args.tag}.json"), "w"), indent=2)
    print(f"[ref] saved ref_{args.tag}.png + noise/sigmas/meta_{args.tag}  "
          f"(size={args.size} steps={args.steps} seed={args.seed})", flush=True)
    print(f"[ref] sigmas={[round(float(s),4) for s in sig]}", flush=True)


if __name__ == "__main__":
    main()
