"""App backend: arbitrary image -> Core AI TripoSplat -> .ply/.splat, with progress on stdout.

Called by the macOS app as a subprocess:
  python app_backend.py --input <img> --out-ply <ply> --out-splat <splat> [--steps 20] [--gaussians 262144]

Emits machine-parseable progress lines: "PROGRESS <stage> <i> <n>" and a final "DONE <ply> <splat>".
Same Core AI path as _run_coreai.py (4 heavy nets on Core AI + eager octree), plus on-the-fly
background removal (BiRefNet) so it accepts raw photos, not just the preprocessed composite.
"""
import sys, os, time, asyncio, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/ — coreai_kit
import numpy as np
import torch
import coreai.runtime as rt
from triposplat import TripoSplatPipeline, encode_image as enc_glue

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "ckpts")
OUTDIR = os.path.join(HERE, "coreai_out")


def log(*a):
    print(*a, flush=True)


class Runner:
    def __init__(self):
        self.loop = asyncio.new_event_loop()

    def load(self, path):
        # default() lets the runtime use GPU/ANE (auto): ~9x faster than cpu_only on the DiT
        # (24s -> 2.6s) with cos vs cpu still 1.000000. cpu_only is only for deterministic parity.
        m = self.loop.run_until_complete(rt.AIModel.load(Path(path), rt.SpecializationOptions.default()))
        return (m, m.load_function("main"))

    def run(self, h, **feed):
        nd = {k: rt.NDArray(np.ascontiguousarray(np.asarray(v))) for k, v in feed.items()}
        return {k: v.numpy() for k, v in self.loop.run_until_complete(h[1](nd)).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-ply", required=True)
    ap.add_argument("--out-splat", default=None)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--gaussians", type=int, default=262144)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log("PROGRESS load 0 1")
    rn = Runner()

    def pick(base, fp32name):  # prefer fp16 (half size, ~lossless), fall back to fp32.
        p16 = f"{OUTDIR}/{base}_fp16.aimodel"  # int8 NOT used: it desaturates this model.
        return p16 if os.path.isdir(p16) else f"{OUTDIR}/{fp32name}"

    h_dino = rn.load(pick("dinov3", "dinov3_fp32.aimodel"))
    h_vae = rn.load(pick("vae", "flux2_vae_enc_fp32.aimodel"))
    h_dit = rn.load(pick("dit", "dit_fp32.aimodel"))
    h_gs = rn.load(pick("gs", "gs_decoder_fp32.aimodel"))

    pipe = TripoSplatPipeline(
        ckpt_path=f"{CK}/diffusion_models/triposplat_fp16.safetensors",
        decoder_path=f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors",
        dinov3_path=f"{CK}/clip_vision/dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path=f"{CK}/vae/flux2-vae.safetensors",
        rmbg_path=f"{CK}/background_removal/birefnet.safetensors",
        device="cpu")
    pipe.decoder.float().eval()
    pipe.rmbg.float().eval()   # background removal runs eager on host

    pipe.dinov3.forward = lambda pixel_values: torch.from_numpy(
        rn.run(h_dino, pixel_values=pixel_values.detach().float().numpy())["feat"])
    pipe.vae_encoder.encode = lambda x, deterministic=True, generator=None: torch.from_numpy(
        rn.run(h_vae, img=x.detach().float().numpy())["feat"])

    def flow_forward(x_t, t, cond):
        o = rn.run(h_dit, latent=x_t["latent"].detach().float().numpy(),
                   camera=x_t["camera"].detach().float().numpy(), t=t.detach().float().numpy(),
                   feature1=cond["feature1"].detach().float().numpy(),
                   feature2=cond["feature2"].detach().float().numpy())
        return {"latent": torch.from_numpy(o["pred_latent"]), "camera": torch.from_numpy(o["pred_camera"])}
    pipe.flow_model.forward = flow_forward
    pipe.flow_model.q_token_length = 8192
    pipe.flow_model.in_channels = 16
    pipe.flow_model.cam_channels = 5

    def gs_forward(x=None, cond=None):
        o = rn.run(h_gs, points=x["points"].detach().float().numpy(), cond=cond.detach().float().numpy())
        return {"features": torch.from_numpy(o["features"])}
    pipe.decoder.gs.forward = gs_forward

    log("PROGRESS preprocess 0 1")
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    torch.manual_seed(args.seed)
    prepared = pipe.preprocess_image(args.input)   # rmbg + crop/resize/composite
    log("PROGRESS encode 0 1")
    cond = enc_glue(prepared, pipe.dinov3, pipe.vae_encoder, generator=gen)

    def cb(i, n):
        log(f"PROGRESS sample {i} {n}")
    log(f"PROGRESS sample 0 {args.steps}")
    out = pipe.sample_latent(cond, steps=args.steps, guidance_scale=3.0, shift=3.0,
                             generator=gen, show_progress=False, callback=cb)

    log("PROGRESS decode 0 1")
    g = pipe.decode_latent(out["latent"], num_gaussians=args.gaussians)

    Path(args.out_ply).parent.mkdir(parents=True, exist_ok=True)
    g.save_ply(args.out_ply)
    if args.out_splat:
        g.save_splat(args.out_splat)
    log(f"DONE {args.out_ply} {args.out_splat or ''}")


if __name__ == "__main__":
    main()
