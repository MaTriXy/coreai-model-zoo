"""Net #5: end-to-end TripoSplat pipeline running the 4 heavy nets on Core AI.

Strategy: load the normal torch pipeline (host glue: preprocess, the EAGER octree sampler,
_build_gaussians, save_ply/splat) and monkeypatch the 4 heavy forwards to call the converted
coreai_out/*.aimodel bundles. Everything else (encode_image padding/layer_norm, FlowEulerCfgSampler,
decode_latent, _build_gaussians) is reused unchanged.

Octree probability decoder stays eager (data-dependent sampler). VAE encode uses the deterministic
(mean) path — the converted bundle is deterministic; matches a deterministic torch reference.

Usage: python _run_coreai.py [steps]   (default 6 for a fast smoke run; use 20 for full quality)
"""
import sys, os, time, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/ — coreai_kit
import numpy as np
import torch
import coreai.runtime as rt
from PIL import Image
from triposplat import TripoSplatPipeline, encode_image as _encode_image_glue

CK = "ckpts"
OUTDIR = "coreai_out"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
PREP = "preprocessed_image.webp"   # reuse the already-preprocessed composite -> skip rmbg


class Runner:
    """Single persistent event loop; load each .aimodel once, call many times (CPU runtime)."""
    def __init__(self):
        self.loop = asyncio.new_event_loop()

    def load(self, path):
        opt = rt.SpecializationOptions.cpu_only()
        model = self.loop.run_until_complete(rt.AIModel.load(Path(path), opt))
        fn = model.load_function("main")
        return (model, fn)

    def run(self, handle, **feed):
        _, fn = handle
        nd = {k: rt.NDArray(np.ascontiguousarray(np.asarray(v))) for k, v in feed.items()}
        res = self.loop.run_until_complete(fn(nd))
        return {k: v.numpy() for k, v in res.items()}


def main():
    print(f"steps={STEPS}", flush=True)
    rn = Runner()
    print("loading Core AI bundles ...", flush=True)
    t0 = time.time()
    h_dino = rn.load(f"{OUTDIR}/dinov3_fp32.aimodel")
    h_vae  = rn.load(f"{OUTDIR}/flux2_vae_enc_fp32.aimodel")
    h_dit  = rn.load(f"{OUTDIR}/dit_fp32.aimodel")
    h_gs   = rn.load(f"{OUTDIR}/gs_decoder_fp32.aimodel")
    print(f"bundles loaded in {time.time()-t0:.1f}s", flush=True)

    # Torch pipeline for host glue (weights of the 4 patched nets are unused).
    print("constructing torch pipeline (host glue) ...", flush=True)
    pipe = TripoSplatPipeline(
        ckpt_path=f"{CK}/diffusion_models/triposplat_fp16.safetensors",
        decoder_path=f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors",
        dinov3_path=f"{CK}/clip_vision/dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path=f"{CK}/vae/flux2-vae.safetensors",
        rmbg_path=f"{CK}/background_removal/birefnet.safetensors",
        device="cpu",
    )
    pipe.decoder.float().eval()   # octree stays eager -> fp32 on CPU

    # ---- monkeypatch the 4 heavy forwards to Core AI ----
    dit_calls = [0]

    def dino_forward(pixel_values):
        o = rn.run(h_dino, pixel_values=pixel_values.detach().float().numpy())
        return torch.from_numpy(o["feat"])
    pipe.dinov3.forward = dino_forward

    def vae_encode(x, deterministic=True, generator=None):
        o = rn.run(h_vae, img=x.detach().float().numpy())
        return torch.from_numpy(o["feat"])
    pipe.vae_encoder.encode = vae_encode

    def flow_forward(x_t, t, cond):
        dit_calls[0] += 1
        o = rn.run(h_dit,
                   latent=x_t["latent"].detach().float().numpy(),
                   camera=x_t["camera"].detach().float().numpy(),
                   t=t.detach().float().numpy(),
                   feature1=cond["feature1"].detach().float().numpy(),
                   feature2=cond["feature2"].detach().float().numpy())
        return {"latent": torch.from_numpy(o["pred_latent"]),
                "camera": torch.from_numpy(o["pred_camera"])}
    pipe.flow_model.forward = flow_forward
    # sampler reads these attributes off flow_model:
    pipe.flow_model.q_token_length = 8192
    pipe.flow_model.in_channels = 16
    pipe.flow_model.cam_channels = 5

    def gs_forward(x=None, cond=None):
        o = rn.run(h_gs,
                   points=x["points"].detach().float().numpy(),
                   cond=cond.detach().float().numpy())
        return {"features": torch.from_numpy(o["features"])}
    pipe.decoder.gs.forward = gs_forward

    # ---- run ----
    prepared = Image.open(PREP).convert("RGB")
    gen = torch.Generator(device="cpu").manual_seed(42)
    torch.manual_seed(42)  # octree.sample uses global torch.rand

    print("encode_image (Core AI DINOv3 + VAE) ...", flush=True)
    t0 = time.time()
    cond = _encode_image_glue(prepared, pipe.dinov3, pipe.vae_encoder, generator=gen)
    print(f"  feature1={tuple(cond['feature1'].shape)} feature2={tuple(cond['feature2'].shape)} "
          f"({time.time()-t0:.1f}s)", flush=True)

    print(f"sample_latent ({STEPS} steps, CFG -> ~{2*STEPS} DiT calls on Core AI) ...", flush=True)
    t0 = time.time()
    out = pipe.sample_latent(cond, steps=STEPS, guidance_scale=3.0, shift=3.0,
                             generator=gen, show_progress=False)
    print(f"  latent={tuple(out['latent'].shape)}  dit_calls={dit_calls[0]}  ({time.time()-t0:.1f}s)",
          flush=True)
    lat = out["latent"]
    print(f"  latent stats: mean={lat.mean():.4f} std={lat.std():.4f} "
          f"min={lat.min():.4f} max={lat.max():.4f} nan={torch.isnan(lat).any().item()}", flush=True)

    print("decode_latent (eager octree.sample -> Core AI gs decoder) ...", flush=True)
    t0 = time.time()
    g = pipe.decode_latent(out["latent"], num_gaussians=262144)
    print(f"  decoded ({time.time()-t0:.1f}s)", flush=True)

    xyz = g._xyz
    print(f"  gaussians={xyz.shape[0]}  xyz range=[{xyz.min():.3f},{xyz.max():.3f}] "
          f"nan={torch.isnan(xyz).any().item()}", flush=True)

    g.save_ply("output_coreai.ply")
    g.save_splat("output_coreai.splat")
    sz_ply = os.path.getsize("output_coreai.ply"); sz_splat = os.path.getsize("output_coreai.splat")
    print(f"=== WROTE output_coreai.ply ({sz_ply/1e6:.1f}MB) + output_coreai.splat ({sz_splat/1e6:.1f}MB) ===",
          flush=True)


main()
