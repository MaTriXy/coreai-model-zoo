"""End-to-end LTX-Video generation with the heavy nets running on Core AI.

Reuses the real torch pipeline (FlowMatch sampler, patchify, indices_grid, tone-map,
decode-noise, decode timestep) but monkeypatches transformer.forward + vae.decoder.forward
(and optionally text_encoder) to call the Core AI .aimodel bundles on GPU.

The bundles MUST have been converted at the SAME (H, W, num_frames, seq) as requested.
Usage: python _run_coreai.py [H W num_frames seed] [--t5] [--cpu]
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import time
import asyncio
from pathlib import Path
import numpy as np
import torch

import _common as C
import coreai.runtime as rt
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

PROMPT = os.environ.get("PROMPT",
          "A clear glass of water on a wooden table, slow motion droplet falling "
          "into it creating ripples, cinematic, soft natural light")
NEG = "worst quality, inconsistent motion, blurry, jittery, distorted"


class Runner:
    """Persistent Core AI loader: load each bundle once, run many times (GPU)."""

    def __init__(self, paths, compute="default"):
        # NOTE: pass an explicit SpecializationOptions, NOT None. AIModel.load(None)
        # trips "MPSGraph Unresolved symbol" on the GPU path; default() works.
        self.opt = (rt.SpecializationOptions.cpu_only() if compute == "cpu"
                    else rt.SpecializationOptions.default())
        self.loop = asyncio.new_event_loop()
        self.fns = {}
        self.models = {}  # keep AIModel refs alive (else GC'd -> garbage outputs)
        for name, p in paths.items():
            m = self.loop.run_until_complete(rt.AIModel.load(Path(p), self.opt))
            self.models[name] = m
            self.fns[name] = m.load_function("main")
            print(f"[runner] loaded {name} <- {p}")

    def run(self, name, feed):
        nd = {k: rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}
        res = self.loop.run_until_complete(self.fns[name](nd))
        return {k: v.numpy() for k, v in res.items()}


def np32(t):
    return t.detach().to("cpu", torch.float32).numpy()


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    H = int(args[0]) if len(args) > 0 else 256
    W = int(args[1]) if len(args) > 1 else 256
    F = int(args[2]) if len(args) > 2 else 25
    seed = int(args[3]) if len(args) > 3 else 42
    use_t5 = "--t5" in sys.argv
    compute = "cpu" if "--cpu" in sys.argv else "default"
    dtype = torch.float32 if "--fp32" in sys.argv else torch.bfloat16
    # Host glue always on CPU: PyTorch's MPS Metal context conflicts with Core AI's
    # MPSGraph (load_function fails). The heavy nets run on Core AI (GPU via compute).
    device = "cpu"
    print(f"[coreai] {H}x{W} x{F}f seed={seed} t5_coreai={use_t5} compute={compute} dtype={dtype}")

    pipe = C.build_pipeline(device=device, dtype=dtype)

    prec = "fp16" if "--fp16" in sys.argv else "fp32"
    paths = {"dit": f"coreai_out/dit_{prec}.aimodel", "vae": f"coreai_out/vae_{prec}.aimodel"}
    if use_t5:
        # T5 encoder overflows in fp16 (well-known); bf16 keeps fp32's exponent range.
        t5_prec = "fp16" if "--t5fp16" in sys.argv else (
            "bf16" if "--t5bf16" in sys.argv else "fp32")
        paths["t5"] = f"coreai_out/t5_{t5_prec}.aimodel"
    runner = Runner(paths, compute=compute)

    # --- DiT: replace forward with the Core AI bundle ---
    dit_calls = [0]
    orig_dit = type(pipe.transformer).forward

    def dit_forward(hidden_states, indices_grid=None, encoder_hidden_states=None,
                    timestep=None, encoder_attention_mask=None, return_dict=True, **kw):
        feed = {
            "hidden_states": np32(hidden_states),
            "indices_grid": np32(indices_grid),
            "encoder_hidden_states": np32(encoder_hidden_states),
            "encoder_attention_mask": np32(encoder_attention_mask),
            "timestep": np32(timestep),
        }
        out = runner.run("dit", feed)["sample"]
        if os.environ.get("DBG2"):
            tref = orig_dit(pipe.transformer, hidden_states, indices_grid=indices_grid,
                            encoder_hidden_states=encoder_hidden_states, timestep=timestep,
                            encoder_attention_mask=encoder_attention_mask,
                            return_dict=False)[0]
            tref = np32(tref)
            print(f"  [dit {dit_calls[0]}] t={float(timestep.flatten()[0]):.4f} "
                  f"bundle.std={out.std():.4f} torch.std={tref.std():.4f} "
                  f"cos={C.cos(out, tref):.6f} maxdiff={np.abs(out-tref).max():.2e}")
        dit_calls[0] += 1
        s = torch.from_numpy(out).to(hidden_states.device, hidden_states.dtype)
        return (s,) if not return_dict else type("O", (), {"sample": s})()

    if "--no-dit" not in sys.argv:
        pipe.transformer.forward = dit_forward

    # --- VAE decoder: replace forward with the Core AI bundle (pure decoder) ---
    orig_dtype = pipe.vae.dtype

    def vae_dec_forward(sample, target_shape=None, timestep=None):
        feed = {"latent": np32(sample), "timestep": np32(timestep.flatten())}
        out = runner.run("vae", feed)["pixels"]
        return torch.from_numpy(out).to(sample.device, sample.dtype)

    if "--no-vae" not in sys.argv:
        pipe.vae.decoder.forward = vae_dec_forward

    # --- optional: T5 on Core AI ---
    if use_t5:
        def t5_forward(input_ids, attention_mask=None, **kw):
            feed = {"input_ids": input_ids.detach().cpu().to(torch.int32).numpy(),
                    "attention_mask": np32(attention_mask)}
            out = runner.run("t5", feed)["text_embeds"]
            emb = torch.from_numpy(out).to(input_ids.device)
            return (emb,)
        pipe.text_encoder.forward = t5_forward

    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()
    out = pipe(
        prompt=PROMPT, negative_prompt=NEG,
        num_inference_steps=8, guidance_scale=1, stg_scale=0, rescaling_scale=1,
        skip_layer_strategy=SkipLayerStrategy.AttentionValues,
        generator=gen, output_type="pt",
        height=H, width=W, num_frames=F, frame_rate=24,
        decode_timestep=0.05, decode_noise_scale=0.025, stochastic_sampling=True,
        is_video=True, vae_per_channel_normalize=True,
        mixed_precision=False, offload_to_cpu=False,
    ).images
    dt = time.time() - t0
    print(f"[coreai] generated {tuple(out.shape)} in {dt:.1f}s ({dit_calls[0]} DiT calls)")

    np.save("coreai_video.npy", np32(out))
    path, shape = C.save_video(out, "coreai.mp4", fps=24)
    print(f"[coreai] wrote {path} {shape}")

    # compare to baseline if present
    if os.path.exists("baseline_video.npy"):
        b = np.load("baseline_video.npy")
        if b.shape == np32(out).shape:
            print(f"[coreai] vs baseline COS = {C.cos(np32(out), b):.6f} "
                  f"maxdiff={np.abs(np32(out)-b).max():.3e}")


if __name__ == "__main__":
    main()
