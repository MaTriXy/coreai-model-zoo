"""Resident Core AI backend for the CoreAIVideo Mac app (LTX-Video 2B → text→video).

Loads the 3 Core AI bundles (DiT fp16 + VAE fp16 + T5 bf16) ONCE, then serves one
generation per stdin line. Reuses LTX's real FlowMatch sampler / patchify / decode-noise;
only the 3 heavy nets run on Core AI. The T5 torch weights are NOT loaded (the bundle does
that compute) — just the tokenizer — so startup is fast and the runtime dir stays small.

Protocol (line-based, stdout):
  stdin  : "<seed>\\t<prompt>\\n"  per request
  stdout : "READY"                         once, after load
           "PROGRESS <stage> <i> <n>"      stage in {load,encode,sample,decode}
           "DONE <mp4path>"                 on success
           "ERROR <message>"               on failure

Fixed resolution = whatever the staged DiT/VAE bundles were converted at (default 512×768×49f).
"""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import sys
import time
import json
import asyncio
import argparse
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# --- fixed geometry (must match the converted DiT/VAE bundles) ---
H, W, F = 512, 768, 49
SEQ = 256
SPATIAL, TEMPORAL, LATENT_CH, CAPTION_CH = 32, 8, 128, 4096
NEG = "worst quality, inconsistent motion, blurry, jittery, distorted"


def log(*a):
    print(*a, flush=True)


def latent_dims():
    return (F - 1) // TEMPORAL + 1, H // SPATIAL, W // SPATIAL


def build_indices_grid(frame_rate=24.0):
    lf, lh, lw = latent_dims()
    fc, hc, wc = torch.meshgrid(torch.arange(0, lf), torch.arange(0, lh),
                                torch.arange(0, lw), indexing="ij")
    lc = torch.stack([fc, hc, wc], 0).reshape(3, -1).unsqueeze(0).float()
    scale = torch.tensor([TEMPORAL, SPATIAL, SPATIAL], dtype=torch.float32)[None, :, None]
    px = lc * scale
    px[:, 0] = (px[:, 0] + 1 - TEMPORAL).clamp(min=0)
    px[:, 0] = px[:, 0] * (1.0 / frame_rate)
    return px


class Runner:
    """Persistent Core AI loader: load each bundle once, run many (GPU via default())."""

    def __init__(self, paths):
        import coreai.runtime as rt
        self.rt = rt
        self.opt = rt.SpecializationOptions.default()
        self.loop = asyncio.new_event_loop()
        self.fns, self.models = {}, {}
        for name, p in paths.items():
            m = self.loop.run_until_complete(rt.AIModel.load(Path(p), self.opt))
            self.models[name] = m  # keep ref alive (GC'd model -> garbage output)
            self.fns[name] = m.load_function("main")

    def run(self, name, feed):
        nd = {k: self.rt.NDArray(np.ascontiguousarray(v)) for k, v in feed.items()}
        res = self.loop.run_until_complete(self.fns[name](nd))
        return {k: v.numpy() for k, v in res.items()}


class StubT5(nn.Module):
    """Stand-in for the T5 encoder: the bundle does the compute, this only satisfies the
    pipeline's `.parameters().device` / `.dtype` / `forward(...)` accesses."""

    def __init__(self):
        super().__init__()
        self._p = nn.Parameter(torch.zeros(1))  # a real cpu param for .parameters()

    @property
    def dtype(self):
        return torch.float32

    @property
    def device(self):  # pipeline._execution_device walks component .device
        return self._p.device

    def forward(self, input_ids, attention_mask=None, **kw):  # patched at runtime
        raise RuntimeError("StubT5.forward should be monkeypatched to the bundle")


def build_lean_pipeline(ckpt, t5_dir):
    """transformer + vae loaded for real (config + un-normalize buffers); T5 = tokenizer +
    stub (no 19 GB weight load). Heavy forwards get monkeypatched to the bundles by the caller."""
    from transformers import T5Tokenizer
    from ltx_video.models.transformers.transformer3d import Transformer3DModel
    from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
    from ltx_video.schedulers.rf import RectifiedFlowScheduler
    from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
    from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
    from safetensors import safe_open

    with safe_open(ckpt, framework="pt") as f:
        allowed = json.loads(f.metadata()["config"]).get("allowed_inference_steps", None)

    vae = CausalVideoAutoencoder.from_pretrained(ckpt).to(torch.float32).eval()
    transformer = Transformer3DModel.from_pretrained(ckpt).to(torch.float32).eval()
    scheduler = RectifiedFlowScheduler.from_pretrained(ckpt)
    tokenizer = T5Tokenizer.from_pretrained(t5_dir, subfolder="tokenizer")
    patchifier = SymmetricPatchifier(patch_size=1)

    pipe = LTXVideoPipeline(
        transformer=transformer, patchifier=patchifier, text_encoder=StubT5(),
        tokenizer=tokenizer, scheduler=scheduler, vae=vae,
        prompt_enhancer_image_caption_model=None,
        prompt_enhancer_image_caption_processor=None,
        prompt_enhancer_llm_model=None, prompt_enhancer_llm_tokenizer=None,
        allowed_inference_steps=allowed,
    )
    return pipe.to("cpu")


def np32(t):
    return t.detach().to("cpu", torch.float32).numpy()


def save_video(images, path, fps=24):
    from PIL import Image
    vid = np32(images)[0].transpose(1, 2, 3, 0)
    vid = (np.clip(vid, 0, 1) * 255).astype(np.uint8)
    d = tempfile.mkdtemp()
    for i, fr in enumerate(vid):
        Image.fromarray(fr).save(f"{d}/f{i:04d}.png")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(fps), "-i", f"{d}/f%04d.png",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path],
                   check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=str(Path.home() / "CoreAIVideoRuntime"))
    ap.add_argument("--coreai", default=str(Path.home() / "Code/coreai"))
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--outdir", default=str(Path(tempfile.gettempdir()) / "coreaivideo_out"))
    args = ap.parse_args()

    rt_dir = Path(args.runtime)
    sys.path.insert(0, args.coreai)            # coreai_kit (unused) / coreai.runtime
    sys.path.insert(0, str(rt_dir))            # ltx_video package + _common (if staged)
    os.chdir(rt_dir)                           # ckpts/ + coreai_out/ are relative
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

    log("PROGRESS load 0 4")
    runner = Runner({
        "dit": "coreai_out/dit_fp16.aimodel",
        "vae": "coreai_out/vae_fp16.aimodel",
        "t5":  "coreai_out/t5_bf16.aimodel",
    })
    log("PROGRESS load 2 4")
    pipe = build_lean_pipeline("ckpts/ltxv-2b-0.9.6-distilled-04-25.safetensors", "ckpts/pixart")
    log("PROGRESS load 4 4")

    def dit_forward(hidden_states, indices_grid=None, encoder_hidden_states=None,
                    timestep=None, encoder_attention_mask=None, return_dict=True, **kw):
        out = runner.run("dit", {
            "hidden_states": np32(hidden_states), "indices_grid": np32(indices_grid),
            "encoder_hidden_states": np32(encoder_hidden_states),
            "encoder_attention_mask": np32(encoder_attention_mask), "timestep": np32(timestep)})["sample"]
        s = torch.from_numpy(out).to(hidden_states.device, hidden_states.dtype)
        return (s,) if not return_dict else type("O", (), {"sample": s})()

    def vae_dec_forward(sample, target_shape=None, timestep=None):
        out = runner.run("vae", {"latent": np32(sample), "timestep": np32(timestep.flatten())})["pixels"]
        return torch.from_numpy(out).to(sample.device, sample.dtype)

    def t5_forward(input_ids, attention_mask=None, **kw):
        out = runner.run("t5", {"input_ids": input_ids.detach().cpu().to(torch.int32).numpy(),
                                "attention_mask": np32(attention_mask)})["text_embeds"]
        return (torch.from_numpy(out).to(input_ids.device),)

    pipe.transformer.forward = dit_forward
    pipe.vae.decoder.forward = vae_dec_forward
    pipe.text_encoder.forward = t5_forward

    log("READY")

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        try:
            seed_s, _, prompt = line.partition("\t")
            seed = int(seed_s)
            steps = args.steps

            def cb(_pipe, i, t, _d):
                log(f"PROGRESS sample {i + 1} {steps}")
            log("PROGRESS encode 0 1")
            gen = torch.Generator(device="cpu").manual_seed(seed)
            out = pipe(
                prompt=prompt, negative_prompt=NEG,
                num_inference_steps=steps, guidance_scale=1, stg_scale=0, rescaling_scale=1,
                skip_layer_strategy=SkipLayerStrategy.AttentionValues,
                generator=gen, output_type="pt",
                height=H, width=W, num_frames=F, frame_rate=24,
                decode_timestep=0.05, decode_noise_scale=0.025, stochastic_sampling=True,
                is_video=True, vae_per_channel_normalize=True,
                mixed_precision=False, offload_to_cpu=False,
                callback_on_step_end=cb,
            ).images
            log("PROGRESS decode 0 1")
            stamp = int(time.time())
            outp = str(Path(args.outdir) / f"gen_{stamp}.mp4")
            save_video(out, outp, fps=24)
            log(f"DONE {outp}")
        except Exception as e:  # noqa: BLE001
            log(f"ERROR {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
