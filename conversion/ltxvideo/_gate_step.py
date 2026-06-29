"""Capture the REAL DiT inputs at sampler step 0 from a torch (fp32) run, then
feed those exact tensors to the Core AI bundle and compare noise_pred. Definitive
integration gate (removes sampler/RNG/precision-rollout from the comparison)."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import types
import numpy as np, torch
import _common as C, coreai_kit as K
from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

PROMPT = ("A clear glass of water on a wooden table, slow motion droplet falling "
          "into it creating ripples, cinematic, soft natural light")

pipe = C.build_pipeline(device="cpu", dtype=torch.float32)
orig = type(pipe.transformer).forward
steps = []

def cap_forward(self, hidden_states, indices_grid=None, encoder_hidden_states=None,
                timestep=None, encoder_attention_mask=None, return_dict=True, **kw):
    out = orig(self, hidden_states, indices_grid=indices_grid,
               encoder_hidden_states=encoder_hidden_states, timestep=timestep,
               encoder_attention_mask=encoder_attention_mask, return_dict=False, **kw)[0]
    steps.append(dict(
        hs=hidden_states.detach().float().numpy(),
        ig=indices_grid.detach().float().numpy(),
        eh=encoder_hidden_states.detach().float().numpy(),
        em=encoder_attention_mask.detach().float().numpy(),
        ts=timestep.detach().float().numpy(),
        out=out.detach().float().numpy(),
    ))
    return (out,)

pipe.transformer.forward = types.MethodType(cap_forward, pipe.transformer)
gen = torch.Generator(device="cpu").manual_seed(42)
pipe(prompt=PROMPT, negative_prompt="bad", num_inference_steps=8, guidance_scale=1,
     stg_scale=0, rescaling_scale=1, skip_layer_strategy=SkipLayerStrategy.AttentionValues,
     generator=gen, output_type="pt", height=256, width=256, num_frames=25, frame_rate=24,
     decode_timestep=0.05, decode_noise_scale=0.025, stochastic_sampling=True,
     is_video=True, vae_per_channel_normalize=True)

print(f"\n[gate] {len(steps)} DiT calls captured")
for i, c in enumerate(steps):
    got = K.run("coreai_out/dit_fp32.aimodel",
                {"hidden_states": c["hs"], "indices_grid": c["ig"],
                 "encoder_hidden_states": c["eh"], "encoder_attention_mask": c["em"],
                 "timestep": c["ts"]}, compute="cpu")["sample"]
    print(f"  step{i} t={c['ts'].ravel()[0]:.4f} COS={C.cos(got, c['out']):.6f} "
          f"maxdiff={np.abs(got-c['out']).max():.2e} torch.std={c['out'].std():.3f}")
