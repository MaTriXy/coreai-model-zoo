# Community port — capture the reference DiT one-forward I/O (oracle for the overlay gate + export inputs).
import json, os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from stable_audio_tools.inference.generation import generate_diffusion_cond

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
cfg = json.load(open(os.path.join(HERE, "model_config.json")))
model = create_model_from_config(cfg)
model.load_state_dict(load_ckpt_state_dict(os.path.join(HERE, "model.safetensors")), strict=False)
model = model.to(DEV).float().eval()

# find the DiffusionTransformer module
dit = None
for m in model.modules():
    if type(m).__name__ == "DiffusionTransformer":
        dit = m; break
assert dit is not None, "DiffusionTransformer not found"
print("[cap] DiT found:", type(dit).__name__)

cap = {}
def pre_hook(mod, args, kwargs):
    if "x" in cap:
        return
    x = args[0] if len(args) > 0 else kwargs.get("x")
    t = args[1] if len(args) > 1 else kwargs.get("t")
    cap["x"] = x.detach().cpu()
    cap["t"] = t.detach().cpu()
    for k in ("cross_attn_cond", "global_embed", "input_concat_cond", "prepend_cond",
              "cross_attn_cond_mask", "global_cond"):
        v = kwargs.get(k)
        if torch.is_tensor(v):
            cap[k] = v.detach().cpu()
    cap["kwarg_keys"] = [k for k in kwargs if torch.is_tensor(kwargs[k])]

def fwd_hook(mod, args, kwargs, output):
    if "out" in cap:
        return
    out = output[0] if isinstance(output, (tuple, list)) else output
    cap["out"] = out.detach().cpu()

h1 = dit.register_forward_pre_hook(pre_hook, with_kwargs=True)
h2 = dit.register_forward_hook(fwd_hook, with_kwargs=True)

with torch.inference_mode():
    generate_diffusion_cond(model=model, steps=8, cfg_scale=1.0,
                            conditioning=[{"prompt": "128 BPM tech house drum loop", "seconds_total": 11}],
                            sample_size=cfg["sample_size"], sample_rate=cfg["sample_rate"], seed=0,
                            device=DEV, return_latents=True)
h1.remove(); h2.remove()

print("[cap] captured DiT inputs:")
for k, v in cap.items():
    if torch.is_tensor(v):
        print(f"   {k}: {tuple(v.shape)} {v.dtype} mean={v.float().mean():.4f} std={v.float().std():.4f}")
    else:
        print(f"   {k}: {v}")
torch.save(cap, os.path.join(HERE, "dit_io.pt"))
print("[cap] saved dit_io.pt")
