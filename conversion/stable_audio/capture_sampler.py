# Community port — capture the exact 8-step sampler trajectory (t, x_in, v_out) to derive the host loop.
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
dit = next(m for m in model.modules() if type(m).__name__ == "DiffusionTransformer")

traj = []
def pre(mod, args, kwargs):
    x = args[0] if args else kwargs.get("x")
    t = args[1] if len(args) > 1 else kwargs.get("t")
    traj.append({"t": float(t.reshape(-1)[0].item()), "x": x.detach().cpu().clone()})
def post(mod, args, kwargs, out):
    o = out[0] if isinstance(out, (tuple, list)) else out
    traj[-1]["v"] = o.detach().cpu().clone()
h1 = dit.register_forward_pre_hook(pre, with_kwargs=True)
h2 = dit.register_forward_hook(post, with_kwargs=True)

with torch.inference_mode():
    lat = generate_diffusion_cond(model=model, steps=8, cfg_scale=1.0,
            conditioning=[{"prompt": "128 BPM tech house drum loop", "seconds_total": 11}],
            sample_size=cfg["sample_size"], sample_rate=cfg["sample_rate"], seed=0, device=DEV,
            return_latents=True)
h1.remove(); h2.remove()
lat = (lat[0] if isinstance(lat, (tuple, list)) else lat).detach().cpu()

ts = [s["t"] for s in traj]
print(f"[samp] {len(traj)} steps; t schedule = {[round(t,4) for t in ts]}")
# verify rf-euler: x_{i+1} ?= x_i + (t_{i+1}-t_i)*v_i   (and the final latent vs last update)
xs = [s["x"] for s in traj] + [lat]
for i in range(len(traj)):
    dt = ts[i+1] - ts[i] if i+1 < len(ts) else (0.0 - ts[i])
    pred = xs[i] + dt * traj[i]["v"]
    cos = torch.nn.functional.cosine_similarity(pred.reshape(-1), xs[i+1].reshape(-1), dim=0).item()
    mae = (pred - xs[i+1]).abs().mean().item()
    print(f"   step {i}: t={ts[i]:.4f} dt={dt:+.4f}  x_next≈x+dt*v cos={cos:.6f} mae={mae:.5f}")

torch.save({"traj": traj, "final_latent": lat, "t_schedule": ts}, os.path.join(HERE, "sampler_traj.pt"))
print("[samp] saved sampler_traj.pt")
