# Community port — fast VAE-decoder export feasibility on a SMALL latent (CPU decode timing + export).
import json, os, sys, time, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))
try:
    torch.__future__.set_swap_module_params_on_conversion(False)  # avoid swap_tensors weakref error on conv export
except Exception:
    pass
import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

cfg = json.load(open(os.path.join(HERE, "model_config.json")))
model = create_model_from_config(cfg)
model.load_state_dict(load_ckpt_state_dict(os.path.join(HERE, "model.safetensors")), strict=False)
model = model.to("cpu").float().eval()

def fold_weight_norm(root):
    """Fold weight_norm parametrizations into plain weights (export can't swap parametrized convs —
    'Couldn't swap Conv1d.bias' / weakref). Known zoo trap (cf Kokoro)."""
    import torch.nn.utils as U
    import torch.nn.utils.parametrize as P
    n = 0
    for mod in root.modules():
        try:
            U.remove_weight_norm(mod); n += 1; continue
        except (ValueError, RuntimeError):
            pass
        if hasattr(mod, "parametrizations") and "weight" in dict(getattr(mod, "parametrizations", {})):
            try:
                P.remove_parametrizations(mod, "weight", leave_parametrized=True); n += 1
            except Exception:
                pass
    return n


nfold = fold_weight_norm(model.pretransform)
print(f"[vae-s] folded weight_norm on {nfold} modules", flush=True)

T = int(sys.argv[1]) if len(sys.argv) > 1 else 16          # latent frames (full = 256)
latent = torch.randn(1, 64, T)
print(f"[vae-s] latent [1,64,{T}] (full=256)", flush=True)


class VAEDecExport(torch.nn.Module):
    def __init__(self, pt): super().__init__(); self.pt = pt
    def forward(self, latent): return self.pt.decode(latent)


import copy  # noqa: E402
m = copy.deepcopy(VAEDecExport(model.pretransform).eval())   # fresh tensors — break parametrization weakrefs
for p in m.parameters():
    p.requires_grad_(False)
t0 = time.time()
with torch.inference_mode():
    ref = m(latent)
print(f"[vae-s] CPU torch decode: {time.time()-t0:.1f}s  out {tuple(ref.shape)}", flush=True)

t0 = time.time()
prog = export_to_coreai(m, {"latent": latent}, dynamic_shapes=None,
                        input_names=("latent",), output_names=("audio",), state_names=None)
print(f"[vae-s] export_to_coreai: {time.time()-t0:.1f}s  ✅", flush=True)
prog.optimize()
import shutil, asyncio, coreai.runtime as rt
out_dir = Path(HERE) / "artifacts" / f"sa_vae_fp16_t{T}"
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
aim = out_dir / out_dir.name
aim = out_dir / f"{out_dir.name}.aimodel"
prog.save_asset(aim, rt.AIModelAssetMetadata())
print(f"[vae-s] saved {aim}", flush=True)

async def gate():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"latent": rt.NDArray(np.ascontiguousarray(latent.numpy().astype(np.float32)))})
    eng = torch.as_tensor(r["audio"].numpy().astype(np.float32))
    cc = torch.nn.functional.cosine_similarity(eng.reshape(-1), ref.reshape(-1), dim=0).item()
    print(f"[vae-s] ENGINE vs torch cos={cc:.6f}  {'PASS' if cc>=0.999 else 'CHECK'}")
asyncio.run(gate())
