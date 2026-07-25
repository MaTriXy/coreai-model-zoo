# Community port — probe: does the Oobleck VAE decoder lower to Core AI directly? (latent -> waveform)
import json, os, sys, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))

import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

cfg = json.load(open(os.path.join(HERE, "model_config.json")))
model = create_model_from_config(cfg)
model.load_state_dict(load_ckpt_state_dict(os.path.join(HERE, "model.safetensors")), strict=False)
model = model.to("cpu").float().eval()

oracle = torch.load(os.path.join(HERE, "ref_oracle.pt"))
latent = oracle["latent"].float().cpu()              # [1,64,256]
ref_audio = oracle["audio"].float().cpu()            # [1,2,524288]


class VAEDecExport(torch.nn.Module):
    def __init__(self, pretransform):
        super().__init__(); self.pt = pretransform

    def forward(self, latent):
        return self.pt.decode(latent)


m = VAEDecExport(model.pretransform).eval()
with torch.inference_mode():
    chk = m(latent)
c = torch.nn.functional.cosine_similarity(chk.reshape(-1), ref_audio.reshape(-1), dim=0).item()
print(f"[vae] torch wrapper vs ref audio cos={c:.6f} | out {tuple(chk.shape)}", flush=True)

print("[vae] attempting export_to_coreai ...", flush=True)
try:
    prog = export_to_coreai(m, {"latent": latent}, dynamic_shapes=None,
                            input_names=("latent",), output_names=("audio",), state_names=None)
    print("[vae] EXPORT OK ✅", flush=True)
    prog.optimize()
    import shutil, asyncio, coreai.runtime as rt
    out_dir = Path(HERE) / "artifacts" / "sa_vae_fp16_probe"
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / "sa_vae_fp16_probe.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    print("[vae] saved", aim, flush=True)

    async def gate():
        gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        r = await fn(inputs={"latent": rt.NDArray(np.ascontiguousarray(latent.numpy().astype(np.float32)))})
        eng = torch.as_tensor(r["audio"].numpy().astype(np.float32))
        cc = torch.nn.functional.cosine_similarity(eng.reshape(-1), ref_audio.reshape(-1), dim=0).item()
        print(f"[vae] ENGINE vs ref audio cos={cc:.6f}  {'PASS' if cc>=0.999 else 'CHECK'}")
    asyncio.run(gate())
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[vae] EXPORT FAILED: {type(e).__name__}: {str(e)[:300]}")
