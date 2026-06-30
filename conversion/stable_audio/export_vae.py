# Community port — export the clean Oobleck decoder to Core AI + engine-gate vs reference.
import os, sys, time, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
from safetensors.torch import load_file
from oobleck_vae import load_decoder

T = int(sys.argv[1]) if len(sys.argv) > 1 else 256
dec = load_decoder(load_file(os.path.join(HERE, "model.safetensors")), torch.float32)
oracle = torch.load(os.path.join(HERE, "ref_oracle.pt"))
latent = oracle["latent"].float()[:, :, :T]
ref_audio = oracle["audio"].float()[:, :, :T * 2048]
print(f"[vae-exp] latent[1,64,{T}] -> audio[1,2,{T*2048}]", flush=True)

t0 = time.time()
prog = export_to_coreai(dec, {"latent": latent}, dynamic_shapes=None,
                        input_names=("latent",), output_names=("audio",), state_names=None)
print(f"[vae-exp] export {time.time()-t0:.1f}s ✅", flush=True)
prog.optimize()
import shutil, asyncio, coreai.runtime as rt
out_dir = Path(HERE) / "artifacts" / f"sa_vae_fp16_s{T}"
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
aim = out_dir / f"{out_dir.name}.aimodel"
prog.save_asset(aim, rt.AIModelAssetMetadata())
import subprocess
sz = subprocess.run(["du", "-sh", str(aim)], capture_output=True, text=True).stdout.split()[0]
print(f"[vae-exp] saved {aim} ({sz})", flush=True)

async def gate():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs={"latent": rt.NDArray(np.ascontiguousarray(latent.numpy().astype(np.float32)))})
    eng = torch.as_tensor(r["audio"].numpy().astype(np.float32))
    cc = torch.nn.functional.cosine_similarity(eng.reshape(-1), ref_audio.reshape(-1), dim=0).item()
    print(f"[vae-exp] ENGINE vs reference audio cos={cc:.6f}  {'PASS' if cc>=0.999 else 'CHECK'}")
asyncio.run(gate())
