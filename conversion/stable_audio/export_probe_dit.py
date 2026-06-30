# Community port — probe: does the reference DiffusionTransformer lower to Core AI directly?
"""Wrap the reference DiT with fixed-shape inputs (from dit_io.pt) and try export_to_coreai.
If it traces + an engine forward matches the reference output (cos), we can skip a full rewrite.
"""
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
dit = next(m for m in model.modules() if type(m).__name__ == "DiffusionTransformer")

io = torch.load(os.path.join(HERE, "dit_io.pt"))
x = io["x"].float(); t = io["t"].float(); cac = io["cross_attn_cond"].float()
ge = io["global_embed"].float(); mask = io["cross_attn_cond_mask"].float()
ref_out = io["out"].float()


class DiTExport(torch.nn.Module):
    def __init__(self, dit):
        super().__init__(); self.dit = dit

    def forward(self, x, t, cross_attn_cond, global_embed, cross_attn_cond_mask):
        return self.dit(x, t, cross_attn_cond=cross_attn_cond, global_embed=global_embed,
                        cross_attn_cond_mask=cross_attn_cond_mask)


m = DiTExport(dit).eval()
# sanity: wrapper reproduces the captured output in torch
with torch.inference_mode():
    chk = m(x, t, cac, ge, mask)
c = torch.nn.functional.cosine_similarity(chk.reshape(-1), ref_out.reshape(-1), dim=0).item()
print(f"[probe] torch wrapper vs captured out cos={c:.6f}")

ref = {"x": x, "t": t, "cross_attn_cond": cac, "global_embed": ge, "cross_attn_cond_mask": mask}
print("[probe] attempting export_to_coreai ...", flush=True)
try:
    prog = export_to_coreai(m, ref, dynamic_shapes=None,
                            input_names=("x", "t", "cross_attn_cond", "global_embed", "cross_attn_cond_mask"),
                            output_names=("v",), state_names=None)
    print("[probe] EXPORT OK — DiT lowers to Core AI ✅", flush=True)
    prog.optimize()
    import shutil, asyncio, coreai.runtime as rt
    out_dir = Path(HERE) / "artifacts" / "sa_dit_fp16_probe"
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / "sa_dit_fp16_probe.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    print("[probe] saved", aim, flush=True)

    # engine gate: GPU forward vs reference DiT output
    async def gate():
        gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
        fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
        def nd(a): return rt.NDArray(np.ascontiguousarray(a.numpy().astype(np.float32)))
        r = await fn(inputs={"x": nd(x), "t": nd(t), "cross_attn_cond": nd(cac),
                             "global_embed": nd(ge), "cross_attn_cond_mask": nd(mask)})
        eng = torch.as_tensor(r["v"].numpy().astype(np.float32))
        cc = torch.nn.functional.cosine_similarity(eng.reshape(-1), ref_out.reshape(-1), dim=0).item()
        print(f"[probe] ENGINE vs reference DiT cos={cc:.6f}  {'PASS' if cc>=0.999 else 'CHECK'}")
    asyncio.run(gate())
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[probe] EXPORT FAILED: {type(e).__name__}: {str(e)[:300]}")
