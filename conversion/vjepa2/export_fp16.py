# Community port — export V-JEPA 2 (SSv2 action classification) to Core AI, fp16, + engine gate.
"""Direct-export probe first (the stable_audio lesson: try wrapping the reference before any
rewrite): VJEPA2ForVideoClassification = ViT-L backbone + attentive pooler + classifier, one graph.
pixel_values_videos [1,16,3,256,256] -> logits [1,174]. fp16 via .half() + fp16 ref inputs."""
import os, sys, shutil, asyncio, numpy as np, torch
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

from transformers import VJEPA2ForVideoClassification
import transformers.models.vjepa2.modeling_vjepa2 as _vj


def _rotate_queries_or_keys(x, pos):
    # Reference copy MINUS the no-op `squeeze(-1)` on a size-D/2 dim (torch ignores it; the
    # Core AI converter maps it to ShrinkDims which requires size 1 and fails). Math unchanged.
    B, num_heads, N, D = x.size()
    omega = torch.arange(D // 2, dtype=x.dtype, device=x.device)
    omega /= D / 2.0
    omega = 1.0 / 10000**omega
    freq = pos.unsqueeze(-1) * omega
    emb_sin = freq.sin().repeat(1, 1, 1, 2)
    emb_cos = freq.cos().repeat(1, 1, 1, 2)
    y = x.unflatten(-1, (-1, 2))
    y1, y2 = y.unbind(dim=-1)
    y = torch.stack((-y2, y1), dim=-1)
    y = y.flatten(-2)
    return (x * emb_cos) + (y * emb_sin)


_vj.rotate_queries_or_keys = _rotate_queries_or_keys

REPO = "facebook/vjepa2-vitl-fpc16-256-ssv2"
model = VJEPA2ForVideoClassification.from_pretrained(REPO, dtype=torch.float32).eval()

px = torch.from_numpy(np.load(os.path.join(HERE, "oracle_input.npy")))          # [1,16,3,256,256] fp32
ref_logits = torch.from_numpy(np.load(os.path.join(HERE, "oracle_logits.npy")))  # [1,174]


class VJEPA2Export(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, pixel_values_videos):
        return self.m(pixel_values_videos=pixel_values_videos).logits


w = VJEPA2Export(model).eval()
with torch.inference_mode():
    chk = w(px)
c = torch.nn.functional.cosine_similarity(chk.reshape(-1), ref_logits.reshape(-1), dim=0).item()
print(f"[gate] torch wrapper vs oracle cos={c:.6f}", flush=True)

# fp16 export (precision follows traced dtype)
w = w.half()
ref = {"pixel_values_videos": px.half()}
print("[export] export_to_coreai (fp16)…", flush=True)
prog = export_to_coreai(w, ref, dynamic_shapes=None,
                        input_names=("pixel_values_videos",), output_names=("logits",), state_names=None)
print("[export] EXPORT OK ✅", flush=True)
prog.optimize()
out_dir = Path(HERE) / "artifacts" / "vjepa2_ssv2_fp16"
if out_dir.exists(): shutil.rmtree(out_dir)
out_dir.mkdir(parents=True)
aim = out_dir / "vjepa2_ssv2_fp16.aimodel"
import coreai.runtime as rt
prog.save_asset(aim, rt.AIModelAssetMetadata())
print("[export] saved", aim, flush=True)


async def gate():
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    inp = rt.NDArray(np.ascontiguousarray(px.numpy().astype(np.float16)))
    out = await fn(inputs={"pixel_values_videos": inp})
    lg = out["logits"].numpy().astype(np.float32)
    cc = float(np.dot(lg.ravel(), ref_logits.numpy().ravel()) /
               (np.linalg.norm(lg) * np.linalg.norm(ref_logits.numpy()) + 1e-9))
    t_ref = np.argsort(-ref_logits.numpy().ravel())[:5]
    t_eng = np.argsort(-lg.ravel())[:5]
    print(f"[gate] ENGINE vs oracle cos={cc:.6f}  top5 ref={t_ref.tolist()} eng={t_eng.tolist()}", flush=True)
    print("[gate] PASS ✅" if cc > 0.99 and t_ref[0] == t_eng[0] else "[gate] FAIL ❌", flush=True)

asyncio.run(gate())
