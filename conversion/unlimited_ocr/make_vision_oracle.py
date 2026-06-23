# P1 vision oracle (CPU fp32): capture the DeepEncoder boundary for ViT export.
#
# DeepEncoder = sam_model (ImageEncoderViT) -> vision_model (VitModel/CLIP, conditioned
# on SAM feats) -> cat([clip[:,1:], sam.flatten(2).permute], -1) -> projector (2048->1280).
# The arrangement (image_newline / view_seperator / tiling) is deterministic and lives
# Swift-side, so the export boundary is image -> projector_out [*, 1280] visual tokens.
#
# We hook the three submodules during a real prefill and save every stage's in/out as
# the gate reference for the Core AI ViT GraphModel export.
import os, sys, time, contextlib, json
import numpy as np
import torch

DEV = "cpu"; DTYPE = torch.float32
torch.manual_seed(0)
torch.Tensor.cuda = lambda self, *a, **k: self.to(DEV)
if hasattr(torch.cuda, "is_available"):
    torch.cuda.is_available = lambda: True
_oa = torch.autocast
def _autocast(dt="cuda", *a, **k):
    return contextlib.nullcontext() if dt == "cuda" else _oa(dt, *a, **k)
torch.autocast = _autocast

CKPT = os.path.join(os.path.dirname(__file__), "ckpt"); sys.path.insert(0, CKPT)
from transformers import AutoModel, AutoTokenizer

IMG = os.environ.get("OCR_IMG", "page_jp.png")
CROP = os.environ.get("OCR_CROP", "0") == "1"   # 0 = Base mode (single 1024^2 global view)

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModel.from_pretrained(
    CKPT, trust_remote_code=True, use_safetensors=True,
    torch_dtype=DTYPE, attn_implementation="eager",
).eval().to(DEV)
mm = model.model
print("[loaded]", type(model).__name__, flush=True)

# ---- hooks: capture every call's inputs/outputs for the 3 DeepEncoder stages ----
CAP = {"sam": [], "vision": [], "proj": []}
def _np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return x
def _mk_hook(key):
    def hook(mod, inp, out):
        ins = [tuple(t.shape) for t in inp if isinstance(t, torch.Tensor)]
        os_ = tuple(out.shape) if isinstance(out, torch.Tensor) else "(non-tensor)"
        CAP[key].append({"in_shapes": ins, "out_shape": os_,
                         "inp": [_np(t) for t in inp if isinstance(t, torch.Tensor)],
                         "out": _np(out)})
        print(f"[hook {key}] in={ins} out={os_}", flush=True)
    return hook
mm.sam_model.register_forward_hook(_mk_hook("sam"))
mm.vision_model.register_forward_hook(_mk_hook("vision"))
mm.projector.register_forward_hook(_mk_hook("proj"))

# ---- run one prefill; vision runs on the first (prefill) forward only ----
t0 = time.time()
out = model.infer(
    tok, prompt="<image>document parsing.",
    image_file=os.path.join(os.path.dirname(__file__), IMG),
    output_path="out/_vision_oracle", base_size=1024, image_size=640, crop_mode=CROP,
    max_length=300, no_repeat_ngram_size=35, ngram_window=128,
    eval_mode=True, save_results=False,
)
print(f"[gen] {time.time()-t0:.1f}s  out_chars={len(out)}", flush=True)

# ---- report + save ----
print("\n" + "=" * 64)
print(f"DeepEncoder stage capture  (img={IMG}, crop={CROP})")
for key in ("sam", "vision", "proj"):
    n = len(CAP[key])
    print(f"  {key:7s}: {n} call(s)" + ("" if not n else
          f"  first in={CAP[key][0]['in_shapes']} out={CAP[key][0]['out_shape']}"))
print("=" * 64)

os.makedirs("out/_vision_oracle", exist_ok=True)
save = {}
for key in ("sam", "vision", "proj"):
    if not CAP[key]:
        continue
    c0 = CAP[key][0]                     # first call = the global-view DeepEncoder pass
    for i, t in enumerate(c0["inp"]):
        save[f"{key}_in{i}"] = t
    save[f"{key}_out"] = c0["out"]
np.savez("out/_vision_oracle/vision_tensors.npz", **save)
meta = {key: [{"in_shapes": c["in_shapes"], "out_shape": c["out_shape"]} for c in CAP[key]]
        for key in CAP}
with open("out/_vision_oracle/vision_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("[saved] out/_vision_oracle/{vision_tensors.npz,vision_meta.json}", flush=True)
print("keys:", sorted(save.keys()), flush=True)
