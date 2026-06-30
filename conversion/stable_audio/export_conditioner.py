# Community port — export the conditioner (T5-base encoder + number conditioner) to Core AI.
"""input_ids[1,64] + attention_mask[1,64] (host T5 tokenizer) + seconds_norm[1] ->
   cross_attn_cond[1,65,768], global_embed[1,768], cond_mask[1,65]. Gate vs the reference conditioner."""
import json, os, sys, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))
import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict

cfg = json.load(open(os.path.join(HERE, "model_config.json")))
model = create_model_from_config(cfg)
model.load_state_dict(load_ckpt_state_dict(os.path.join(HERE, "model.safetensors")), strict=False)
model = model.to("cpu").eval()

t5cond = model.conditioner.conditioners["prompt"]
numcond = model.conditioner.conditioners["seconds_total"]
t5cond.model.float()                                   # T5 lives in __dict__ — cast explicitly to fp32
PROMPT, SECS = "128 BPM tech house drum loop", 11.0

# reference conditioning (what the DiT actually receives)
with torch.inference_mode():
    ctens = model.conditioner([{"prompt": PROMPT, "seconds_total": SECS}], "cpu")
    ci = model.get_conditioning_inputs(ctens)
ref_cross = ci["cross_attn_cond"].float()
ref_global = ci["global_cond"].float()
ref_mask = ci["cross_attn_mask"].float()
print(f"[cond] ref cross={tuple(ref_cross.shape)} global={tuple(ref_global.shape)} mask={tuple(ref_mask.shape)}")

# tokenize on host (this runs in Swift via a T5 tokenizer)
enc = t5cond.tokenizer([PROMPT], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
input_ids = enc["input_ids"]                            # [1,64] int
attn = enc["attention_mask"].float()                    # [1,64]
secs_norm = torch.tensor([(SECS - numcond.min_val) / (numcond.max_val - numcond.min_val)], dtype=torch.float32)


class CondExport(torch.nn.Module):
    def __init__(self, t5, proj_out, num_embedder):
        super().__init__(); self.t5 = t5; self.proj_out = proj_out; self.num = num_embedder

    def forward(self, input_ids, attention_mask, seconds_norm):
        emb = self.t5(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]  # [1,64,768]
        emb = self.proj_out(emb) * attention_mask.unsqueeze(-1)                                   # zero padding
        num = self.num(seconds_norm).unsqueeze(1)                                                 # [1,1,768]
        cross = torch.cat([emb, num], dim=1)                                                      # [1,65,768]
        ones = torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype)
        mask = torch.cat([attention_mask, ones], dim=1)                                           # [1,65]
        return cross, num.squeeze(1), mask


m = CondExport(t5cond.model, t5cond.proj_out, numcond.embedder).eval()
with torch.inference_mode():
    cross, gemb, mask = m(input_ids, attn, secs_norm)
def cos(a, b): return torch.nn.functional.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()
print(f"[cond] wrapper vs ref: cross cos={cos(cross, ref_cross):.6f}  global cos={cos(gemb, ref_global):.6f}  "
      f"mask match={torch.allclose(mask, ref_mask)}")

print("[cond] attempting export ...", flush=True)
try:
    prog = export_to_coreai(m, {"input_ids": input_ids.int(), "attention_mask": attn, "seconds_norm": secs_norm},
                            dynamic_shapes=None, input_names=("input_ids", "attention_mask", "seconds_norm"),
                            output_names=("cross_attn_cond", "global_embed", "cond_mask"), state_names=None)
    print("[cond] EXPORT OK ✅", flush=True)
    import shutil, asyncio, coreai.runtime as rt
    prog.optimize()
    out_dir = Path(HERE) / "artifacts" / "sa_cond_fp16"
    if out_dir.exists(): shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    print("[cond] saved", aim, flush=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print(f"[cond] EXPORT FAILED: {type(e).__name__}: {str(e)[:200]}")
