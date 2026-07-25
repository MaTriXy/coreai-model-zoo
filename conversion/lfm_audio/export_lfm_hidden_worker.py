import argparse
from pathlib import Path

import torch
import torch.nn as nn

import export_lfm2_embeds_decode as L
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.lfm2 import DECODE_STATE_NAMES, build_decode_state
from coreai_models.primitives.macos.cache import KVCache

# Hidden-output LFM2 embeds-decode graph for TTS: same stateful S=1 graph as
# export_worker, but the output is the last-token HIDDEN [1,1,2048] (post
# embedding_norm, pre lm_head) — the audio path feeds this to the depthformer;
# the host applies lm_head only for the text phase. Run as a SUBPROCESS.

ap = argparse.ArgumentParser()
ap.add_argument("out_dir")
ap.add_argument("--max-ctx", type=int, default=4096)
args = ap.parse_args()

DTYPE = torch.float16
cfg = L.load_lfm_config()
base = L.load_backbone(cfg)                     # Lfm2EmbedsForCausalLMStateful (fp16, fp32 attn)


class HiddenOut(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.model                # Lfm2Model

    def forward(self, inputs_embeds, position_ids, k_cache, v_cache, conv_state):
        h = self.model.forward_stateful_embeds(
            inputs_embeds, position_ids, KVCache(k_cache, v_cache), conv_state)
        return h[:, -1:, :]


model = HiddenOut(base).eval()

inputs_embeds = torch.randn(1, 1, cfg.hidden_size, dtype=DTYPE)
position_ids = torch.arange(65, dtype=torch.int32).unsqueeze(0)
state = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
ref = {"inputs_embeds": inputs_embeds, "position_ids": position_ids,
       "k_cache": state["k_cache"], "v_cache": state["v_cache"], "conv_state": state["conv_state"]}
seq_pos = torch.export.Dim("seq_pos", min=1, max=args.max_ctx - 1)
k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
dyn = {"inputs_embeds": None, "position_ids": {1: seq_pos},
       "k_cache": {KVCache.seq_len_dim(): k_seq}, "v_cache": {KVCache.seq_len_dim(): v_seq},
       "conv_state": None}
specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
prog = export_to_coreai(model, ref, dynamic_shapes=dyn,
                        input_names=("inputs_embeds", "position_ids"), output_names=("hidden",),
                        state_names=DECODE_STATE_NAMES, externalize_modules=specs)
prog.optimize()

import coreai.runtime as rt  # noqa: E402

out = Path(args.out_dir)
if out.exists():
    import shutil
    shutil.rmtree(out)
out.mkdir(parents=True)
aim = out / f"{out.name}.aimodel"
meta = rt.AIModelAssetMetadata()
meta.license = "lfm-open-license-v1.0"
prog.save_asset(aim, meta)
sz = sum(f.stat().st_size for f in aim.rglob("*") if f.is_file()) / 1e6
print(f"[save] {aim} ({sz:.1f} MB)")
print("EXPORT_OK", aim)
