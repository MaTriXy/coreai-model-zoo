# Community port — NOT an Apple model.
"""Ground-truth oracle from the OFFICIAL ustcwhy/BitVLA code (fork transformers, on-the-fly W1.58-A8).
Builds LlavaForConditionalGeneration (hardwired to BitNet LLM + BitLinear SigLIP in this fork),
loads lxsy/bitvla-bf16, and dumps:
  - pixel_values (official SiglipImageProcessor) so the port is compared on identical inputs,
  - image_features = multi_modal_projector(vision_tower(...).hidden_states[-1])  (the S2 contract),
  - full-forward logits at the last position for a text+image prompt (sanity).

Run in the isolated oracle venv (system torch + fork transformers):
  ~/code/coreai/_bitvla_oracle_venv/bin/python \
    ~/code/coreai/coreai-models-community/conversion/bitvla/oracle_official.py --image <png>
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from safetensors.torch import load_file
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import work_path  # noqa: E402

CKDIR = str(work_path("_bitvla_ckpt", "bitvla_bf16"))
OUT = str(work_path("_bitvla_ckpt", "oracle.npz"))


def build_model():
    from transformers import LlavaConfig, LlavaForConditionalGeneration
    cfg_d = json.load(open(f"{CKDIR}/config.json"))
    # drop fields LlavaConfig won't accept cleanly; keep text_config/vision_config/image_token_index
    for k in ("norm_stats", "n_action_bins", "auto_map", "architectures"):
        cfg_d.pop(k, None)
    cfg = LlavaConfig(**cfg_d)
    cfg.text_config.vit_weight_bits = getattr(cfg.vision_config, "vit_weight_bits", 1)
    model = LlavaForConditionalGeneration(cfg)
    sd = load_file(f"{CKDIR}/model.safetensors")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("  e.g. missing:", missing[:5])
    if unexpected:
        print("  e.g. unexpected:", unexpected[:5])
    return model.eval().float(), cfg


SYS = ("System: A chat between a curious human and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the human's questions.<|eot_id|>")
IMG_TOK = 128260   # config.image_token_index (LlavaForConditionalGeneration splice key)


def gen_action(model, cfg, tok, pv, instruction, new=8):
    """Autoregressive (causal) action-token generation, OpenVLA-style. Returns generated token ids."""
    pre = SYS + "Human: "
    post = "\n" + f"What action should the robot take to {instruction}?" + "<|eot_id|>Assistant: "
    pre_ids = tok(pre, return_tensors="pt", add_special_tokens=True).input_ids
    post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
    img_ids = torch.full((1, 256), IMG_TOK, dtype=pre_ids.dtype)
    input_ids = torch.cat([pre_ids, img_ids, post_ids], dim=1)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        out = model.generate(input_ids=input_ids, attention_mask=attn, pixel_values=pv,
                             do_sample=False, max_new_tokens=new, use_cache=True)
    return out[0, input_ids.shape[1]:].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--instruction", default="pick up the object")
    args = ap.parse_args()

    from transformers import SiglipImageProcessor
    from PIL import Image

    proc = SiglipImageProcessor.from_pretrained(CKDIR)
    pv = proc(images=Image.open(args.image).convert("RGB"), return_tensors="pt")["pixel_values"].float()
    print(f"[pixel_values] {tuple(pv.shape)} mean {pv.mean():.4f} std {pv.std():.4f}")

    model, cfg = build_model()
    vfl = cfg.vision_feature_layer
    with torch.no_grad():
        vout = model.vision_tower(pv, output_hidden_states=True)
        feat = vout.hidden_states[vfl]                       # [1,256,1152]
        img_embeds = model.multi_modal_projector(feat)       # [1,256,2560]
    print(f"[vision] feat {tuple(feat.shape)} std {feat.std():.4f}; "
          f"img_embeds {tuple(img_embeds.shape)} std {img_embeds.std():.4f}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(CKDIR)
    act_ids = gen_action(model, cfg, tok, pv, args.instruction, new=8)
    print(f"[action] instruction={args.instruction!r} -> token ids {act_ids}")
    print(f"         decoded: {[tok.decode([t]) if t < 128010 else f'<{t}>' for t in act_ids]}")

    np.savez(OUT,
             pixel_values=pv.numpy().astype(np.float32),
             vision_feat=feat.numpy().astype(np.float32),
             img_embeds=img_embeds.numpy().astype(np.float32),
             action_ids=np.array(act_ids, dtype=np.int64),
             instruction=args.instruction)
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
