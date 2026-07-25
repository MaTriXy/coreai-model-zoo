# Community port — NOT an Apple model.
"""Precompute the small host-side data the on-device BitVLA demo needs, so the phone carries NO
tokenizer and NO 656MB embed table — just preset-instruction text embeds + a 256-row action-token
embed table (for autoregressive feedback) + the action norm stats.

Outputs (to exports/bitvla_device_data/):
  e_pre.f16            [Npre, 2560]   embeds of "<SYS>Human: "  (shared by all prompts)
  e_post_<k>.f16       [Npost,2560]   embeds of "\\n<instruction><|eot_id|>Assistant: " per preset
  act_embed.f16        [256, 2560]    embed_tokens[128012:128268] (action-token feedback)
  manifest.json        shapes + preset list + ACT_LO + norm_stats(bridge_orig)

  cd ~/code/coreai/coreai-models && .venv/bin/python \
    ../coreai-models-community/conversion/bitvla/prep_device_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import exports_dir, work_path  # noqa: E402

CKDIR = str(work_path("_bitvla_ckpt", "bitvla_bf16"))
CK = f"{CKDIR}/model.safetensors"
OUT = exports_dir() / "bitvla_device_data"
ACT_LO, N_BINS = 128012, 256
SYS = ("System: A chat between a curious human and an artificial intelligence assistant. "
       "The assistant gives helpful, detailed, and polite answers to the human's questions.<|eot_id|>")
PRESETS = [
    "pick up the remote",          # preset 0 = the oracle prompt (on-device parity vs official)
    "pick up the object",
    "open the top drawer",
    "move the gripper to the left",
    "stack the blocks",
]
SAMPLE_IMG = str(work_path("_bitvla_repo", "transformers",
                           "tests/fixtures/tests_samples/COCO/000000039769.png"))


def w16(path: Path, t: torch.Tensor):
    path.write_bytes(t.detach().to(torch.float16).contiguous().numpy().tobytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(CKDIR)
    embed_w = load_file(CK)["language_model.model.embed_tokens.weight"].float()

    pre_ids = tok(SYS + "Human: ", return_tensors="pt", add_special_tokens=True).input_ids
    e_pre = F.embedding(pre_ids, embed_w)[0]                       # [Npre, 2560]
    w16(OUT / "e_pre.f16", e_pre)

    manifest = {"hidden": 2560, "act_lo": ACT_LO, "n_bins": N_BINS,
                "e_pre_len": int(e_pre.shape[0]), "presets": [], "image_token_count": 256}
    for k, instr in enumerate(PRESETS):
        post = "\n" + f"What action should the robot take to {instr}?" + "<|eot_id|>Assistant: "
        post_ids = tok(post, return_tensors="pt", add_special_tokens=False).input_ids
        e_post = F.embedding(post_ids, embed_w)[0]
        w16(OUT / f"e_post_{k}.f16", e_post)
        manifest["presets"].append({"text": instr, "e_post_len": int(e_post.shape[0])})

    act_embed = embed_w[ACT_LO:ACT_LO + N_BINS]                    # [256, 2560]
    w16(OUT / "act_embed.f16", act_embed)

    # bundle the COCO sample image (on-device parity vs the official oracle for preset 0)
    import shutil
    if Path(SAMPLE_IMG).exists():
        shutil.copy(SAMPLE_IMG, OUT / "sample.png")

    ns = json.load(open(f"{CKDIR}/config.json"))["norm_stats"]["bridge_orig"]["action"]
    bins = np.linspace(-1, 1, N_BINS)
    manifest["bin_centers"] = ((bins[:-1] + bins[1:]) / 2.0).tolist()
    manifest["norm_q01"] = list(map(float, ns["q01"]))
    manifest["norm_q99"] = list(map(float, ns["q99"]))
    manifest["norm_mask"] = [bool(m) for m in ns.get("mask", [True] * 7)]
    manifest["unnorm_key"] = "bridge_orig"
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum(p.stat().st_size for p in OUT.glob("*.f16"))
    print(f"wrote {OUT} ({total/1e6:.1f} MB f16 + manifest); presets={len(PRESETS)}")
    print("e_pre", tuple(e_pre.shape), "act_embed", tuple(act_embed.shape))


if __name__ == "__main__":
    main()
