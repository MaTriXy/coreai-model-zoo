"""Verify the Base-mode prefix ARRANGEMENT recipe (the glue the Swift app implements)
and export the constants it needs. Reconstructs the decoder prefix from the vision
features + the fixed prompt and checks it equals the oracle's `dec_embeds`.

Recipe (image_size=640, crop_mode=False), from modeling_unlimitedocr.py:551-582:
  1. vision .aimodel(image[1,3,640,640]) -> patches [100,1280]  (10x10 grid)
  2. view [10,10,1280]; append image_newline after each row -> [10,11,1280] -> [110,1280];
     append view_seperator -> [111,1280]  (the arranged visual block)
  3. input_ids = [BOS=0] + [<image>=128815]x111 + tokenize("document parsing.")  (115, fixed)
  4. inputs_embeds = embed_tokens(input_ids); masked_scatter the 111 visual features into
     the input_ids==128815 positions -> prefix [1,115,1280]

Run:  cd ~/code/coreai/coreai-models && PYTHONPATH=../_unlimited_ocr \
          .venv/bin/python ../_unlimited_ocr/arrange_assets.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from model import ocr_decoder_from_safetensors

HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpt"
IMAGE_TOKEN_ID = 128815
OUT = HERE / "out/_swift_assets"


def main() -> None:
    z = np.load(HERE / "out/_oracle/oracle_tensors.npz")
    zv = np.load(HERE / "out/_vision_oracle/vision_tensors.npz")
    input_ids = torch.from_numpy(z["token_ids"][0][:115].astype(np.int64))  # [115] fixed prompt
    oracle_prefix = torch.from_numpy(z["dec_embeds"]).float()  # [1,115,1280]
    patches = torch.from_numpy(zv["proj_out"]).float()  # [1,100,1280] vision features

    model = ocr_decoder_from_safetensors(str(CKPT), target_dtype=torch.float32)
    embed = model.model.embed_tokens
    st = glob.glob(str(CKPT / "model-*.safetensors"))[0]
    with safe_open(st, framework="pt", device="cpu") as f:
        image_newline = f.get_tensor("model.image_newline").float()  # [1280]
        view_seperator = f.get_tensor("model.view_seperator").float()  # [1280]

    # --- arrange: 100 patches -> 10x10 + row newlines (110) + seperator (111) ---
    n_dim = patches.shape[-1]
    h = w = int(round(patches.shape[1] ** 0.5))  # 10
    grid = patches.reshape(h, w, n_dim)
    grid = torch.cat([grid, image_newline.view(1, 1, n_dim).expand(h, 1, n_dim)], dim=1)  # [10,11,1280]
    visual = torch.cat([grid.reshape(-1, n_dim), view_seperator.view(1, n_dim)], dim=0)  # [111,1280]

    # --- embed prompt + scatter visual into <image> positions ---
    with torch.no_grad():
        prefix = embed(input_ids.unsqueeze(0)).float()  # [1,115,1280]
    mask = (input_ids == IMAGE_TOKEN_ID)
    assert int(mask.sum()) == visual.shape[0], f"{int(mask.sum())} image slots != {visual.shape[0]} features"
    prefix[0, mask] = visual.to(prefix.dtype)

    cos = torch.nn.functional.cosine_similarity(prefix, oracle_prefix, dim=-1)
    mad = (prefix - oracle_prefix).abs().max()
    print(f"[arrange] reconstructed prefix vs oracle dec_embeds: "
          f"per-token cos mean {cos.mean():.6f} min {cos.min():.6f}  max|Δ| {mad:.3e}")
    ok = cos.min() > 0.9999
    print("✅ ARRANGEMENT RECIPE VERIFIED" if ok else "❌ MISMATCH")

    # --- export the Swift constants ---
    OUT.mkdir(parents=True, exist_ok=True)
    embed.weight.detach().to(torch.float16).numpy().tofile(OUT / "embed_tokens.f16")  # [129280,1280]
    image_newline.to(torch.float16).numpy().tofile(OUT / "image_newline.f16")  # [1280]
    view_seperator.to(torch.float16).numpy().tofile(OUT / "view_seperator.f16")  # [1280]
    np.array(input_ids.tolist(), dtype=np.int32).tofile(OUT / "prompt_input_ids.i32")  # [115]
    (OUT / "recipe.json").write_text(
        '{\n'
        '  "image_size": 640, "crop_mode": false, "patch_grid": 10, "visual_tokens": 111,\n'
        '  "prefix_len": 115, "image_token_id": 128815, "bos": 0, "eos": 1, "hidden": 1280,\n'
        '  "preprocess": "ImageOps.pad to 640x640 (pad color = mean*255), normalize mean=std=0.5",\n'
        '  "arrange": "vision[100,1280] -> view[10,10] -> +image_newline per row [110] -> +view_seperator [111]",\n'
        '  "scatter": "embed_tokens(prompt_input_ids); fill input_ids==128815 positions (in order) with the 111 visual features",\n'
        '  "files": {"embed": "embed_tokens.f16 [129280,1280] fp16", "newline": "image_newline.f16 [1280]",\n'
        '            "seperator": "view_seperator.f16 [1280]", "prompt": "prompt_input_ids.i32 [115]"}\n'
        '}\n')
    print(f"[export] Swift assets -> {OUT}/  ({', '.join(p.name for p in sorted(OUT.iterdir()))})")


if __name__ == "__main__":
    main()
