#!/usr/bin/env python3
"""Gate the re-authored North-Micro-Vision against the fp32 oracle, before any export.

Two stages against `_smoke/north_micro_vision_instruct_ref.npz`
(`_smoke/northmv_ref.py`, transformers git main -- 5.15.0 does not know
`cohere_compass` at all):

  A. VISION -- the claim under test is that this checkpoint's tower IS the
     Qwen3-VL tower at SigLIP2-SO400M dimensions, so the zoo's existing
     `Qwen3VLVisionEncoder` is reused rather than re-authored. Weights load with
     zero missing/unexpected keys; this checks that it also COMPUTES the same
     thing, seam by seam, at the fixture's own grid.

  B. FULL CHAIN -- extension ids `V + slot` + deepstack + the parallel Cohere
     decoder, against `logits_last` and 48 greedy ids. Token-exact or it failed.

Run (the Core AI export venv; no transformers needed):
    ../coreai-models/.venv/bin/python _smoke/test_northmv_torch_ladder.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from coreai_models.models.macos.cohere_compass import (
    CohereCompassPipelinedForCausalLM,
    cohere_compass_config_from_dict,
    vision_encoder_from_hf,
    _load_state_dict,
)

DEFAULT_REF = Path(__file__).parent / "north_micro_vision_instruct_ref.npz"


def cos(a: torch.Tensor, b: np.ndarray) -> float:
    x = a.detach().double().reshape(-1)
    y = torch.from_numpy(np.asarray(b, dtype=np.float64)).reshape(-1)
    return float(torch.dot(x, y) / (x.norm() * y.norm()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="CohereLabs/North-Micro-Vision-Instruct")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--skip-decoder", action="store_true")
    args = ap.parse_args()

    ref = np.load(args.ref)
    print(f"oracle {args.ref}")
    print(f"  hf_id={ref['_meta_hf_id']} transformers={ref['_meta_transformers']}")
    t, ph, pw = (int(v) for v in ref["image_grid_thw"][0])
    merge = 2
    grid_h, grid_w = ph // merge, pw // merge
    n_tokens = grid_h * grid_w
    print(f"  grid_thw {(t, ph, pw)} -> {ph * pw} patches -> merged {grid_h}x{grid_w} "
          f"= {n_tokens} image tokens")

    failures: list[str] = []

    def check(name: str, got: torch.Tensor, want: np.ndarray) -> None:
        c = cos(got, want)
        mx = float((got.detach().float() - torch.from_numpy(want.astype(np.float32))).abs().max())
        ok = c >= args.tol
        print(f"  {'PASS' if ok else 'FAIL'} {name:24s} cos {c:.6f}  max|d| {mx:.3e}")
        if not ok:
            failures.append(name)

    # ---------------- A. vision --------------------------------------------
    print("\nA. vision tower (the zoo's Qwen3-VL encoder, reused)")
    vis = vision_encoder_from_hf(
        args.hf_id, target_dtype=torch.float32, grid_h=grid_h, grid_w=grid_w
    )
    patches = torch.from_numpy(ref["pixel_values"]).float()
    print(f"  patches {tuple(patches.shape)} -> tower expects "
          f"({vis.n_patches}, {vis.vcfg.in_channels * vis.vcfg.temporal_patch_size * vis.vcfg.patch_size ** 2})")

    with torch.no_grad():
        x = vis.patch_proj(patches) + vis.pos_embed_const
        cosb, sinb = vis.cos_const, vis.sin_const
        deepstack = []
        k = 0
        for i, blk in enumerate(vis.blocks):
            x = blk(x, cosb, sinb)
            if i == 0:
                check("vision_layer0", x, ref["vision_layer0"])
            elif i == len(vis.blocks) // 2:
                check("vision_layer_mid", x, ref["vision_layer_mid"])
            elif i == len(vis.blocks) - 1:
                check("vision_layer_last", x, ref["vision_layer_last"])
            if i in vis.deepstack_visual_indexes:
                deepstack.append(vis.deepstack_merger_list[k](x))
                k += 1
        image_embeds = vis.merger(x)
        deepstack_embeds = torch.cat(deepstack, dim=0)
    check("image_features", image_embeds, ref["image_features"])
    print(f"  image_embeds {tuple(image_embeds.shape)} | deepstack {tuple(deepstack_embeds.shape)}")

    if args.skip_decoder:
        print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
        return 0 if not failures else 1

    # ---------------- B. full chain ----------------------------------------
    print("\nB. full chain (parallel Cohere decoder, extension-id splice)")
    raw, _ = _load_state_dict(args.hf_id, "model.language_model.", torch.float32)
    cfg = cohere_compass_config_from_dict(raw)
    image_token_id = int(raw["image_token_id"])
    dec = CohereCompassPipelinedForCausalLM.from_hf(
        args.hf_id, target_dtype=torch.float32, grid_h=grid_h, grid_w=grid_w
    )

    ids = torch.from_numpy(ref["input_ids"][0].astype(np.int64)).clone()
    img_pos = (ids == image_token_id).nonzero().reshape(-1)
    assert img_pos.numel() == n_tokens, (
        f"prompt has {img_pos.numel()} image placeholders, tower made {n_tokens}"
    )
    img_start = int(img_pos[0])
    ids[img_pos] = cfg.vocab_size + torch.arange(n_tokens, dtype=torch.int64)
    ids = ids.unsqueeze(0).to(torch.int32)
    prompt_len = ids.shape[1]

    # Qwen3-VL rope-shift contract: an image consumes only max(H, W) rope
    # positions, so text after it shifts back by N - max(H, W).
    shift_start = torch.tensor([img_start + n_tokens], dtype=torch.int32)
    shift_amount = torch.tensor([n_tokens - max(grid_h, grid_w)], dtype=torch.int32)
    print(f"  img_start {img_start} | shift_start {int(shift_start)} "
          f"amount {int(shift_amount)}")

    max_seq = prompt_len + args.max_new_tokens + 8
    kshape = (cfg.num_hidden_layers, 1, cfg.num_key_value_heads, max_seq, cfg.head_dim)
    k_cache = torch.zeros(kshape)
    v_cache = torch.zeros(kshape)
    positions = torch.arange(max_seq, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        logits = dec(ids, positions[:, :prompt_len], image_embeds, deepstack_embeds,
                     shift_start, shift_amount, k_cache, v_cache)
    check("logits_last", logits[0, -1], ref["logits_last"])

    # The oracle stops at EOS, so it may be shorter than --max-new-tokens; compare
    # exactly the sequence it produced rather than padding the question.
    want_ids = ref["gen_ids"].astype(np.int64)
    n_gen = min(args.max_new_tokens, want_ids.size)
    got = [int(logits[0, -1].argmax())]
    with torch.no_grad():
        for step in range(1, n_gen):
            logits = dec(
                torch.tensor([[got[-1]]], dtype=torch.int32),
                positions[:, : prompt_len + step],
                image_embeds, deepstack_embeds, shift_start, shift_amount,
                k_cache, v_cache,
            )
            got.append(int(logits[0, -1].argmax()))
    got_arr = np.array(got, dtype=np.int64)
    want_ids = want_ids[: got_arr.size]
    n_match = int((got_arr == want_ids).sum())
    exact = n_match == got_arr.size
    print(f"  {'PASS' if exact else 'FAIL'} greedy {n_match}/{got_arr.size} token-exact")
    if not exact:
        first = int(np.argmax(got_arr != want_ids))
        print(f"    first divergence at {first}: got {got_arr[first]} want {want_ids[first]}")
        failures.append("gen_ids")

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
