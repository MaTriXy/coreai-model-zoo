#!/usr/bin/env python3
"""Gate the re-authored LFM2.5-VL against the fp32 oracle, before any export.

Two stages, both against `_smoke/lfm2_5_vl_450m_ref.npz` (written by
`_smoke/lfm25vl_ref.py` on transformers >= 5 -- see that file for why 4.x is
not an acceptable oracle for this checkpoint):

  A. VISION LADDER -- patch embeddings + position resize, encoder layer 0,
     encoder mid layer, post_layernorm, and the projector's image_features,
     each at cos >= 0.999 (--tol).

     The oracle ran at the image's NATIVE NaFlex grid (640x480 -> 26x36 patches
     padded to 1024) with a padding mask. The authored module has no mask: at a
     fixed full grid every patch is real. Those are the same computation -- a
     masked key contributes nothing to an unpadded query -- so the ladder feeds
     the first 26*36 patches at grid (26,36) and compares against the oracle's
     first 26*36 rows. The shipped export bakes 32x32 instead (one 512x512 tile
     -> 256 tokens); that configuration gets its own oracle.

  B. FULL CHAIN -- splice the projected image tokens into the LFM2 decoder via
     the extension-id contract (image ids -> V + slot) and check logits_last
     plus 48 greedy token ids against the oracle. Token-exact or it failed.

Run (the Core AI export venv; NO transformers needed here):
    ../coreai-models/.venv/bin/python _smoke/test_lfm25vl_torch_ladder.py \
        [--hf-id LiquidAI/LFM2.5-VL-450M] [--ref _smoke/lfm2_5_vl_450m_ref.npz]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from coreai_models.models.macos.lfm2 import build_decode_state
from coreai_models.models.macos.lfm2_vl import (
    Lfm2VlPipelinedForCausalLM,
    Lfm2VlVisionEncoder,
    lfm2_vl_configs_from_dict,
    load_lfm2_vl_state_dict,
)

DEFAULT_REF = Path(__file__).parent / "lfm2_5_vl_450m_ref.npz"


def cos(a: torch.Tensor, b: np.ndarray) -> float:
    """Cosine in float64. In float32 the reduction over ~10^6 elements carries
    ~1e-4 of error -- enough to print 1.000088 for two IDENTICAL tensors, which
    makes every digit of the gate meaningless."""
    x = a.detach().double().reshape(-1)
    y = torch.from_numpy(np.asarray(b, dtype=np.float64)).reshape(-1)
    return float(torch.dot(x, y) / (x.norm() * y.norm()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-id", default="LiquidAI/LFM2.5-VL-450M")
    ap.add_argument("--ref", default=str(DEFAULT_REF))
    ap.add_argument("--tol", type=float, default=0.999)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument(
        "--dtype",
        default="fp32",
        choices=["fp32", "fp16"],
        help="authoring dtype. fp16 is the SHIP dtype: run it to attribute an "
        "engine-vs-oracle gap to dtype rather than to the port.",
    )
    ap.add_argument(
        "--host-patches",
        action="store_true",
        help="feed the NumPy host's patches (_smoke/lfm25vl_preprocess.py) instead of "
        "the oracle's pixel_values. The host's resize differs from Pillow's by at most "
        "one 0-255 level; this says whether that reaches the tokens.",
    )
    args = ap.parse_args()

    ref = np.load(args.ref)
    print(f"oracle {args.ref}")
    print(f"  hf_id={ref['_meta_hf_id']} transformers={ref['_meta_transformers']}")

    grid_h, grid_w = (int(v) for v in ref["spatial_shapes"][0])
    n_patch = grid_h * grid_w
    if args.host_patches:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from lfm25vl_preprocess import _fixture_image, preprocess  # noqa: E402

        host = preprocess(_fixture_image())
        if host.shape[0] != n_patch:
            raise SystemExit(
                f"host emits {host.shape[0]} patches, this oracle is {n_patch} "
                "(--host-patches only fits the fixed-grid oracle)"
            )
        pixel_values = torch.from_numpy(host).float()
        print("  patches from the NumPy host, not the oracle")
    else:
        pixel_values = torch.from_numpy(ref["pixel_values"][0, :n_patch]).float()
    mask = ref["pixel_attention_mask"][0]
    assert int(mask.sum()) == n_patch, (
        f"oracle mask has {int(mask.sum())} real patches, grid says {n_patch}"
    )
    print(f"  grid {grid_h}x{grid_w} = {n_patch} patches (of {mask.shape[0]} padded)")

    failures: list[str] = []
    dtype = torch.float32 if args.dtype == "fp32" else torch.float16

    def check(name: str, got: torch.Tensor, want: np.ndarray) -> None:
        got = got.float()
        c = cos(got, want)
        ok = c >= args.tol
        mx = float((got.detach().float() - torch.from_numpy(want.astype(np.float32))).abs().max())
        print(f"  {'PASS' if ok else 'FAIL'} {name:24s} cos {c:.6f}  max|d| {mx:.3e}")
        if not ok:
            failures.append(name)

    # ---------------- A. vision ladder -------------------------------------
    print(f"\nA. vision ladder ({args.dtype}, grid {grid_h}x{grid_w})")
    enc = Lfm2VlVisionEncoder.from_hf(
        args.hf_id, target_dtype=dtype, grid_h=grid_h, grid_w=grid_w
    )
    pixel_values = pixel_values.to(dtype)

    with torch.no_grad():
        x = enc.patch_embedding(pixel_values) + enc.pos_embed_const
        check("vision_embeddings", x, ref["vision_embeddings"][0, :n_patch])
        mid = len(enc.layers) // 2
        for i, layer in enumerate(enc.layers):
            x = layer(x)
            if i == 0:
                check("vision_layer0", x, ref["vision_layer0"][0, :n_patch])
            elif i == mid:
                check("vision_layer_mid", x, ref["vision_layer_mid"][0, :n_patch])
        x = enc.post_layernorm(x)
        check("vision_post_layernorm", x, ref["vision_post_layernorm"][0, :n_patch])
        image_embeds = enc.project(x)
        want_feats = ref["image_features"].reshape(-1, ref["image_features"].shape[-1])
        check("image_features", image_embeds, want_feats)

    n_img = image_embeds.shape[0]
    print(f"  {n_img} image tokens")

    # ---------------- B. full chain ----------------------------------------
    print(f"\nB. full chain ({args.dtype} decoder, extension-id splice)")
    raw, _ = load_lfm2_vl_state_dict(args.hf_id, "model.language_model.", torch.float32)
    _, text_cfg = lfm2_vl_configs_from_dict(raw)
    image_token_id = int(raw["image_token_id"])

    dec = Lfm2VlPipelinedForCausalLM.from_hf(
        args.hf_id, target_dtype=dtype, n_image_tokens=n_img,
        fp32_attn_proj=(dtype == torch.float16),
    )

    ids = torch.from_numpy(ref["input_ids"][0].astype(np.int64)).clone()
    img_pos = (ids == image_token_id).nonzero().reshape(-1)
    assert img_pos.numel() == n_img, (
        f"prompt has {img_pos.numel()} image placeholders, encoder made {n_img} tokens"
    )
    ids[img_pos] = text_cfg.vocab_size + torch.arange(n_img, dtype=torch.int64)
    ids = ids.unsqueeze(0).to(torch.int32)
    prompt_len = ids.shape[1]

    max_seq = prompt_len + args.max_new_tokens + 8
    state = build_decode_state(text_cfg, max_seq_len=max_seq, dtype=dtype)
    positions = torch.arange(max_seq, dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        logits = dec(
            ids,
            positions[:, :prompt_len],
            image_embeds,
            state["k_cache"],
            state["v_cache"],
            state["conv_state"],
        )
    check("logits_last", logits[0, -1], ref["logits_last"])

    want_ids = ref["gen_ids"].astype(np.int64)
    got_ids: list[int] = []
    nxt = int(logits[0, -1].argmax())
    got_ids.append(nxt)
    with torch.no_grad():
        for step in range(1, args.max_new_tokens):
            pos = prompt_len + step
            logits = dec(
                torch.tensor([[nxt]], dtype=torch.int32),
                positions[:, :pos],
                image_embeds,
                state["k_cache"],
                state["v_cache"],
                state["conv_state"],
            )
            nxt = int(logits[0, -1].argmax())
            got_ids.append(nxt)

    got = np.array(got_ids, dtype=np.int64)
    n_match = int((got == want_ids[: len(got)]).sum())
    exact = n_match == len(got)
    print(f"  {'PASS' if exact else 'FAIL'} greedy {n_match}/{len(got)} token-exact")
    if not exact:
        first = int(np.argmax(got != want_ids[: len(got)]))
        print(f"    first divergence at {first}: got {got[first]} want {want_ids[first]}")
        failures.append("gen_ids")

    print("\nALL PASS" if not failures else f"\nFAILED: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
