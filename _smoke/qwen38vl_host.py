#!/usr/bin/env python3
"""Qwen3.8-27B VL host contract, in NumPy: mRoPE position planes + embed splice.

This is the code any driver (python gate today, Swift host later) runs between
the tokenizer, the vision tower and the embeddings-input decoder. Mirrors HF
``Qwen3_5Model.get_rope_index`` for the image case, and is asserted against the
oracle's CAPTURED rope positions in the suite fixture (never trusted blind).

Facts easy to get wrong and silent when wrong:
  * an image consumes only ``max(H, W) // merge`` rope positions — the text
    after it does NOT resume at (start + n_image_tokens);
  * decode-step positions are ``cache_position + rope_delta`` on ALL THREE
    planes, where ``rope_delta = max_prefill_position + 1 - prompt_len``.
"""
from __future__ import annotations

import numpy as np

IMAGE_TOKEN_ID = 248056
MERGE = 2


def mrope_positions(
    ids: np.ndarray,
    grids: list[tuple[int, int, int]],
    image_token_id: int = IMAGE_TOKEN_ID,
    merge: int = MERGE,
) -> tuple[np.ndarray, int]:
    """(pos [3, S] int32, rope_delta) for a prompt with 0+ images.

    ``grids`` are the (t, h, w) PATCH grids per image, in prompt order.
    """
    S = len(ids)
    pos = np.zeros((3, S), dtype=np.int32)
    current = 0
    img_iter = iter(grids)
    i = 0
    while i < S:
        if ids[i] == image_token_id:
            t, h, w = next(img_iter)
            lh, lw = h // merge, w // merge
            n = t * lh * lw
            if not np.all(ids[i : i + n] == image_token_id):
                raise ValueError(f"expected {n} contiguous image tokens at {i}")
            rows = np.repeat(np.arange(lh, dtype=np.int32), lw)
            cols = np.tile(np.arange(lw, dtype=np.int32), lh)
            for f in range(t):
                sl = slice(i + f * lh * lw, i + (f + 1) * lh * lw)
                pos[0, sl] = current + f
                pos[1, sl] = current + rows
                pos[2, sl] = current + cols
            current += max(lh, lw)
            i += n
        else:
            pos[:, i] = current
            current += 1
            i += 1
    rope_delta = int(pos.max()) + 1 - S
    return pos, rope_delta


def splice_embeds(
    ids: np.ndarray,
    embed_table: np.ndarray,
    image_embeds: np.ndarray | None,
    image_token_id: int = IMAGE_TOKEN_ID,
) -> np.ndarray:
    """[S, hidden]: text rows gathered from the table, tower rows at image slots."""
    x = embed_table[ids.astype(np.int64)]
    if image_embeds is not None:
        img_pos = np.nonzero(ids == image_token_id)[0]
        if img_pos.size != image_embeds.shape[0]:
            raise ValueError(
                f"{img_pos.size} image tokens vs {image_embeds.shape[0]} tower rows")
        x[img_pos] = image_embeds.astype(x.dtype)
    return x
