"""Gate the app-facing assets against the oracle, before a line of Swift is written.

  encoder v2 (input_ids in-graph)  vs the pipeline's caption embeds
  t_embedder graph                 vs rm.t_embedder
  rope tables + coordinate rule    vs rm.rope_embedder over the real DiT prep

If these three pass, a Swift host only needs: a tokenizer, three graphs, and a table
lookup. It never needs the 778 MB embedding matrix, the timestep MLP, or a RoPE port.
"""
import asyncio
import json
import os

import numpy as np
import torch

import coreai.runtime as rt

HERE = os.path.dirname(os.path.abspath(__file__))
ORA = os.path.join(HERE, "oracle")
GLUE = os.path.join(HERE, "glue")


def corr(a, b):
    return float(np.corrcoef(a.flatten(), b.flatten())[0, 1])


def load_rope_tables():
    meta = json.load(open(os.path.join(GLUE, "rope_meta.json")))
    out = []
    for i, ax in enumerate(meta["axes"]):
        a = np.fromfile(os.path.join(GLUE, f"rope_axis{i}.f32"), "<f4")
        out.append(a.reshape(ax["rows"], ax["freqs"], 2))   # [..., (cos,sin)]
    return meta, out


def rope_lookup(tables, coords):
    """coords [N,3] -> cos [N,64], sin [N,64] — what Swift will do."""
    cos = np.concatenate([tables[i][coords[:, i], :, 0] for i in range(3)], -1)
    sin = np.concatenate([tables[i][coords[:, i], :, 1] for i in range(3)], -1)
    return cos, sin


async def main():
    meta_o = json.load(open(os.path.join(ORA, "meta.json")))
    Lc = meta_o["cap_cond_L"]
    prompt = meta_o["prompt"]
    cap_ref = np.fromfile(os.path.join(ORA, "cap_cond.f32"), "<f4").reshape(Lc, 2560)

    from diffusers import ZImageTransformer2DModel
    from transformers import AutoTokenizer
    M = "Tongyi-MAI/Z-Image-Turbo"
    rm = ZImageTransformer2DModel.from_pretrained(
        M, subfolder="transformer", torch_dtype=torch.float32).eval()
    tok = AutoTokenizer.from_pretrained(M, subfolder="tokenizer")

    # ---------- 1. encoder v2 (input_ids) ----------
    L = 64
    s = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=True)
    ids_valid = tok(s, return_tensors="pt").input_ids
    Lv = ids_valid.shape[1]
    pad_id = tok.pad_token_id or 0
    ids = torch.full((1, L), pad_id, dtype=torch.int32)
    ids[0, :Lv] = ids_valid[0, :Lv].to(torch.int32)
    neg = torch.finfo(torch.bfloat16).min
    m = torch.triu(torch.full((L, L), neg), 1)
    m[:, Lv:] = neg
    mask = m[None, None].to(torch.bfloat16)

    enc = (await rt.AIModel.load(
        "exports/zimage_encoder_seq64_full_bf16_ids/zimage_encoder_seq64_full_bf16_ids.aimodel",
        rt.SpecializationOptions.default())).load_function("main")
    r = await enc(inputs={"input_ids": rt.NDArray(ids.contiguous()),
                          "mask": rt.NDArray(mask.contiguous())})
    pen = r["penultimate"].numpy().astype(np.float32)[0, :Lc]
    c = corr(pen, cap_ref)
    print(f"[gate] encoder v2 (input_ids, tokens={Lv}) vs oracle cap: corr {c:.6f}  "
          f"{'PASS' if c > 0.999 else 'FAIL'}")

    # ---------- 2. t_embedder graph ----------
    te = (await rt.AIModel.load(
        "glue/zimage_t_embedder_fp32/zimage_t_embedder_fp32.aimodel",
        rt.SpecializationOptions.default())).load_function("main")
    worst = 1.0
    sig = np.fromfile(os.path.join(ORA, "sigmas.f32"), "<f4")
    for step in range(8):
        t = float(1.0 - sig[step])
        got = (await te(inputs={"timestep": rt.NDArray(np.array([t], "<f4"))}))["adaln"].numpy()
        want = np.fromfile(os.path.join(ORA, f"adaln_{step}.f32"), "<f4").reshape(1, 256)
        worst = min(worst, corr(got, want))
    print(f"[gate] t_embedder graph vs oracle adaln (8 steps): worst corr {worst:.6f}  "
          f"{'PASS' if worst > 0.9999 else 'FAIL'}")

    # ---------- 3. rope tables + coordinate rule ----------
    _, tables = load_rope_tables()
    ok = True
    for lat, Lcap in ((32, 18), (64, 18), (64, 45), (128, 18)):
        cap = torch.randn(Lcap, 2560)
        latent = torch.randn(16, 1, lat, lat)
        _, _, _, x_pos, cap_pos, _, cap_pad = rm.patchify_and_embed([latent], [cap], 2, 1)
        n_cap = len(cap_pad[0])
        g = lat // 2
        # the rule a Swift host will use, with no access to the model:
        cap_coords = np.stack([np.arange(1, n_cap + 1), np.zeros(n_cap, int), np.zeros(n_cap, int)], 1)
        hw = np.stack(np.meshgrid(np.arange(g), np.arange(g), indexing="ij"), -1).reshape(-1, 2)
        img_coords = np.concatenate([np.full((g * g, 1), n_cap + 1), hw], 1)

        want_x = rm.rope_embedder(x_pos[0])[:g * g]
        want_c = rm.rope_embedder(cap_pos[0])[:n_cap]
        gx_cos, gx_sin = rope_lookup(tables, img_coords)
        gc_cos, gc_sin = rope_lookup(tables, cap_coords)
        e = max(np.abs(gx_cos - want_x.real.numpy()).max(), np.abs(gx_sin - want_x.imag.numpy()).max(),
                np.abs(gc_cos - want_c.real.numpy()).max(), np.abs(gc_sin - want_c.imag.numpy()).max())
        good = e < 1e-6
        ok &= good
        print(f"[gate] rope rule lat={lat:3d} n_cap={n_cap:3d}: max|d| {e:.2e}  {'ok' if good else 'MISMATCH'}")
    print(f"[gate] rope tables + coordinate rule: {'PASS' if ok else 'FAIL'}")


asyncio.run(main())
