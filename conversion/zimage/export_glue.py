"""Everything a host app needs besides the three big graphs.

A Swift host should not reimplement Z-Image's timestep MLP or its 3-axis RoPE, and
it must not carry the 778 MB token-embedding matrix. So:

  t_embedder.aimodel   timestep [1] -> adaln [1,256]     (a ~2 MB Core AI graph)
  rope_axis{0,1,2}.f32 the RopeEmbedder's per-axis tables

RopeEmbedder is literally `cat([freqs[i][ids[:, i]] for i in 0..2], -1)` — a per-axis
lookup — so three 1-D tables reproduce it exactly for any resolution and any prompt
length. Coordinates (verified against patchify_and_embed):

  caption token i : (i + 1, 0, 0)             for all i in [0, n_cap)
  image token k   : (n_cap + 1, h, w)         k = h * (W/2) + w

Run (coreai-models venv, from conversion/zimage/): python export_glue.py
"""
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class TEmbed(nn.Module):
    """timestep (normalized, (1000 - t)/1000) -> adaln [1,256]."""

    def __init__(self, rm):
        super().__init__()
        self.t_embedder = rm.t_embedder
        self.register_buffer("scale", torch.tensor(float(rm.t_scale)))

    def forward(self, timestep):
        return self.t_embedder(timestep * self.scale)


def main():
    out = Path("glue")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir()

    from diffusers import ZImageTransformer2DModel
    print("[glue] loading transformer ...", flush=True)
    rm = ZImageTransformer2DModel.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", subfolder="transformer", torch_dtype=torch.float32).eval()

    # --- RoPE tables (force lazy init, then dump) ---
    _ = rm.rope_embedder(torch.zeros(1, 3, dtype=torch.long))
    tables = rm.rope_embedder.freqs_cis
    meta = {"axes": []}
    for i, t in enumerate(tables):
        cos = t.real.contiguous().float().numpy()
        sin = t.imag.contiguous().float().numpy()
        np.ascontiguousarray(np.stack([cos, sin], -1), "<f4").tofile(out / f"rope_axis{i}.f32")
        meta["axes"].append({"rows": int(t.shape[0]), "freqs": int(t.shape[1])})
        print(f"[glue] rope_axis{i}: rows={t.shape[0]} freqs={t.shape[1]} (cos,sin interleaved)")

    # --- sanity: table lookup + concat == rope_embedder ---
    ids = torch.tensor([[33, 5, 7], [1, 0, 0]])
    want = rm.rope_embedder(ids)
    got_r, got_i = [], []
    for row in ids:
        cr = torch.cat([tables[i].real[row[i]] for i in range(3)])
        ci = torch.cat([tables[i].imag[row[i]] for i in range(3)])
        got_r.append(cr); got_i.append(ci)
    ok = (torch.allclose(want.real, torch.stack(got_r)) and
          torch.allclose(want.imag, torch.stack(got_i)))
    print(f"[glue] per-axis lookup reproduces rope_embedder: {ok}")
    assert ok

    meta["seq_multi_of"] = 32
    meta["latent_scaling"] = 0.3611
    meta["latent_shift"] = 0.1159
    meta["note"] = ("caption token i -> coord (i+1,0,0); image token (h,w) -> "
                    "coord (n_cap+1,h,w). Concat the three axis lookups for cos and sin.")
    json.dump(meta, open(out / "rope_meta.json", "w"), indent=2)

    # --- t_embedder graph ---
    from coreai_models.export.macos import export_to_coreai
    import coreai.runtime as rt
    wrap = TEmbed(rm).eval().to(torch.float32)
    ref = {"timestep": torch.tensor([0.5], dtype=torch.float32)}
    print("[glue] exporting t_embedder graph ...", flush=True)
    prog = export_to_coreai(wrap, ref, dynamic_shapes={"timestep": None},
                            input_names=("timestep",), output_names=("adaln",))
    prog.optimize()
    d = out / "zimage_t_embedder_fp32"
    d.mkdir()
    prog.save_asset(d / "zimage_t_embedder_fp32.aimodel", rt.AIModelAssetMetadata())

    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(out) for f in fs)
    print(f"[glue] wrote glue/ ({tot/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
