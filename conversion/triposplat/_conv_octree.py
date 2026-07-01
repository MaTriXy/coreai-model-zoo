"""Net #4a: convert the octree probability decoder (decoder.octree) for on-device.
It's cross-ONLY attention (queries don't interact), so a fixed query length L works — at runtime
pad the level's L_lv voxels up to L and take the first L_lv logits (no mask needed).
Inputs: x=parent_coords_norm(1,L,3), l=res(1,) int, cond=latent(1,8192,16). Output: logits(1,L,8)."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Code/coreai"))
import torch, torch.nn as nn
import coreai_kit
from triposplat import load_decoder

CK, OUT = "ckpts", "coreai_out"
L = 8192  # max voxels = num_points (262144 / 32)


def main():
    dec = load_decoder(f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors", device="cpu", dtype=torch.float16)
    oct_ = dec.octree.float().eval()

    class W(nn.Module):
        def __init__(s): super().__init__(); s.m = oct_
        def forward(s, x, l, cond):
            return s.m(x, l, cond)["logits"]   # l2 defaults None (additional_level_embed=False)

    g = torch.Generator().manual_seed(0)
    x = torch.rand(1, L, 3, generator=g)
    l = torch.tensor([16.0])   # res as float32 (level_embedding upcasts anyway); int64 input errors the runtime
    cond = torch.randn(1, 8192, 16, generator=g)
    w = W().eval()
    with torch.no_grad():
        ref = w(x, l, cond)
    print(f"octree out logits={tuple(ref.shape)}; converting (optimize=False) ...", flush=True)
    coreai_kit.convert(w, (x, l, cond), ["x", "l", "cond"], ["logits"], f"{OUT}/octree_fp32.aimodel", optimize=False)
    outs = coreai_kit.run(f"{OUT}/octree_fp32.aimodel", {"x": x.numpy(), "l": l.numpy(), "cond": cond.numpy()}, compute="cpu")
    a = ref.flatten().double(); b = torch.tensor(outs["logits"]).flatten().double()
    cos = float((a @ b) / (a.norm() * b.norm()))
    md = float((ref - torch.tensor(outs["logits"])).abs().max())
    print(f"=== OCTREE GATE: cos={cos:.6f} maxdiff={md:.4e} ===", flush=True)


if __name__ == "__main__":
    main()
