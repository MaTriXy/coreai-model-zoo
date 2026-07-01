"""Net #4b+: convert gs-decoder + build_gaussians + Gaussian .ply-activation math into ONE net.
Input: points(1,N,3) + cond/latent(1,8192,16). Output: ply_attrs (N*32, 14) =
[xyz(3), f_dc(3), opacity(1), scale(3), rot(4)] in .ply column order (normals=0 added in Swift; the
fixed y-up DEFAULT_TRANSFORM is SKIPPED — the on-device viewer's up=(0,-1,0) handles orientation, and
skipping it avoids un-convertible quat->matrix->quat branching).
Gated against the real Python _build_gaussians + Gaussian._get_ply_data(transform=None)."""
import sys, os, math
sys.path.insert(0, os.path.expanduser("~/Code/coreai"))
import torch, torch.nn as nn, torch.nn.functional as F
import coreai_kit
from triposplat import load_decoder, _build_gaussians

CK, OUT = "ckpts", "coreai_out"
N = 8192   # decoder tokens; *32 gaussians = 262144


def main():
    dec = load_decoder(f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors", device="cpu", dtype=torch.float16)
    gs = dec.gs.float().eval()
    lay = gs.layout
    rep = gs.rep_config
    ng = rep["num_gaussians"]                       # 32
    aabb = torch.tensor([-0.5, -0.5, -0.5, 1.0, 1.0, 1.0])
    opacity_bias_val = math.log(rep["opacity_bias"] / (1 - rep["opacity_bias"]))      # inverse_sigmoid
    sb = rep["scaling_bias"]
    scale_bias = sb + math.log(-math.expm1(-sb))                                       # inverse_softplus
    min_kernel = rep["filter_kernel_size_3d"]

    class W(nn.Module):
        def __init__(s): super().__init__(); s.gs = gs
        def forward(s, points, cond):
            feats = s.gs(x={"points": points}, cond=cond)["features"]       # (1,N,480)
            B, Np = points.shape[0], points.shape[1]
            offset = s.gs._get_offset(feats)                                # (1,N,ng,3)
            xyz = (offset + points[:, :, None, :]).reshape(B, Np * ng, 3)
            xyz = xyz * aabb[3:] + aabb[:3]
            def grab(key, last):
                r = lay[key]["range"]
                return feats[:, :, r[0]:r[1]].reshape(B, Np * ng, last)
            f_dc = grab("_features_dc", 3)
            opacity = grab("_opacity", 1) * rep["lr"]["_opacity"] + opacity_bias_val
            scaling_raw = grab("_scaling", 3) * rep["lr"]["_scaling"]
            get_scaling = torch.sqrt(F.softplus(scaling_raw + scale_bias) ** 2 + min_kernel ** 2)
            scale = torch.log(get_scaling)
            rot = grab("_rotation", 4) * rep["lr"]["_rotation"]
            rot = rot + torch.tensor([1.0, 0.0, 0.0, 0.0])
            return torch.cat([xyz, f_dc, opacity, scale, rot], dim=-1).reshape(Np * ng, 14)

    g = torch.Generator().manual_seed(0)
    points = torch.rand(1, N, 3, generator=g)
    cond = torch.randn(1, 8192, 16, generator=g)
    w = W().eval()
    with torch.no_grad():
        feats = gs(x={"points": points}, cond=cond)
        gauss = _build_gaussians(gs, {"points": points}, feats)[0]
        xyz, _, f_dc, op, sc, rot = gauss._get_ply_data(transform=None)
        import numpy as np
        ref = torch.tensor(np.concatenate([xyz, f_dc, op, sc, rot], axis=1)).float()   # (N*32,14)
        mine_eager = w(points, cond)
    def masked_cos(a, b):  # ignore non-finite (Python inverse_sigmoid saturates to ±inf on random)
        af, bf = a.flatten().double(), b.flatten().double()
        m = torch.isfinite(af) & torch.isfinite(bf)
        return float((af[m] @ bf[m]) / (af[m].norm() * bf[m].norm())), int((~m).sum())
    cose, ninf = masked_cos(ref, mine_eager)
    print(f"eager W vs Python cos={cose:.6f} (non-finite skipped: {ninf})", flush=True)

    print("converting decode net ...", flush=True)
    coreai_kit.convert(w, (points, cond), ["points", "cond"], ["ply"], f"{OUT}/decode_fp32.aimodel", optimize=False)
    outs = coreai_kit.run(f"{OUT}/decode_fp32.aimodel", {"points": points.numpy(), "cond": cond.numpy()}, compute="cpu")
    conv = torch.tensor(outs["ply"])
    cos, ninf = masked_cos(ref, conv)
    # also gate converted-vs-eager (both finite -> the true conversion fidelity)
    cos2 = float((mine_eager.flatten().double() @ conv.flatten().double()) /
                 (mine_eager.flatten().double().norm() * conv.flatten().double().norm()))
    print(f"=== DECODE GATE: vs_python_cos={cos:.6f} (skip {ninf}) | convert_cos(vs eager)={cos2:.6f} ===", flush=True)


if __name__ == "__main__":
    main()
