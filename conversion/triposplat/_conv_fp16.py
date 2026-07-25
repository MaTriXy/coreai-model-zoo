"""Convert a net to fp16 weights (io kept fp32 so the app needs no change) + gate cos vs fp32 eager.
Usage: python _conv_fp16.py {dit|dinov3|gs|vae}"""
import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/ — coreai_kit
import torch, torch.nn as nn
import coreai_kit
from triposplat import load_flow_model, load_decoder, load_dinov3, load_vae_encoder

CK, OUT = "ckpts", "coreai_out"


class FP16IO(nn.Module):
    """Wrap an fp16 model so the graph io stays fp32 (cast in -> half, out -> float)."""
    def __init__(s, inner): super().__init__(); s.inner = inner
    def forward(s, *a):
        a = [x.half() if torch.is_floating_point(x) else x for x in a]
        o = s.inner(*a)
        return tuple(x.float() for x in o) if isinstance(o, (tuple, list)) else o.float()


def build(net, dtype):
    g = torch.Generator().manual_seed(0)
    if net == "dit":
        fm = load_flow_model(f"{CK}/diffusion_models/triposplat_fp16.safetensors", device="cpu", dtype=dtype).eval()
        class W(nn.Module):
            def __init__(s): super().__init__(); s.m = fm
            def forward(s, latent, camera, t, feature1, feature2):
                o = s.m({"latent": latent, "camera": camera}, t, {"feature1": feature1, "feature2": feature2})
                return o["latent"], o["camera"]
        ex = (torch.randn(1, 8192, 16, generator=g), torch.randn(1, 1, 5, generator=g),
              torch.tensor([1000.0]), torch.randn(1, 4101, 1280, generator=g), torch.randn(1, 4101, 128, generator=g))
        return W().eval(), ex, ["latent", "camera", "t", "feature1", "feature2"], ["pred_latent", "pred_camera"]
    if net == "gs":
        dec = load_decoder(f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors", device="cpu", dtype=dtype)
        gs = dec.gs.eval()
        class W(nn.Module):
            def __init__(s): super().__init__(); s.m = gs
            def forward(s, points, cond): return s.m(x={"points": points}, cond=cond)["features"]
        ex = (torch.rand(1, 8192, 3, generator=g), torch.randn(1, 8192, 16, generator=g))
        return W().eval(), ex, ["points", "cond"], ["features"]
    if net == "dinov3":
        d = load_dinov3(f"{CK}/clip_vision/dino_v3_vit_h.safetensors", device="cpu", dtype=dtype).eval()
        return d, (torch.randn(1, 3, 1024, 1024, generator=g),), ["pixel_values"], ["feat"]
    if net == "vae":
        e = load_vae_encoder(f"{CK}/vae/flux2-vae.safetensors", device="cpu", dtype=dtype).eval()
        class W(nn.Module):
            def __init__(s): super().__init__(); s.e = e
            def forward(s, x): return s.e.encode(x, deterministic=True)
        return W().eval(), (torch.randn(1, 3, 1024, 1024, generator=g),), ["img"], ["feat"]
    raise SystemExit(net)


def bsize(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs) if os.path.isdir(p) else 0


def main():
    net = sys.argv[1]
    m32, ex, inames, onames = build(net, torch.float32)
    with torch.no_grad():
        ref = m32(*ex)
    refs = list(ref) if isinstance(ref, (tuple, list)) else [ref]
    del m32

    m16, _, _, _ = build(net, torch.float16)
    model = FP16IO(m16).eval()
    out_path = f"{OUT}/{net}_fp16.aimodel"
    print(f"[{net}] converting fp16 (optimize=False) ...", flush=True)
    coreai_kit.convert(model, ex, inames, onames, out_path, optimize=False)
    fp16 = bsize(out_path)
    fp32 = bsize(f"{OUT}/{net}_fp32.aimodel") or bsize(f"{OUT}/gs_decoder_fp32.aimodel") or bsize(f"{OUT}/flux2_vae_enc_fp32.aimodel")

    outs = coreai_kit.run(out_path, {n: x.detach().numpy() for n, x in zip(inames, ex)}, compute="cpu")
    for on, r in zip(onames, refs):
        a = r.flatten().double(); b = torch.tensor(outs[on]).flatten().double()
        cos = float((a @ b) / (a.norm() * b.norm()))
        md = float((r - torch.tensor(outs[on])).abs().max())
        print(f"=== [{net}] {on}: cos={cos:.6f} maxdiff={md:.4e} ===", flush=True)
    print(f"=== [{net}] SIZE fp32={fp32/1e6:.0f}MB fp16={fp16/1e6:.0f}MB ratio={fp16/max(fp32,1):.2f} ===", flush=True)


if __name__ == "__main__":
    main()
