"""Net #2: convert Flux2 VAE encoder (conditioning) to Core AI + gate."""
import sys
from pathlib import Path
import os
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/ — coreai_kit
import torch, torch.nn as nn
from torchvision import transforms
import coreai_kit
from triposplat import TripoSplatPipeline

CK = "ckpts"
pipe = TripoSplatPipeline(
    ckpt_path=f"{CK}/diffusion_models/triposplat_fp16.safetensors",
    decoder_path=f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path=f"{CK}/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path=f"{CK}/vae/flux2-vae.safetensors",
    rmbg_path=f"{CK}/background_removal/birefnet.safetensors",
    device="cpu",
)
enc = pipe.vae_encoder.float().eval()

class W(nn.Module):
    def __init__(self, e): super().__init__(); self.e = e
    def forward(self, x): return self.e.encode(x, deterministic=True)
w = W(enc).eval()

img = pipe.preprocess_image("static/example_inputs/building_stone_house.webp")
x = transforms.ToTensor()(img).unsqueeze(0).float() * 2 - 1
print("vae_enc input:", tuple(x.shape), flush=True)
with torch.no_grad():
    ref = w(x)
print("vae_enc output:", tuple(ref.shape), flush=True)

ok, maxdiff, outs = coreai_kit.verify(
    w, (x,), ["img"], ["feat"], "coreai_out/flux2_vae_enc_fp32.aimodel", atol=2e-2)
a = ref.flatten().double(); b = torch.tensor(outs["feat"]).flatten().double()
cos = float((a @ b) / (a.norm() * b.norm()))
print(f"=== VAE-ENC GATE: ok={ok} maxdiff={maxdiff:.4e} cos={cos:.6f} ===", flush=True)
