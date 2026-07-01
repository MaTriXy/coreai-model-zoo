"""Net #1: convert TripoSplat DINOv3 ViT-H encoder to Core AI + gate vs torch eager."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/Code/coreai"))  # coreai_kit
import torch, torch.nn.functional as F
from torchvision import transforms
import coreai_kit
from triposplat import TripoSplatPipeline, _DINOV3_NORMALIZE

CK = "ckpts"
pipe = TripoSplatPipeline(
    ckpt_path              = f"{CK}/diffusion_models/triposplat_fp16.safetensors",
    decoder_path           = f"{CK}/vae/triposplat_vae_decoder_fp16.safetensors",
    dinov3_path            = f"{CK}/clip_vision/dino_v3_vit_h.safetensors",
    flux2_vae_encoder_path = f"{CK}/vae/flux2-vae.safetensors",
    rmbg_path              = f"{CK}/background_removal/birefnet.safetensors",
    device                 = "cpu",
)
dino = pipe.dinov3.float().eval()

img = pipe.preprocess_image("static/example_inputs/building_stone_house.webp")
x = _DINOV3_NORMALIZE(transforms.ToTensor()(img).unsqueeze(0).float())
print("dinov3 input:", tuple(x.shape), flush=True)

with torch.no_grad():
    ref = dino(x)
print("dinov3 output:", tuple(ref.shape), flush=True)

ok, maxdiff, outs = coreai_kit.verify(
    dino, (x,), ["pixel_values"], ["feat"],
    "coreai_out/dinov3_fp32.aimodel", atol=2e-2)
# cosine too
import numpy as np
a = ref.flatten().double(); b = torch.tensor(outs["feat"]).flatten().double()
cos = float((a @ b) / (a.norm() * b.norm()))
print(f"=== DINOv3 GATE: ok={ok} maxdiff={maxdiff:.4e} cos={cos:.6f} ===", flush=True)
