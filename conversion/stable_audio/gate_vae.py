# Community port — gate the clean Oobleck decoder overlay vs the reference (ref_oracle.pt).
import os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from safetensors.torch import load_file
from oobleck_vae import load_decoder

sd = load_file(os.path.join(HERE, "model.safetensors"))
dec = load_decoder(sd, torch.float32)
print(f"[vae-gate] clean decoder built + loaded ({sum(p.numel() for p in dec.parameters())/1e6:.1f}M params)")

oracle = torch.load(os.path.join(HERE, "ref_oracle.pt"))
latent = oracle["latent"].float()       # [1,64,256]
ref_audio = oracle["audio"].float()     # [1,2,524288]

t0 = time.time()
with torch.inference_mode():
    my_audio = dec(latent)
print(f"[vae-gate] decode {time.time()-t0:.1f}s  out={tuple(my_audio.shape)}")

cos = torch.nn.functional.cosine_similarity(my_audio.reshape(-1), ref_audio.reshape(-1), dim=0).item()
mae = (my_audio - ref_audio).abs().mean().item()
print(f"[vae-gate] my vs reference audio  cos={cos:.6f}  MAE={mae:.5f}  "
      f"{'PASS' if cos >= 0.999 else 'FAIL'}")
