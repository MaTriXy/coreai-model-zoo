# Community port — NOT an Apple model.
"""Torch parity gate for the VoxCPM2 (2B) feat_decoder (LocDiT 12L + CFM) and feat_encoder (LocEnc 12L).

My exportable overlays (`feat_decoder_v2`, `feat_encoder_v2`) vs the OFFICIAL OpenBMB modules
(`_ref_v2/voxref/...`, the github source reassembled into an importable package), both loaded from the
real VoxCPM2 checkpoint, fed identical seeded inputs. Three checks:

  1. LocDiT estimator forward (deterministic): (x, mu, t, cond, dt) -> velocity
  2. full UnifiedCFM euler (z seed-matched: both sides draw the SAME randn after manual_seed(0))
  3. LocEnc forward: [B,T,P,64] -> [B,T,H]

Pass = cos >= 0.999 on all three.

  coreai-models/.venv/bin/python gate_v2_feat_torch.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "_ref_v2"))

from feat_decoder import CFMDecoder  # noqa: E402
from feat_decoder_v2 import LocDiTV2, dit_cfg, load_feat_decoder_v2  # noqa: E402
from feat_encoder_v2 import load_feat_encoder_v2  # noqa: E402
from voxref.modules.locdit import CfmConfig, UnifiedCFM, VoxCPMLocDiTV2  # noqa: E402
from voxref.modules.locenc import VoxCPMLocEnc  # noqa: E402
from voxref.modules.minicpm4 import MiniCPM4Config  # noqa: E402

DTYPE = torch.float32
PATCH = 4
FEAT = 64


def snap() -> str:
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]


def load_sd() -> dict:
    from safetensors.torch import load_file
    return load_file(snap() + "/model.safetensors")


def full_cfg() -> dict:
    return json.load(open(snap() + "/config.json"))


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def official_dit_config(lm_cfg: dict, dit_cfg_json: dict) -> MiniCPM4Config:
    """Mirror VoxCPM2Model: copy lm_config, override hidden/ffn/heads/layers/kv_channels, vocab 0."""
    c = dict(lm_cfg)
    c["hidden_size"] = dit_cfg_json["hidden_dim"]
    c["intermediate_size"] = dit_cfg_json["ffn_dim"]
    c["num_attention_heads"] = dit_cfg_json["num_heads"]
    c["num_hidden_layers"] = dit_cfg_json["num_layers"]
    c["kv_channels"] = dit_cfg_json["kv_channels"]
    c["vocab_size"] = 0
    return MiniCPM4Config(**c)


def load_official(module: torch.nn.Module, sd: dict, prefix: str, name: str):
    sub = {k[len(prefix):]: v.to(DTYPE) for k, v in sd.items() if k.startswith(prefix)}
    miss, unexp = module.load_state_dict(sub, strict=False)
    miss = [m for m in miss if not any(t in m for t in ("inv_freq", "cos_cached", "sin_cached"))]
    if miss:
        raise RuntimeError(f"official {name}: unloaded {miss[:6]}")
    return module.to(DTYPE).eval()


def gate_estimator(sd, lm_cfg, dit_json, short_factor) -> float:
    print("\n=== feat_decoder estimator (LocDiT 12L) ===")
    off = official_dit_config(lm_cfg, dit_json)
    off_est = load_official(VoxCPMLocDiTV2(off, in_channels=FEAT), sd, "feat_decoder.estimator.", "estimator")
    my_est = load_feat_decoder_v2(sd, short_factor).estimator  # CFMDecoder.estimator

    torch.manual_seed(0)
    N = 2  # cfg batch of 2
    H_mu = lm_cfg["hidden_size"]      # mu = 2048 (concat of two 1024 dit projections)
    x = torch.randn(N, FEAT, PATCH, dtype=DTYPE)
    mu = torch.randn(N, H_mu, dtype=DTYPE)
    t = torch.rand(N, dtype=DTYPE)
    cond = torch.randn(N, FEAT, PATCH, dtype=DTYPE)
    dt = torch.zeros(N, dtype=DTYPE)
    with torch.inference_mode():
        o = off_est(x, mu, t, cond, dt)
        m = my_est(x, mu, t, cond, dt)
    c = cos(m, o)
    print(f"  estimator cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}  (shapes off={tuple(o.shape)} my={tuple(m.shape)})")
    return c


def gate_cfm(sd, lm_cfg, dit_json, short_factor) -> float:
    print("\n=== feat_decoder full CFM (euler 10-step, cfg 2.0) ===")
    off = official_dit_config(lm_cfg, dit_json)
    off_est = load_official(VoxCPMLocDiTV2(off, in_channels=FEAT), sd, "feat_decoder.estimator.", "estimator")
    off_cfm = UnifiedCFM(in_channels=FEAT, cfm_params=CfmConfig(), estimator=off_est, mean_mode=False).eval()
    my_cfm = load_feat_decoder_v2(sd, short_factor)

    H_mu = lm_cfg["hidden_size"]
    torch.manual_seed(1)
    mu = torch.randn(1, H_mu, dtype=DTYPE)
    cond = torch.randn(1, FEAT, PATCH, dtype=DTYPE)

    with torch.inference_mode():
        torch.manual_seed(0)
        o = off_cfm(mu, n_timesteps=10, patch_size=PATCH, cond=cond,
                    temperature=1.0, cfg_value=2.0, sway_sampling_coef=1.0, use_cfg_zero_star=True)
        torch.manual_seed(0)
        z = torch.randn(1, FEAT, PATCH, dtype=DTYPE)  # same draw the official makes first
        m = my_cfm(mu, cond, z)
    c = cos(m, o)
    print(f"  CFM cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}  (off={tuple(o.shape)} my={tuple(m.shape)})")
    return c


def gate_encoder(sd, lm_cfg, enc_json, short_factor) -> float:
    print("\n=== feat_encoder (LocEnc 12L) ===")
    off_cfg = official_dit_config(lm_cfg, enc_json)
    off_enc = load_official(VoxCPMLocEnc(off_cfg, input_dim=FEAT), sd, "feat_encoder.", "encoder")
    my_enc = load_feat_encoder_v2(sd, short_factor)

    torch.manual_seed(0)
    x = torch.randn(1, 3, PATCH, FEAT, dtype=DTYPE)  # [B, T, P, D]
    with torch.inference_mode():
        o = off_enc(x)
        m = my_enc(x)
    c = cos(m, o)
    print(f"  encoder cos={c:.6f}  {'OK' if c >= 0.999 else 'FAIL'}  (off={tuple(o.shape)} my={tuple(m.shape)})")
    return c


def main():
    sd = load_sd()
    cfg = full_cfg()
    lm = cfg["lm_config"]
    short_factor = lm["rope_scaling"]["short_factor"]
    print(f"[cfg] dit={cfg['dit_config']['num_layers']}L enc={cfg['encoder_config']['num_layers']}L "
          f"patch={cfg['patch_size']} feat={cfg['feat_dim']} head_dim={cfg['dit_config']['kv_channels']} "
          f"|short_factor|={len(short_factor)}")
    cs = [
        gate_estimator(sd, lm, cfg["dit_config"], short_factor),
        gate_cfm(sd, lm, cfg["dit_config"], short_factor),
        gate_encoder(sd, lm, cfg["encoder_config"], short_factor),
    ]
    lo = min(cs)
    print(f"\n>>> min cos = {lo:.6f}  ->  {'GATE PASS' if lo >= 0.999 else 'GATE FAIL'}")
    sys.exit(0 if lo >= 0.999 else 1)


if __name__ == "__main__":
    main()
