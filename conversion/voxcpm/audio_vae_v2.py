# Community port — NOT an Apple model.
"""VoxCPM2 AudioVAE decoder (latents[1,64,T] -> 48kHz wav, 1920x upsample).

Same DAC-style causal-conv decoder family as v1 (`audio_vae.py`), scaled + sample-rate conditioned:
* decoder_dim 2048 (v1 1536); rates [8,6,5,2,2,2] = 1920x (v1 [8,8,5,2]=640x) -> 6 upsample blocks,
  final channels 2048//2^6 = 32.
* SampleRateConditionLayer (cond_type "scale_bias") applied to EACH block's INPUT: x = x*scale + bias,
  with scale/bias = Embedding(4, ch)[sr_idx]. The decode sample-rate is fixed (out_sample_rate 48000 ->
  bucketize([20000,30000,40000]) = idx 3), so the per-channel scale/bias vectors are BAKED at load into
  constant buffers (no embedding lookup in the graph).
* use_noise_block=False => fully DETERMINISTIC. weight_norm FOLDED at load (Kokoro lesson #1). Causal
  convs = left-pad + valid conv; causal transpose-convs = convT + right-trim (reused from v1).

Mirrors `_ref_v2/voxref/modules/audiovae/audio_vae_v2.py` (official AudioVAEV2.decode).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from audio_vae import Snake1d, CausalConv1d, DecoderBlock, _fold_wn  # reuse v1 primitives

DEC_DIM = 2048
LATENT = 64
RATES = [8, 6, 5, 2, 2, 2]
SR_BIN_BOUNDARIES = [20000, 30000, 40000]
OUT_SR = 48000


class AudioVAEDecoderV2(nn.Module):
    """depthwise=True, SR-conditioned decoder. forward: latents[B,64,T] -> wav[B,1,1920T]."""

    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList()
        self.model.append(CausalConv1d(LATENT, LATENT, 7, pad=3, groups=LATENT))  # 0 depthwise
        self.model.append(CausalConv1d(LATENT, DEC_DIM, 1))                       # 1 pointwise
        self.block_input_dims = []
        for i, stride in enumerate(RATES):
            ci = DEC_DIM // 2 ** i
            co = DEC_DIM // 2 ** (i + 1)
            self.block_input_dims.append(ci)
            self.model.append(DecoderBlock(ci, co, stride, groups=co))           # 2..7
        ch = DEC_DIM // 2 ** len(RATES)                                           # 32
        self.model.append(Snake1d(ch))                                           # 8
        self.model.append(CausalConv1d(ch, 1, 7, pad=3))                         # 9 (Tanh in forward)
        # baked SR-cond affine (one per block), filled by the loader
        for j, ci in enumerate(self.block_input_dims):
            self.register_buffer(f"cond_scale_{j}", torch.ones(1, ci, 1))
            self.register_buffer(f"cond_bias_{j}", torch.zeros(1, ci, 1))

    def forward(self, z):
        x = self.model[0](z)
        x = self.model[1](x)
        for j in range(len(self.block_input_dims)):
            x = x * getattr(self, f"cond_scale_{j}") + getattr(self, f"cond_bias_{j}")
            x = self.model[2 + j](x)
        x = self.model[8](x)
        x = self.model[9](x)
        return torch.tanh(x)


def _sr_idx() -> int:
    return int(torch.bucketize(torch.tensor(OUT_SR), torch.tensor(SR_BIN_BOUNDARIES)).item())  # = 3


def load_audio_vae_v2(audiovae_sd: dict, dtype=torch.float32) -> AudioVAEDecoderV2:
    """Load decoder.* from audiovae.pth: fold weight_norm, '.conv.' remap, bake SR-cond affine."""
    m = AudioVAEDecoderV2().to(dtype).eval()
    own = m.state_dict()
    src = {k[len("decoder."):]: v for k, v in audiovae_sd.items() if k.startswith("decoder.")}

    # --- bake SR-cond affine from sr_cond_model.{2..7}.{scale,bias}_embed.weight[sr_idx] ---
    idx = _sr_idx()
    baked = {}
    for j in range(len(m.block_input_dims)):
        sk = f"sr_cond_model.{2 + j}.scale_embed.weight"
        bk = f"sr_cond_model.{2 + j}.bias_embed.weight"
        baked[f"cond_scale_{j}"] = src[sk][idx].reshape(1, -1, 1)
        baked[f"cond_bias_{j}"] = src[bk][idx].reshape(1, -1, 1)

    # --- fold weight_norm for the conv path (model.* only; skips sr_cond_model + buffers) ---
    folded = {}
    bases = {k.rsplit(".", 1)[0] for k in src
             if k.startswith("model.") and k.endswith((".weight_g", ".weight_v"))}
    for b in bases:
        folded[b + ".weight"] = _fold_wn(src[b + ".weight_g"], src[b + ".weight_v"])
    for k, v in src.items():
        if not k.startswith("model.") or k.endswith((".weight_g", ".weight_v")):
            continue
        folded[k] = v

    # --- remap: insert '.conv.' before final weight/bias where our CausalConv* nests it ---
    remap = dict(baked)
    for k, v in folded.items():
        if k in own:
            remap[k] = v
        else:
            head, tail = k.rsplit(".", 1)
            cand = head + ".conv." + tail
            remap[cand if cand in own else k] = v

    missing = [k for k in own if k not in remap]
    if missing:
        raise RuntimeError(f"audio_vae_v2: {len(missing)} unloaded, e.g. {missing[:6]}")
    m.load_state_dict({k: v.to(dtype) for k, v in remap.items() if k in own}, strict=True, assign=True)
    return m
