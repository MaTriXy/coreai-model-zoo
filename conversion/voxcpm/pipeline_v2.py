# Community port — NOT an Apple model.
"""Self-contained VoxCPM2 (2B) generator from my exportable overlays — the host-loop REFERENCE that
the Swift/CoreAIKit host will mirror (cf v1 `generate.py`). Loads everything from the raw checkpoint
(model.safetensors + audiovae.pth): backbone (base 28L + residual 8L no-rope), CFMDecoder(LocDiT 12L),
LocEnc 12L, AudioVAE v2 (48kHz), and the host-glue Linears (embed/proj/fsq/stop) — NO official package.

The v2 AR dataflow (verified bit-exact per-component + e2e vs the official model in the gates):
  prefill: combined = text_mask*embed(text)*scale_emb + feat_mask*enc_to_lm(LocEnc(feat))
           base_lm -> enc_outputs; lm_h = (fsq on audio pos)[-1]; res_in = fusion(cat(enc, feat_emb));
           residual_lm -> res_h
  step:    mu = cat(lm_to_dit(lm_h), res_to_dit(res_h))            # 2048 -> two DiT tokens
           pred = CFM(mu, cond=prev_patch, z)                       # [1,4,64]
           curr = enc_to_lm(LocEnc(pred))                           # feedback embed
           stop = argmax(stop_head(silu(stop_proj(lm_h))))
           lm_h = fsq(base_lm.decode(curr, pos));  res_h = residual_lm.decode(fusion(cat(lm_h,curr)), pos)
"""
from __future__ import annotations

import glob
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from minicpm4 import load_backbone, build_kv_state
from feat_decoder_v2 import load_feat_decoder_v2
from feat_encoder_v2 import load_feat_encoder_v2
from audio_vae_v2 import load_audio_vae_v2

PATCH = 4
FEAT = 64
FSQ_SCALE = 9
AUDIO_START = 101


def snap() -> str:
    return sorted(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--openbmb--VoxCPM2/snapshots/*")))[-1]


def _linear(sd, prefix, dtype):
    w = sd[prefix + ".weight"].to(dtype)
    out_f, in_f = w.shape
    has_b = (prefix + ".bias") in sd
    m = nn.Linear(in_f, out_f, bias=has_b).to(dtype).eval()
    m.weight = nn.Parameter(w, requires_grad=False)
    if has_b:
        m.bias = nn.Parameter(sd[prefix + ".bias"].to(dtype), requires_grad=False)
    return m


class VoxCPM2Pipeline:
    def __init__(self, dtype=torch.float32, buf: int = 4096):
        self.dtype = dtype
        self.buf = buf
        from safetensors.torch import load_file
        sd = load_file(snap() + "/model.safetensors")
        cfg = json.load(open(snap() + "/config.json"))
        lm = cfg["lm_config"]
        sf = lm["rope_scaling"]["short_factor"]
        bb = dict(hidden_size=lm["hidden_size"], intermediate_size=lm["intermediate_size"],
                  num_attention_heads=lm["num_attention_heads"],
                  num_key_value_heads=lm["num_key_value_heads"],
                  head_dim=lm["kv_channels"], short_factor=sf)
        self.base = load_backbone(sd, "base_lm.", 28, 73448, buf, dtype, no_rope=False, **bb)
        self.res = load_backbone(sd, "residual_lm.", 8, 0, buf, dtype, no_rope=True, **bb)
        self.cfm = load_feat_decoder_v2(sd, sf)
        self.enc = load_feat_encoder_v2(sd, sf)
        vae_ck = torch.load(snap() + "/audiovae.pth", map_location="cpu", weights_only=True)
        self.vae = load_audio_vae_v2(vae_ck.get("state_dict", vae_ck), dtype)
        # host glue
        self.embed = self.base.embed_tokens
        self.enc_to_lm = _linear(sd, "enc_to_lm_proj", dtype)
        self.lm_to_dit = _linear(sd, "lm_to_dit_proj", dtype)
        self.res_to_dit = _linear(sd, "res_to_dit_proj", dtype)
        self.fusion = _linear(sd, "fusion_concat_proj", dtype)
        self.fsq_in = _linear(sd, "fsq_layer.in_proj", dtype)
        self.fsq_out = _linear(sd, "fsq_layer.out_proj", dtype)
        self.stop_proj = _linear(sd, "stop_proj", dtype)
        self.stop_head = _linear(sd, "stop_head", dtype)
        self.tokenizer = None

    def fsq(self, h):
        h = torch.tanh(self.fsq_in(h))
        h = torch.round(h * FSQ_SCALE) / FSQ_SCALE
        return self.fsq_out(h)

    def _tok(self, text: str):
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(snap(), trust_remote_code=True)
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return torch.tensor(ids + [AUDIO_START], dtype=torch.long).unsqueeze(0)

    @torch.inference_mode()
    def generate(self, text: str, *, min_len=2, max_len=1000, zs=None, seed=0):
        """Zero-shot. zs: optional list of pre-drawn CFM noise [1,64,4] (for parity replay);
        else draws its own. Returns latents [1,64,T*4]."""
        text_token = self._tok(text)
        T = text_token.shape[1]
        combined = self.embed(text_token).to(self.dtype) * 1.0          # zero-shot: feat_mask 0
        kb, vb = build_kv_state(self.base.cfg, self.buf, self.dtype)
        enc_outputs = self.base.prefill(combined, kb, vb)
        lm_h = enc_outputs[:, -1, :]                                    # text-masked -> no FSQ
        res_in = self.fusion(torch.cat((enc_outputs, torch.zeros_like(enc_outputs)), dim=-1))
        krb, vrb = build_kv_state(self.res.cfg, self.buf, self.dtype)
        res_h = self.res.prefill(res_in, krb, vrb)[:, -1, :]

        prefix_cond = torch.zeros(1, PATCH, FEAT, dtype=self.dtype)
        if zs is None and seed is not None:
            torch.manual_seed(seed)
        lat = []
        for i in range(max_len):
            mu = torch.cat((self.lm_to_dit(lm_h), self.res_to_dit(res_h)), dim=-1)
            z = zs[i] if zs is not None else torch.randn(1, FEAT, PATCH, dtype=self.dtype)
            pred = self.cfm(mu, prefix_cond.transpose(1, 2).contiguous(), z).transpose(1, 2)   # [1,4,64]
            curr = self.enc_to_lm(self.enc(pred.unsqueeze(1)))                                  # [1,1,2048]
            lat.append(pred.unsqueeze(1))
            prefix_cond = pred
            stop = self.stop_head(F.silu(self.stop_proj(lm_h))).argmax(-1)[0].item()
            if i > min_len and stop == 1 and zs is None:
                break
            if zs is not None and i + 1 >= len(zs):
                break
            pos = torch.tensor([T + i], dtype=torch.int32)
            lm_h = self.fsq(self.base.decode(curr[:, 0:1, :], pos, kb, vb).reshape(1, -1))
            res_in2 = self.fusion(torch.cat((lm_h, curr[:, 0, :]), dim=-1))
            res_h = self.res.decode(res_in2.unsqueeze(1), pos, krb, vrb).reshape(1, -1)
        return rearrange(torch.cat(lat, dim=1), "b t p d -> b d (t p)", p=PATCH)

    @torch.inference_mode()
    def synth(self, text: str, **kw):
        """text -> 48kHz waveform [N]."""
        lat = self.generate(text, **kw)
        return self.vae(lat.to(self.dtype)).reshape(-1)
