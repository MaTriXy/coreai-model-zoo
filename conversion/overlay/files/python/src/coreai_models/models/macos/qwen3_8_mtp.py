# Qwen3.8 MTP-head drafter (DeepSeek-V3-style multi-token-prediction head) for the
# Core AI authoring path.
#
# Community port — NOT an Apple model. The Qwen3.8 checkpoints ship a trained
# 1-layer drafter under `mtp.*` (config: mtp_num_hidden_layers=1,
# mtp_use_dedicated_embeddings=False) that transformers 5.x does NOT implement
# (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`), so the forward here is
# authored from the safetensors shapes and gated empirically (teacher-forced
# accuracy / spec-decode acceptance vs the target's own stream — there is no HF
# oracle to diff against).
#
#   inputs per step: token t (last committed / previously drafted), hidden h
#                    (the target row that produced t, or the drafter's own
#                    recurrent hidden on later steps)
#   x  = fc( concat( RMSNorm_emb(embed(t)), RMSNorm_hid(h) ) )   # fc: [2d -> d]
#   x  = decoder_layer(x)     # mtp.layers.0: gated full attention + SwiGLU MLP,
#                             # same block family as the target's full layers
#   h' = norm(x)
#   logits = lm_head(h')      # head weight SHARED with the target (untied ckpt)
#   next step feeds (argmax(logits), recurrent hidden)
#
# Ambiguities the checkpoint cannot answer, A/B-gated 2026-08-16 in
# ondevice/_mtp_alpha_probe.py (27B fp16 torch, code+free streams):
#   * concat order into fc: [emb | hidden] WINS decisively (swapped -> acc 0.00)
#   * recurrent hidden fed to the next draft step: post-`norm` h' WINS
#     (v2 mean accept-len 5.86 vs 5.21 code, 1.57 vs 1.50 free)
#   * target-side hidden (the caller's choice): POST-final-norm WINS
#     (v2 depth-1 acc 0.984 vs 0.945 code) — feed the verify bundle's
#     `--emit-hidden post` output.
# Context matters enormously: fresh-KV drafting (v1) gets depth-1 acc ~0.69;
# committed-context replay (v2) gets 0.98 on code. Host loops must be v2.
#
# All RMSNorms use the (1 + weight) zero-centered gain like every other norm in
# this family (mtp.layers.0.input_layernorm mean ~0.04 — plain gain would zero
# the stream).
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNormPlusOne

from .qwen3_5 import Qwen3_5Config, Qwen3_5DecoderLayer

# HF checkpoint prefixes
MTP_PREFIX = "mtp."
EMBED_KEY = "model.language_model.embed_tokens.weight"
HEAD_KEY = "lm_head.weight"

MTP_STATE_NAMES = ("keyCache", "valueCache")


def mtp_block_config(cfg: Qwen3_5Config) -> Qwen3_5Config:
    """1-layer full-attention view of the target config (layer 0 must be full:
    interval=1 makes ``is_full(0)`` true; the GDN fields ride along unused)."""
    import dataclasses

    return dataclasses.replace(
        cfg, num_hidden_layers=1, full_attention_interval=1, layer_types=["full_attention"]
    )


class Qwen3_8MTPDrafter(nn.Module):
    """Stateful S-token MTP drafter graph: (input_ids [b,s], hidden_in [b,s,d],
    position_ids [b,seq], k_cache/v_cache [1,b,n_kv,cap,hd]) -> (logits, hidden_out).

    ``position_ids`` carries the full drafted length so ``offset = seq - s``
    (the decode-graph convention); v1 host use is fresh-KV per round, S=1 steps
    at positions 0..k-1.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, d)
        self.pre_fc_norm_embedding = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer(mtp_block_config(config), 0)])
        self.norm = RMSNormPlusOne(d, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        rd = config.rotary_dim
        inv_freq = 1.0 / (config.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # A/B-gated conventions (see module docstring; probe winners as defaults).
        self.concat_order = "eh"  # "eh" = [emb|hidden] (DeepSeek), "he" = swapped
        self.recurrent_hidden = "postnorm"  # "prenorm" = residual x, "postnorm" = h'

    def rope_cos_sin(self, position_ids: torch.Tensor):
        freqs = position_ids[..., None].float() * self.inv_freq
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_in: torch.Tensor,
        position_ids: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s = input_ids.shape[1]
        seq_len = position_ids.shape[1]
        offset = seq_len - s
        e = self.pre_fc_norm_embedding(self.embed_tokens(input_ids))
        h = self.pre_fc_norm_hidden(hidden_in)
        parts = (e, h) if self.concat_order == "eh" else (h, e)
        x = self.fc(torch.cat(parts, dim=-1))
        q_pos = position_ids.narrow(1, offset, s)
        cos, sin = self.rope_cos_sin(q_pos)
        kv = KVCache(k_cache, v_cache)
        x = self.layers[0](x, cos, sin, kv_cache=kv, offset=offset,
                           seq_len=seq_len, full_idx=0)
        hn = self.norm(x)
        logits = self.lm_head(hn)
        hidden_out = x if self.recurrent_hidden == "prenorm" else hn
        return logits, hidden_out

    def build_kv_state(self, cap: int, batch: int = 1,
                       dtype: torch.dtype = torch.float16) -> dict[str, torch.Tensor]:
        cfg = self.config
        shape = (1, batch, cfg.num_key_value_heads, cap, cfg.head_dim)
        return {"k_cache": torch.zeros(shape, dtype=dtype),
                "v_cache": torch.zeros(shape, dtype=dtype)}

    @classmethod
    def from_checkpoint(
        cls,
        huggingface_model_id: str,
        target_dtype: torch.dtype = torch.float16,
    ) -> "Qwen3_8MTPDrafter":
        """Load the mtp.* head + the shared embed/lm_head tables straight from the
        (multimodal) checkpoint safetensors. ~2.5 GB of unique weights at fp16
        plus the two 1.27 GB tables."""
        import glob
        import os

        from huggingface_hub import snapshot_download
        from safetensors import safe_open
        from transformers import AutoConfig

        from .qwen3_5 import qwen3_5_config_from_hf

        model_dir = snapshot_download(
            huggingface_model_id,
            allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
        )
        raw = AutoConfig.from_pretrained(model_dir)
        config = qwen3_5_config_from_hf(getattr(raw, "text_config", raw))

        model = cls(config)
        model.to(dtype=target_dtype)

        sd: dict[str, torch.Tensor] = {}
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith(MTP_PREFIX):
                        local = key[len(MTP_PREFIX):]
                    elif key == EMBED_KEY:
                        local = "embed_tokens.weight"
                    elif key == HEAD_KEY:
                        local = "lm_head.weight"
                    else:
                        continue
                    sd[local] = f.get_tensor(key).to(target_dtype)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # inv_freq is non-persistent (rebuilt below); anything else missing is real.
        real_missing = [k for k in missing if "inv_freq" not in k]
        if real_missing or unexpected:
            raise RuntimeError(f"MTP load mismatch: missing={real_missing} "
                               f"unexpected={unexpected}")
        rd = config.rotary_dim
        model.inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, rd, 2).float() / rd)
        )
        return model
