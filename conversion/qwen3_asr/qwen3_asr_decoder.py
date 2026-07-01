# Community port — NOT an Apple model.
"""Qwen3-ASR text decoder for the Core AI pipelined fast path.

The Qwen3-ASR ``thinker`` decoder is **standard Qwen3** (28L, GQA 16:8, head_dim 128, QK-RMSNorm,
SwiGLU, tied head, vocab 151936). Its config declares interleaved MRoPE, but ``get_rope_index``
feeds 3 *identical* position copies for ASR, so MRoPE collapses bit-exactly to plain 1-D RoPE
(theta 1e6) — verified token-for-token vs the HF oracle (see conversion/qwen3_asr/STATE.md). We
therefore strip ``rope_scaling`` and reuse the zoo's ``qwen3.py`` blocks unchanged.

Audio is injected via the zoo's id-space trick (mirrors qwen2_5_omni_thinker): audio placeholder
ids = ``V + slot``; the graph ``index_select``s row ``slot`` of the static ``audio_embeds [N, 2048]``
input (the AuT encoder's output) at the embedding step. The host rewrites the prompt's
``<|audio_pad|>`` (151676) positions to ``V + 0..N-1`` and passes the encoder tokens.

Bundle: ``input_ids, position_ids, audio_embeds, keyCache, valueCache -> logits``.
"""
from __future__ import annotations

import glob
import json

import torch
import torch.nn as nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from coreai_models.models.macos.qwen3 import Qwen3Model
from coreai_models.primitives.macos.cache import KVCache

PIPELINED_STATE_NAMES = ("keyCache", "valueCache")


def _text_qwen3_config(hf_id: str) -> Qwen3Config:
    """Read the thinker text_config from config.json and normalize to plain Qwen3Config (1-D RoPE).

    Parsed directly (no AutoConfig) so this module does not depend on the custom `qwen_asr`
    `qwen3_asr` config being registered.
    """
    from huggingface_hub import hf_hub_download

    d = json.load(open(hf_hub_download(hf_id, "config.json")))
    tc = dict(d["thinker_config"]["text_config"] if "thinker_config" in d else d["text_config"])
    tc["rope_scaling"] = None          # MRoPE is a no-op for ASR -> plain 1-D RoPE (theta 1e6)
    tc["model_type"] = "qwen3"
    tc.pop("dtype", None)
    return Qwen3Config(**tc)


class Qwen3ASRDecoderPipelined(nn.Module):
    """Engine-shaped Qwen3-ASR decoder; audio embeds injected via ``V + slot`` ids."""

    coreai_externalize_specs: tuple = ()

    def __init__(self, config: Qwen3Config, n_audio_tokens: int) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.n_audio_tokens = int(n_audio_tokens)

    def forward(
        self,
        input_ids: torch.Tensor,     # [1, s] int32; audio tokens = V + slot
        position_ids: torch.Tensor,  # [1, total] int32 sequential ramp
        audio_embeds: torch.Tensor,  # [N, h] static input (AuT encoder output)
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        m = self.model
        V = self.config.vocab_size
        N = self.n_audio_tokens
        b, s = input_ids.shape

        ids = input_ids
        is_aud = ids >= V
        slot = (ids - V).clamp(0, N - 1)
        e_txt = m.embed_tokens(ids.clamp(0, V - 1))
        e_aud = audio_embeds.index_select(0, slot.reshape(-1)).reshape(b, s, -1)
        x = torch.where(is_aud.unsqueeze(-1), e_aud.to(e_txt.dtype), e_txt)

        cache = KVCache(k_cache, v_cache)
        for layer in m.layers:
            x = layer(x, position_ids, cache)
        return self.lm_head(m.norm(x))

    # -- loading ------------------------------------------------------------
    @classmethod
    def from_hf(cls, hf_id: str, n_audio_tokens: int,
                target_dtype: torch.dtype = torch.float16) -> "Qwen3ASRDecoderPipelined":
        from huggingface_hub import snapshot_download
        from safetensors import safe_open

        cfg = _text_qwen3_config(hf_id)
        model = cls(cfg, n_audio_tokens).to(dtype=target_dtype)

        snap = snapshot_download(hf_id, allow_patterns=["*.safetensors"])
        sd = {}
        for path in glob.glob(snap + "/*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as f:
                for k in f.keys():  # noqa: SIM118
                    if k.startswith("thinker.model.") or k == "thinker.lm_head.weight":
                        sd[k[len("thinker."):]] = f.get_tensor(k).to(target_dtype)

        # remap to the zoo qwen3 tree: fuse qkv + qk_norm (USE_FUSED_KV recipe)
        out = {}
        hd, nh, nkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
        for i in range(cfg.num_hidden_layers):
            pre = f"model.layers.{i}.self_attn."
            qw = sd.pop(pre + "q_proj.weight"); kw = sd.pop(pre + "k_proj.weight"); vw = sd.pop(pre + "v_proj.weight")
            out[pre + "qkv_proj.weight"] = torch.cat([qw, kw, vw], dim=0)
            qn = sd.pop(pre + "q_norm.weight"); kn = sd.pop(pre + "k_norm.weight")
            out[pre + "qk_norm.weight"] = torch.cat(
                [qn.view(1, 1, hd).expand(nh, 1, hd), kn.view(1, 1, hd).expand(nkv, 1, hd)], dim=0).contiguous()
        for k, v in sd.items():
            out[k] = v  # model.layers.*.{o_proj,mlp.*,*layernorm}, model.embed_tokens, model.norm, lm_head
        if cfg.tie_word_embeddings:
            out.pop("lm_head.weight", None)

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if k != "lm_head.weight"]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.embed_tokens.weight
        return model.eval()

    def build_export_spec(self, target_dtype: torch.dtype, max_context_length: int,
                          trace_kv_len: int, trace_query: int = 8, trace_past: int = 64) -> dict:
        cfg = self.config
        N, h = self.n_audio_tokens, cfg.hidden_size
        input_ids = torch.randint(1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
        position_ids = torch.arange(trace_past + trace_query, dtype=torch.int32).unsqueeze(0)
        k_cache = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads,
                              trace_kv_len, cfg.head_dim, dtype=target_dtype)
        v_cache = torch.zeros_like(k_cache)
        reference_inputs = {
            "input_ids": input_ids, "position_ids": position_ids,
            "audio_embeds": torch.zeros(N, h, dtype=target_dtype),
            "k_cache": k_cache, "v_cache": v_cache,
        }
        seq_pos = torch.export.Dim("seq_pos", min=2, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        ids_shape = None if trace_query == 1 else {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)}
        dynamic_shapes = {
            "input_ids": ids_shape, "position_ids": {1: seq_pos}, "audio_embeds": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq}, "v_cache": {KVCache.seq_len_dim(): v_seq},
        }
        return {
            "reference_inputs": reference_inputs, "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "audio_embeds"),
            "output_names": ("logits",), "state_names": PIPELINED_STATE_NAMES,
        }
