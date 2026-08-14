# LFM2.5-VL (Liquid AI) for the Core AI authoring path — SigLIP2-NaFlex vision
# tower + 2-layer projector, plus the text-decoder rider that splices the
# projected image tokens into the already-shipped LFM2 decoder (lfm2.py).
#
# Community port — NOT an Apple model.
#
# TWO graphs, the zoo's VLM shape:
#
#   * ``Lfm2VlVisionEncoder`` — runs ONCE per image as a plain ``.aimodel``:
#     ``patches [n_patch, C*P*P] -> image_embeds [N, text_hidden]`` at a FIXED
#     patch grid (default 32x32 = one 512x512 tile -> 256 tokens, which is
#     exactly the checkpoint's ``max_image_tokens``). Everything positional is
#     a constant at a fixed grid, so the graph is a plain ViT + MLP.
#
#   * ``Lfm2VlPipelinedForCausalLM`` — the LFM2 hybrid decoder on the pipelined
#     engine contract with ``image_embeds [N, h]`` riding the static-input hook:
#     the host rewrites the prompt's ``<image>`` ids to EXTENSION ids ``V + slot``
#     and in-graph ``embedding = ids < V ? embed_tokens[ids] : image_embeds[ids - V]``.
#     With zero image embeds and no extension ids the graph IS the shipped LFM2
#     text decoder.
#
# What this checkpoint family does that a SigLIP port from MiniCPM-V/Qwen-VL
# would get wrong — both are visible in the weight shapes, and both are silent:
#
#   1. ``embeddings.patch_embedding.weight`` is [d, 768] — a LINEAR over
#      already-flattened 16x16x3 patches, not a Conv2d over an image. The host
#      patchifies (SigLIP2 NaFlex), the graph never sees a picture.
#   2. ``embeddings.position_embedding.weight`` is [256, d] — a 16x16 grid that
#      is BILINEARLY RESIZED (antialias=True) to the actual patch grid, per
#      image. At a fixed grid that resize is a load-time constant; getting it
#      wrong costs cosine, not a crash.
#
# And the trap that costs a day: the projector has NO LayerNorm on this
# checkpoint (``projector_use_layernorm: false``, no such weights in the
# safetensors). transformers 4.57.6 applies one unconditionally, and
# nn.LayerNorm's default init (weight 1, bias 0) makes the difference invisible
# — no warning, no garbage, just a quietly wrong reference. Build oracles for
# this family on transformers >= 5.
#
# Projector = pixel_unshuffle(factor 2) -> linear_1 -> gelu(EXACT erf, not the
# tower's tanh approximation) -> linear_2. The unshuffle's axis order follows
# HF's ``Lfm2VlMultiModalProjector.pixel_unshuffle`` exactly (it names dim 1
# "width" and dim 2 "height", which reads backwards; the tensor it is handed is
# [1, patch_rows, patch_cols, d]).
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.models.macos.lfm2 import (
    DECODE_STATE_NAMES,
    Lfm2Config,
    Lfm2ForCausalLMStateful,
    Lfm2Model,
    build_decode_state,
    lfm2_config_from_dict,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.sdpa import SDPA

__all__ = [
    "Lfm2VlVisionConfig",
    "Lfm2VlVisionEncoder",
    "Lfm2VlPipelinedForCausalLM",
    "lfm2_text_core_from_hf",
    "lfm2_vl_configs_from_dict",
    "load_lfm2_vl_state_dict",
]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Lfm2VlVisionConfig:
    """Vision tower + projector geometry, parsed from the raw config.json."""

    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    patch_size: int = 16
    num_patches: int = 256          # position-embedding table: 16x16
    layer_norm_eps: float = 1e-6
    # projector
    downsample_factor: int = 2
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    projector_use_layernorm: bool = False
    text_hidden_size: int = 1024

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def patch_dim(self) -> int:
        return self.num_channels * self.patch_size * self.patch_size

    @property
    def pos_grid(self) -> int:
        return int(round(self.num_patches**0.5))


def lfm2_vl_configs_from_dict(raw: dict) -> tuple[Lfm2VlVisionConfig, Lfm2Config]:
    """(vision config, text config) from a raw LFM2.5-VL config.json."""
    v = raw["vision_config"]
    if v.get("projector_use_layernorm") or raw.get("projector_use_layernorm"):
        # No such weights ship with LFM2.5-VL; a checkpoint that sets this would
        # need the extra LayerNorm authored, and silently loses cosine without.
        raise NotImplementedError(
            "projector_use_layernorm=true is not authored (LFM2.5-VL ships false)"
        )
    vision = Lfm2VlVisionConfig(
        hidden_size=v["hidden_size"],
        intermediate_size=v["intermediate_size"],
        num_hidden_layers=v["num_hidden_layers"],
        num_attention_heads=v["num_attention_heads"],
        num_channels=v.get("num_channels", 3),
        patch_size=v.get("patch_size", 16),
        num_patches=v.get("num_patches", 256),
        layer_norm_eps=v.get("layer_norm_eps", 1e-6),
        downsample_factor=raw.get("downsample_factor", 2),
        projector_hidden_size=raw["projector_hidden_size"],
        projector_bias=raw.get("projector_bias", True),
        projector_use_layernorm=False,
        text_hidden_size=raw["text_config"]["hidden_size"],
    )
    return vision, lfm2_config_from_dict(raw["text_config"])


def _snapshot(hf_id_or_dir: str) -> str:
    if os.path.isdir(hf_id_or_dir):
        return hf_id_or_dir
    from huggingface_hub import snapshot_download

    return snapshot_download(
        hf_id_or_dir,
        allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"],
    )


def load_lfm2_vl_state_dict(
    hf_id_or_dir: str, prefix: str, dtype: torch.dtype
) -> tuple[dict, dict[str, torch.Tensor]]:
    """(raw config.json, {key_without_prefix: tensor}) for one sub-tree."""
    import glob

    from safetensors import safe_open

    model_dir = _snapshot(hf_id_or_dir)
    with open(os.path.join(model_dir, "config.json")) as f:
        raw = json.load(f)
    files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {model_dir}")
    sd: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith(prefix):
                    sd[key.removeprefix(prefix)] = f.get_tensor(key).to(dtype)
    if not sd:
        raise RuntimeError(f"no keys under prefix {prefix!r} in {model_dir}")
    return raw, sd


# --------------------------------------------------------------------------- #
# Vision tower (SigLIP2 NaFlex at a fixed grid)
# --------------------------------------------------------------------------- #
class _VisionAttention(nn.Module):
    """Full bidirectional ViT attention. No rotary, no mask: at a fixed full
    grid every patch is real, so the NaFlex padding mask is all-ones."""

    def __init__(self, vcfg: Lfm2VlVisionConfig) -> None:
        super().__init__()
        self.num_heads = vcfg.num_attention_heads
        self.head_dim = vcfg.head_dim
        d = vcfg.hidden_size
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        shape = (1, n, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = self.sdpa(q, k, v)                       # [1, heads, n, hd]
        out = out.transpose(1, 2).reshape(n, self.num_heads * self.head_dim)
        return self.out_proj(out)


class _VisionMLP(nn.Module):
    def __init__(self, vcfg: Lfm2VlVisionConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(vcfg.hidden_size, vcfg.intermediate_size, bias=True)
        self.fc2 = nn.Linear(vcfg.intermediate_size, vcfg.hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # config hidden_act = "gelu_pytorch_tanh"
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class _VisionBlock(nn.Module):
    def __init__(self, vcfg: Lfm2VlVisionConfig) -> None:
        super().__init__()
        eps = vcfg.layer_norm_eps
        self.layer_norm1 = nn.LayerNorm(vcfg.hidden_size, eps=eps)
        self.layer_norm2 = nn.LayerNorm(vcfg.hidden_size, eps=eps)
        self.self_attn = _VisionAttention(vcfg)
        self.mlp = _VisionMLP(vcfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class Lfm2VlVisionEncoder(nn.Module):
    """Fixed-grid LFM2.5-VL vision tower + projector.

    ``patches [grid_h*grid_w, C*P*P] -> image_embeds [N, text_hidden]`` with
    N = (grid_h//f) * (grid_w//f) tokens in row-major order, f = downsample
    factor. The default 32x32 grid is one 512x512 tile -> 256 tokens.

    The grid is a python constant, so the position-embedding resize folds to a
    baked buffer and the projector's unshuffle folds to static reshapes.
    """

    def __init__(self, vcfg: Lfm2VlVisionConfig, grid_h: int = 32, grid_w: int = 32) -> None:
        super().__init__()
        f = vcfg.downsample_factor
        if grid_h % f or grid_w % f:
            raise ValueError(f"grid {grid_h}x{grid_w} not divisible by downsample factor {f}")
        self.vcfg = vcfg
        self.grid_h, self.grid_w = grid_h, grid_w
        self.n_patches = grid_h * grid_w
        self.n_tokens = (grid_h // f) * (grid_w // f)

        self.patch_embedding = nn.Linear(vcfg.patch_dim, vcfg.hidden_size, bias=True)
        self.layers = nn.ModuleList(
            [_VisionBlock(vcfg) for _ in range(vcfg.num_hidden_layers)]
        )
        self.post_layernorm = nn.LayerNorm(vcfg.hidden_size, eps=vcfg.layer_norm_eps)

        in_ch = vcfg.hidden_size * f * f
        self.linear_1 = nn.Linear(in_ch, vcfg.projector_hidden_size, bias=vcfg.projector_bias)
        self.linear_2 = nn.Linear(
            vcfg.projector_hidden_size, vcfg.text_hidden_size, bias=vcfg.projector_bias
        )

        self.register_buffer(
            "pos_embed_const",
            torch.zeros(self.n_patches, vcfg.hidden_size),
            persistent=False,
        )

    # -- graph --------------------------------------------------------------

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedding(patches) + self.pos_embed_const.to(patches.dtype)
        for layer in self.layers:
            x = layer(x)
        x = self.post_layernorm(x)
        return self.project(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """post_layernorm hidden [n_patch, d] -> image_embeds [N, text_hidden]."""
        f = self.vcfg.downsample_factor
        h, w, d = self.grid_h, self.grid_w, self.vcfg.hidden_size
        # HF Lfm2VlMultiModalProjector.pixel_unshuffle, on [1, h, w, d]:
        #   [1,h,w,d] -> [1,h,w/f,d*f] -> permute -> [1,w/f,h,d*f]
        #             -> [1,w/f,h/f,d*f^2] -> permute -> [1,h/f,w/f,d*f^2]
        u = x.reshape(1, h, w // f, d * f)
        u = u.permute(0, 2, 1, 3)
        u = u.reshape(1, w // f, h // f, d * f * f)
        u = u.permute(0, 2, 1, 3)
        u = self.linear_1(u)
        u = F.gelu(u)                       # projector_hidden_act = "gelu" (exact)
        u = self.linear_2(u)
        return u.reshape(self.n_tokens, self.vcfg.text_hidden_size)

    # -- constants ----------------------------------------------------------

    def _init_positional_constants(self, pos_table: torch.Tensor) -> None:
        """Bake the NaFlex position-embedding resize for the fixed grid.

        Mirrors ``Siglip2VisionEmbeddings.resize_positional_embeddings``: the
        16x16 table is bilinearly resized WITH antialias to the patch grid, in
        fp32 (HF upcasts on CPU for the same reason).
        """
        side = self.vcfg.pos_grid
        table = pos_table.float().reshape(side, side, -1)
        table = table.permute(2, 0, 1).unsqueeze(0)             # [1, d, side, side]
        resized = F.interpolate(
            table,
            size=(self.grid_h, self.grid_w),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        pe = resized.reshape(self.vcfg.hidden_size, self.n_patches).transpose(0, 1)
        self.pos_embed_const.copy_(pe.to(self.pos_embed_const.dtype))

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id_or_dir: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 32,
        grid_w: int = 32,
    ) -> "Lfm2VlVisionEncoder":
        raw, sd = load_lfm2_vl_state_dict(
            hf_id_or_dir, "model.vision_tower.vision_model.", torch.float32
        )
        _, proj = load_lfm2_vl_state_dict(
            hf_id_or_dir, "model.multi_modal_projector.", torch.float32
        )
        vcfg, _ = lfm2_vl_configs_from_dict(raw)
        model = cls(vcfg, grid_h=grid_h, grid_w=grid_w).float()

        out: dict[str, torch.Tensor] = {}
        out["patch_embedding.weight"] = sd.pop("embeddings.patch_embedding.weight")
        out["patch_embedding.bias"] = sd.pop("embeddings.patch_embedding.bias")
        pos_table = sd.pop("embeddings.position_embedding.weight")
        for key, tensor in sd.items():
            out[key.replace("encoder.layers.", "layers.")] = tensor
        for key, tensor in proj.items():
            out[key] = tensor

        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith("pos_embed_const")]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        model._init_positional_constants(pos_table)
        model = model.to(dtype=target_dtype)
        model.eval()
        return model


# --------------------------------------------------------------------------- #
# Text decoder rider (LFM2 hybrid + image_embeds static input)
# --------------------------------------------------------------------------- #
def _text_state_dict(sd: dict[str, torch.Tensor], fp32_attn_proj: bool) -> dict:
    """VL ``model.language_model.*`` weights -> the lfm2 authoring tree.

    The only differences from the plain LFM2 checkpoint are the key prefix
    (stripped by the caller) and HF's w1/w3/w2 MLP names. ``fp32_attn_proj``
    keeps the four attention projections in fp32 on an fp16 load — lfm2.py
    IMPORTANT #2: the GPU delegate's fused attention prologue loses ~1.3% in
    fp16 under a dynamic graph, which LFM2.5's large q/k-norm gains compound
    across the stack into garbage logits.
    """
    import re

    mlp_map = {
        ".feed_forward.w1.": ".feed_forward.gate_proj.",
        ".feed_forward.w3.": ".feed_forward.up_proj.",
        ".feed_forward.w2.": ".feed_forward.down_proj.",
    }
    attn_proj_re = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$")
    out: dict[str, torch.Tensor] = {}
    for key, tensor in sd.items():
        local = key
        for old, new in mlp_map.items():
            if old in local:
                local = local.replace(old, new)
                break
        if fp32_attn_proj and attn_proj_re.search(local):
            tensor = tensor.to(torch.float32)
        out["model." + local] = tensor
    return out


def lfm2_text_core_from_hf(
    hf_id_or_dir: str,
    target_dtype: torch.dtype = torch.float16,
    fp32_attn_proj: bool = True,
) -> Lfm2ForCausalLMStateful:
    """The VL checkpoint's decoder as a PLAIN LFM2 text model — no image input.

    Two uses, both real: it is a working 350M LFM2 text bundle on its own, and
    it is the only way to get a Mac decode number for this port, because
    ``llm-benchmark`` cannot bind the VLM bundle's ``image_embeds`` buffer (the
    same proxy the MiniCPM-V-4.6 card documents). Decode cost is the same graph
    plus one bound buffer, so the proxy measures the shipped decoder's speed —
    but it IS a proxy, and a card that reports it should say so.
    """
    raw, sd = load_lfm2_vl_state_dict(hf_id_or_dir, "model.language_model.", target_dtype)
    _, text_cfg = lfm2_vl_configs_from_dict(raw)
    model = Lfm2ForCausalLMStateful(text_cfg).to(dtype=target_dtype)
    missing, unexpected = model.load_state_dict(
        _text_state_dict(sd, fp32_attn_proj), strict=False, assign=True
    )
    missing = [k for k in missing if not k.endswith(("lm_head.weight", "inv_freq"))]
    if missing or unexpected:
        raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
    if text_cfg.tie_embedding:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()
    model.eval()
    return model



class Lfm2VlPipelinedForCausalLM(nn.Module):
    """LFM2.5-VL text decoder on the pipelined-engine contract.

    ``(input_ids [1,s] dyn, position_ids [1,total] dyn, image_embeds [N,h],
       keyCache/valueCache/convState) -> logits``

    Image tokens arrive as EXTENSION ids ``V + slot`` (slot = 0..N-1 in the
    vision encoder's row-major token order); everything else is the shipped
    LFM2 decoder, states and fused conv write-back included.
    """

    def __init__(self, config: Lfm2Config, n_image_tokens: int = 256) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embedding:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.n_image_tokens = n_image_tokens
        self.last_token_only = False

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        conv_state: torch.Tensor,
    ) -> torch.Tensor:
        V = self.config.vocab_size
        N = self.n_image_tokens
        b, s = input_ids.shape

        is_img = input_ids >= V
        slot = (input_ids - V).clamp(0, N - 1).reshape(-1)
        e_txt = self.model.embed_tokens(input_ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, slot).reshape(b, s, -1)
        embeds = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)

        h = self.model.forward_stateful_embeds(
            embeds, position_ids, KVCache(k_cache, v_cache), conv_state
        )
        if self.last_token_only:
            h = h[:, -1:, :]
        return self.lm_head(h)

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id_or_dir: str,
        target_dtype: torch.dtype = torch.float16,
        n_image_tokens: int = 256,
        fp32_attn_proj: bool = True,
    ) -> "Lfm2VlPipelinedForCausalLM":
        """Load the text side of an LFM2.5-VL checkpoint.

        The decoder IS the shipped LFM2 decoder — the whole difference is that
        its keys live under ``model.language_model.`` and its config under
        ``text_config``. ``fp32_attn_proj`` keeps the four attention
        projections fp32 on an fp16 load (lfm2.py IMPORTANT #2: the GPU
        delegate's fused attention prologue loses ~1.3% in fp16 under a dynamic
        graph, which LFM2.5's large q/k-norm gains compound into garbage).
        """
        raw, sd = load_lfm2_vl_state_dict(
            hf_id_or_dir, "model.language_model.", target_dtype
        )
        _, text_cfg = lfm2_vl_configs_from_dict(raw)
        model = cls(text_cfg, n_image_tokens=n_image_tokens)
        model = model.to(dtype=target_dtype)

        out = _text_state_dict(sd, fp32_attn_proj)
        missing, unexpected = model.load_state_dict(out, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(("lm_head.weight", "inv_freq"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        if text_cfg.tie_embedding:
            model.lm_head.weight = model.model.embed_tokens.weight
        model.model.reset_buffers()
        model.eval()
        return model

    # -- export -------------------------------------------------------------

    def build_export_spec(
        self,
        target_dtype: torch.dtype,
        max_context_length: int,
        trace_kv_len: int,
        trace_query: int = 1,
        trace_past: int = 64,
        static_ids: bool | None = None,
    ) -> dict:
        """Pipelined-engine spec: the lfm2 hybrid spec + the image_embeds
        static input (shape-fixed, like every other zoo VLM rider)."""
        cfg = self.config
        input_ids = torch.randint(1, cfg.vocab_size, (1, trace_query), dtype=torch.int32)
        position_ids = torch.arange(
            trace_past + trace_query, dtype=torch.int32
        ).unsqueeze(0)
        state = build_decode_state(cfg, max_seq_len=trace_kv_len, dtype=target_dtype)
        reference_inputs = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "image_embeds": torch.zeros(
                self.n_image_tokens, cfg.hidden_size, dtype=target_dtype
            ),
            "k_cache": state["k_cache"],
            "v_cache": state["v_cache"],
            "conv_state": state["conv_state"],
        }
        if static_ids is None:
            static_ids = trace_query == 1
        pos_min = max(2, trace_query) if static_ids else 2
        seq_pos = torch.export.Dim("seq_pos", min=pos_min, max=max_context_length - 1)
        k_seq = torch.export.Dim("k_seq", min=trace_kv_len, max=max_context_length)
        v_seq = torch.export.Dim("v_seq", min=trace_kv_len, max=max_context_length)
        ids_shape = (
            None
            if static_ids
            else {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 2)}
        )
        dynamic_shapes = {
            "input_ids": ids_shape,
            "position_ids": {1: seq_pos},
            "image_embeds": None,
            "k_cache": {KVCache.seq_len_dim(): k_seq},
            "v_cache": {KVCache.seq_len_dim(): v_seq},
            "conv_state": None,
        }
        return {
            "reference_inputs": reference_inputs,
            "dynamic_shapes": dynamic_shapes,
            "input_names": ("input_ids", "position_ids", "image_embeds"),
            "output_names": ("logits",),
            "state_names": DECODE_STATE_NAMES,
        }
