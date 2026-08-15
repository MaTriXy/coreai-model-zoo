"""Qwen3.5-family vision tower shaped as a FIXED-GRID one-shot encoder .aimodel.

Community port — NOT an Apple model. The vision half of the Qwen3.8-27B VLM
(`model.visual.*`, ~400M params): ``patches [n_patch, C*T*P*P] -> image_embeds
[N, out_hidden]``, run ONCE per image as a plain ``.aimodel``; the text decoder
consumes the rows through its embeddings-input variant (qwen3_5.py).

Mirrors HF ``Qwen3_5VisionModel`` (transformers >= 5.12) with everything
positional baked as constants for one canonical grid — the same shape as the
shipped Qwen3-VL tower (qwen3_vl_pipelined.py), minus deepstack
(``deepstack_visual_indexes: []`` on this family):

* patches come from the processor in merge-block-major order, already
  rescaled+normalized (mean/std 0.5); patch vector layout is (C, T, P, P) —
  the Conv3D patch embed collapses to a Linear over that flat vector.
* learned pos-embed: bilinear interpolation of the 48x48 ``pos_embed`` table
  to the (H, W) patch grid, gathered in block-major order — baked ``[n_patch, h]``.
* 2D rotary (theta 10000): head_dim 72, per-axis freqs over head_dim//2, layout
  [row(18) | col(18) | row(18) | col(18)], standard rotate-half over the full
  head_dim — baked cos/sin ``[1, 1, n_patch, 72]``.
* 27 bidirectional blocks: pre-LN (LayerNorm eps 1e-6), fused qkv (bias),
  gelu-tanh MLP 1152->4304.
* merger: LayerNorm(1152) pre-shuffle -> [N, 4608] -> Linear -> exact GELU
  -> Linear -> [N, 5120].

Weights load straight from the checkpoint safetensors + raw config.json (the
export venv's transformers has no qwen3_5 classes; none are needed).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coreai_models.primitives.macos.sdpa import SDPA


class _VisionAttention(nn.Module):
    """Full bidirectional ViT attention with constant 2D rotary cos/sin."""

    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        self.num_heads = vcfg["num_heads"]
        self.head_dim = vcfg["hidden_size"] // vcfg["num_heads"]
        self.qkv = nn.Linear(vcfg["hidden_size"], vcfg["hidden_size"] * 3, bias=True)
        self.proj = nn.Linear(vcfg["hidden_size"], vcfg["hidden_size"], bias=True)
        self.sdpa = SDPA(is_causal=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # x [n, d]; cos/sin [1, 1, n, head_dim]
        n = x.shape[0]
        qkv = self.qkv(x).reshape(n, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(1, 2, 0, 3).unsqueeze(1)  # [3, 1, heads, n, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        def rot(t):
            t1 = t[..., : self.head_dim // 2]
            t2 = t[..., self.head_dim // 2 :]
            return torch.cat((-t2, t1), dim=-1)

        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        out = self.sdpa(q, k, v)  # [1, heads, n, hd]
        out = out.permute(0, 2, 1, 3).reshape(n, self.num_heads * self.head_dim)
        return self.proj(out)


class _VisionMLP(nn.Module):
    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(vcfg["hidden_size"], vcfg["intermediate_size"], bias=True)
        self.linear_fc2 = nn.Linear(vcfg["intermediate_size"], vcfg["hidden_size"], bias=True)

    def forward(self, x):
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(x), approximate="tanh"))


class _VisionBlock(nn.Module):
    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(vcfg["hidden_size"], eps=1e-6)
        self.norm2 = nn.LayerNorm(vcfg["hidden_size"], eps=1e-6)
        self.attn = _VisionAttention(vcfg)
        self.mlp = _VisionMLP(vcfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class _VisionMerger(nn.Module):
    """LayerNorm pre-shuffle -> merge^2 concat -> Linear -> exact GELU -> Linear."""

    def __init__(self, vcfg: dict) -> None:
        super().__init__()
        self.hidden_size = vcfg["hidden_size"] * (vcfg["spatial_merge_size"] ** 2)
        self.norm = nn.LayerNorm(vcfg["hidden_size"], eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_fc2 = nn.Linear(self.hidden_size, vcfg["out_hidden_size"], bias=True)

    def forward(self, x):
        x = self.norm(x).reshape(-1, self.hidden_size)
        return self.linear_fc2(nn.functional.gelu(self.linear_fc1(x)))


class Qwen3_5VisionEncoder(nn.Module):
    """Fixed-grid Qwen3.5 vision tower.

    ``patches [n_patch, in_ch*T*P*P] -> image_embeds [N, out_hidden]`` with
    n_patch = (merge*grid_h)*(merge*grid_w) in the processor's block-major
    patch order and N = grid_h*grid_w merged tokens.
    """

    coreai_externalize_specs: tuple = ()

    def __init__(self, vcfg: dict, grid_h: int = 16, grid_w: int = 16) -> None:
        super().__init__()
        self.vcfg = vcfg
        self.grid_h, self.grid_w = grid_h, grid_w
        merge = vcfg["spatial_merge_size"]
        self.n_patches = (grid_h * merge) * (grid_w * merge)
        patch_dim = vcfg["in_channels"] * vcfg["temporal_patch_size"] * vcfg["patch_size"] ** 2

        self.patch_proj = nn.Linear(patch_dim, vcfg["hidden_size"], bias=True)
        self.blocks = nn.ModuleList([_VisionBlock(vcfg) for _ in range(vcfg["depth"])])
        self.merger = _VisionMerger(vcfg)

        self.register_buffer(
            "pos_embed_const", torch.zeros(self.n_patches, vcfg["hidden_size"]),
            persistent=False)
        head_dim = vcfg["hidden_size"] // vcfg["num_heads"]
        self.register_buffer(
            "cos_const", torch.zeros(1, 1, self.n_patches, head_dim), persistent=False)
        self.register_buffer(
            "sin_const", torch.zeros(1, 1, self.n_patches, head_dim), persistent=False)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(patches)
        x = x + self.pos_embed_const.to(x.dtype)
        cos = self.cos_const.to(x.dtype)
        sin = self.sin_const.to(x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        return self.merger(x)

    # -- constants ----------------------------------------------------------

    def _init_positional_constants(self, pos_embed_table: torch.Tensor) -> None:
        """Bake bilinear pos-embed interpolation + 2D rotary for the fixed grid.

        Mirrors HF ``get_vision_bilinear_indices_and_weights`` +
        ``get_vision_position_ids`` (block-major order) exactly, in fp32.
        """
        vcfg = self.vcfg
        merge = vcfg["spatial_merge_size"]
        H, W = self.grid_h * merge, self.grid_w * merge
        side = int(vcfg["num_position_embeddings"] ** 0.5)
        table = pos_embed_table.float()

        h_idxs = torch.linspace(0, side - 1, H)
        w_idxs = torch.linspace(0, side - 1, W)
        h_f = h_idxs.int()
        w_f = w_idxs.int()
        h_c = (h_f + 1).clamp(max=side - 1)
        w_c = (w_f + 1).clamp(max=side - 1)
        dh = h_idxs - h_f
        dw = w_idxs - w_f

        idx = [
            (h_f[:, None] * side + w_f[None, :]).flatten(),
            (h_f[:, None] * side + w_c[None, :]).flatten(),
            (h_c[:, None] * side + w_f[None, :]).flatten(),
            (h_c[:, None] * side + w_c[None, :]).flatten(),
        ]
        wgt = [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten(),
            ((1 - dh)[:, None] * dw[None, :]).flatten(),
            (dh[:, None] * (1 - dw)[None, :]).flatten(),
            (dh[:, None] * dw[None, :]).flatten(),
        ]
        pe = sum(table[i.long()] * w[:, None] for i, w in zip(idx, wgt))  # [H*W, d] raster
        pe = (
            pe.view(H // merge, merge, W // merge, merge, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(H * W, -1)
        )  # block-major
        self.pos_embed_const.copy_(pe.to(self.pos_embed_const.dtype))

        # 2D rotary (block-major coords), HF Qwen3_5VisionRotaryEmbedding:
        # per-axis freqs over dim = head_dim // 2, [row | col] then tiled x2.
        head_dim = vcfg["hidden_size"] // vcfg["num_heads"]
        rot_dim = head_dim // 2
        inv = 1.0 / (10000.0 ** (torch.arange(0, rot_dim, 2).float() / rot_dim))
        br = torch.arange(H // merge)
        bc = torch.arange(W // merge)
        ir = torch.arange(merge)
        ic = torch.arange(merge)
        rowg = (br[:, None, None, None] * merge + ir[None, None, :, None]).expand(
            H // merge, W // merge, merge, merge).reshape(-1)
        colg = (bc[None, :, None, None] * merge + ic[None, None, None, :]).expand(
            H // merge, W // merge, merge, merge).reshape(-1)
        coords = torch.stack([rowg, colg], dim=-1).float()       # [n, 2]
        freqs = coords[..., None] * inv[None, None, :]           # [n, 2, rot/4]
        freqs = freqs.flatten(1)                                  # [n, rot/2]
        emb = torch.cat([freqs, freqs], dim=-1)                  # [n, head_dim]
        self.cos_const.copy_(emb.cos().view(1, 1, self.n_patches, head_dim))
        self.sin_const.copy_(emb.sin().view(1, 1, self.n_patches, head_dim))

    # -- loading ------------------------------------------------------------

    @classmethod
    def from_hf(
        cls,
        hf_id: str,
        target_dtype: torch.dtype = torch.float16,
        grid_h: int = 16,
        grid_w: int = 16,
    ) -> "Qwen3_5VisionEncoder":
        cfg, sd = _load_visual_state_dict(hf_id, torch.float32)
        vcfg = cfg["vision_config"]
        if vcfg.get("deepstack_visual_indexes"):
            raise NotImplementedError(
                f"deepstack indexes {vcfg['deepstack_visual_indexes']} — this tower "
                "is the no-deepstack Qwen3.5 shape")
        model = cls(vcfg, grid_h=grid_h, grid_w=grid_w).float()

        conv_w = sd.pop("patch_embed.proj.weight")  # [d, C, T, P, P]
        sd["patch_proj.weight"] = conv_w.reshape(conv_w.shape[0], -1)
        sd["patch_proj.bias"] = sd.pop("patch_embed.proj.bias")
        pos_table = sd.pop("pos_embed.weight")
        missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
        missing = [k for k in missing if not k.endswith(
            ("pos_embed_const", "cos_const", "sin_const"))]
        if missing or unexpected:
            raise RuntimeError(f"load mismatch: missing={missing} unexpected={unexpected}")
        model._init_positional_constants(pos_table)
        model = model.to(dtype=target_dtype)
        model.eval()
        return model


def _load_visual_state_dict(hf_id: str, dtype: torch.dtype):
    """(config_dict, {stripped_key: tensor}) for keys under ``model.visual.``."""
    import glob
    import json

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    model_dir = snapshot_download(
        hf_id, allow_patterns=["*.safetensors", "*.safetensors.index.json", "config.json"])
    with open(f"{model_dir}/config.json") as f:
        cfg = json.load(f)
    sd = {}
    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                if key.startswith("model.visual."):
                    sd[key.removeprefix("model.visual.")] = f.get_tensor(key).to(dtype)
    return cfg, sd
