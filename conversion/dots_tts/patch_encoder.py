# Community port — NOT an Apple model.
"""Static-KV patch_encoder (VAESemanticEncoder.decode_patch) for the dots.tts audio loop.

Per-patch streaming re-encoder: a causal Conv1d downsample (k=2, stride=2; carries a 1-frame
`conv_tail`) -> in_proj -> a 24-layer causal transformer over a FIXED [1,16,1000,64]/layer KV
buffer -> project 2 tokens -> one LLM embedding [1,1,1536].

⚠️ Despite config.json PatchEncoder qk_norm/rotary_bias=True, the upstream `SuperviseEncoder`
does NOT forward them to `MultiHeadAttention` (only hidden/num_heads/ffn/norm_layer) → the actual
module is a PLAIN causal transformer: **no qk_norm, no RoPE** (confirmed by the checkpoint — there
are no `attn.q_norm/k_norm` keys). RMSNorm pre-norm + MHA + SiLU MLP. Positions drive only the KV
write slot + causal mask (not RoPE). So this is simpler than the backbone (no GQA either: nh=nkv=16).

Reformulated to coreai static KV (mutable_slice_update write + whole-buffer read + runtime causal
mask), mirroring `backbone.py`; gated cos=1.000000 vs the upstream oracle fixture.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreai_models.primitives._ops import mutable_slice_update
from coreai_models.primitives.macos.sdpa import SDPA

from backbone import write_kv_range, causal_buffer_mask  # reuse the proven static-KV helpers


class PECfg:
    def __init__(self, cfg: dict):
        P = cfg["PatchEncoder"]
        self.hidden = P["hidden_size"]          # 1024
        self.n_layers = P["num_layers"]         # 24
        self.n_heads = P["num_heads"]           # 16
        self.head_dim = self.hidden // self.n_heads  # 64
        self.ffn = P["ffn_hidden_size"]         # 4096
        self.input_dim = P["input_dim"]         # 128
        self.patch_size = int(cfg["patch_size"])  # 4
        self.in_ds_rate = 2
        self.out_ds_rate = self.patch_size // self.in_ds_rate  # 2
        self.out_dim = cfg["latent_dim"] * cfg["patch_size"] // 1  # placeholder; set by out_proj shape
        self.rms_eps = 1e-5  # nn.RMSNorm default; overridden to match upstream if needed


class _Mlp(nn.Module):
    def __init__(self, h, ffn):
        super().__init__()
        self.fc1 = nn.Linear(h, ffn, bias=True)
        self.fc2 = nn.Linear(ffn, h, bias=True)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class _Attn(nn.Module):
    def __init__(self, cfg: PECfg):
        super().__init__()
        self.nh, self.hd = cfg.n_heads, cfg.head_dim
        H = cfg.hidden
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=True)


class _Layer(nn.Module):
    def __init__(self, cfg: PECfg):
        super().__init__()
        self.attn = _Attn(cfg)
        self.attn_norm = nn.RMSNorm(cfg.hidden)
        self.ffn_norm = nn.RMSNorm(cfg.hidden)
        self.ffn = _Mlp(cfg.hidden, cfg.ffn)


class PatchEncoderDecode(nn.Module):
    """decode_patch as a coreai static-KV graph.

    inputs : latent_patch [1,4,128], conv_tail [1,128,1], pos (int32 [1] = first write slot)
    state  : keyCache/valueCache [24,1,16,BUF,64]
    outputs: embedding [1,1,1536], new_conv_tail [1,128,1]
    """

    def __init__(self, cfg: PECfg, buf: int):
        super().__init__()
        self.cfg = cfg
        self.buf = buf
        H = cfg.hidden
        # downsample: causal Conv1d(128,128,k=2,stride=2); conv_tail (1 left-pad frame) fed in
        self.ds_proj = nn.Conv1d(cfg.input_dim, cfg.input_dim, kernel_size=cfg.in_ds_rate,
                                 stride=cfg.in_ds_rate)
        self.in_proj = nn.Linear(cfg.input_dim, H)
        self.layers = nn.ModuleList([_Layer(cfg) for _ in range(cfg.n_layers)])
        self.out_proj = nn.Linear(H * cfg.out_ds_rate, cfg.out_dim)
        self.sdpa = SDPA(is_causal=False)

    def _attn(self, layer, x, q_pos, write_start, mask, k_cache, v_cache, li):
        a = layer.attn
        b, q, _ = x.shape
        nh, hd = a.nh, a.hd
        query = a.q_proj(x).reshape(b, q, nh, hd).permute(0, 2, 1, 3)
        key = a.k_proj(x).reshape(b, q, nh, hd).permute(0, 2, 1, 3)
        value = a.v_proj(x).reshape(b, q, nh, hd).permute(0, 2, 1, 3)
        # no qk_norm, no RoPE (upstream defaults)
        write_kv_range(k_cache, li, write_start, key)
        write_kv_range(v_cache, li, write_start, value)
        full_k = k_cache.narrow(0, li, 1).squeeze(0)
        full_v = v_cache.narrow(0, li, 1).squeeze(0)
        out = self.sdpa(query, full_k, full_v, attn_mask=mask).permute(0, 2, 1, 3).reshape(b, q, nh * hd)
        return a.o_proj(out)

    def forward(self, latent_patch, conv_tail, pos, k_cache, v_cache):
        cfg = self.cfg
        # ---- causal downsample step ----
        raw = latent_patch.transpose(1, 2)                       # [1,128,4]
        conv_input = torch.cat([conv_tail, raw], dim=-1)         # [1,128,1+4]
        projected = self.ds_proj(conv_input).transpose(1, 2)     # [1,2,128]
        new_conv_tail = raw[..., -1:]                            # [1,128,1]
        x = self.in_proj(projected)                             # [1,2,1024]

        # ---- 24L causal transformer over the fixed KV buffer ----
        q = cfg.out_ds_rate                                      # 2
        q_pos = (pos.reshape(1, 1) + torch.arange(q, device=pos.device, dtype=torch.int32).reshape(1, q))
        mask = causal_buffer_mask(q_pos, self.buf)
        for i, layer in enumerate(self.layers):
            h = layer.attn_norm(x)
            x = x + self._attn(layer, h, q_pos, pos, mask, k_cache, v_cache, i)
            x = x + layer.ffn(layer.ffn_norm(x))

        # ---- project 2 tokens -> 1 embedding ----
        z = x.reshape(1, cfg.out_ds_rate * cfg.hidden)          # rearrange "b (s d) h -> b s (d h)", s=1
        emb = self.out_proj(z).reshape(1, 1, cfg.out_dim)       # [1,1,1536]
        return emb, new_conv_tail


def build_kv_state(cfg: PECfg, buf: int, dtype=torch.float32):
    n, nh, hd = cfg.n_layers, cfg.n_heads, cfg.head_dim
    return (torch.zeros(n, 1, nh, buf, hd, dtype=dtype), torch.zeros(n, 1, nh, buf, hd, dtype=dtype))


def load_patch_encoder(state_dict: dict, cfg_json: dict, buf: int, dtype=torch.float32):
    cfg = PECfg(cfg_json)
    # out_dim from the checkpoint out_proj shape
    cfg.out_dim = state_dict["patch_encoder.out_proj.weight"].shape[0]
    m = PatchEncoderDecode(cfg, buf).to(dtype).eval()
    P = "patch_encoder."
    g = lambda k: state_dict[P + k].to(dtype)
    with torch.no_grad():
        m.ds_proj.weight.copy_(g("ds_proj.weight")); m.ds_proj.bias.copy_(g("ds_proj.bias"))
        m.in_proj.weight.copy_(g("in_proj.weight")); m.in_proj.bias.copy_(g("in_proj.bias"))
        m.out_proj.weight.copy_(g("out_proj.weight")); m.out_proj.bias.copy_(g("out_proj.bias"))
        for i, layer in enumerate(m.layers):
            lp = f"encoder.layers.{i}."
            layer.attn.q_proj.weight.copy_(g(lp + "attn.q_proj.weight"))
            layer.attn.k_proj.weight.copy_(g(lp + "attn.k_proj.weight"))
            layer.attn.v_proj.weight.copy_(g(lp + "attn.v_proj.weight"))
            layer.attn.o_proj.weight.copy_(g(lp + "attn.o_proj.weight"))
            layer.attn.o_proj.bias.copy_(g(lp + "attn.o_proj.bias"))
            layer.attn_norm.weight.copy_(g(lp + "attn_norm.weight"))
            layer.ffn_norm.weight.copy_(g(lp + "ffn_norm.weight"))
            layer.ffn.fc1.weight.copy_(g(lp + "ffn.fc1.weight")); layer.ffn.fc1.bias.copy_(g(lp + "ffn.fc1.bias"))
            layer.ffn.fc2.weight.copy_(g(lp + "ffn.fc2.weight")); layer.ffn.fc2.bias.copy_(g(lp + "ffn.fc2.bias"))
    return m, cfg
