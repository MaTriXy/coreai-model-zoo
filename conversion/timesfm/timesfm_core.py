"""Re-authored, torch.export-clean TimesFM 2.5 graph core.

Graph boundary = pure feed-forward transformer over patch tokens:
    tok_in (B,N,2P) -> input_ff_layer -> 20 decoder layers
      -> output_projection_point (B,N,H*Q), output_projection_quantiles (B,N,Lq*Q)
No data-dependent control flow, no KV cache, static shapes. Host does all RevIN/flip.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


class ResidualBlock(nn.Module):
    def __init__(self, in_dims, hid_dims, out_dims, bias):
        super().__init__()
        self.input_layer = nn.Linear(in_dims, hid_dims, bias=bias)
        self.output_layer = nn.Linear(hid_dims, out_dims, bias=bias)
        self.residual_layer = nn.Linear(in_dims, out_dims, bias=bias)

    def forward(self, x):
        h = F.silu(self.input_layer(x))
        return self.output_layer(h) + self.residual_layer(x)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg["heads"]
        self.head_dim = cfg["head_dim"]
        d = cfg["hidden"]
        self.q_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, d, bias=False)
        self.q_norm = RMSNorm(self.head_dim, cfg["eps"])
        self.k_norm = RMSNorm(self.head_dim, cfg["eps"])
        self.scaling = nn.Parameter(torch.ones(self.head_dim))

    def forward(self, x, cos, sin, attn_bias):
        B, N, _ = x.shape
        shp = (B, N, self.n_heads, self.head_dim)
        q = self.q_proj(x).view(shp).transpose(1, 2)  # B,h,N,hd
        k = self.k_proj(x).view(shp).transpose(1, 2)
        v = self.v_proj(x).view(shp).transpose(1, 2)
        # RoPE (cos/sin: B,N,hd -> unsqueeze head dim)
        c = cos.unsqueeze(1)
        s = sin.unsqueeze(1)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        q = self.q_norm(q)
        k = self.k_norm(k)
        scale = F.softplus(self.scaling).mul(1.442695041 / math.sqrt(self.head_dim))
        q = q * scale[None, None, None, :]
        aw = torch.matmul(q, k.transpose(2, 3)) + attn_bias  # scaling folded into q
        aw = F.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
        o = torch.matmul(aw, v).transpose(1, 2).reshape(B, N, -1)
        return self.o_proj(o)


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["hidden"]
        self.self_attn = Attention(cfg)
        self.input_layernorm = RMSNorm(d, cfg["eps"])
        self.post_attention_layernorm = RMSNorm(d, cfg["eps"])
        self.pre_feedforward_layernorm = RMSNorm(d, cfg["eps"])
        self.post_feedforward_layernorm = RMSNorm(d, cfg["eps"])
        self.mlp_fc1 = nn.Linear(d, cfg["inter"], bias=False)
        self.mlp_fc2 = nn.Linear(cfg["inter"], d, bias=False)

    def forward(self, x, cos, sin, attn_bias):
        r = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin, attn_bias)
        x = self.post_attention_layernorm(x) + r
        r = x
        x = self.pre_feedforward_layernorm(x)
        x = self.mlp_fc2(F.silu(self.mlp_fc1(x)))
        x = self.post_feedforward_layernorm(x) + r
        return x


class TimesFmCore(nn.Module):
    """The exportable graph. Inputs: tok_in (B,N,2P), cos/sin (B,N,hd), attn_bias (B,1,N,N).
    Outputs: proj_point (B,N,H*Q), proj_q (B,N,Lq*Q)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        P = cfg["patch"]
        d = cfg["hidden"]
        Q = cfg["q"] + 1
        self.input_ff_layer = ResidualBlock(2 * P, d, d, bias=True)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg["layers"])])
        self.output_projection_point = ResidualBlock(d, d, cfg["horizon"] * Q, bias=False)
        self.output_projection_quantiles = ResidualBlock(d, d, cfg["oql"] * Q, bias=False)

    def forward(self, tok_in, cos, sin, attn_bias):
        x = self.input_ff_layer(tok_in)
        for layer in self.layers:
            x = layer(x, cos, sin, attn_bias)
        return self.output_projection_point(x), self.output_projection_quantiles(x)


class EngineCore:
    """Callable matching TimesFmCore.forward, backed by a loaded Core AI graph function.

    Pass `model.load_function("main")` and the bundle dtype. Usable as the `core` argument to
    host_forecast.forecast(). coreai.runtime is imported lazily so this module still imports in a
    plain torch/transformers env (for the oracle)."""

    def __init__(self, fn, dtype):
        self.fn, self.dtype = fn, dtype

    def to(self, *a, **k):
        return self

    def __call__(self, tok_in, cos, sin, attn_bias):
        import asyncio
        import numpy as np
        import coreai.runtime as rt
        d = self.dtype
        out = asyncio.run(self.fn({
            "tok_in": rt.NDArray(tok_in.to(d).numpy()),
            "cos": rt.NDArray(cos.to(d).numpy()),
            "sin": rt.NDArray(sin.to(d).numpy()),
            "attn_bias": rt.NDArray(attn_bias.to(d).numpy()),
        }))
        pp = torch.tensor(out["proj_point"].numpy().astype(np.float32))
        pq = torch.tensor(out["proj_q"].numpy().astype(np.float32))
        return pp, pq


def rope_cos_sin(position_ids, head_dim, theta=10000.0):
    """position_ids: (B,N) float -> cos,sin (B,N,head_dim)."""
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    freqs = position_ids.float().unsqueeze(-1) * inv.view(1, 1, -1)  # B,N,hd/2
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _map_key(k):
    if k.startswith("model.input_ff_layer."):
        return k[len("model."):]
    if k.startswith("model.layers."):
        # raw checkpoint uses mlp.ff0/ff1 (transformers remaps to fc1/fc2)
        return (k[len("model."):]
                .replace(".mlp.ff0.", ".mlp_fc1.").replace(".mlp.ff1.", ".mlp_fc2.")
                .replace(".mlp.fc1.", ".mlp_fc1.").replace(".mlp.fc2.", ".mlp_fc2."))
    if k.startswith("output_projection_point.") or k.startswith("output_projection_quantiles."):
        return k
    return None  # rotary buffers etc.


def load_core_from_safetensors(path, cfg):
    """Load TimesFmCore weights directly from a model.safetensors (no transformers dep)."""
    from safetensors.torch import load_file
    sd = load_file(path)
    core = TimesFmCore(cfg)
    new = {}
    for k, v in sd.items():
        nk = _map_key(k)
        if nk is not None:
            new[nk] = v
    missing, unexpected = core.load_state_dict(new, strict=False)
    assert not [m for m in missing if "inv_freq" not in m], f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return core.eval()


def load_core_from_hf(hf_model, cfg):
    """Copy weights from a loaded HF TimesFm2_5ModelForPrediction into TimesFmCore."""
    core = TimesFmCore(cfg)
    sd = hf_model.state_dict()
    new = {}
    for k, v in sd.items():
        nk = k
        if k.startswith("model.input_ff_layer."):
            nk = k[len("model."):]
        elif k.startswith("model.layers."):
            # model.layers.i.mlp.fc1 -> layers.i.mlp_fc1
            nk = k[len("model."):].replace(".mlp.fc1.", ".mlp_fc1.").replace(".mlp.fc2.", ".mlp_fc2.")
        elif k.startswith("output_projection_point.") or k.startswith("output_projection_quantiles."):
            nk = k
        else:
            continue  # rotary_emb buffers etc.
        new[nk] = v
    missing, unexpected = core.load_state_dict(new, strict=False)
    assert not [m for m in missing if "inv_freq" not in m], f"missing: {missing}"
    assert not unexpected, f"unexpected: {unexpected}"
    return core.eval()
