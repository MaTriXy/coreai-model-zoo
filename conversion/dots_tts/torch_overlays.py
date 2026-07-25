"""dots.tts Core-AI-shaped torch overlays — clean reimplementations of each exportable
submodule in the form that maps 1:1 onto the Core AI export graph (baked RoPE, explicit
GQA, static-KV-capable), gated cos>=0.999 against the torch oracle (oracle.py fixtures).

Stage-by-stage the ladder adds a class here; each gate_*.py loads the matching oracle
fixture, runs the overlay, and asserts cosine >= 0.999.

Weights load straight from the upstream `model.safetensors` (keys under `llm.`,
`velocity_field_predictor.`, `patch_encoder.`, the small projections). Everything is
plain torch — no HF modeling code — so the same module bodies transcribe to the coreai
converter.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- utils
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """HF-exact RMSNorm: normalize in fp32, scale by weight, cast back."""
    dt = x.dtype
    x = x.to(torch.float32)
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (weight * x.to(dt))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: [B, H, S, D]; cos,sin: [S, D] -> broadcast over B,H
    cos = cos[None, None]
    sin = sin[None, None]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# --------------------------------------------------------------------- Qwen2.5 trunk
class Qwen2Backbone(nn.Module):
    """Stock Qwen2.5-1.5B trunk for the dots.tts audio loop.

    Runs `inputs_embeds` [B,S,H] -> (`hidden` [B,S,H] = final-RMSNorm output = HF
    `hidden_states[-1]`, `logits` [B,S,V] = hidden @ tied-embed.T). The dots.tts decode
    only consumes `hidden` (+ the separate eos head); `logits` is computed for parity
    with the oracle but is dead weight in the real loop.

    Baked RoPE (cos/sin from positions), explicit GQA (repeat kv), causal mask — the
    static-KV decode form (q=1, cached k/v) is the same math with S=1 and prior k/v.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.H = cfg["hidden_size"]
        self.n_layers = cfg["num_hidden_layers"]
        self.n_heads = cfg["num_attention_heads"]
        self.n_kv = cfg["num_key_value_heads"]
        self.head_dim = self.H // self.n_heads
        self.eps = cfg["rms_norm_eps"]
        self.theta = cfg["rope_theta"]
        self.n_rep = self.n_heads // self.n_kv
        # params registered as buffers/parameters, filled by load_from_state_dict
        self.embed_tokens = nn.Parameter(torch.zeros(cfg["vocab_size"], self.H), requires_grad=False)
        self.norm_w = nn.Parameter(torch.zeros(self.H), requires_grad=False)
        self.layers = nn.ModuleList([_Qwen2Layer(cfg) for _ in range(self.n_layers)])

    def _rope(self, S, device, dtype):
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))
        pos = torch.arange(S, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)  # [S, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    @torch.no_grad()
    def forward(self, inputs_embeds: torch.Tensor):
        B, S, _ = inputs_embeds.shape
        dtype = inputs_embeds.dtype
        cos, sin = self._rope(S, inputs_embeds.device, dtype)
        # causal additive mask [S,S]
        mask = torch.full((S, S), float("-inf"), device=inputs_embeds.device, dtype=torch.float32).triu(1)
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, cos, sin, mask, self.n_rep, self.n_heads, self.n_kv, self.head_dim, self.eps)
        h = rms_norm(h, self.norm_w, self.eps)
        logits = F.linear(h, self.embed_tokens)  # tied lm_head
        return h, logits

    @torch.no_grad()
    def load_upstream(self, sd: dict):
        """Load from a `llm.*`-prefixed slice of the upstream state dict."""
        g = lambda k: sd[f"llm.model.{k}"]
        self.embed_tokens.copy_(g("embed_tokens.weight"))
        self.norm_w.copy_(g("norm.weight"))
        for i, layer in enumerate(self.layers):
            layer.load_upstream(sd, i)


class _Qwen2Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, nh, nkv = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"]
        hd = H // nh
        self.q = nn.Linear(H, nh * hd, bias=True)
        self.k = nn.Linear(H, nkv * hd, bias=True)
        self.v = nn.Linear(H, nkv * hd, bias=True)
        self.o = nn.Linear(nh * hd, H, bias=False)
        inter = cfg["intermediate_size"]
        self.gate = nn.Linear(H, inter, bias=False)
        self.up = nn.Linear(H, inter, bias=False)
        self.down = nn.Linear(inter, H, bias=False)
        self.in_ln = nn.Parameter(torch.zeros(H), requires_grad=False)
        self.post_ln = nn.Parameter(torch.zeros(H), requires_grad=False)

    def forward(self, h, cos, sin, mask, n_rep, nh, nkv, hd, eps):
        B, S, H = h.shape
        r = h
        x = rms_norm(h, self.in_ln, eps)
        q = self.q(x).view(B, S, nh, hd).transpose(1, 2)   # [B,nh,S,hd]
        k = self.k(x).view(B, S, nkv, hd).transpose(1, 2)  # [B,nkv,S,hd]
        v = self.v(x).view(B, S, nkv, hd).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        # GQA repeat
        k = k.repeat_interleave(n_rep, dim=1)
        v = v.repeat_interleave(n_rep, dim=1)
        attn = torch.matmul(q.float(), k.float().transpose(-1, -2)) / (hd ** 0.5)
        attn = attn + mask
        attn = attn.softmax(-1).to(v.dtype)
        o = torch.matmul(attn, v)  # [B,nh,S,hd]
        o = o.transpose(1, 2).reshape(B, S, nh * hd)
        h = r + self.o(o)
        r = h
        x = rms_norm(h, self.post_ln, eps)
        h = r + self.down(F.silu(self.gate(x)) * self.up(x))
        return h

    @torch.no_grad()
    def load_upstream(self, sd, i):
        p = f"llm.model.layers.{i}."
        self.q.weight.copy_(sd[p + "self_attn.q_proj.weight"]); self.q.bias.copy_(sd[p + "self_attn.q_proj.bias"])
        self.k.weight.copy_(sd[p + "self_attn.k_proj.weight"]); self.k.bias.copy_(sd[p + "self_attn.k_proj.bias"])
        self.v.weight.copy_(sd[p + "self_attn.v_proj.weight"]); self.v.bias.copy_(sd[p + "self_attn.v_proj.bias"])
        self.o.weight.copy_(sd[p + "self_attn.o_proj.weight"])
        self.gate.weight.copy_(sd[p + "mlp.gate_proj.weight"])
        self.up.weight.copy_(sd[p + "mlp.up_proj.weight"])
        self.down.weight.copy_(sd[p + "mlp.down_proj.weight"])
        self.in_ln.copy_(sd[p + "input_layernorm.weight"])
        self.post_ln.copy_(sd[p + "post_attention_layernorm.weight"])


# ------------------------------------------------------------------ DiT flow head
def _timestep_embedding(t: torch.Tensor, dim: int = 256, max_period: float = 10000.0):
    """Sinusoidal timestep embedding — matches TimestepEmbedder.timestep_embedding."""
    half = dim // 2
    freqs = torch.exp(-torch.log(torch.tensor(max_period)) *
                      torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t[:, None].float() * freqs[None]
    # return in the input dtype so the fp16 export doesn't mix fp32 into the fp16 time-MLP
    # (no-op for the fp32 oracle gates)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(t.dtype)


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _dit_rope(pos_ids: torch.Tensor, dim: int, theta: float):
    """Return cos/sin [..,S,dim] for the DiT rotary (θ=1e4). pos_ids [B,S] or [S]."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=pos_ids.device) / dim))
    pos = pos_ids.to(torch.float32)
    if pos.dim() == 1:
        freqs = torch.outer(pos, inv_freq)          # [S, dim/2]
    else:
        freqs = torch.einsum("bi,j->bij", pos, inv_freq)  # [B,S,dim/2]
    emb = torch.cat((freqs, freqs), dim=-1)          # [...,S,dim]
    return emb.cos(), emb.sin()


def _apply_dit_rope(t, cos, sin):
    # t: [B,H,S,d]; cos/sin: [B,S,d] (or [S,d]) -> add head axis
    if cos.dim() == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)     # [B,1,S,d]
    else:
        cos, sin = cos[None, None], sin[None, None]        # [1,1,S,d]
    # cast back to t's dtype so the fp16 export keeps attention in fp16
    # (fp32 cos/sin would otherwise promote q,k to fp32; no-op for the fp32 oracle gates)
    return (t * cos + rotate_half(t) * sin).to(t.dtype)


class DiTOverlay(nn.Module):
    """dots.tts flow head (velocity_field_predictor) — 18-block AdaLN DiT.

    x [B,S,1024] (LLM-hidden-space seq w/ noise latent projected in) + timesteps + g_cond
    (+ duration for meanflow) -> velocity [B,S,128]. qk_norm RMSNorm, RoPE θ1e4, GELU-tanh
    FFN, parameter-free LayerNorm modulated by adaLN. One unrolled denoise step; the solver
    (soar CFG N-step / meanflow NFE-step) wraps this at the host level.
    """

    def __init__(self, cfg: dict, mode: str = "flow_matching"):
        super().__init__()
        self.H = cfg["hidden_size"]           # 1024
        self.n_heads = cfg["num_heads"]       # 16
        self.head_dim = self.H // self.n_heads
        self.n_layers = cfg["num_layers"]     # 18
        self.ffn = cfg["ffn_hidden_size"]     # 4096
        self.theta = cfg["rotary_theta"]      # 1e4
        self.mode = mode
        self.in_dim = cfg.get("in_dim", 1024)
        self.out_dim = cfg.get("out_dim", 128)

        self.input_layer = nn.Linear(self.in_dim, self.H)
        # time embedder MLP: Linear(256,H) -> SiLU -> Linear(H,H)
        self.t_fc1 = nn.Linear(256, self.H); self.t_fc2 = nn.Linear(self.H, self.H)
        if mode == "meanflow":
            self.d_fc1 = nn.Linear(256, self.H); self.d_fc2 = nn.Linear(self.H, self.H)
        self.blocks = nn.ModuleList([_DiTBlock(self.H, self.n_heads, self.ffn) for _ in range(self.n_layers)])
        # output layer (FinalLayer)
        self.out_adaLN = nn.Linear(self.H, 2 * self.H)
        self.out_linear = nn.Linear(self.H, self.out_dim)
        self.ln_eps = 1e-5

    def _time_mlp(self, t):
        return self.t_fc2(F.silu(self.t_fc1(_timestep_embedding(t))))

    def _dur_mlp(self, d):
        return self.d_fc2(F.silu(self.d_fc1(_timestep_embedding(d))))

    @torch.no_grad()
    def forward(self, x, timesteps, attn_mask=None, pos_ids=None, g_cond=None, duration=None):
        c = self._time_mlp(timesteps)
        if self.mode == "meanflow" and duration is not None:
            c = c + self._dur_mlp(duration)
        if g_cond is not None:
            c = c + g_cond
        B, S, _ = x.shape
        cos, sin = _dit_rope(pos_ids if pos_ids is not None else torch.arange(S, device=x.device),
                             self.head_dim, self.theta)
        # additive attention bias from boolean mask [.,S,S] (1=attend)
        bias = None
        if attn_mask is not None:
            m = attn_mask
            if m.dim() == 3:
                m = m.unsqueeze(1)  # [.,1,S,S]
            bias = torch.zeros_like(m, dtype=x.dtype)
            bias = bias.masked_fill(m.to(torch.bool).logical_not(), float("-inf"))
        h = self.input_layer(x)
        for blk in self.blocks:
            h = blk(h, c, cos, sin, bias, self.n_heads, self.head_dim, self.ln_eps)
        # final layer
        shift, scale = self.out_adaLN(F.silu(c)).chunk(2, dim=1)
        h = _modulate(F.layer_norm(h, (self.H,), eps=self.ln_eps), shift, scale)
        return self.out_linear(h)

    @torch.no_grad()
    def load_upstream(self, sd: dict):
        P = "velocity_field_predictor."
        self.input_layer.weight.copy_(sd[P + "input_layer.weight"]); self.input_layer.bias.copy_(sd[P + "input_layer.bias"])
        self.t_fc1.weight.copy_(sd[P + "time_embedder.mlp.0.weight"]); self.t_fc1.bias.copy_(sd[P + "time_embedder.mlp.0.bias"])
        self.t_fc2.weight.copy_(sd[P + "time_embedder.mlp.2.weight"]); self.t_fc2.bias.copy_(sd[P + "time_embedder.mlp.2.bias"])
        if self.mode == "meanflow":
            self.d_fc1.weight.copy_(sd[P + "duration_embedder.mlp.0.weight"]); self.d_fc1.bias.copy_(sd[P + "duration_embedder.mlp.0.bias"])
            self.d_fc2.weight.copy_(sd[P + "duration_embedder.mlp.2.weight"]); self.d_fc2.bias.copy_(sd[P + "duration_embedder.mlp.2.bias"])
        for i, blk in enumerate(self.blocks):
            blk.load_upstream(sd, P + f"blocks.{i}.")
        self.out_adaLN.weight.copy_(sd[P + "output_layer.adaLN_modulation.1.weight"])
        self.out_adaLN.bias.copy_(sd[P + "output_layer.adaLN_modulation.1.bias"])
        self.out_linear.weight.copy_(sd[P + "output_layer.linear.weight"]); self.out_linear.bias.copy_(sd[P + "output_layer.linear.bias"])


class _DiTBlock(nn.Module):
    def __init__(self, H, n_heads, ffn):
        super().__init__()
        hd = H // n_heads
        self.q = nn.Linear(H, H, bias=False); self.k = nn.Linear(H, H, bias=False); self.v = nn.Linear(H, H, bias=False)
        self.o = nn.Linear(H, H, bias=True)
        self.q_norm = nn.RMSNorm(hd); self.k_norm = nn.RMSNorm(hd)
        self.fc1 = nn.Linear(H, ffn, bias=True); self.fc2 = nn.Linear(ffn, H, bias=True)
        self.adaLN = nn.Linear(H, 6 * H, bias=True)

    def forward(self, x, c, cos, sin, bias, n_heads, hd, eps):
        B, S, H = x.shape
        sa, ca, ga, sf, cf, gf = self.adaLN(F.silu(c)).chunk(6, dim=1)
        # attention
        xn = _modulate(F.layer_norm(x, (H,), eps=eps), sa, ca)
        q = self.q(xn).view(B, S, n_heads, hd).transpose(1, 2)
        k = self.k(xn).view(B, S, n_heads, hd).transpose(1, 2)
        v = self.v(xn).view(B, S, n_heads, hd).transpose(1, 2)
        q = self.q_norm(q); k = self.k_norm(k)
        q = _apply_dit_rope(q, cos, sin); k = _apply_dit_rope(k, cos, sin)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (hd ** 0.5)
        if bias is not None:
            attn = attn + bias
        attn = attn.softmax(-1)
        o = torch.matmul(attn, v).transpose(1, 2).reshape(B, S, H)
        x = x + ga.unsqueeze(1) * self.o(o)
        # ffn
        xn = _modulate(F.layer_norm(x, (H,), eps=eps), sf, cf)
        ff = self.fc2(F.gelu(self.fc1(xn), approximate="tanh"))
        x = x + gf.unsqueeze(1) * ff
        return x

    @torch.no_grad()
    def load_upstream(self, sd, p):
        self.q.weight.copy_(sd[p + "attn.q_proj.weight"]); self.k.weight.copy_(sd[p + "attn.k_proj.weight"]); self.v.weight.copy_(sd[p + "attn.v_proj.weight"])
        self.o.weight.copy_(sd[p + "attn.o_proj.weight"]); self.o.bias.copy_(sd[p + "attn.o_proj.bias"])
        self.q_norm.weight.copy_(sd[p + "attn.q_norm.weight"]); self.k_norm.weight.copy_(sd[p + "attn.k_norm.weight"])
        self.fc1.weight.copy_(sd[p + "ffn.fc1.weight"]); self.fc1.bias.copy_(sd[p + "ffn.fc1.bias"])
        self.fc2.weight.copy_(sd[p + "ffn.fc2.weight"]); self.fc2.bias.copy_(sd[p + "ffn.fc2.bias"])
        self.adaLN.weight.copy_(sd[p + "adaLN_modulation.1.weight"]); self.adaLN.bias.copy_(sd[p + "adaLN_modulation.1.bias"])
