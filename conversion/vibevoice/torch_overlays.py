"""Standalone, transformers-free re-authoring of the VibeVoice-Realtime-0.5B exportable submodules.

Each overlay mirrors the upstream module numerically but depends only on torch (so it loads in the
coreai export venv, which has its own transformers). `load_upstream(state_dict)` pulls the matching
weights straight out of the model.safetensors by key prefix. Gated cos>=0.999 against oracle_ref.npz.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Diffusion head (prediction_head): adaLN-modulated FFN stack, v-prediction target.
#   (noisy[B,64], timesteps[B], condition[B,896]) -> eps[B,64]
# ============================================================================
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            out = out * self.weight
        return out


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def forward(self, t):
        # Recompute freqs in fp32 each call (128 elems, negligible): the sinusoid MUST be at full
        # precision — upstream casts to the working dtype only after cos/sin. If freqs lived as an
        # fp16 buffer (module.to(fp16)), low-magnitude frequencies collapse and the embedding drifts
        # (torch overlay reads cos~0.79 vs the fp32 oracle, even though the exported graph is fine).
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(self.mlp[0].weight.dtype)
        return self.mlp(emb)


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ffn_dim):
        super().__init__()
        self.gate_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(embed_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, embed_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HeadLayer(nn.Module):
    def __init__(self, embed_dim, ffn_dim, cond_dim, norm_eps=1e-5):
        super().__init__()
        self.ffn = FeedForwardNetwork(embed_dim, ffn_dim)
        self.norm = RMSNorm(embed_dim, eps=norm_eps)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * embed_dim, bias=False))

    def forward(self, x, c):
        shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)
        return x + gate * self.ffn(modulate(self.norm(x), shift, scale))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size, cond_size, norm_eps=1e-5):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size, eps=norm_eps, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, output_size, bias=False)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(cond_size, 2 * hidden_size, bias=False))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiffusionHeadOverlay(nn.Module):
    """Mirrors VibeVoiceDiffusionHead (hidden 896, latent 64, 4 layers, ffn_ratio 3.0)."""
    def __init__(self, hidden_size=896, latent_size=64, head_layers=4, head_ffn_ratio=3.0, rms_norm_eps=1e-5):
        super().__init__()
        self.noisy_images_proj = nn.Linear(latent_size, hidden_size, bias=False)
        self.cond_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.t_embedder = TimestepEmbedder(hidden_size)
        ffn_dim = int(hidden_size * head_ffn_ratio)
        self.layers = nn.ModuleList([
            HeadLayer(hidden_size, ffn_dim, hidden_size, norm_eps=rms_norm_eps) for _ in range(head_layers)
        ])
        self.final_layer = FinalLayer(hidden_size, latent_size, hidden_size, norm_eps=rms_norm_eps)

    def forward(self, noisy_images, timesteps, condition):
        x = self.noisy_images_proj(noisy_images)
        t = self.t_embedder(timesteps)
        c = self.cond_proj(condition) + t
        for layer in self.layers:
            x = layer(x, c)
        return self.final_layer(x, c)

    def load_upstream(self, sd, prefix="model.prediction_head."):
        want = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = self.load_state_dict(want, strict=False)
        # norm_final has no weight (elementwise_affine=False); freqs is a non-persistent buffer
        missing = [m for m in missing if not m.endswith("t_embedder.freqs")]
        assert not missing, f"diffusion head missing weights: {missing}"
        assert not unexpected, f"diffusion head unexpected weights: {unexpected}"
        return self


# ============================================================================
# Acoustic connector (SpeechConnector): 64 -> 896 feedback projection.
#   latent[B,1,64] -> embed[B,1,896]
# ============================================================================
class ConnectorOverlay(nn.Module):
    def __init__(self, input_dim=64, output_dim=896, eps=1e-6):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = RMSNorm(output_dim, eps=eps)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features):
        return self.fc2(self.norm(self.fc1(features)))

    def load_upstream(self, sd, prefix="model.acoustic_connector."):
        want = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
        missing, unexpected = self.load_state_dict(want, strict=False)
        assert not missing and not unexpected, f"connector m={missing} u={unexpected}"
        return self
