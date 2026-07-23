"""Standalone non-streaming re-authoring of the VibeVoice acoustic-tokenizer DECODER (causal ConvNeXt
VAE vocoder). Transformers-free so it loads in the coreai export venv. Numerically identical to the
upstream TokenizerDecoder non-streaming path (norm='none', pad_mode='constant', causal, RMSNorm,
depthwise_conv mixer, disable_last_norm). Whole-sequence decode: latent[1,64,T] -> audio[1,1,3200*T].

decoder config (0.5B, resolved from config.json):
  dimension=64, channels=1, n_filters=32, ratios=[8,5,5,4,2,2], depths=[8,3,3,3,3,3,3] (rev encoder),
  kernel_size=7, last_kernel_size=7, ffn_expansion=4, layer_scale_init_value=1e-6, layernorm_eps=1e-5.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total=0):
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x, paddings, mode="constant", value=0.0):
    return F.pad(x, paddings, mode, value)


def unpad1d(x, paddings):
    pl, pr = paddings
    end = x.shape[-1] - pr
    return x[..., pl:end]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            out = out * self.weight
        return out


class ConvRMSNorm(RMSNorm):
    """RMSNorm over channels for [B,C,T] tensors (transpose, norm, transpose back)."""
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            out = out * self.weight
        return out.transpose(1, 2)


class SConv1d(nn.Module):
    """Causal Conv1d with left constant padding (non-streaming)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, dilation=1, groups=1, bias=True, pad_mode="constant"):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, stride, dilation=dilation, groups=groups, bias=bias)
        self.kernel_size, self.stride, self.dilation = kernel_size, stride, dilation
        self.pad_mode = pad_mode
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)

    def forward(self, x):
        # All decoder SConv1d are stride=1 -> get_extra_padding is provably 0; skipping it drops the
        # math.ceil() guard that would otherwise block a dynamic (variable-T) torch.export.
        extra = 0 if self.stride == 1 else get_extra_padding_for_conv1d(x, self.kernel_size, self.stride, self.padding_total)
        x = pad1d(x, (self.padding_total, extra), mode=self.pad_mode, value=0.0)
        return self.conv(x)


class SConvTranspose1d(nn.Module):
    """Causal ConvTranspose1d with right trim (non-streaming, trim_right_ratio=1.0)."""
    def __init__(self, in_ch, out_ch, kernel_size, stride, bias=True, trim_right_ratio=1.0):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(in_ch, out_ch, kernel_size, stride, bias=bias)
        self.padding_total = kernel_size - stride
        self.trim_right_ratio = trim_right_ratio

    def forward(self, x):
        y = self.convtr(x)
        pr = math.ceil(self.padding_total * self.trim_right_ratio)
        pl = self.padding_total - pr
        if pl + pr > 0:
            y = unpad1d(y, (pl, pr))
        return y


class FFN(nn.Module):
    def __init__(self, embed_dim, ffn_dim, bias=False):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.gelu = nn.GELU()  # erf gelu == transformers ACT2FN['gelu']
        self.linear2 = nn.Linear(ffn_dim, embed_dim, bias=bias)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class Block1D(nn.Module):
    """ConvNeXt-style: depthwise-conv mixer + FFN, each with RMSNorm + layer-scale gamma residual."""
    def __init__(self, dim, kernel_size=7, ffn_expansion=4, eps=1e-5, bias=True):
        super().__init__()
        self.norm = ConvRMSNorm(dim, eps=eps)
        self.ffn_norm = ConvRMSNorm(dim, eps=eps)
        self.mixer = nn.Module()
        self.mixer.conv = SConv1d(dim, dim, kernel_size, groups=dim, bias=bias, pad_mode="constant")
        self.ffn = FFN(dim, ffn_expansion * dim, bias=bias)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.ffn_gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mixer.conv(x)
        x = x * self.gamma.unsqueeze(-1)
        x = residual + x
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        x = x * self.ffn_gamma.unsqueeze(-1)
        return residual + x


class DecoderOverlay(nn.Module):
    def __init__(self, dimension=64, channels=1, n_filters=32, ratios=(8, 5, 5, 4, 2, 2),
                 depths=(8, 3, 3, 3, 3, 3, 3), kernel_size=7, last_kernel_size=7, eps=1e-5):
        super().__init__()
        self.depths = list(depths)
        self.ratios = list(ratios)
        nd = len(self.depths)
        stem = nn.Sequential(SConv1d(dimension, n_filters * 2 ** (nd - 1), kernel_size, pad_mode="constant"))
        self.upsample_layers = nn.ModuleList([stem])
        for i in range(len(self.ratios)):
            in_ch = n_filters * (2 ** (nd - 1 - i))
            out_ch = n_filters * (2 ** (nd - 1 - i - 1))
            self.upsample_layers.append(nn.Sequential(
                SConvTranspose1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2, stride=self.ratios[i])))
        self.stages = nn.ModuleList()
        for i in range(nd):
            in_ch = n_filters * (2 ** (nd - 1 - i))
            self.stages.append(nn.Sequential(*[Block1D(in_ch, kernel_size=kernel_size, eps=eps)
                                               for _ in range(self.depths[i])]))
        self.norm = nn.Identity()  # disable_last_norm=True
        self.head = SConv1d(in_ch, channels, last_kernel_size, pad_mode="constant")

    def forward(self, latents):  # latents [B,64,T]
        x = latents
        for i in range(len(self.depths)):
            for layer in self.upsample_layers[i]:
                x = layer(x)
            for block in self.stages[i]:
                x = block(x)
        return self.head(self.norm(x))

    def load_upstream(self, sd, prefix="model.acoustic_tokenizer.decoder."):
        want = {}
        for k, v in sd.items():
            if not k.startswith(prefix):
                continue
            kk = k[len(prefix):]
            # upstream wraps conv as SConv1d.conv=NormConv1d(conv=Conv1d) -> "...conv.conv.weight",
            # and the block mixer adds one more level (Convlayer.conv=SConv1d) -> "mixer.conv.conv.conv.".
            # Our SConv1d.conv IS the Conv1d, so a single left-to-right non-overlapping ".conv.conv."->
            # ".conv." collapse is exactly right for every case: stem/head/upsample (2->1 level) and the
            # depthwise mixer (3->2 levels, "mixer.conv.conv.conv."->"mixer.conv.conv.").
            kk = kk.replace(".conv.conv.", ".conv.")
            kk = kk.replace(".convtr.convtr.", ".convtr.")
            want[kk] = v
        missing, unexpected = self.load_state_dict(want, strict=False)
        assert not missing, f"decoder missing: {missing[:8]} ... ({len(missing)})"
        assert not unexpected, f"decoder unexpected: {unexpected[:8]} ... ({len(unexpected)})"
        return self
