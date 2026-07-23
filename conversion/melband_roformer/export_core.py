"""Export-boundary core for Mel-Band RoFormer.

Reference forward = host_stft -> [ gather(freq_indices) -> band_split ->
axial rotary transformer x depth -> mask_estimator -> band-average -> complex
mask multiply ] -> host_istft.  The [...] part is the exportable Core AI graph;
STFT/iSTFT stay host-side (Stable Audio idiom). Everything in the core is REAL
arithmetic (real/imag as last dim = 2): the band-average scatter becomes a
constant matmul A, the complex multiply becomes real ops. So the graph has no
complex tensors / scatter_add -> clean to export.

I/O:  core(stft_real [b, F2=2*(n_fft//2+1), T, 2]) -> masked_real [b, F2, T, 2]
"""
import torch
import torch.nn as nn
from einops import rearrange


class SepCore(nn.Module):
    def __init__(self, model):
        super().__init__()
        # reuse the reference submodules verbatim (weights come for free)
        self.band_split = model.band_split
        self.layers = model.layers
        self.mask_estimator = model.mask_estimators[0]
        self.audio_channels = model.audio_channels

        fi = model.freq_indices.long()                 # [F_sel] indices into F2=(f s)
        self.register_buffer("freq_indices", fi, persistent=False)

        n_freqs = model.num_bands_per_freq.numel()      # 1025
        F2 = n_freqs * model.audio_channels             # 2050
        # band-average matrix A[out, j] = 1{freq_indices[j]==out} / denom[out]
        denom = model.num_bands_per_freq.repeat_interleave(model.audio_channels).clamp(min=1e-8)  # [(f s)]
        A = torch.zeros(F2, fi.numel())
        A[fi, torch.arange(fi.numel())] = 1.0
        A = A / denom[:, None]
        self.register_buffer("A", A, persistent=False)  # [F2, F_sel]

    def forward(self, stft_real):
        # stft_real: [b, F2, T, 2]
        b = stft_real.shape[0]
        # gather selected freqs (overlapping mel bands)
        x = stft_real.index_select(1, self.freq_indices)          # [b, F_sel, T, 2]
        x = rearrange(x, "b f t c -> b t (f c)")                  # [b, T, F_sel*2]
        x = self.band_split(x)                                     # [b, T, nbands, dim]

        for time_tf, freq_tf in self.layers:
            x = rearrange(x, "b t f d -> (b f) t d")
            x = time_tf(x)
            x = rearrange(x, "(b f) t d -> (b t) f d", b=b)
            x = freq_tf(x)
            x = rearrange(x, "(b t) f d -> b t f d", b=b)

        m = self.mask_estimator(x)                                 # [b, T, F_sel*2]
        m = rearrange(m, "b t (f c) -> b f t c", c=2)             # [b, F_sel, T, 2]

        # band-average (constant matmul over freq) -> [b, F2, T, 2]
        m = torch.einsum("of,bftc->botc", self.A, m)

        # complex multiply  stft_real * m  (real arithmetic)
        sr, si = stft_real[..., 0], stft_real[..., 1]
        mr, mi = m[..., 0], m[..., 1]
        out_r = sr * mr - si * mi
        out_i = sr * mi + si * mr
        return torch.stack([out_r, out_i], dim=-1)                # [b, F2, T, 2]


class HostDSP:
    """host-side STFT / iSTFT matching the reference torch.stft kwargs."""
    def __init__(self, model):
        self.k = dict(model.stft_kwargs)                 # n_fft, hop_length, win_length, normalized
        self.win = model.stft_window_fn()                # hann(win_length)
        self.ch = model.audio_channels

    def stft(self, raw_audio):
        # raw_audio [b, s, T_samp] -> stft_real [b, F2, T, 2]
        b, s, _ = raw_audio.shape
        x = rearrange(raw_audio, "b s t -> (b s) t")
        z = torch.stft(x, **self.k, window=self.win.to(x.device), return_complex=True)
        z = torch.view_as_real(z)                        # [(b s), f, T, 2]
        z = rearrange(z, "(b s) f t c -> b (f s) t c", s=s)
        return z

    def istft(self, masked_real, length=None):
        # masked_real [b, F2, T, 2] -> audio [b, s, T_samp]
        b = masked_real.shape[0]
        z = rearrange(masked_real, "b (f s) t c -> (b s) f t c", s=self.ch)
        z = torch.view_as_complex(z.contiguous())
        a = torch.istft(z, **self.k, window=self.win.to(z.device), return_complex=False, length=length)
        return rearrange(a, "(b s) t -> b s t", s=self.ch)
