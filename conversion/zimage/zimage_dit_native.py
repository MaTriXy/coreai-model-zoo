"""Core AI export wrapper over the STOCK diffusers Z-Image blocks.

A hand-rolled GPU-clean DiT (the LiteRT/Android port's design: manual multi-head
view/transpose/matmul attention) does NOT convert on Core AI — it fails the
versioned-IR pass ("cannot unwrap empty odiec_module_t"). Core AI wants attention
expressed via its SDPA composite op, which the native diffusers attention
(dispatch_attention_fn) produces. So this wrapper reuses the real
noise_refiner / context_refiner / layers / final modules verbatim; the ONLY change
is swapping the attention processor's export-hostile `view_as_complex` RoPE for a
bit-exact real interleaved form fed precomputed cos/sin.

Graph contract (host does patchify / RoPE / pad-mask / unpatchify — zimage_host.py):
    img_tokens [1,n_img,64], cap_feats [1,n_cap,2560], adaln [1,256],
    x_cos/x_sin [1,n_img,64], cap_cos/cap_sin [1,n_cap,64] (hd/2=64),
    x_pad_mask [1,n_img,1], cap_pad_mask [1,n_cap,1]
Output: velocity patches [1, n_img+n_cap, 64] (host slices img[:, :n_img] + unpatchify)
"""
import torch
import torch.nn as nn

from diffusers.models.transformers.transformer_z_image import (
    ZSingleStreamAttnProcessor, dispatch_attention_fn)


@torch.no_grad()
def rescale_residual(dit, k: float):
    """Divide the residual stream by k — output-exact, complements rescale_fp16_safe.

    Handles the overflow that lives INSIDE the norms rather than at a module output:
    the residual peaks at ~6.7e3, so `mean(x^2)` = 4.4e7 >> 65504. (diffusers' RMSNorm
    upcasts the variance to fp32, but that upcast does not survive the fp16 graph.)

    RMSNorm is scale-invariant, so norm1(x/k) == norm1(x) and every attention/FFN
    input is unchanged. Scaling the entry points by 1/k and the output-side norm
    *weights* by 1/k carries x/k through every block; norms that READ the rescaled
    residual (norm1, norm_final) need eps/k^2 to stay exact.

    Neither transform alone is enough: rescaling only the residual leaves w2/to_out
    at 3.1e5, and rescaling only w2/to_out leaves mean(x^2) at 4.4e7. Apply both.
    """
    def _scale_last_linear(mod):
        # x_embed is a Linear; cap_embed is Sequential(RMSNorm, Linear) — the leading
        # RMSNorm is scale-invariant, so only its Linear needs scaling.
        lins = [m for m in mod.modules() if isinstance(m, nn.Linear)]
        assert lins, f"no Linear found in {type(mod).__name__}"
        lin = lins[-1]
        lin.weight.div_(k)
        if lin.bias is not None:
            lin.bias.div_(k)

    _scale_last_linear(dit.x_embed)
    _scale_last_linear(dit.cap_embed)
    dit.x_pad_token.div_(k)
    dit.cap_pad_token.div_(k)
    k2 = k * k
    for stack in (dit.noise_refiner, dit.context_refiner, dit.layers):
        for blk in stack:
            blk.attention_norm2.weight.div_(k)   # scales the attention residual update
            blk.ffn_norm2.weight.div_(k)         # scales the FFN residual update
            blk.attention_norm1.eps /= k2        # these norms read the rescaled residual
            blk.ffn_norm1.eps /= k2
    dit.final.norm_final.eps /= k2               # affine-free LayerNorm on the residual
    return dit


@torch.no_grad()
def rescale_fp16_safe(dit, c: float):
    """Shrink the ONLY tensors that overflow fp16 — output-exact weight transform.

    iOS AOT accepts fp16 activations only (bf16 is rejected, fp32 makes the compiler
    materialize the int8 weights). But Z-Image NaNs in fp16 at sampler step 2.

    Measured (fp32, real step-2 inputs, per-module max): exactly 18 of 515 modules
    exceed 65504, and they are all of two kinds —
        feed_forward.w2        max 3.1e5   (FFN output projection)
        attention.to_out[0]    max 1.2e5   (attention output projection)
    i.e. the linears that PRODUCE the residual update. The residual stream itself
    peaks at ~6.7e3 and never overflows (rescaling *it* changes nothing — verified).

    Both are immediately followed by an RMSNorm (`ffn_norm2` / `attention_norm2`),
    and RMSNorm is scale-invariant: norm2(y/c) == norm2(y). So dividing w2 / to_out
    by c shrinks the overflowing intermediate by c while leaving the block output
    mathematically unchanged. Only those norms' eps must shrink by c^2, since
    rsqrt(mean(y^2/c^2) + eps) == c*rsqrt(mean(y^2) + c^2*eps).

    No new ops enter the graph — hand-rolled norms NaN inside Core AI (see
    knowledge/zimage-port.md). Fix the weights, not the graph.
    """
    c2 = c * c
    for stack in (dit.noise_refiner, dit.context_refiner, dit.layers):
        for blk in stack:
            for lin in (blk.attention.to_out[0], blk.feed_forward.w2):
                lin.weight.div_(c)
                if lin.bias is not None:
                    lin.bias.div_(c)
            blk.attention_norm2.eps /= c2
            blk.ffn_norm2.eps /= c2
    return dit


class RealRopeAttnProcessor(ZSingleStreamAttnProcessor):
    """ZSingleStreamAttnProcessor with export-clean real RoPE.

    freqs_cis arrives as a REAL packed tensor [B, N, hd] = concat(cos, sin) along
    the last dim (hd/2 each). The interleaved rope
        out[2i]   = x[2i]*cos_i - x[2i+1]*sin_i
        out[2i+1] = x[2i]*sin_i + x[2i+1]*cos_i
    is bit-exact to view_as_complex((x0+i x1)*(cos+i sin)) but uses only
    reshape/stack/mul (no complex ops).
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, freqs_cis=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        def apply_rotary_emb(x_in, cos, sin):
            # x_in [B,N,heads,hd]; cos/sin [B,N,1,hd/2]
            hd = x_in.shape[-1]
            xp = x_in.reshape(*x_in.shape[:-1], hd // 2, 2)
            x0 = xp[..., 0]
            x1 = xp[..., 1]
            o0 = x0 * cos - x1 * sin
            o1 = x0 * sin + x1 * cos
            return torch.stack([o0, o1], dim=-1).reshape(x_in.shape)

        if freqs_cis is not None:
            half = freqs_cis.shape[-1] // 2
            cos = freqs_cis[..., :half].unsqueeze(2)   # [B,N,1,hd/2]
            sin = freqs_cis[..., half:].unsqueeze(2)
            query = apply_rotary_emb(query, cos, sin)
            key = apply_rotary_emb(key, cos, sin)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False, backend=self._attention_backend,
            parallel_config=self._parallel_config)

        hidden_states = hidden_states.flatten(2, 3).to(dtype)
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output


class NativeZDiT(nn.Module):
    def __init__(self, rm, n_layers=None, residual_scale=1.0, update_scale=1.0):
        super().__init__()
        self.x_embed = rm.all_x_embedder["2-1"]
        self.cap_embed = rm.cap_embedder
        self.noise_refiner = rm.noise_refiner
        self.context_refiner = rm.context_refiner
        self.layers = rm.layers if n_layers is None else rm.layers[:n_layers]
        self.final = rm.all_final_layer["2-1"]
        self.register_buffer("x_pad_token", rm.x_pad_token.detach().clone())
        self.register_buffer("cap_pad_token", rm.cap_pad_token.detach().clone())
        # swap every block's attention processor to the export-clean real-rope one
        for stack in (self.noise_refiner, self.context_refiner, self.layers):
            for blk in stack:
                old = blk.attention.processor
                proc = RealRopeAttnProcessor()
                proc._attention_backend = getattr(old, "_attention_backend", None)
                proc._parallel_config = getattr(old, "_parallel_config", None)
                blk.attention.processor = proc
        # both transforms are needed for fp16 and both are output-exact:
        # update_scale fixes w2/to_out (3.1e5), residual_scale fixes mean(x^2) (4.4e7)
        if update_scale != 1.0:
            rescale_fp16_safe(self, update_scale)
        if residual_scale != 1.0:
            rescale_residual(self, residual_scale)

    def forward(self, img_tokens, cap_feats, adaln, x_cos, x_sin, cap_cos, cap_sin,
                x_pad_mask, cap_pad_mask):
        x = self.x_embed(img_tokens)
        x = x * (1.0 - x_pad_mask) + self.x_pad_token * x_pad_mask
        x_freqs = torch.cat([x_cos, x_sin], dim=-1)          # [1,n_img,hd]
        for blk in self.noise_refiner:
            x = blk(x, None, x_freqs, adaln, None)
        cap = self.cap_embed(cap_feats)
        cap = cap * (1.0 - cap_pad_mask) + self.cap_pad_token * cap_pad_mask
        cap_freqs = torch.cat([cap_cos, cap_sin], dim=-1)    # [1,n_cap,hd]
        for blk in self.context_refiner:
            cap = blk(cap, None, cap_freqs, None, None)
        unified = torch.cat([x, cap], dim=1)
        uni_freqs = torch.cat([x_freqs, cap_freqs], dim=1)
        for blk in self.layers:
            unified = blk(unified, None, uni_freqs, adaln, None)
        return self.final(unified, adaln)
