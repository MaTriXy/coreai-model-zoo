"""Host-side prep for the Z-Image DiT graph — faithful reproduction of the
diffusers ZImageTransformer2DModel internal prep (patchify / position-ids / RoPE /
unpatchify), reused by both the parity check and the Core AI host engine.

Uses the real model's own weight-free helpers (patchify_and_embed, rope_embedder,
unpatchify) so it is bit-exact to the pipeline. Returns the DiT graph inputs
at n_cap = the valid caption length (no padding => no attention mask needed:
valid-only attention == the pipeline's pad-masked attention for the image outputs).
"""
import torch


def build_dit_inputs(rm, latent, cap, patch=2, f_patch=1):
    """latent [C,1,H,W], cap [L,2560] -> DiT graph inputs (cos/sin [1,1,n,hd/2]).

    Mirrors the model's own prep: caption is padded UP to a multiple of
    SEQ_MULTI_OF (=32) via patchify_and_embed; padded positions carry the learned
    pad token (substituted in-graph via cap_pad_mask) and rope of coord (0,0,0).
    The pad tokens are real attention context (the pipeline does not mask them),
    so n_cap = round_up(L, 32) must match the pipeline exactly.

    Returns dict with img_tokens/cap_feats/x_cos/x_sin/cap_cos/cap_sin/
    x_pad_mask/cap_pad_mask, plus x_size and n_img/n_cap for unpatchify.
    """
    device = latent.device
    xt, capf, x_size, x_pos, cap_pos, x_pad, cap_pad = rm.patchify_and_embed(
        [latent], [cap], patch, f_patch)
    n_img = xt[0].shape[0]
    n_cap = capf[0].shape[0]                            # already padded to mult of 32
    img_tokens = xt[0][None]                            # [1, n_img, 64]
    cap_feats = capf[0][None]                           # [1, n_cap, 2560]
    xf = rm.rope_embedder(x_pos[0])[:n_img]            # complex [n_img, hd/2]
    cf = rm.rope_embedder(cap_pos[0])[:n_cap]
    xc, xs = xf.real[None, None], xf.imag[None, None]  # [1,1,n_img,64]
    cc, cs = cf.real[None, None], cf.imag[None, None]  # [1,1,n_cap,64]
    x_pad_mask = x_pad[0].float()[None, :, None]        # [1,n_img,1]
    cap_pad_mask = cap_pad[0].float()[None, :, None]    # [1,n_cap,1]
    return dict(
        img_tokens=img_tokens, cap_feats=cap_feats, x_cos=xc, x_sin=xs,
        cap_cos=cc, cap_sin=cs, x_pad_mask=x_pad_mask, cap_pad_mask=cap_pad_mask,
        x_size=x_size, n_img=n_img, n_cap=n_cap)


def build_native_inputs(rm, latent, cap, patch=2, f_patch=1):
    """Same as build_dit_inputs but cos/sin shaped [1,n,hd/2] for NativeZDiT
    (which packs freqs as concat(cos,sin) and unsqueezes the head axis in-graph)."""
    d = build_dit_inputs(rm, latent, cap, patch, f_patch)
    for k in ("x_cos", "x_sin", "cap_cos", "cap_sin"):
        d[k] = d[k][:, 0]                                   # [1,1,n,hd/2] -> [1,n,hd/2]
    return d


def adaln_from_t(rm, t_norm):
    """Host adaLN: t_embedder(t_norm * t_scale) -> [1,256]. t_norm = (1000-t)/1000."""
    t = torch.tensor([t_norm], dtype=torch.float32)
    return rm.t_embedder(t * rm.t_scale)


def unpatchify_velocity(rm, unified, x_size, n_img, patch=2, f_patch=1):
    """unified [1, n_img+n_cap, 64] -> velocity latent [C,1,H,W] (image slice)."""
    img = unified[0][:n_img]
    return rm.unpatchify([img], x_size, patch, f_patch, None)[0]
