"""Build the mixed-bit extract artifacts from the OFFICIAL google checkpoint.

Replaces the .litertlm reverse-engineering (litertlm_peek) as the source of the Gemma-4 E2B
mobile mixed-bit transplant. Reads google/gemma-4-E2B-it-qat-mobile-transformers (HF, Apache-2.0)
and emits the SAME four artifacts the pipelined export already consumes, so the export + the
3/3 id-exact gate run UNCHANGED:

  gemma4e2b_mixedbit_weights.safetensors   packed codes (our layout) + per-row .scale
  gemma4e2b_mixedbit_manifest.json         {shape, bits, qdim, zp_all_zero}
  gemma4e2b_fp32_norms.safetensors         fp32 norms (from the official plain tensors)
  final_norm.f32.npy                       final RMSNorm weight

Packing is done by UNPACKING the official codes and REPACKING into our kernel layout (int4
low-nibble-first signed, int2 bits[1:0]-first signed, int8 signed), so it is robust to whatever
byte order the official checkpoint uses; the equivalence test locks that the dequantized values
match the litertlm extract bit-for-bit. The (1+w) norm shift is applied per NORM_SHIFT below,
which the equivalence test determines.

Run:
  .venv/bin/python scripts/gemma4_official_to_mixedbit_extract.py \
      --official src_models/gemma-4-E2B-it-qat-mobile-transformers \
      --out out/gemma4e2b_extract_official
"""
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

torch.set_grad_enabled(False)

# ---------- official unpack (convention LOCKED by the equivalence test) ----------
# equiv test result: official int4/int2 = UNSIGNED code minus midpoint (zp = 2^(bits-1)); int8 =
# signed. Byte-encoding differs from our two's-complement extract, but the dequantized VALUES are
# bit-identical (max|Δ|=0 across all 13 families). int4 low-nibble-first, int2 bits[1:0]-first.
OFF_INT4_ORDER = "low_first"   # nibble order
OFF_INT2_ORDER = "low_first"   # 2-bit slot order
NORM_SHIFT = 0.0               # direct assignment (equiv test: |Δdirect| << |Δ(off+1)|)

def off_unpack(op: torch.Tensor, rows: int, cols: int, bits: int) -> torch.Tensor:
    """Official packed U8/I8 -> SIGNED int codes [rows, cols] (int16), in our sign convention."""
    if bits == 8:
        return op.view(torch.int8).reshape(rows, cols).to(torch.int16)
    if bits == 4:
        b = op.reshape(rows, cols // 2).to(torch.int16)
        n_lo, n_hi = b & 0xF, b >> 4
        a, c = (n_lo, n_hi) if OFF_INT4_ORDER == "low_first" else (n_hi, n_lo)
        out = torch.empty(rows, cols, dtype=torch.int16)
        out[:, 0::2] = a; out[:, 1::2] = c
        return out - 8                    # unsigned nibble (0..15) -> signed (-8..7)
    b = op.reshape(rows, cols // 4).to(torch.int16)
    seq = (0, 2, 4, 6) if OFF_INT2_ORDER == "low_first" else (6, 4, 2, 0)
    out = torch.empty(rows, cols, dtype=torch.int16)
    for slot, s in enumerate(seq): out[:, slot::4] = (b >> s) & 3
    return out - 2                        # unsigned 2-bit (0..3) -> signed (-2..1)

# ---------- our-layout repack (inverse of the kernel unpackers) ----------
def our_pack(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Signed int codes [rows, cols] -> flat uint8 in OUR kernel layout."""
    rows, cols = codes.shape
    if bits == 8:
        return codes.to(torch.int8).reshape(-1).view(torch.uint8).contiguous()
    if bits == 4:
        c = (codes & 0xF).to(torch.uint8).reshape(rows, cols // 2, 2)  # low nibble first
        packed = (c[..., 0] | (c[..., 1] << 4)).reshape(-1).contiguous()
        return packed
    c = (codes & 0x3).to(torch.uint8).reshape(rows, cols // 4, 4)      # bits[1:0] first
    packed = (c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)).reshape(-1).contiguous()
    return packed

def round_trip(op, rows, cols, bits):
    """Return (packed_our_layout uint8 flat, codes int16) for an official weight tensor."""
    codes = off_unpack(op, rows, cols, bits)
    return our_pack(codes, bits), codes

# ---------- name maps ----------
LI = lambda li, s: f"model.language_model.layers.{li}.{s}"
# our_key suffix -> (official suffix, bits) for per-layer weights
def layer_weight_map(li: int, mlp_bits: int):
    return {
        f"mlp.gating1": (LI(li, "mlp.gate_proj.weight"), mlp_bits),
        f"mlp.gating2": (LI(li, "mlp.up_proj.weight"), mlp_bits),
        f"mlp.down":    (LI(li, "mlp.down_proj.weight"), mlp_bits),
        f"attn.q":      (LI(li, "self_attn.q_proj.weight"), 4),
        f"attn.o":      (LI(li, "self_attn.o_proj.weight"), 4),
        f"ple.gate":    (LI(li, "per_layer_input_gate.weight"), 8),
        f"ple.proj":    (LI(li, "per_layer_projection.weight"), 8),
    }
# k/v only for the non-shared (producer) layers
KV_MAP = lambda li: {
    "attn.k": (LI(li, "self_attn.k_proj.weight"), 4),
    "attn.v": (LI(li, "self_attn.v_proj.weight"), 4),
}
def NORM_MAP(li: int) -> dict:
    m = {
        f"layer_{li:02d}.pre_attention_norm":        LI(li, "input_layernorm.weight"),
        f"layer_{li:02d}.post_attention_norm":       LI(li, "post_attention_layernorm.weight"),
        f"layer_{li:02d}.pre_ffw_norm":              LI(li, "pre_feedforward_layernorm.weight"),
        f"layer_{li:02d}.post_ffw_norm":             LI(li, "post_feedforward_layernorm.weight"),
        f"layer_{li:02d}.post_per_layer_input_norm": LI(li, "post_per_layer_input_norm.weight"),
        f"layer_{li:02d}.query_norm":                LI(li, "self_attn.q_norm.weight"),
    }
    if li < FIRST_SHARED:  # kv-shared layers (L15+) compute no K -> no k_norm
        m[f"layer_{li:02d}.key_norm"] = LI(li, "self_attn.k_norm.weight")
    return m

L = 35
FIRST_SHARED = 15  # L0-14 = int4 mlp + own k/v; L15-34 = int2 mlp, kv-shared

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--official", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    off = safe_open(str(Path(args.official) / "model.safetensors"), framework="pt")
    okeys = set(off.keys())
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    tensors, manifest, norms = {}, {}, {}

    def add_weight(our_key, oname, bits):
        assert oname in okeys, oname
        op = off.get_tensor(oname)
        rows = op.shape[0]
        cols = op.shape[1] * (8 // bits) if op.dtype == torch.uint8 else op.shape[1]
        # official scale
        sname = oname.replace(".weight", ".weight_scale")
        sc = off.get_tensor(sname).float().reshape(-1)  # [rows]
        packed, _ = round_trip(op, rows, cols, bits)
        tensors[our_key] = packed
        tensors[our_key + ".scale"] = sc.contiguous()
        manifest[our_key] = {"shape": [rows, cols], "bits": bits, "qdim": 0, "zp_all_zero": True}

    # per-layer weights + norms
    mlp_bits_per_layer = [4 if li < FIRST_SHARED else 2 for li in range(L)]
    for li in range(L):
        for suf, (oname, bits) in layer_weight_map(li, mlp_bits_per_layer[li]).items():
            add_weight(f"decode.layer_{li:02d}.{suf}", oname, bits)
        if li < FIRST_SHARED:
            for suf, (oname, bits) in KV_MAP(li).items():
                add_weight(f"decode.layer_{li:02d}.{suf}", oname, bits)
        for our_n, off_n in NORM_MAP(li).items():
            v = off.get_tensor(off_n).float() + NORM_SHIFT
            norms[our_n] = v.contiguous()
        # skip_scale (layer end residual scalar) = official layer_scalar [1]
        norms[f"layer_{li:02d}.skip_scale"] = off.get_tensor(LI(li, "layer_scalar")).float().reshape(1).contiguous()

    # lm_head (int2), embed (int2)
    add_weight("decode.lm_head", "lm_head.weight", 2)
    # embed uses embedding_quantized + embedding_scale
    ep = off.get_tensor("model.language_model.embed_tokens.embedding_quantized")
    es = off.get_tensor("model.language_model.embed_tokens.embedding_scale").float().reshape(-1)
    er, ec = ep.shape[0], ep.shape[1] * 4
    tensors["embed.composite"], _ = (our_pack(off_unpack(ep, er, ec, 2), 2), None)
    tensors["embed.composite.scale"] = es.contiguous()
    manifest["embed.composite"] = {"shape": [er, ec], "bits": 2, "qdim": 0, "zp_all_zero": True}

    # PLE tables: official concatenated [262144, 4480] int4 (35 tables x 128 bytes) + scale [262144,35]
    pp = off.get_tensor("model.language_model.embed_tokens_per_layer.embedding_quantized")
    ps = off.get_tensor("model.language_model.embed_tokens_per_layer.embedding_scale").float()  # [V,35]
    V = pp.shape[0]
    HPLE = 256  # per-layer input width
    bytes_per_table = HPLE // 2  # int4 -> 128
    for i in range(L):
        key = "ple_table.composite" + ("" if i == 0 else str(i))
        sub = pp[:, i * bytes_per_table:(i + 1) * bytes_per_table].contiguous()  # [V,128] u8
        codes = off_unpack(sub, V, HPLE, 4)
        tensors[key] = our_pack(codes, 4)
        tensors[key + ".scale"] = ps[:, i].contiguous()
        manifest[key] = {"shape": [V, HPLE], "bits": 4, "qdim": 0, "zp_all_zero": True}

    # ple.model_proj: official BF16 [8960,1536] -> our int8 per-row (export requantizes per-block-32)
    mp = off.get_tensor("model.language_model.per_layer_model_projection.weight").float()
    mp_scale = mp.abs().amax(dim=1).clamp_min(1e-12) / 127.0
    mp_codes = torch.round(mp / mp_scale.unsqueeze(1)).clamp(-127, 127).to(torch.int16)
    tensors["decode.ple.model_proj"] = our_pack(mp_codes, 8)
    tensors["decode.ple.model_proj.scale"] = mp_scale.contiguous()
    manifest["decode.ple.model_proj"] = {"shape": list(mp.shape), "bits": 8, "qdim": 0, "zp_all_zero": True}

    # extra norms
    norms["ple_projection_norm"] = off.get_tensor(
        "model.language_model.per_layer_projection_norm.weight").float() + NORM_SHIFT
    final_norm = (off.get_tensor("model.language_model.norm.weight").float() + NORM_SHIFT).numpy()

    # write
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(out / "gemma4e2b_mixedbit_weights.safetensors"))
    save_file({k: v.contiguous() for k, v in norms.items()},
              str(out / "gemma4e2b_fp32_norms.safetensors"))
    np.save(out / "final_norm.f32.npy", final_norm.astype(np.float32))
    (out / "gemma4e2b_mixedbit_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(tensors)} weight tensors, {len(norms)} norms, manifest {len(manifest)} entries -> {out}")
    print(f"NORM_SHIFT={NORM_SHIFT}  OFF_INT4_ORDER={OFF_INT4_ORDER}  OFF_INT2_ORDER={OFF_INT2_ORDER}")

if __name__ == "__main__":
    main()
