"""Prove google/gemma-4-E2B-it-qat-mobile-transformers == our litertlm mixed-bit extract.

Our extract is already validated bit-exact vs the .litertlm (3/3 id-exact gate, cos>=0.9987 vs
MLX oracle). So it is the ground truth. For each weight family we dequantize OUR side (known
convention) and the OFFICIAL side under candidate packing conventions; the correct convention
yields dequant-cos ~= 1.0 and max-abs-diff ~= 0. This (1) proves the official checkpoint is the
same QAT run and (2) locks the official packing convention + scale layout for the converter.
Also compares norms (official plain tensors vs our reverse-extracted fp32 norms) to determine the
(1+w) shift convention.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import code_path  # noqa: E402

OFFICIAL = code_path("litertlm-convert", "src_models", "gemma-4-E2B-it-qat-mobile-transformers")
EXTRACT = code_path("litertlm-convert", "out", "gemma4e2b_extract")
torch.set_grad_enabled(False)

# ---- our extract dequant (known-good convention: int2 bits[1:0]-first signed, int4 low-nibble
#      first signed, int8 signed; per-row scale, zp=0) ----
def ours_codes(key, w, man):
    m = man[key]; rows, cols = m["shape"]; bits = m["bits"]; p = w.get_tensor(key)
    if bits == 8:
        return p.view(torch.int8).reshape(rows, cols).float()
    if bits == 4:
        b = p.reshape(rows, cols // 2).to(torch.int16)
        c = torch.stack([b & 0xF, b >> 4], -1).reshape(rows, cols)
        return torch.where(c >= 8, c - 16, c).float()
    b = p.reshape(rows, cols // 4).to(torch.int16)
    c = torch.stack([(b >> s) & 3 for s in (0, 2, 4, 6)], -1).reshape(rows, cols)
    return torch.where(c >= 2, c - 4, c).float()

def ours_scale(key, w):
    return w.get_tensor(key + ".scale").float()  # (rows,)

# ---- official candidate unpackers (order x sign/offset) ----
def off_codes(op, rows, cols, bits, order, mode):
    if bits == 8:
        base = op.view(torch.int8).reshape(rows, cols).float() if mode == "signed" \
               else op.reshape(rows, cols).float() - (128 if mode == "off_mid" else 0)
        return base
    if bits == 4:
        b = op.reshape(rows, cols // 2).to(torch.int16)
        n_lo, n_hi = b & 0xF, b >> 4
        a, c = (n_lo, n_hi) if order == "low_first" else (n_hi, n_lo)
        out = torch.empty(rows, cols, dtype=torch.int16)
        out[:, 0::2] = a; out[:, 1::2] = c
        if mode == "signed":  return torch.where(out >= 8, out - 16, out).float()
        if mode == "off_mid": return (out - 8).float()
        return out.float()
    b = op.reshape(rows, cols // 4).to(torch.int16)
    seq = (0, 2, 4, 6) if order == "low_first" else (6, 4, 2, 0)
    out = torch.empty(rows, cols, dtype=torch.int16)
    for slot, s in enumerate(seq): out[:, slot::4] = (b >> s) & 3
    if mode == "signed":  return torch.where(out >= 2, out - 4, out).float()
    if mode == "off_mid": return (out - 2).float()
    return out.float()

def cos(a, b):
    a = a.flatten().double(); b = b.flatten().double()
    return float((a @ b) / (a.norm() * b.norm() + 1e-30))

def find_off_scale(off, wname):
    for cand in (wname.replace(".weight", ".weight_scale"), wname + "_scale",
                 wname.replace(".weight", ".scale"),
                 wname.replace("_quantized", "_scale")):
        if cand in off.keys(): return cand, off.get_tensor(cand).float()
    return None, None

def main():
    ow = safe_open(str(OFFICIAL / "model.safetensors"), framework="pt")
    okeys = set(ow.keys())
    print(f"[official] {len(okeys)} tensors\n[official] all scale-like tensor names:")
    for k in sorted(okeys):
        if "scale" in k.lower() and ("layers.0." in k or "layers" not in k):
            print("   ", k, tuple(ow.get_tensor(k).shape), ow.get_tensor(k).dtype)

    w = safe_open(str(EXTRACT / "gemma4e2b_mixedbit_weights.safetensors"), framework="pt")
    man = json.loads((EXTRACT / "gemma4e2b_mixedbit_manifest.json").read_text())

    PAIRS = [  # our_key, official weight name, bits
        ("decode.layer_00.mlp.gating1", "model.language_model.layers.0.mlp.gate_proj.weight", 4),
        ("decode.layer_00.mlp.gating2", "model.language_model.layers.0.mlp.up_proj.weight", 4),
        ("decode.layer_00.mlp.down",    "model.language_model.layers.0.mlp.down_proj.weight", 4),
        ("decode.layer_20.mlp.gating1", "model.language_model.layers.20.mlp.gate_proj.weight", 2),
        ("decode.layer_20.mlp.down",    "model.language_model.layers.20.mlp.down_proj.weight", 2),
        ("decode.layer_00.attn.q",      "model.language_model.layers.0.self_attn.q_proj.weight", 4),
        ("decode.layer_00.attn.k",      "model.language_model.layers.0.self_attn.k_proj.weight", 4),
        ("decode.layer_00.attn.v",      "model.language_model.layers.0.self_attn.v_proj.weight", 4),
        ("decode.layer_00.attn.o",      "model.language_model.layers.0.self_attn.o_proj.weight", 4),
        ("decode.layer_00.ple.gate",    "model.language_model.layers.0.per_layer_input_gate.weight", 8),
        ("decode.layer_00.ple.proj",    "model.language_model.layers.0.per_layer_projection.weight", 8),
        ("decode.lm_head",              "lm_head.weight", 2),
        ("embed.composite",             "model.language_model.embed_tokens.embedding_quantized", 2),
    ]
    print("\n" + "=" * 100)
    summary = []
    for our_key, oname, bits in PAIRS:
        if oname not in okeys:
            print(f"MISSING official: {oname}"); summary.append((our_key, "MISSING")); continue
        m = man[our_key]; rows, cols = m["shape"]
        oc_codes = ours_codes(our_key, w, man); osc = ours_scale(our_key, w)
        ours_deq = oc_codes * osc.unsqueeze(1)
        op = ow.get_tensor(oname)
        sname, off_sc = find_off_scale(ow, oname)
        print(f"\n### {our_key}  <->  {oname}   bits={bits} shape={rows}x{cols}")
        print(f"    official packed dtype={op.dtype} numel={op.numel()} (expect {rows*cols*bits//8})")
        print(f"    official scale: {sname} {tuple(off_sc.shape) if off_sc is not None else None}")
        best = None
        for order in ("low_first", "high_first"):
            for mode in ("signed", "off_mid", "unsigned"):
                try: ocodes = off_codes(op, rows, cols, bits, order, mode)
                except Exception as e: continue
                # code-exact vs ours
                exact = float((ocodes == oc_codes).float().mean())
                # dequant using official scale if per-row shaped, else our scale
                if off_sc is not None and off_sc.numel() == rows:
                    deq = ocodes * off_sc.reshape(rows, 1)
                elif off_sc is not None and off_sc.shape == (rows, 1):
                    deq = ocodes * off_sc
                else:
                    deq = ocodes * osc.unsqueeze(1)  # fallback: our scale
                dcos = cos(deq, ours_deq)
                rec = (order, mode, exact, dcos)
                if best is None or exact > best[2] or (exact == best[2] and abs(dcos) > abs(best[3])):
                    best = rec
                if exact > 0.5 or abs(dcos) > 0.5:
                    print(f"      {order:10s}/{mode:8s}  code-exact={exact:6.4f}  dequant-cos={dcos:+.6f}")
        # with the winning convention, report max abs diff
        o, mo, ex, dc = best
        ocodes = off_codes(op, rows, cols, bits, o, mo)
        if off_sc is not None and off_sc.numel() == rows:
            deq = ocodes * off_sc.reshape(rows, 1)
            scale_cos = cos(off_sc.flatten(), osc.flatten())
        else:
            deq = ocodes * osc.unsqueeze(1); scale_cos = float("nan")
        madiff = float((deq - ours_deq).abs().max())
        print(f"    >>> BEST {o}/{mo}: code-exact={ex:.4f} dequant-cos={dc:+.6f} "
              f"scale-cos={scale_cos:.6f} max|Δdequant|={madiff:.3e}")
        summary.append((our_key, f"{o}/{mo} exact={ex:.3f} cos={dc:+.4f} maxΔ={madiff:.1e}"))

    # ---- PLE tables: official [V,4480] 128-byte layer-major blocks vs our per-table extract ----
    print("\n" + "=" * 100 + "\nPLE TABLES (official 128B blocks vs our ple_table.compositeN):")
    if "model.language_model.embed_tokens_per_layer.embedding_quantized" in okeys:
        pp = ow.get_tensor("model.language_model.embed_tokens_per_layer.embedding_quantized")
        ps = ow.get_tensor("model.language_model.embed_tokens_per_layer.embedding_scale").float()
        V = pp.shape[0]; HPLE = 256; bpt = HPLE // 2
        for i in (0, 1, 34):
            key = "ple_table.composite" + ("" if i == 0 else str(i))
            our_codes = ours_codes(key, w, man); our_sc = ours_scale(key, w)
            ours_deq = our_codes * our_sc.unsqueeze(1)
            sub = pp[:, i * bpt:(i + 1) * bpt].contiguous()
            oc = off_codes(sub, V, HPLE, 4, "low_first", "signed")
            deq = oc * ps[:, i].reshape(V, 1)
            print(f"  table {i:2d}: code-exact={float((oc==our_codes).float().mean()):.4f} "
                  f"dequant-cos={cos(deq,ours_deq):+.6f} scale-cos={cos(ps[:,i],our_sc):.6f} "
                  f"max|Δ|={float((deq-ours_deq).abs().max()):.3e}")

    # ---- model_proj: official BF16 vs our int8 (expect high cos, NOT bit-exact) ----
    if "model.language_model.per_layer_model_projection.weight" in okeys:
        mp = ow.get_tensor("model.language_model.per_layer_model_projection.weight").float()
        our_deq = ours_codes("decode.ple.model_proj", w, man) * ours_scale("decode.ple.model_proj", w).unsqueeze(1)
        r = min(mp.shape[0], our_deq.shape[0])
        print(f"\nMODEL_PROJ (official bf16 vs our int8): cos={cos(mp[:r], our_deq[:r]):+.6f} "
              f"max|Δ|={float((mp[:r]-our_deq[:r]).abs().max()):.3e} (int8 grid diff expected)")

    # ---- norms: official plain tensors vs our reverse-extracted fp32 norms ----
    print("\n" + "=" * 100 + "\nNORMS (official plain vs our fp32_norms; test direct vs +1 shift):")
    nrm = safe_open(str(EXTRACT / "gemma4e2b_fp32_norms.safetensors"), framework="pt")
    NORM_PAIRS = [
        ("layer_00.pre_attention_norm", "model.language_model.layers.0.input_layernorm.weight"),
        ("layer_00.post_attention_norm", "model.language_model.layers.0.post_attention_layernorm.weight"),
        ("layer_00.pre_ffw_norm", "model.language_model.layers.0.pre_feedforward_layernorm.weight"),
        ("layer_00.query_norm", "model.language_model.layers.0.self_attn.q_norm.weight"),
    ]
    for ours_n, off_n in NORM_PAIRS:
        try: on = nrm.get_tensor(ours_n).float()
        except Exception: print(f"  our norm missing: {ours_n}"); continue
        if off_n not in okeys: print(f"  official norm missing: {off_n}"); continue
        of = ow.get_tensor(off_n).float()
        d_direct = float((of - on).abs().max()); d_plus1 = float((of + 1.0 - on).abs().max())
        print(f"  {ours_n:32s} our[min/max]={on.min():+.3f}/{on.max():+.3f}  "
              f"off[min/max]={of.min():+.3f}/{of.max():+.3f}  |Δdirect|={d_direct:.3e}  |Δ(off+1)|={d_plus1:.3e}")

    print("\n" + "=" * 100 + "\nSUMMARY:")
    for k, s in summary: print(f"  {k:32s} {s}")

if __name__ == "__main__":
    main()
