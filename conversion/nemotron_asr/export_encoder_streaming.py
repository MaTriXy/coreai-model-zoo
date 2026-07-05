"""Phase 6 — re-author the Nemotron 3.5 ASR encoder in cache-aware STREAMING form, export the
static graphs, and gate them chunk-by-chunk against oracle_stream_en_US.npz
(run in the MAIN coreai-models venv).

Weight-sharing split (the 600M conformer exists once on disk, in TWO halves):

  pre_first   : mel[1,25,128]                      -> embeds[1,4,1024] + conv2d caches (x3)
                (first chunk: causal init pad baked in — HF Conv2dCacheLayer init_pad)
  pre         : mel[1,32,128] + conv2d caches (x3) -> embeds[1,4,1024] + conv2d caches (x3)
  conformer_a : x[1,4,1024] + neg_mask[1,1,4,60] + k/v_cache[12,8,56,128] + conv_cache[12,1024,8]
                -> x + updated caches                      (layers 0-11)
  conformer_b : x + one_hot[1,128] + neg_mask + caches (x12 layers)
                -> enc_proj[1,4,640] + updated caches      (layers 12-23 + prompt fusion + projector)

WHY TWO HALVES: the single 24-layer graph (2.4 GB resources.bin) AOT-compiles fine but the
on-device loader rejects it with an instant POSIX-2 ENOENT (iPhone 17 Pro, iOS 27, h18p) —
bisected: 1-layer and 12-layer AOT bundles of the SAME topology load fine, and every file was
byte-verified on device, so the trigger is the >2 GB multi-I/O bundle itself. 12+12 keeps each
`.aimodelc` at ~1.1 GB (device-proven) for one extra ~1 ms graph call per 320 ms chunk.

Cache design (validated against HF DynamicCache shapes in the streaming oracle):
  * KV cache = fixed 56-slot right-aligned rolling window per layer. In steady state the
    chunked_limited window is exactly cache(56)+chunk(4)=60 keys, all visible; during the
    first 14 chunks the host masks the not-yet-filled left slots via the additive neg_mask
    input (-inf). Rel-pos distances are unchanged by right-alignment, so pos_emb is a fixed
    [1,119,1024] constant.
  * conv2d caches hold the last freq-padded time row per subsampling stage
    ([1,1,1,131] / [1,256,1,68] / [1,256,1,36]); conv1d caches hold the last 8 post-GLU
    frames per conformer layer ([12,1024,8] per half). Zero-init == HF zero-init.

Gate ladder: eager fp32 per-chunk (embeds + enc_proj + cache tensors + per-layer hidden
states vs oracle) -> fp16 export -> engine per-chunk cos on cpu/gpu.

Run:  coreai-models/.venv/bin/python export_encoder_streaming.py [--dtype float16] [--skip-export]
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from export_encoder import (
    CONV_K, HDIM, HEADS, HID, MEL, NUM_PROMPTS, SLIDING_WINDOW,
    Encoder, Subsampling, build_pos_emb, load_weights,
)

HERE = Path(__file__).resolve().parent
LOOKAHEAD = 3
Q = LOOKAHEAD + 1                    # enc frames per chunk
KV = SLIDING_WINDOW - 1 + Q          # 60 keys in steady state
NLAYERS = 24
SPLIT = 12                           # conformer_a = layers [0,12), conformer_b = [12,24)
CHUNK_FIRST = 1 + 8 * LOOKAHEAD      # 25 mel
CHUNK_NEXT = 8 * (LOOKAHEAD + 1)     # 32 mel


# --------------------------------------------------------------------------- pre-encode graphs
def _freq_pad(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (2, 1))          # causal freq pad (k-1, s-1)


class PreEncodeFirst(nn.Module):
    """First chunk: no cache inputs; time left-pad k-1 (cache zeros + init_pad) baked in."""

    def __init__(self, sub: Subsampling):
        super().__init__()
        self.sub = sub

    def forward(self, mel):  # [1,25,128]
        xf = _freq_pad(mel.unsqueeze(1))                                   # [1,1,25,131]
        c0 = xf[:, :, -1:, :]
        h = torch.relu(self.sub.conv_in(F.pad(xf, (0, 0, 2, 0))))          # [1,256,13,65]
        s1, s2 = self.sub.layers
        h1f = _freq_pad(h)                                                 # [1,256,13,68]
        c1 = h1f[:, :, -1:, :]
        h = torch.relu(s1.pointwise_conv(s1.depthwise_conv(F.pad(h1f, (0, 0, 2, 0)))))  # [1,256,7,33]
        h2f = _freq_pad(h)                                                 # [1,256,7,36]
        c2 = h2f[:, :, -1:, :]
        h = torch.relu(s2.pointwise_conv(s2.depthwise_conv(F.pad(h2f, (0, 0, 2, 0)))))  # [1,256,4,17]
        h = h.transpose(1, 2).reshape(1, h.shape[2], -1)
        return self.sub.linear(h), c0, c1, c2


class PreEncode(nn.Module):
    """Steady chunk: time left context comes from the conv2d caches (1 row per stage)."""

    def __init__(self, sub: Subsampling):
        super().__init__()
        self.sub = sub

    def forward(self, mel, cache0, cache1, cache2):  # [1,32,128] + caches
        xf = _freq_pad(mel.unsqueeze(1))                                   # [1,1,32,131]
        c0 = xf[:, :, -1:, :]
        h = torch.relu(self.sub.conv_in(torch.cat([cache0, xf], dim=2)))   # 33 rows -> [1,256,16,65]
        s1, s2 = self.sub.layers
        h1f = _freq_pad(h)                                                 # [1,256,16,68]
        c1 = h1f[:, :, -1:, :]
        h = torch.relu(s1.pointwise_conv(s1.depthwise_conv(torch.cat([cache1, h1f], dim=2))))  # [1,256,8,33]
        h2f = _freq_pad(h)                                                 # [1,256,8,36]
        c2 = h2f[:, :, -1:, :]
        h = torch.relu(s2.pointwise_conv(s2.depthwise_conv(torch.cat([cache2, h2f], dim=2))))  # [1,256,4,17]
        h = h.transpose(1, 2).reshape(1, h.shape[2], -1)
        return self.sub.linear(h), c0, c1, c2


# --------------------------------------------------------------------------- conformer halves
class ConformerStreamPart(nn.Module):
    """Conformer layers [lo, hi) over one 4-frame chunk with explicit KV / depthwise-conv
    caches; the head part appends prompt fusion + projector. Weights are shared with the
    offline Encoder re-author."""

    def __init__(self, enc: Encoder, lo: int, hi: int, head: bool):
        super().__init__()
        self.enc = enc
        self.lo, self.hi, self.head = lo, hi, head
        self.register_buffer("pos_emb", build_pos_emb(KV), persistent=False)  # [1,2*KV-1,HID]
        self.collect_layer_hs: list[torch.Tensor] | None = None

    def forward(self, x, neg_mask, k_cache, v_cache, conv_cache, one_hot=None):
        # x [1,Q,HID]  neg_mask [1,1,Q,KV] additive  k/v [n,H,KV-Q,D]  conv [n,HID,K-1]
        k_outs, v_outs, c_outs = [], [], []
        for i, blk in enumerate(self.enc.layers[self.lo: self.hi]):
            x = x + 0.5 * blk.feed_forward1(blk.norm_feed_forward1(x))

            a = blk.self_attn
            hn = blk.norm_self_att(x)
            q = a.q_proj(hn).view(1, Q, HEADS, HDIM).transpose(1, 2)
            k_new = a.k_proj(hn).view(1, Q, HEADS, HDIM).transpose(1, 2)
            v_new = a.v_proj(hn).view(1, Q, HEADS, HDIM).transpose(1, 2)
            k = torch.cat([k_cache[i].unsqueeze(0), k_new], dim=2)          # [1,H,KV,D]
            v = torch.cat([v_cache[i].unsqueeze(0), v_new], dim=2)
            k_outs.append(k[0, :, Q:, :])
            v_outs.append(v[0, :, Q:, :])
            q_u = q + a.bias_u.view(1, HEADS, 1, HDIM)
            q_v = q + a.bias_v.view(1, HEADS, 1, HDIM)
            rel_k = a.relative_k_proj(self.pos_emb).view(1, -1, HEADS, HDIM)
            bd = q_v @ rel_k.permute(0, 2, 3, 1)                            # [1,H,Q,2*KV-1]
            bd = a._rel_shift(bd)[..., :KV] * a.scaling + neg_mask
            w = torch.softmax(q_u @ k.transpose(-2, -1) * a.scaling + bd, dim=-1)
            attn = (w @ v).transpose(1, 2).reshape(1, Q, HID)
            x = x + a.o_proj(attn)

            cm = blk.conv
            g = F.glu(cm.pointwise_conv1(blk.norm_conv(x).transpose(1, 2)), dim=1)  # [1,HID,Q]
            gp = torch.cat([conv_cache[i].unsqueeze(0), g], dim=-1)         # [1,HID,K-1+Q]
            c_outs.append(gp[0, :, Q:])
            y = cm.depthwise_conv(gp)                                       # [1,HID,Q]
            y = F.silu(cm.norm(y.transpose(1, 2))).transpose(1, 2)
            y = cm.pointwise_conv2(y).transpose(1, 2)                       # [1,Q,HID]
            x = x + y

            x = x + 0.5 * blk.feed_forward2(blk.norm_feed_forward2(x))
            x = blk.norm_out(x)
            if self.collect_layer_hs is not None:
                self.collect_layer_hs.append(x.detach().clone())

        k_out, v_out, c_out = torch.stack(k_outs), torch.stack(v_outs), torch.stack(c_outs)
        if not self.head:
            return x, k_out, v_out, c_out
        oh = one_hot[:, None, :].expand(-1, Q, -1)
        fused = self.enc.prompt_linear_2(torch.relu(self.enc.prompt_linear_1(torch.cat([x, oh], dim=-1))))
        return self.enc.projector(fused), k_out, v_out, c_out


# NOTE the input is named "embeds", NOT "x": the on-device AOT loader rejects conformer_b with a
# first input named "x" (instant POSIX-2 ENOENT at load; renaming the SAME graph fixed it —
# bisected on iPhone 17 Pro / iOS 27). Keep long descriptive I/O names.
class ConformerA(ConformerStreamPart):
    def __init__(self, enc: Encoder):
        super().__init__(enc, 0, SPLIT, head=False)

    def forward(self, embeds, neg_mask, k_cache, v_cache, conv_cache):
        return super().forward(embeds, neg_mask, k_cache, v_cache, conv_cache)


class ConformerB(ConformerStreamPart):
    def __init__(self, enc: Encoder):
        super().__init__(enc, SPLIT, NLAYERS, head=True)

    def forward(self, embeds, one_hot, neg_mask, k_cache, v_cache, conv_cache):
        return super().forward(embeds, neg_mask, k_cache, v_cache, conv_cache, one_hot=one_hot)


def build_neg_mask(chunk_idx: int, dtype=torch.float32) -> torch.Tensor:
    """[1,1,Q,KV] additive mask: -inf on cache slots not yet filled (right-aligned window)."""
    valid = min(KV - Q, Q * chunk_idx)
    neg = torch.zeros(1, 1, Q, KV, dtype=dtype)
    neg[..., : KV - Q - valid] = float("-inf")
    return neg


def mel_chunks(mel: torch.Tensor):
    yield mel[:, :CHUNK_FIRST]
    for s in range(CHUNK_FIRST, mel.shape[1], CHUNK_NEXT):
        yield mel[:, s: s + CHUNK_NEXT]


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.reshape(-1), b.reshape(-1), dim=0).item()


# --------------------------------------------------------------------------- gate/export
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--oracle", default="oracle_stream_en_US.npz")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    d = np.load(HERE / args.oracle)
    mel = torch.from_numpy(d["mel"]).float()                        # [1,1465,128]
    one_hot = torch.from_numpy(d["one_hot"]).float()[None]          # [1,128]
    embeds_gold = torch.from_numpy(d["embeds_stream"]).float()      # [46,4,1024]
    enc_gold = torch.from_numpy(d["enc_stream"]).float()            # [184,640]
    assert int(d["num_lookahead_tokens"]) == LOOKAHEAD

    enc = Encoder(mel.shape[1], LOOKAHEAD).eval()
    load_weights(enc)
    pre_first = PreEncodeFirst(enc.subsampling).eval()
    pre = PreEncode(enc.subsampling).eval()
    conf_a, conf_b = ConformerA(enc).eval(), ConformerB(enc).eval()

    # ---- 1. eager fp32 per-chunk gate ----
    with torch.no_grad():
        kA = torch.zeros(SPLIT, HEADS, KV - Q, HDIM); vA = kA.clone(); ccA = torch.zeros(SPLIT, HID, CONV_K - 1)
        kB = kA.clone(); vB = kA.clone(); ccB = ccA.clone()
        c0 = c1 = c2 = None
        worst_e, worst_p = 1.0, 1.0
        proj_all = []
        for i, chunk in enumerate(mel_chunks(mel)):
            if i == 0:
                e, c0, c1, c2 = pre_first(chunk)
            else:
                e, c0, c1, c2 = pre(chunk, c0, c1, c2)
            worst_e = min(worst_e, cos(e[0], embeds_gold[i]))
            conf_a.collect_layer_hs = [] if i < 2 else None
            conf_b.collect_layer_hs = [] if i < 2 else None
            neg = build_neg_mask(i)
            x, kA, vA, ccA = conf_a(e, neg, kA, vA, ccA)
            proj, kB, vB, ccB = conf_b(x, one_hot, neg, kB, vB, ccB)
            proj_all.append(proj[0])
            worst_p = min(worst_p, cos(proj[0], enc_gold[4 * i: 4 * i + 4]))
            if i < 2:  # cache + per-layer debug tensors recorded by the oracle
                gk = torch.from_numpy(d[f"k0_c{i}"]).float()
                gcc = torch.from_numpy(d[f"conv1d0_c{i}"]).float()
                lhs = torch.stack(conf_a.collect_layer_hs + conf_b.collect_layer_hs)[:, 0]
                glhs = torch.from_numpy(d[f"layer_hs_c{i}"]).float()
                subs = [c0, c1, c2]
                sub_cos = min(cos(subs[s][0], torch.from_numpy(d[f"sub{s}_c{i}"]).float()) for s in range(3))
                print(f"  [debug c{i}] k0 cos {cos(kA[0, :, -gk.shape[1]:], gk):.6f} "
                      f"conv1d0 cos {cos(ccA[0], gcc):.6f} sub min-cos {sub_cos:.6f} "
                      f"layer-hs min-cos {min(cos(lhs[j], glhs[j]) for j in range(NLAYERS)):.6f}")
        conf_a.collect_layer_hs = conf_b.collect_layer_hs = None
        proj_cat = torch.cat(proj_all)
        pertok = F.cosine_similarity(proj_cat, enc_gold, dim=-1)
    print(f"[eager fp32] worst per-chunk embeds cos {worst_e:.6f}  enc_proj cos {worst_p:.6f}  "
          f"per-frame mean {pertok.mean():.6f} min {pertok.min():.6f} max|Δ| {(proj_cat - enc_gold).abs().max():.2e}")
    if pertok.mean() < 0.999 or worst_e < 0.999:
        print("❌ streaming re-author DIVERGES — fix before export")
        raise SystemExit(1)
    print("✅ eager streaming re-author matches the HF streaming oracle")
    if args.skip_export:
        return

    # ---- 2. export the four graphs ----
    import asyncio
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    z = lambda *s: torch.zeros(*s, dtype=dtype)

    def export(mod, example, ins, outs, name):
        prog = export_to_coreai(mod.to(dtype), example, dynamic_shapes=None,
                                input_names=ins, output_names=outs, state_names=None,
                                externalize_modules=[])
        prog.optimize()
        path = art / f"nemotron_asr_stream_{name}_{args.dtype}.aimodel"
        shutil.rmtree(path, ignore_errors=True)
        meta = rt.AIModelAssetMetadata()
        meta.license = "openmdw-1.1"
        prog.save_asset(path, meta)
        sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        print(f"[save] {path.name} ({sz:.1f} MB)")
        return path

    print(f"[export] streaming graphs ({args.dtype}) -> Core AI ...", flush=True)
    pf_path = export(pre_first, {"mel": z(1, CHUNK_FIRST, MEL)},
                     ("mel",), ("embeds", "cache0", "cache1", "cache2"), "pre_first")
    pr_path = export(pre, {"mel": z(1, CHUNK_NEXT, MEL), "cache0": z(1, 1, 1, 131),
                           "cache1": z(1, 256, 1, 68), "cache2": z(1, 256, 1, 36)},
                     ("mel", "cache0", "cache1", "cache2"),
                     ("embeds", "cache0_out", "cache1_out", "cache2_out"), "pre")
    ca_path = export(conf_a, {"embeds": z(1, Q, HID), "neg_mask": z(1, 1, Q, KV),
                              "k_cache": z(SPLIT, HEADS, KV - Q, HDIM),
                              "v_cache": z(SPLIT, HEADS, KV - Q, HDIM),
                              "conv_cache": z(SPLIT, HID, CONV_K - 1)},
                     ("embeds", "neg_mask", "k_cache", "v_cache", "conv_cache"),
                     ("embeds_out", "k_out", "v_out", "conv_out"), "conformer_a")
    cb_path = export(conf_b, {"embeds": z(1, Q, HID), "one_hot": z(1, NUM_PROMPTS),
                              "neg_mask": z(1, 1, Q, KV),
                              "k_cache": z(SPLIT, HEADS, KV - Q, HDIM),
                              "v_cache": z(SPLIT, HEADS, KV - Q, HDIM),
                              "conv_cache": z(SPLIT, HID, CONV_K - 1)},
                     ("embeds", "one_hot", "neg_mask", "k_cache", "v_cache", "conv_cache"),
                     ("enc_proj", "k_out", "v_out", "conv_out"), "conformer_b")

    # ---- 3. engine per-chunk gate ----
    async def gate(unit: str):
        opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
                else rt.SpecializationOptions.from_preferred_compute_unit_kind(getattr(rt.ComputeUnitKind, unit)()))
        fns = {}
        for name, path in (("pf", pf_path), ("pr", pr_path), ("ca", ca_path), ("cb", cb_path)):
            m = await rt.AIModel.load(str(path), opts)
            fns[name] = m.load_function("main")
        nd = lambda t: rt.NDArray(t.to(dtype).numpy())
        zn = lambda *s: rt.NDArray(np.zeros(s, dtype=np.float16 if dtype == torch.float16 else np.float32))
        kA, vA = zn(SPLIT, HEADS, KV - Q, HDIM), zn(SPLIT, HEADS, KV - Q, HDIM)
        ccA = zn(SPLIT, HID, CONV_K - 1)
        kB, vB = zn(SPLIT, HEADS, KV - Q, HDIM), zn(SPLIT, HEADS, KV - Q, HDIM)
        ccB = zn(SPLIT, HID, CONV_K - 1)
        c0 = c1 = c2 = None
        proj_all, dt_ms = [], []
        for i, chunk in enumerate(mel_chunks(mel)):
            t0 = time.perf_counter()
            if i == 0:
                r = await fns["pf"]({"mel": nd(chunk)})
                e, c0, c1, c2 = (r[x] for x in ("embeds", "cache0", "cache1", "cache2"))
            else:
                r = await fns["pr"]({"mel": nd(chunk), "cache0": c0, "cache1": c1, "cache2": c2})
                e, c0, c1, c2 = (r[x] for x in ("embeds", "cache0_out", "cache1_out", "cache2_out"))
            neg = nd(build_neg_mask(i))
            r = await fns["ca"]({"embeds": e, "neg_mask": neg, "k_cache": kA, "v_cache": vA, "conv_cache": ccA})
            xh, kA, vA, ccA = r["embeds_out"], r["k_out"], r["v_out"], r["conv_out"]
            r = await fns["cb"]({"embeds": xh, "one_hot": nd(one_hot), "neg_mask": neg,
                                 "k_cache": kB, "v_cache": vB, "conv_cache": ccB})
            kB, vB, ccB = r["k_out"], r["v_out"], r["conv_out"]
            dt_ms.append((time.perf_counter() - t0) * 1e3)
            proj_all.append(torch.from_numpy(r["enc_proj"].numpy().astype(np.float32))[0])
        proj_cat = torch.cat(proj_all)
        pt = F.cosine_similarity(proj_cat, enc_gold, dim=-1)
        ok = pt.mean() > 0.999 and pt.min() > 0.99
        print(f"[gate {unit}] per-frame cos mean {pt.mean():.6f} min {pt.min():.6f} "
              f"max|Δ| {(proj_cat - enc_gold).abs().max():.3f}  "
              f"chunk {np.mean(dt_ms[1:]):.1f}ms avg / {np.max(dt_ms[1:]):.1f}ms max "
              f"(audio 320ms) -> {'PASS' if ok else 'FAIL'}")
        return ok

    for unit in ("cpu", "gpu"):
        asyncio.run(gate(unit))


if __name__ == "__main__":
    main()
