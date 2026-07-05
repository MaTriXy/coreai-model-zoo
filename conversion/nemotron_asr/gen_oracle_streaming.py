"""Phase 5 — Nemotron 3.5 ASR STREAMING golden oracle (run in the ISOLATED tf-source env).

Drives the HF cache-aware streaming path (use_cache=True, chunked mel) chunk by chunk and
records everything the Core-AI streaming export gates against:

  chunk protocol (lookahead=3, subsampling=8):  25 mel first, then 32 mel  ->  4 enc frames each
  caches: attention KV sliding window (56 left-context frames) + conv padding caches
          (3x subsampling CausalConv2d + 24x depthwise CausalConv1d)

The mel is sliced from ONE offline (center=True) pass — the HF feature extractor guarantees
per-chunk STFT reproduces those frames, so mel slicing isolates the encoder-graph math from
the frontend (the Swift mel gate covers the frontend separately).

Saves oracle_stream_en_US.npz:
    mel               [1,1465,128]  offline mel, truncated to 25 + 45*32 frames
    one_hot           [128]         language prompt one-hot (en-US = 0)
    embeds_stream     [46,4,1024]   per-chunk subsampling output (post linear) — gates pre graphs
    enc_stream        [184,640]     per-chunk enc_proj concat — gates the conformer graph (GOLDEN)
    enc_off           [184,640]     offline enc_proj on the same 1465 mel (agreement stat)
    tokens            [U]           streaming generate() token ids (no blank/start)
    tokens_off        [U']          offline generate() token ids on the same mel
    text / text_off                 decoded transcripts
    layer_hs_c0/c1    [24,4,1024]   per-conformer-layer hidden states for chunks 0/1 (debug)
    k0_c0,v0_c0,k0_c1,v0_c1         layer-0 attention KV cache after chunks 0/1 (debug)
    conv1d0_c0/c1     [1024,8]      layer-0 depthwise conv cache after chunks 0/1 (debug)
    sub{0,1,2}_c0/c1                subsampling conv2d caches after chunks 0/1 (debug)
  + scalars: prompt_id, num_lookahead_tokens, chunk_first, chunk_next, n_chunks, T

Run:  ~/code/coreai/_nemotron_oracle/.venv/bin/python gen_oracle_streaming.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

MODEL = "nvidia/nemotron-3.5-asr-streaming-0.6b"
HERE = Path(__file__).resolve().parent


def cache_layer_tensors(cache, layer_idx: int):
    """Best-effort (keys, values) of a DynamicCache layer across transformers versions."""
    if hasattr(cache, "layers"):
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="en-US")
    ap.add_argument("--out", default="oracle_stream_en_US.npz")
    args = ap.parse_args()
    out = HERE / args.out

    import librosa
    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

    processor = AutoProcessor.from_pretrained(MODEL)
    model = Nemotron3_5AsrForRNNT.from_pretrained(MODEL, dtype=torch.float32).eval()
    cfg = model.config
    blank, start = cfg.blank_token_id, cfg.blank_token_id
    lookahead = 3
    sub = cfg.encoder_config.subsampling_factor              # 8
    chunk_first = 1 + sub * lookahead                        # 25
    chunk_next = sub * (lookahead + 1)                       # 32

    wav, _ = librosa.load(librosa.example("libri1"), sr=16000, mono=True)
    wav = wav[: int(16.0 * 16000)]
    inputs = processor(wav, sampling_rate=16000, language=args.language, return_tensors="pt")
    mel_full = inputs["input_features"]
    prompt_ids = inputs["prompt_ids"]
    n_chunks = 1 + (mel_full.shape[1] - chunk_first) // chunk_next
    L = chunk_first + (n_chunks - 1) * chunk_next            # 25 + 45*32 = 1465
    mel = mel_full[:, :L]
    print(f"mel full {tuple(mel_full.shape)} -> stream-aligned L={L} ({n_chunks} chunks: "
          f"{chunk_first} then {chunk_next}) prompt_id={int(prompt_ids[0])} lookahead={lookahead}")

    # ---------------- offline reference on the SAME truncated mel ----------------
    gen_off = model.generate(input_features=mel, prompt_ids=prompt_ids, num_lookahead_tokens=lookahead)
    text_off = processor.batch_decode(gen_off.sequences)[0]
    tokens_off = [t for t in gen_off.sequences[0].tolist() if t not in (start, blank)]
    enc_off = model.get_audio_features(input_features=mel, prompt_ids=prompt_ids,
                                       num_lookahead_tokens=lookahead).pooler_output[0]
    T = enc_off.shape[0]
    print(f"[offline] T={T} tokens={len(tokens_off)} text: {text_off!r}")
    assert T == 4 * n_chunks, f"enc frames {T} != 4*{n_chunks}"

    # ---------------- streaming: chunked get_audio_features with caches ----------------
    embeds_log: list[torch.Tensor] = []
    layer_hs: list[list[torch.Tensor]] = []

    def sub_hook(_m, _i, o):
        embeds_log.append(o.detach().clone())

    def layer_hook(_m, _i, o):
        layer_hs[-1].append((o[0] if isinstance(o, tuple) else o).detach().clone())

    hooks = [model.encoder.subsampling.register_forward_hook(sub_hook)]
    hooks += [blk.register_forward_hook(layer_hook) for blk in model.encoder.layers]

    def mel_chunk(i: int) -> torch.Tensor:
        if i == 0:
            return mel[:, :chunk_first]
        s = chunk_first + (i - 1) * chunk_next
        return mel[:, s: s + chunk_next]

    pkv, pad = None, None
    enc_chunks: list[torch.Tensor] = []
    debug: dict[str, np.ndarray] = {}
    for i in range(n_chunks):
        layer_hs.append([])
        kw = {} if pkv is None else {"past_key_values": pkv, "padding_cache": pad}
        o = model.get_audio_features(input_features=mel_chunk(i), prompt_ids=prompt_ids,
                                     num_lookahead_tokens=lookahead, use_cache=True,
                                     output_attention_mask=False, **kw)
        pkv, pad = o.past_key_values, o.padding_cache
        enc_chunks.append(o.pooler_output[0].clone())
        if i < 2:
            k0, v0 = cache_layer_tensors(pkv, 0)
            debug[f"k0_c{i}"] = k0[0].numpy().astype(np.float32)
            debug[f"v0_c{i}"] = v0[0].numpy().astype(np.float32)
            debug[f"conv1d0_c{i}"] = pad.layers["conv.0"].cache[0].numpy().astype(np.float32)
            for s_idx in range(3):
                debug[f"sub{s_idx}_c{i}"] = pad.layers[f"subsampling.{s_idx}"].cache[0].numpy().astype(np.float32)
            debug[f"layer_hs_c{i}"] = torch.stack(layer_hs[i])[:, 0].numpy().astype(np.float32)
        if i == 0:
            k0, _ = cache_layer_tensors(pkv, 0)
            print(f"[cache shapes after c0] kv {tuple(k0.shape)} "
                  f"conv1d {tuple(pad.layers['conv.0'].cache.shape)} "
                  + " ".join(f"sub{s} {tuple(pad.layers[f'subsampling.{s}'].cache.shape)}" for s in range(3)))
        if i in (1, 14, 15):
            k0, _ = cache_layer_tensors(pkv, 0)
            print(f"[cache after c{i}] kv len={k0.shape[2]}")
    for h in hooks:
        h.remove()

    enc_stream = torch.cat(enc_chunks, dim=0)                # [T,640]
    embeds_stream = torch.stack(embeds_log)[:, 0]            # [46,4,1024]
    d = (enc_stream - enc_off).abs()
    cos = torch.nn.functional.cosine_similarity(enc_stream, enc_off, dim=-1)
    print(f"[stream vs offline enc_proj] cos mean {cos.mean():.7f} min {cos.min():.7f} max|Δ| {d.max():.2e}")

    # ---------------- streaming generate() (official chunk-generator path) ----------------
    def chunk_gen():
        for i in range(n_chunks):
            yield mel_chunk(i)

    gen_s = model.generate(input_features=chunk_gen(), prompt_ids=prompt_ids,
                           num_lookahead_tokens=lookahead)
    text_s = processor.batch_decode(gen_s.sequences)[0]
    tokens_s = [t for t in gen_s.sequences[0].tolist() if t not in (start, blank)]
    match = tokens_s == tokens_off
    print(f"[streaming generate] tokens={len(tokens_s)} text: {text_s!r}")
    print(f"[check] streaming tokens == offline tokens: {match}")

    one_hot = torch.nn.functional.one_hot(prompt_ids[0], num_classes=cfg.num_prompts).float()
    np.savez_compressed(
        out,
        mel=mel.numpy().astype(np.float32),
        one_hot=one_hot.numpy().astype(np.float32),
        embeds_stream=embeds_stream.numpy().astype(np.float32),
        enc_stream=enc_stream.numpy().astype(np.float32),
        enc_off=enc_off.numpy().astype(np.float32),
        tokens=np.array(tokens_s, dtype=np.int64),
        tokens_off=np.array(tokens_off, dtype=np.int64),
        text=np.array(text_s), text_off=np.array(text_off),
        prompt_id=np.array(int(prompt_ids[0])),
        num_lookahead_tokens=np.array(lookahead),
        chunk_first=np.array(chunk_first), chunk_next=np.array(chunk_next),
        n_chunks=np.array(n_chunks), T=np.array(T),
        **debug,
    )
    print(f"[save] {out} ({out.stat().st_size / 1e6:.1f} MB)  "
          f"{'✅ streaming path reproduces offline' if match else '❌ MISMATCH — investigate before export'}")


if __name__ == "__main__":
    main()
