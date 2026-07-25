"""dots.tts torch oracle — golden per-stage fixtures for the Core AI port ladder.

Runs the upstream DotsTtsModel on a FIXED (text, seed), captures:
  * every torch.randn draw (the flow-matching initial noise, for deterministic replay),
  * first-call INPUTS + OUTPUT of each exportable submodule (backbone llm, patch_encoder,
    DiT velocity field, the small projections, vocoder),
  * the final 48 kHz waveform.
Saves oracle_ref.npz + oracle.wav. Each Core AI overlay is later gated cos>=0.999 against
the matching fixture (mirrors VoxCPM scratchpad/oracle.py + the MLX port's tests/).

v2 (2026-07-05): fills the 4 v1 gaps —
  * llm.in_* / dit.kw_* (inputs were passed as kwargs → forward_pre_hook(with_kwargs=True)),
  * patch_encoder.* / vocoder.* (called via METHODS decode_patch / inference_from_latents,
    not .forward → the bound methods are wrapped on the instance; upstream runs optimize=False
    so _get_compiled_method returns the instance attribute = our wrapper).

Run (from repo root):
  W=/private/tmp/.../scratchpad/dots_tts
  PYTHONPATH="$W/_shims:$W/dots.tts/src" $W/venv/bin/python conversion/dots_tts/oracle.py \
      --src $W/weights/dots.tts-soar --out conversion/dots_tts/artifacts --num-steps 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _to_np(x):
    if isinstance(x, torch.Tensor):
        # MUST copy: torch .numpy() on a CPU-fp32 tensor aliases storage, so recording a
        # tensor that is later mutated in place (e.g. the patch_encoder KV workspace, which
        # index_copy_'s across patches) would otherwise capture its FINAL state, not this call.
        return np.array(x.detach().to(torch.float32).cpu().numpy(), copy=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default="Hello from Core A I.")
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="")  # e.g. "mf" to name the npz oracle_ref_mf.npz
    ap.add_argument("--e2e", action="store_true",
                    help="also capture the full-utterance E2E seed: generation_schedule, "
                         "audio_span_token_ids, prefill inputs_embeds, fm_null_g_cond, per-patch golden latents")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fixtures: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    def rec(key, val):
        """Record a tensor / int / (nested tuple of tensors) under `key` (first write wins)."""
        if key in fixtures:
            return
        if isinstance(val, torch.Tensor):
            arr = _to_np(val)
            if arr is not None:
                fixtures[key] = arr
        elif isinstance(val, (int, float, bool)):
            fixtures[key] = np.array([val])
        elif isinstance(val, (tuple, list)):
            for i, v in enumerate(val):
                rec(f"{key}_{i}", v)

    # ---- record every torch.randn (flow noise) deterministically ----
    torch.manual_seed(args.seed)
    _orig_randn = torch.randn
    randn_log: list[np.ndarray] = []

    def _randn_spy(*a, **k):
        t = _orig_randn(*a, **k)
        randn_log.append(_to_np(t))
        return t

    torch.randn = _randn_spy  # type: ignore

    from dots_tts.runtime import DotsTtsRuntime

    rt = DotsTtsRuntime.from_pretrained(args.src, precision="float32")
    model = rt.model

    handles = []

    # ---- forward hooks: first-call OUTPUT of each exportable submodule ----
    def make_out_hook(name):
        def hook(mod, inputs, output):
            if f"{name}._out_done" in seen:
                return
            seen.add(f"{name}._out_done")
            # positional inputs (usually empty here — kwargs captured by pre-hook)
            for i, t in enumerate(inputs):
                rec(f"{name}.in{i}", t)
            outs = output
            if hasattr(output, "logits") or hasattr(output, "last_hidden_state"):
                outs = tuple(
                    getattr(output, a)
                    for a in ("last_hidden_state", "logits", "hidden_states")
                    if getattr(output, a, None) is not None
                )
                # hidden_states is a tuple of layers → keep only the last
            if isinstance(outs, torch.Tensor):
                outs = (outs,)
            if isinstance(outs, (tuple, list)):
                for i, t in enumerate(outs):
                    if isinstance(t, (tuple, list)):
                        rec(f"{name}.out{i}", t[-1])  # e.g. hidden_states → last layer
                    else:
                        rec(f"{name}.out{i}", t)
        return hook

    # ---- pre hooks (with kwargs): first-call INPUTS incl. kwargs ----
    def make_pre_hook(name, kw_keys):
        def pre(mod, pargs, pkwargs):
            if f"{name}._in_done" in seen:
                return
            seen.add(f"{name}._in_done")
            for i, t in enumerate(pargs):
                rec(f"{name}.in{i}", t)
            for k in kw_keys:
                if k in pkwargs and pkwargs[k] is not None:
                    rec(f"{name}.kw_{k}", pkwargs[k])
        return pre

    llm = model.core.llm
    dit = model.core.velocity_field_predictor
    handles.append(llm.register_forward_hook(make_out_hook("llm")))
    handles.append(llm.register_forward_pre_hook(
        make_pre_hook("llm", ["inputs_embeds", "input_ids", "attention_mask"]),
        with_kwargs=True))
    handles.append(dit.register_forward_hook(make_out_hook("dit")))
    handles.append(dit.register_forward_pre_hook(
        make_pre_hook("dit", ["x", "timesteps", "duration", "attn_mask", "pos_ids",
                              "g_cond", "hidden_size", "patch_size"]),
        with_kwargs=True))
    for n in ("hidden_proj", "latent_proj", "coordinate_proj", "eos_proj"):
        handles.append(getattr(model.core, n).register_forward_hook(make_out_hook(n)))

    # ---- wrap METHOD-dispatched modules (patch_encoder.decode_patch, vocoder.inference_from_latents) ----
    pe = model.core.patch_encoder
    _orig_decode_patch = pe.decode_patch

    def _decode_patch_wrap(latent_patch, conv_tail, layer_caches, positions):
        embedding, new_tail = _orig_decode_patch(latent_patch, conv_tail, layer_caches, positions)
        if "patch_encoder._done" not in seen:
            seen.add("patch_encoder._done")
            rec("patch_encoder.in_latent_patch", latent_patch)
            rec("patch_encoder.in_conv_tail", conv_tail)
            rec("patch_encoder.in_positions", positions)
            rec("patch_encoder.in_layer_caches", layer_caches)
            rec("patch_encoder.out_embedding", embedding)
            rec("patch_encoder.out_conv_tail", new_tail)
        return embedding, new_tail

    pe.decode_patch = _decode_patch_wrap

    voc = model.vocoder
    _orig_infer = voc.inference_from_latents

    def _infer_wrap(x, *a, **k):
        wav = _orig_infer(x, *a, **k)
        if "vocoder._done" not in seen:
            seen.add("vocoder._done")
            rec("vocoder.in_x", x)
            rec("vocoder.out_wav", wav)
        return wav

    voc.inference_from_latents = _infer_wrap

    # ---- E2E seed capture (full utterance): schedule + prefill embeds + null g_cond + golden latents ----
    if args.e2e:
        # generation_schedule + audio span ids come through _prefill; grab them + the prefill embeds
        _orig_prefill = model._prefill
        _orig_bpe = model._build_prefill_inputs_embeds

        def _bpe_wrap(gs, **kw):
            emb = _orig_bpe(gs, **kw)
            if "e2e.prefill_embeds" not in fixtures:
                fixtures["e2e.prefill_embeds"] = _to_np(emb)          # [1,prefill_end,1536]
            return emb
        model._build_prefill_inputs_embeds = _bpe_wrap

        def _prefill_wrap(gs, *, state, span_positions, prompt_patches, prompt_patch_embeddings, audio_placeholder_ids):
            fixtures["e2e.generation_schedule"] = np.asarray(gs.detach().cpu().numpy(), dtype=np.int64)
            fixtures["e2e.span_positions"] = np.asarray(span_positions.detach().cpu().numpy(), dtype=np.int64)
            fixtures["e2e.audio_span_token_ids"] = np.asarray(sorted(audio_placeholder_ids), dtype=np.int64)
            pe_ = _orig_prefill(gs, state=state, span_positions=span_positions, prompt_patches=prompt_patches,
                                prompt_patch_embeddings=prompt_patch_embeddings, audio_placeholder_ids=audio_placeholder_ids)
            fixtures["e2e.prefill_end"] = np.array([pe_])
            if getattr(state, "fm_null_g_cond", None) is not None:
                fixtures["e2e.fm_null_g_cond"] = _to_np(state.fm_null_g_cond)
            return pe_
        model._prefill = _prefill_wrap

        # capture each golden denormalized payload latent (what generate() yields)
        _orig_gls = model._generate_latents_stream

        def _gls_wrap(data, **kw):
            n = 0
            for lat in _orig_gls(data, **kw):
                fixtures[f"e2e.latent{n}"] = _to_np(lat)
                n += 1
                yield lat
            fixtures["e2e.num_latents"] = np.array([n])
        model._generate_latents_stream = _gls_wrap

    # ---- run generation ----
    result = rt.generate(text=args.text, num_steps=args.num_steps)
    for h in handles:
        h.remove()
    pe.decode_patch = _orig_decode_patch
    voc.inference_from_latents = _orig_infer
    torch.randn = _orig_randn  # type: ignore

    # ---- extract waveform ----
    wav = None
    for attr in ("audio", "waveform", "wav", "samples"):
        v = result.get(attr) if isinstance(result, dict) else getattr(result, attr, None)
        if v is not None:
            wav = _to_np(v) if isinstance(v, torch.Tensor) else np.asarray(v, dtype=np.float32)
            break
    if wav is not None:
        fixtures["wav"] = wav.reshape(-1)

    # ---- noise log ----
    for i, z in enumerate(randn_log[:128]):
        fixtures[f"randn{i}"] = z
    fixtures["_meta_num_randn"] = np.array([len(randn_log)])
    fixtures["_meta_num_steps"] = np.array([args.num_steps])

    stem = "oracle_ref" + (f"_{args.tag}" if args.tag else "")
    np.savez(out / f"{stem}.npz", **fixtures)
    if wav is not None:
        import soundfile as sf
        wname = "oracle" + (f"_{args.tag}" if args.tag else "") + ".wav"
        sf.write(str(out / wname), wav.reshape(-1), 48000)

    print("=== ORACLE CAPTURED ===")
    for k in sorted(fixtures):
        if k.startswith("randn") and k != "randn0":
            continue
        v = fixtures[k]
        print(f"  {k:32s} shape={tuple(v.shape)}")
    print(f"num torch.randn draws = {len(randn_log)}; wav = {None if wav is None else wav.reshape(-1).shape}")
    print(f"-> {out / (stem + '.npz')}")


if __name__ == "__main__":
    main()
