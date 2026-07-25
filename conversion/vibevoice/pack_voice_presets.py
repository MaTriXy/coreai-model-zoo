"""Pack the upstream voice presets (.pt torch pickles) into flat blobs a Swift host can read,
so CoreAIKit needs neither torch nor the `vibevoice` package to speak in any of the 25 voices.

Per voice (`voices/<name>/`):
  main_{k,v}.f16    (nl=4, 1, 2, L_main, 64)   context-LM prefill KV     ("lm")
  tts_{k,v}.f16     (nl=20, 1, 2, L_tts, 64)   speech-LM prefill KV      ("tts_lm")
  neg_{k,v}.f16     (nl=20, 1, 2, L_neg, 64)   CFG-negative prefill KV   ("neg_tts_lm")
  negtts_last.f16   (1, 896)                   first negative condition
  meta.json                                    prefill lengths + language tag

Shared (`glue/`):
  type_emb.f16   (2,896)   tts_input_types (speech=0 / text=1)
  eos_fc{1,2}_{w,b}.f32    the EOS classifier (host-side, 2 small matmuls)
  glue.json                hidden/vae_dim/hop/windows/cfg/scaling/bias/layers + DPMSolver++ schedule
  tokenizer.json, tokenizer_config.json        Qwen2.5-0.5B tokenizer (the processor's tokenizer)

The text path is just `tokenizer.encode(text.strip() + "\\n", add_special_tokens=False)` followed by
an `embed_tokens` lookup (the fp16 table ships at the repo root), which is what the Swift runtime does.

    <coreai-venv>/bin/python pack_voice_presets.py --out voices_coreai
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache

HERE = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402
SNAP = hf_snapshot("microsoft/VibeVoice-Realtime-0.5B", "model.safetensors")
VOICES = HERE / "_code" / "demo" / "voices" / "streaming_model"


def stack_kv(entry):
    pk = entry["past_key_values"] if isinstance(entry, dict) else entry.past_key_values
    ks = pk.key_cache if hasattr(pk, "key_cache") else [layer[0] for layer in pk]
    vs = pk.value_cache if hasattr(pk, "value_cache") else [layer[1] for layer in pk]
    K = torch.stack([k.detach().float() for k in ks], dim=0)     # (nl,1,nkv,L,hd)
    V = torch.stack([v.detach().float() for v in vs], dim=0)
    return K.numpy(), V.numpy()


def last_hidden(entry):
    h = entry["last_hidden_state"] if isinstance(entry, dict) else entry.last_hidden_state
    return h[:, -1].detach().float().numpy()


def w16(path, arr):
    np.ascontiguousarray(arr, dtype=np.float16).tofile(path)


def w32(path, arr):
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "voices_coreai"))
    ap.add_argument("--ddpm-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--oracle", default=str(HERE / "artifacts/oracle_ref.npz"))
    a = ap.parse_args()
    out = Path(a.out)
    (out / "glue").mkdir(parents=True, exist_ok=True)
    (out / "voices").mkdir(parents=True, exist_ok=True)

    sd = load_file(SNAP)
    z = np.load(a.oracle)
    sys.path.insert(0, str(HERE))
    from dpm_solver import DPMSolverMultistepScheduler

    # ---- shared glue ----
    w16(out / "glue" / "type_emb.f16", sd["model.tts_input_types.weight"].float().numpy())
    for k, nm in [("fc1.weight", "eos_fc1_w"), ("fc1.bias", "eos_fc1_b"),
                  ("fc2.weight", "eos_fc2_w"), ("fc2.bias", "eos_fc2_b")]:
        w32(out / "glue" / f"{nm}.f32", sd[f"tts_eos_classifier.{k}"].float().numpy())

    s = DPMSolverMultistepScheduler(num_train_timesteps=1000, beta_schedule="cosine",
                                    prediction_type="v_prediction")
    s.set_timesteps(a.ddpm_steps)
    at = s.alpha_t.numpy().astype(np.float64)
    sg = s.sigma_t.numpy().astype(np.float64)
    lam = np.log(at) - np.log(sg)
    sched = [{"t": int(t), "alpha": float(at[t]), "sigma": float(sg[t]), "lambda": float(lam[t])}
             for t in s.timesteps]
    glue = {
        "hidden": 896, "vae_dim": 64, "hop": 3200, "text_window": 5, "speech_window": 6,
        "cfg": a.cfg, "ddpm_steps": a.ddpm_steps, "schedule": sched,
        "scaling": float(z["speech_scaling_factor"]), "bias": float(z["speech_bias_factor"]),
        "main_layers": 4, "tts_layers": 20, "n_kv": 2, "head_dim": 64,
        "sample_rate": 24000, "decoder_frames": 64,
        "tokenizer": "Qwen/Qwen2.5-0.5B", "text_suffix": "\n",
    }
    (out / "glue" / "glue.json").write_text(json.dumps(glue, indent=2))

    # ---- per-voice prefill caches ----
    index = []
    for pt in sorted(VOICES.glob("*.pt")):
        with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
            pre = torch.load(pt, map_location="cpu", weights_only=False)
        d = out / "voices" / pt.stem
        d.mkdir(parents=True, exist_ok=True)
        lens = {}
        for nm, key in [("main", "lm"), ("tts", "tts_lm"), ("neg", "neg_tts_lm")]:
            K, V = stack_kv(pre[key])
            w16(d / f"{nm}_k.f16", K)
            w16(d / f"{nm}_v.f16", V)
            lens[f"{nm}_prefill_len"] = int(K.shape[3])
        w16(d / "negtts_last.f16", last_hidden(pre["neg_tts_lm"]))
        lang = pt.stem.split("-")[0] if "-" in pt.stem else "en"
        meta = {"name": pt.stem, "language": lang, **lens}
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
        index.append(meta)
        print(f"  {pt.stem:24s} main={lens['main_prefill_len']:4d} "
              f"tts={lens['tts_prefill_len']:4d} neg={lens['neg_prefill_len']}")

    (out / "voices" / "index.json").write_text(json.dumps({"voices": index}, indent=2))
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"packed {len(index)} voices -> {out}  ({total:.0f} MB)")
    print("NOTE: fetch tokenizer.json + tokenizer_config.json from Qwen/Qwen2.5-0.5B into glue/ "
          "(the upload script does this).")


if __name__ == "__main__":
    main()
