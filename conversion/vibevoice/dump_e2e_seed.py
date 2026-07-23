"""Dump everything the coreai-venv host needs to replicate generate() for one (voice, script)
without the vibevoice pkg: streamed text ids, per-stream prefill KV (seed the static-KV buffers),
and the neg-tts prefill last-hidden (the first speech token's CFG-negative condition).

  _oracle/.venv/bin/python dump_e2e_seed.py --voice ..._man.pt --script "Speaker 1: ..."

Layout (artifacts/e2e_seed.npz):
  tts_text_ids [1,Ttext]                     the script tokens streamed in windows of 5
  main_prefill_len, tts_prefill_len          int (72 / 253 for the sample)
  lm_k / lm_v      [4,1,2,72,64]             main LM prefill KV (post-RoPE, as cached upstream)
  tts_k / tts_v    [20,1,2,253,64]           tts LM prefill KV
  negtts_k/negtts_v[20,1,2,1,64]             neg tts LM prefill KV
  negtts_last_hidden [1,896]                 first speech token's CFG-negative condition
"""
import os, warnings, argparse
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

HERE = Path(__file__).resolve().parent
MODEL = "microsoft/VibeVoice-Realtime-0.5B"


def stack_kv(pk):
    ks = pk.key_cache if hasattr(pk, "key_cache") else [layer[0] for layer in pk]
    vs = pk.value_cache if hasattr(pk, "value_cache") else [layer[1] for layer in pk]
    K = torch.stack([k.detach().float() for k in ks], dim=0)  # (nl,1,nkv,L,hd)
    V = torch.stack([v.detach().float() for v in vs], dim=0)
    return K.numpy(), V.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=str(HERE / "_code/demo/voices/streaming_model/en-Frank_man.pt"))
    ap.add_argument("--script", default="Speaker 1: Hello, this is a quick test.")
    ap.add_argument("--out", default=str(HERE / "artifacts/e2e_seed.npz"))
    a = ap.parse_args()

    proc = VibeVoiceStreamingProcessor.from_pretrained(MODEL)
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prefill = torch.load(a.voice, map_location="cpu", weights_only=False)
    inputs = proc.process_input_with_cached_prompt(
        text=a.script.replace("’", "'"), cached_prompt=prefill, padding=True,
        return_tensors="pt", return_attention_mask=True)

    lm_k, lm_v = stack_kv(prefill["lm"].past_key_values)
    tts_k, tts_v = stack_kv(prefill["tts_lm"].past_key_values)
    nt_k, nt_v = stack_kv(prefill["neg_tts_lm"].past_key_values)

    out = {
        "tts_text_ids": inputs["tts_text_ids"].numpy().astype(np.int64),
        "main_prefill_len": np.array([lm_k.shape[3]], np.int64),
        "tts_prefill_len": np.array([tts_k.shape[3]], np.int64),
        "lm_k": lm_k, "lm_v": lm_v, "tts_k": tts_k, "tts_v": tts_v,
        "negtts_k": nt_k, "negtts_v": nt_v,
        "negtts_last_hidden": prefill["neg_tts_lm"].last_hidden_state[:, -1].detach().float().numpy(),
    }
    np.savez(a.out, **out)
    print("tts_text_ids", out["tts_text_ids"].shape, "->", out["tts_text_ids"].tolist())
    print("main_prefill_len", int(out["main_prefill_len"][0]), "tts_prefill_len", int(out["tts_prefill_len"][0]))
    print("lm_k", lm_k.shape, "tts_k", tts_k.shape, "negtts_k", nt_k.shape)
    print("-> ", a.out)


if __name__ == "__main__":
    main()
