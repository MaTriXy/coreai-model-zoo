"""Architecture trace for VibeVoice-Realtime-0.5B: hook the key submodules during one short
generation and record their input/output shapes + dtypes — this fixes the Core AI export boundary
(main LM / tts LM / diffusion prediction head / acoustic tokenizer decode).

Run in the oracle venv:  _oracle/.venv/bin/python capture_arch.py
"""
import os, copy, warnings
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "microsoft/VibeVoice-Realtime-0.5B"
VOICE = os.path.join(HERE, "_code/demo/voices/streaming_model/en-Frank_man.pt")
SCRIPT = "Speaker 1: Hello, this is a quick test."
DEV = "mps"


def shp(x):
    if torch.is_tensor(x): return f"{tuple(x.shape)}:{str(x.dtype).replace('torch.','')}"
    if isinstance(x, (list, tuple)): return "[" + ", ".join(shp(e) for e in x[:3]) + ("..." if len(x) > 3 else "") + "]"
    if hasattr(x, "last_hidden_state"): return f"OutputWithPast(hs={shp(x.last_hidden_state)})"
    return type(x).__name__


def main():
    proc = VibeVoiceStreamingProcessor.from_pretrained(MODEL)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa", device_map=None)
    model.to(DEV).eval(); model.set_ddpm_inference_steps(num_steps=5)

    seen = {}
    def hook(name):
        def f(mod, inp, out):
            if name not in seen:   # first call only
                seen[name] = (f"in={shp(inp)}", f"out={shp(out)}")
        return f
    m = model.model
    targets = {
        "language_model(main Qwen2.5)": m.language_model,
        "tts_language_model": getattr(m, "tts_language_model", None),
        "prediction_head(diffusion)": getattr(m, "prediction_head", None),
        "acoustic_tokenizer": getattr(m, "acoustic_tokenizer", None),
        "acoustic_tokenizer.decoder": getattr(getattr(m, "acoustic_tokenizer", None), "decoder", None),
        "tts_eos_classifier": getattr(model, "tts_eos_classifier", None),
    }
    handles = [mod.register_forward_hook(hook(n)) for n, mod in targets.items() if mod is not None]

    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prefill = torch.load(VOICE, map_location=DEV, weights_only=False)
    inputs = proc.process_input_with_cached_prompt(
        text=SCRIPT.replace("’", "'"), cached_prompt=prefill, padding=True,
        return_tensors="pt", return_attention_mask=True)
    inputs = {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    print("=== generation inputs ===")
    for k, v in inputs.items(): print(f"  {k}: {shp(v)}")
    print("=== prefill (voice cache) keys ===", list(prefill.keys()) if isinstance(prefill, dict) else type(prefill))

    out = model.generate(**inputs, max_new_tokens=None, cfg_scale=1.5, tokenizer=proc.tokenizer,
                         generation_config={"do_sample": False}, verbose=False,
                         all_prefilled_outputs=copy.deepcopy(prefill))
    for h in handles: h.remove()

    print("\n=== component I/O (first call) — the export boundary ===")
    for n in targets:
        if n in seen: print(f"[{n}]\n    {seen[n][0]}\n    {seen[n][1]}")
    print("\n=== outputs ===")
    print("  speech_outputs[0]:", shp(out.speech_outputs[0]))
    print("  sequences:", shp(out.sequences))
    # count DDPM head calls (sanity on the sampling loop)
    print("\n=== module param counts (M) ===")
    for n, mod in targets.items():
        if mod is not None:
            p = sum(x.numel() for x in mod.parameters()) / 1e6
            print(f"  {n}: {p:.1f}M")


if __name__ == "__main__":
    main()
