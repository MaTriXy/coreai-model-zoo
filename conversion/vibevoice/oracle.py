"""VibeVoice-Realtime-0.5B torch oracle — golden per-stage fixtures for the Core AI port ladder.

Runs the upstream streaming inference model on a FIXED (voice, script, ddpm-steps, seed) and captures
the first-call INPUT+OUTPUT of every exportable submodule, the full DDPM trajectory of the first speech
token, the streaming speech latents, and the final 24 kHz waveform. Each Core AI overlay is later gated
cos>=0.999 against the matching fixture (mirrors conversion/{dots_tts,voxcpm}/oracle.py).

Export boundary captured here:
  * language_model      (main Qwen2, 4 lower layers + embeds) : inputs_embeds -> last_hidden_state
  * tts_language_model  (Qwen2, 20 upper layers)             : inputs_embeds -> last_hidden_state
  * prediction_head     (diffusion head, adaLN FF)           : (noisy[2,64], t[2], cond[2,896]) -> eps[2,64]
  * acoustic_connector  (SpeechConnector 64->896)            : latent -> embed
  * acoustic_tokenizer.decode (causal-conv VAE, streaming)   : scaled_latent -> audio_chunk
  * tts_eos_classifier  (BinaryClassifier 896->1)            : hidden -> logit
  * DDPM sampler        (DPMSolverMultistep, cosine, v-pred) : first-token step-by-step trajectory
  * speech_scaling_factor / speech_bias_factor scalars

Run in the oracle venv:
  _oracle/.venv/bin/python oracle.py --out artifacts --ddpm-steps 5
"""
import os, copy, argparse, warnings
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "microsoft/VibeVoice-Realtime-0.5B"


def to_np(x):
    if torch.is_tensor(x):
        return np.array(x.detach().to(torch.float32).cpu().numpy(), copy=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "artifacts"))
    ap.add_argument("--voice", default=os.path.join(HERE, "_code/demo/voices/streaming_model/en-Frank_man.pt"))
    ap.add_argument("--script", default="Speaker 1: Hello, this is a quick test.")
    ap.add_argument("--ddpm-steps", type=int, default=5)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dev", default="mps")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = args.dev
    fixtures: dict[str, np.ndarray] = {}
    seen: set[str] = set()

    def rec(key, val):
        if key in fixtures:
            return
        arr = to_np(val)
        if arr is not None:
            fixtures[key] = arr

    # ---- deterministic noise: spy on torch.randn (the DDPM initial noise per speech token) ----
    torch.manual_seed(args.seed)
    _orig_randn = torch.randn
    randn_log: list[np.ndarray] = []

    def randn_spy(*a, **k):
        t = _orig_randn(*a, **k)
        randn_log.append(to_np(t))
        return t
    torch.randn = randn_spy  # type: ignore

    proc = VibeVoiceStreamingProcessor.from_pretrained(MODEL)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa", device_map=None)
    model.to(dev).eval(); model.set_ddpm_inference_steps(num_steps=args.ddpm_steps)

    m = model.model
    rec("speech_scaling_factor", m.speech_scaling_factor)
    rec("speech_bias_factor", m.speech_bias_factor)

    handles = []

    def out_hook(name, keys=("last_hidden_state",)):
        def hook(mod, inp, output):
            if f"{name}._o" in seen:
                return
            seen.add(f"{name}._o")
            for i, t in enumerate(inp):
                rec(f"{name}.in{i}", t)
            if isinstance(output, torch.Tensor):
                rec(f"{name}.out", output)
            else:
                for a in keys:
                    v = getattr(output, a, None)
                    if v is not None:
                        rec(f"{name}.out_{a}", v)
        return hook

    handles.append(m.language_model.register_forward_hook(out_hook("mainlm")))
    handles.append(m.tts_language_model.register_forward_hook(out_hook("ttslm")))
    handles.append(m.acoustic_connector.register_forward_hook(out_hook("conn")))
    handles.append(model.tts_eos_classifier.register_forward_hook(out_hook("eos")))

    # ---- diffusion head: record the FIRST call (single DDPM step, batch=2 cond+uncond) ----
    ph = m.prediction_head
    _orig_ph = ph.forward

    def ph_wrap(noisy_images, timesteps, condition):
        eps = _orig_ph(noisy_images, timesteps, condition)
        if "dhead._o" not in seen:
            seen.add("dhead._o")
            rec("dhead.in_noisy", noisy_images)
            rec("dhead.in_t", timesteps)
            rec("dhead.in_cond", condition)
            rec("dhead.out_eps", eps)
        return eps
    ph.forward = ph_wrap

    # ---- DDPM sampler: capture the FULL first-token trajectory (init noise, per-step eps + prev_sample) ----
    traj = {"n": 0, "done": False, "steps": []}
    _orig_sample = model.sample_speech_tokens

    def sample_wrap(condition, neg_condition, cfg_scale=3.0):
        first = not traj["done"]
        if first:
            rec("ddpm.cond_pos", condition)
            rec("ddpm.cond_neg", neg_condition)
        # replicate to intercept scheduler.step for the first token only
        model.model.noise_scheduler.set_timesteps(model.ddpm_inference_steps)
        cond = torch.cat([condition, neg_condition], dim=0).to(model.model.prediction_head.device)
        speech = torch.randn(cond.shape[0], model.config.acoustic_vae_dim).to(cond)
        if first:
            rec("ddpm.init_noise", speech)
            fixtures["ddpm.timesteps"] = np.asarray(
                [float(t) for t in model.model.noise_scheduler.timesteps], dtype=np.float32)
        for si, t in enumerate(model.model.noise_scheduler.timesteps):
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = model.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=cond)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech_next = model.model.noise_scheduler.step(eps, t, speech).prev_sample
            if first:
                rec(f"ddpm.step{si}_in", speech)
                rec(f"ddpm.step{si}_eps", eps[: len(eps) // 2])
                rec(f"ddpm.step{si}_out", speech_next)
            speech = speech_next
        result = speech[: len(speech) // 2]
        if first:
            rec("ddpm.final_latent", result)
            traj["done"] = True
        return result
    model.sample_speech_tokens = sample_wrap

    # ---- acoustic decode: wrap first call (scaled_latent -> audio_chunk) + record every latent ----
    ac = m.acoustic_tokenizer
    _orig_decode = ac.decode
    latents_log: list[np.ndarray] = []

    def decode_wrap(latents, **k):
        audio = _orig_decode(latents, **k)
        latents_log.append(to_np(latents))
        if "dec._o" not in seen:
            seen.add("dec._o")
            rec("dec.in_scaled_latent", latents)
            rec("dec.out_audio", audio)
        return audio
    ac.decode = decode_wrap

    # ---- load voice prefill cache + build inputs ----
    with torch.serialization.safe_globals([BaseModelOutputWithPast, DynamicCache]):
        prefill = torch.load(args.voice, map_location=dev, weights_only=False)
    inputs = proc.process_input_with_cached_prompt(
        text=args.script.replace("’", "'"), cached_prompt=prefill, padding=True,
        return_tensors="pt", return_attention_mask=True)
    inputs = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    print("=== generation inputs ===")
    for k, v in inputs.items():
        print(f"  {k}: {tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}")

    out_gen = model.generate(**inputs, max_new_tokens=None, cfg_scale=args.cfg, tokenizer=proc.tokenizer,
                             generation_config={"do_sample": False}, verbose=False, show_progress_bar=False,
                             all_prefilled_outputs=copy.deepcopy(prefill))

    for h in handles:
        h.remove()
    ph.forward = _orig_ph
    ac.decode = _orig_decode
    model.sample_speech_tokens = _orig_sample
    torch.randn = _orig_randn  # type: ignore

    # ---- outputs: waveform + speech latents ----
    wav = out_gen.speech_outputs[0]
    fixtures["wav"] = to_np(wav).reshape(-1)
    for i, lat in enumerate(latents_log):
        fixtures[f"latent{i}"] = lat
    fixtures["num_latents"] = np.array([len(latents_log)])
    for i, z in enumerate(randn_log[:64]):
        fixtures[f"randn{i}"] = z
    fixtures["num_randn"] = np.array([len(randn_log)])
    fixtures["ddpm_steps"] = np.array([args.ddpm_steps])
    fixtures["cfg"] = np.array([args.cfg])

    np.savez(str(out / "oracle_ref.npz"), **fixtures)
    import soundfile as sf
    sf.write(str(out / "oracle.wav"), fixtures["wav"], 24000)

    print("\n=== ORACLE CAPTURED ===")
    for k in sorted(fixtures):
        if (k.startswith("randn") and k != "randn0") or (k.startswith("latent") and k not in ("latent0",)):
            continue
        v = fixtures[k]
        print(f"  {k:28s} shape={tuple(v.shape)}")
    print(f"num latents={len(latents_log)}  num randn={len(randn_log)}  wav={fixtures['wav'].shape}")
    print(f"-> {out/'oracle_ref.npz'}  +  oracle.wav (24kHz)")


if __name__ == "__main__":
    main()
