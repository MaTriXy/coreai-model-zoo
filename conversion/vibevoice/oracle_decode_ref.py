"""Non-streaming golden for the acoustic decoder gate (run in the oracle venv).

The oracle wav is built by the *streaming* decoder (per-frame chunks, some tail frames dropped after
EOS). For a clean component gate we want a non-streaming decode of a known latent stack. This:
  * loads the real acoustic tokenizer,
  * non-streaming-decodes the concatenated oracle latents (all frames, and the first `wav_frames`),
  * confirms non-streaming ~= the streamed wav (justifies shipping whole-sequence non-streaming decode),
  * saves artifacts/dec_ref.npz {latents_all[1,64,N], audio_full[1,1,3200N], audio_wavN, wav}.

  _oracle/.venv/bin/python oracle_decode_ref.py
"""
import os, warnings
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

HERE = Path(__file__).resolve().parent
MODEL = "microsoft/VibeVoice-Realtime-0.5B"
DEV = "mps"


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    n = min(a.numel(), b.numel())
    return torch.nn.functional.cosine_similarity(a[:n], b[:n], dim=0).item()


def main():
    z = np.load(HERE / "artifacts/oracle_ref.npz")
    N = int(z["num_latents"][0])
    lat = np.concatenate([z[f"latent{i}"] for i in range(N)], axis=0)  # (N,1,64)
    lat = torch.from_numpy(lat).permute(1, 2, 0).contiguous().to(DEV)   # (1,64,N)  scaled latents
    wav = z["wav"]; wav_frames = wav.shape[0] // 3200

    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="sdpa", device_map=None).to(DEV).eval()
    ac = model.model.acoustic_tokenizer

    with torch.no_grad():
        audio_full = ac.decode(lat, use_cache=False).float().cpu().numpy()          # (1,1,3200N)
        audio_wavN = ac.decode(lat[:, :, :wav_frames], use_cache=False).float().cpu().numpy()

    print(f"N latents={N}  wav={wav.shape} ({wav_frames} frames)  audio_full={audio_full.shape}")
    print(f"non-streaming(first {wav_frames}) vs streamed wav: cos={cos(audio_wavN, wav):.6f}")
    print(f"non-streaming(all {N})        len={audio_full.shape[-1]}")

    np.savez(str(HERE / "artifacts/dec_ref.npz"),
             latents_all=lat.cpu().numpy(), audio_full=audio_full,
             latents_wavN=lat[:, :, :wav_frames].cpu().numpy(), audio_wavN=audio_wavN, wav=wav)
    print(f"-> artifacts/dec_ref.npz")


if __name__ == "__main__":
    main()
