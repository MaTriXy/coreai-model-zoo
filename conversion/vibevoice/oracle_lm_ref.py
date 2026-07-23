"""Golden hidden states for the two Qwen2 backbones (run in the oracle venv).

Fresh causal prefill (no voice cache) of a fixed inputs_embeds sequence through the upstream
language_model (4L, norm=Identity) and tts_language_model (20L, norm=real). The standalone
backbone.py is later gated cos>=0.999 against these — validates RoPE/GQA/qkv-bias/eps/norm.

  _oracle/.venv/bin/python oracle_lm_ref.py
"""
import os, warnings
os.environ["HF_HUB_DISABLE_XET"] = "1"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference

HERE = Path(__file__).resolve().parent
MODEL = "microsoft/VibeVoice-Realtime-0.5B"
DEV = "cpu"  # fp32 CPU for a clean numeric golden (no MPS fp drift)


def main():
    torch.manual_seed(0)
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager", device_map=None).to(DEV).eval()
    m = model.model
    T = 40
    ids = torch.arange(2, 2 + T, dtype=torch.long, device=DEV).unsqueeze(0)
    emb = m.get_input_embeddings()(ids)  # (1,T,896)
    attn = torch.ones(1, T, dtype=torch.long, device=DEV)

    with torch.no_grad():
        main_out = m.language_model(inputs_embeds=emb, attention_mask=attn, use_cache=False,
                                    output_hidden_states=False, return_dict=True).last_hidden_state
        tts_out = m.tts_language_model(inputs_embeds=emb, attention_mask=attn, use_cache=False,
                                       output_hidden_states=False, return_dict=True).last_hidden_state

    np.savez(str(HERE / "artifacts/lm_ref.npz"),
             emb=emb.detach().float().numpy(), main_hidden=main_out.detach().float().numpy(),
             tts_hidden=tts_out.detach().float().numpy())
    print(f"emb {tuple(emb.shape)}  main_hidden {tuple(main_out.shape)}  tts_hidden {tuple(tts_out.shape)}")
    print(f"main_hidden std={main_out.std():.4f}  tts_hidden std={tts_out.std():.4f}")
    print("-> artifacts/lm_ref.npz")


if __name__ == "__main__":
    main()
