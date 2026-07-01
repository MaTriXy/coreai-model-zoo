"""P1 end-to-end parity: my static encoder -> masked_scatter -> Qwen3 decode == golden gen_ids.

Proves AuTEncoderStatic is a drop-in: builds inputs_embeds from the oracle prompt ids, scatters
MY encoder's 55 tokens at the audio positions, runs the (standard Qwen3) thinker decoder greedily,
and checks the generated ids match oracle_tokens.npz['gen_ids'] token-for-token.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

OFFICIAL = "/tmp/qwen3-asr-official"
sys.path.insert(0, OFFICIAL)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_encoder import AuTEncoderStatic, build_attn_bias  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_TOKEN_ID = 151676
EOS = [151643, 151645]
OUTDIR = Path(__file__).resolve().parent


@torch.no_grad()
def main() -> None:
    d = np.load(OUTDIR / "oracle_tokens.npz")
    input_ids = torch.from_numpy(d["input_ids"]).long()              # [1, 70]
    input_features = torch.from_numpy(d["input_features"]).float()   # [1, 128, 417]
    golden_gen = torch.from_numpy(d["gen_ids"]).long().tolist()
    N = int(d["encoder_out"].shape[0])

    print("loading model ...", flush=True)
    model = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    thinker = model.thinker
    tower = thinker.audio_tower

    chunk_mel = tower.n_window * 2
    K = math.ceil(input_features.shape[-1] / chunk_mel)
    enc = AuTEncoderStatic(tower, K).eval()
    mel = torch.nn.functional.pad(input_features, (0, K * chunk_mel - input_features.shape[-1]))
    bias = build_attn_bias(K * enc.tok_per_chunk, N, window=enc.window)
    audio_embeds = enc(mel, bias)[:N].to(torch.float32)             # [N, 2048] from MY encoder

    inputs_embeds = thinker.get_input_embeddings()(input_ids)        # [1, 70, 2048]
    mask = (input_ids == AUDIO_TOKEN_ID).unsqueeze(-1).expand_as(inputs_embeds)
    inputs_embeds = inputs_embeds.masked_scatter(mask, audio_embeds.to(inputs_embeds.dtype))
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)

    out = thinker.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=128,
        do_sample=False,
        eos_token_id=EOS,
        pad_token_id=151643,
    )
    gen = out[0].tolist()
    # generate() with inputs_embeds returns only newly generated ids
    print(f"\nmine  ({len(gen)}): {gen}", flush=True)
    print(f"golden({len(golden_gen)}): {golden_gen}", flush=True)
    match = gen[:len(golden_gen)] == golden_gen
    print("\n=== DECODE PARITY ===")
    print("PASS — token-for-token match" if match else "FAIL — mismatch", flush=True)


if __name__ == "__main__":
    main()
