"""P2 decoder weight-map check: my Qwen3ASRDecoderPipelined (zoo qwen3 from_hf) logits vs the
official thinker decoder, single prefill on the ja1 prompt (fp32). Confirms the checkpoint remap
(fused qkv / qk_norm, tied head, rope_scaling stripped) is correct, decoupled from the engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

OFFICIAL = "/tmp/qwen3-asr-official"
sys.path.insert(0, OFFICIAL)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration  # noqa: E402
from qwen3_asr_decoder import Qwen3ASRDecoderPipelined  # noqa: E402

MODEL = "Qwen/Qwen3-ASR-1.7B"
AUDIO_TOKEN_ID = 151676
V = 151936
CACHE_LEN = 128
OUTDIR = Path(__file__).resolve().parent


@torch.no_grad()
def main() -> None:
    d = np.load(OUTDIR / "oracle_tokens.npz")
    ids = torch.from_numpy(d["input_ids"]).long()           # [1, 70]
    audio = torch.from_numpy(d["encoder_out"]).float()      # [N, 2048] golden encoder
    N = audio.shape[0]
    golden_first = int(d["gen_ids"][0])                     # 11528

    # --- official reference logits (last position) ---
    print("loading official model (fp32) ...", flush=True)
    off = Qwen3ASRForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    th = off.thinker
    emb = th.get_input_embeddings()(ids)
    mask = (ids == AUDIO_TOKEN_ID).unsqueeze(-1).expand_as(emb)
    emb = emb.masked_scatter(mask, audio.to(emb.dtype))
    attn = torch.ones_like(ids)
    ref_logits = th(inputs_embeds=emb, attention_mask=attn).logits[0, -1].float()

    # --- my decoder ---
    print("loading my decoder (zoo qwen3 from_hf, fp32) ...", flush=True)
    dec = Qwen3ASRDecoderPipelined.from_hf(MODEL, n_audio_tokens=N, target_dtype=torch.float32)

    # rewrite audio positions to V + slot (0..N-1, in order)
    ids2 = ids.clone()
    aud_pos = (ids[0] == AUDIO_TOKEN_ID).nonzero(as_tuple=True)[0]
    assert len(aud_pos) == N, f"{len(aud_pos)} != {N}"
    ids2[0, aud_pos] = V + torch.arange(N)
    s = ids2.shape[1]
    pos = torch.arange(s, dtype=torch.int32).unsqueeze(0)
    cfg = dec.config
    kc = torch.zeros(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, CACHE_LEN, cfg.head_dim)
    vc = torch.zeros_like(kc)
    mine_logits = dec(ids2, pos, audio, kc, vc)[0, -1].float()

    cos = torch.nn.functional.cosine_similarity(mine_logits, ref_logits, dim=0).item()
    top1_mine = int(mine_logits.argmax()); top1_ref = int(ref_logits.argmax())
    print(f"\n=== DECODER LOGITS PARITY (last pos) ===")
    print(f"cos={cos:.6f}  max|Δ|={(mine_logits-ref_logits).abs().max():.4e}")
    print(f"argmax mine={top1_mine}  ref={top1_ref}  golden_first={golden_first}")
    ok = cos > 0.999 and top1_mine == top1_ref == golden_first
    print("PASS" if ok else "FAIL", flush=True)


if __name__ == "__main__":
    main()
