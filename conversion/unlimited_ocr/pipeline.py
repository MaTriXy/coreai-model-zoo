"""CAPSTONE: full image->markdown OCR via the Core AI engine, end-to-end, glued.

  preprocessed image [1,3,640,640]
    -> DeepEncoder .aimodel            -> visual tokens [1,100,1280]   (fp16, engine)
    -> arrange (10x10 + image_newline rows + view_seperator) -> [111,1280]
    -> scatter into embed_tokens(prompt_input_ids)           -> prefix [1,115,1280]
    -> unified decoder bundle: prefill(prefix) + decode loop  -> token ids
    -> detokenize                                            -> markdown

This is the exact recipe the Swift app implements (assets in out/_swift_assets/). Uses the
validated preprocessed input sam_in0 (raw-image->640 pad+normalize 0.5 is done host-side).

Run:  cd ~/code/coreai/coreai-models && PYTHONPATH=../_unlimited_ocr \
          .venv/bin/python ../_unlimited_ocr/pipeline.py [--no-repeat-ngram 35]
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

import numpy as np
import torch

from model import DECODE_STATE_NAMES, build_decode_state, ocr_decoder_from_safetensors

HERE = Path(__file__).resolve().parent
CKPT = HERE / "ckpt"
ASSETS = HERE / "out/_swift_assets"
VISION = HERE / "out/_vision_export/_vision_export.aimodel"
DECODER = HERE / "out/_dec_unified/_dec_unified.aimodel"
CACHE_LEN, DTYPE, EOS, IMG_TOK = 2048, torch.float16, 1, 128815


def argmax_blocked(logits, gen, n):
    if n <= 0 or len(gen) < n - 1:
        return int(np.argmax(logits))
    order = np.argsort(logits)[::-1]
    pre = tuple(gen[-(n - 1):])
    banned = {gen[i + n - 1] for i in range(len(gen) - n + 1) if tuple(gen[i:i + n - 1]) == pre}
    return next((int(t) for t in order if int(t) not in banned), int(order[0]))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-repeat-ngram", type=int, default=35)
    ap.add_argument("--max", type=int, default=600)
    args = ap.parse_args()

    import coreai.runtime as rt
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(CKPT / "tokenizer.json"))
    model = ocr_decoder_from_safetensors(str(CKPT), target_dtype=DTYPE)  # embed table
    image_newline = torch.from_numpy(np.fromfile(ASSETS / "image_newline.f16", np.float16).astype(np.float32))
    view_seperator = torch.from_numpy(np.fromfile(ASSETS / "view_seperator.f16", np.float16).astype(np.float32))
    prompt_ids = torch.from_numpy(np.fromfile(ASSETS / "prompt_input_ids.i32", np.int32).astype(np.int64))  # [115]
    sam_in0 = np.load(HERE / "out/_vision_oracle/vision_tensors.npz")["sam_in0"].astype(np.float16)  # [1,3,640,640]

    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    # 1) vision encoder (engine, fp16)
    vm = await rt.AIModel.load(str(VISION), gpu)
    vout = await vm.load_function("main")(inputs={"image": rt.NDArray(np.ascontiguousarray(sam_in0))})
    patches = torch.from_numpy(vout["visual_tokens"].numpy().astype(np.float32))  # [1,100,1280]
    del vm

    # 2) arrange -> 111 visual features ; 3) scatter into embedded prompt -> prefix
    n = patches.shape[-1]
    g = patches.reshape(10, 10, n)
    g = torch.cat([g, image_newline.view(1, 1, n).expand(10, 1, n)], dim=1)  # [10,11,1280]
    visual = torch.cat([g.reshape(-1, n), view_seperator.view(1, n)], dim=0)  # [111,1280]
    with torch.no_grad():
        prefix = model.model.embed_tokens(prompt_ids.unsqueeze(0)).float()  # [1,115,1280]
    prefix[0, prompt_ids == IMG_TOK] = visual
    prefix = prefix.to(DTYPE)

    # 4) decoder: prefill + greedy decode
    dm = await rt.AIModel.load(str(DECODER), gpu)
    fn_prefill, fn_decode = dm.load_function("prefill"), dm.load_function("decode")
    state = {DECODE_STATE_NAMES[0]: rt.NDArray(np.ascontiguousarray(build_decode_state(model.config, CACHE_LEN, DTYPE)["k_cache"].numpy())),
             DECODE_STATE_NAMES[1]: rt.NDArray(np.ascontiguousarray(build_decode_state(model.config, CACHE_LEN, DTYPE)["v_cache"].numpy()))}
    Lm = prefix.shape[1]
    t0 = time.time()
    r = await fn_prefill(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(prefix.numpy()))}, state=state)
    gen, tk = [], argmax_blocked(r["logits"].numpy().astype(np.float32).reshape(-1), [], args.no_repeat_ngram)
    with torch.no_grad():
        for p in range(Lm, Lm + args.max):
            gen.append(tk)
            if tk == EOS:
                break
            emb = model.model.embed_tokens(torch.tensor([[tk]])).to(DTYPE)
            res = await fn_decode(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(emb.numpy())),
                                          "pos": rt.NDArray(np.ascontiguousarray(np.array([p], np.int32)))}, state=state)
            tk = argmax_blocked(res["logits"].numpy().astype(np.float32).reshape(-1), gen, args.no_repeat_ngram)
    dt = time.time() - t0

    md = tok.decode([t for t in gen if t != EOS], skip_special_tokens=False)
    (HERE / "out/_pipeline_ocr.md").write_text(md)
    print(f"[pipeline] image->markdown via engine: {len(gen)} tokens, {dt:.1f}s ({len(gen)/dt:.0f} tok/s), "
          f"EOS={'yes' if gen[-1]==EOS else 'NO'}, fp16 vision .aimodel", flush=True)
    print("=" * 70 + "\n" + md)


if __name__ == "__main__":
    asyncio.run(main())
