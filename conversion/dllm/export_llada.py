# Community port — NOT an Apple model.
"""Export the LLaDA-8B dLLM backbone to a Core AI .aimodel + engine-gate vs the fp32 torch overlay
(which is bit-exact vs the official model — gate_llada_torch.py — so the overlay IS the oracle).

ONE bundle: a fixed-seq BIDIRECTIONAL forward (no KV, no mask) — the dLLM canvas is always the full
length S, every denoising step is one full forward over it. input_ids[1,S] int32 -> logits[1,S,vocab].

  python export_llada.py --mode int4hu --seq 128         # body int4 per-block-32 + int8 head (iPhone ~5GB)
  python export_llada.py --mode int4lin --seq 128        # body int4 + fp16 head
  python export_llada.py --mode fp16 --seq 128           # validate export path / engine parity first

Engine gate: (1) engine logits cos vs fp32-torch on the canvas (fp16 ~1.0; int4 ~0.996, bar 0.99),
(2) drive the real decode loop through the engine and print the generated text (judge coherence — the
right metric for a diffusion LM, NOT token id match). Mac GPU.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "d3LLM_LLaDA"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "_ref"))

import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.export.compression import quantize_pytorch_model  # noqa: E402

from llada import load_backbone  # noqa: E402
from generate_llada import generate as my_generate  # noqa: E402
from gate_llada_torch import load_weights  # noqa: E402

# keep SDPA + (baked) RoPE decomposed in-graph instead of externalized — VoxCPM2 bidirectional idiom
_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]

ART = HERE / "artifacts"
MASK_ID = 126336
OVERWRITE = False


def cos(a, b) -> float:
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _du(p) -> str:
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    print(f"  saving {aim.name} ...", flush=True)
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


class LLaDAForward(nn.Module):
    """input_ids[1,S] (int32) -> logits[1,S,vocab]. Fixed-shape bidirectional full forward, no KV."""

    def __init__(self, bb):
        super().__init__()
        self.bb = bb

    def forward(self, input_ids):
        return self.bb(input_ids)


# weight-only quant config (coreai-opt PT2E). Body = int4/int8 per-block-32 (axis=1). Embedding kept
# fp16. lm_head (`^ff_out$`, anchored so block down-projs `blocks.N.ff_out` are NOT matched) = fp16 or int8.
def quant_config(body_dtype: str, head_dtype):
    cfg = {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {
                "dtype": body_dtype, "qscheme": "symmetric_with_clipping",
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None},
        # exclude non-Linear weights: my custom RMSNorm (rank-1 gain) + the token Embedding.
        "module_type_configs": {"llada.RMSNorm": None,
                                "torch.nn.modules.sparse.Embedding": None},
        "module_name_configs": {},
    }
    if head_dtype is None:
        cfg["module_name_configs"][r"^ff_out$"] = None                      # head stays fp16
    else:
        cfg["module_name_configs"][r"^ff_out$"] = {
            "op_state_spec": {"weight": {
                "dtype": head_dtype, "qscheme": "symmetric",                # absmax: big-vocab head is fat-tailed
                "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None}
    return cfg


async def run(mode: str, seq: int, prompt: str):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(WEIGHTS), trust_remote_code=True)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   add_generation_prompt=True, tokenize=False)
    ids = torch.tensor(tok(text)["input_ids"], dtype=torch.long).unsqueeze(0)
    L = ids.shape[1]
    if L >= seq:
        raise SystemExit(f"prompt_len {L} >= seq {seq}; raise --seq")
    gen = seq - L
    print(f"[export] mode={mode} seq={seq} prompt_len={L} gen={gen}")

    DT = torch.float16
    sd = load_weights(32)
    bb = load_backbone(sd, 32, seq, DT)
    del sd

    # fp32 oracle logits + decode tokens on a full canvas
    bb32 = bb  # fp16 overlay is the export source; compare engine vs this same torch overlay
    canvas = torch.full((1, seq), MASK_ID, dtype=torch.long)
    canvas[:, :L] = ids
    with torch.inference_mode():
        ref_logits = bb32(canvas).float()

    model = LLaDAForward(bb).eval()
    ref_ids = canvas.to(torch.int32)
    out_dir = ART / f"llada8b_{mode}_s{seq}"
    aim = out_dir / f"{out_dir.name}.aimodel"

    if aim.exists() and not OVERWRITE:
        print(f"  reusing existing bundle {out_dir.name} ({_du(aim)})", flush=True)
    else:
        if mode != "fp16":
            body = "int4" if mode.startswith("int4") else "int8"
            head_dtype = "int8" if mode.endswith("hu") else None   # int8 head for *hu, fp16 for *lin
            print(f"  quantizing: body={body} head={'int8' if head_dtype else 'fp16'} ...", flush=True)
            model = quantize_pytorch_model(model, (ref_ids,), {"input_ids": None},
                                           quant_config(body, head_dtype))
        prog = export_to_coreai(model, {"input_ids": ref_ids}, dynamic_shapes=None,
                                input_names=("input_ids",), output_names=("logits",), state_names=None)
        aim = _save(prog, out_dir)
        print(f"  -> {out_dir.name} ({_du(aim)})", flush=True)

    # ---- engine gate ----
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")

    async def engine_forward(x_long):
        r = await fn(inputs={"input_ids": rt.NDArray(np.ascontiguousarray(x_long.to(torch.int32).numpy()))})
        return torch.as_tensor(r["logits"].numpy().astype(np.float32))

    eng_logits = await engine_forward(canvas)
    c = cos(eng_logits[:, L:], ref_logits[:, L:])
    print(f"  engine vs fp16-torch logits cos = {c:.6f}  {'OK' if c >= 0.99 else 'LOW'}")

    # drive the real decode loop through the engine: my_generate is sync, so run it in a worker
    # thread and schedule each engine forward back onto this (free) event loop.
    main_loop = asyncio.get_running_loop()

    def blocking_forward(z):
        return asyncio.run_coroutine_threadsafe(engine_forward(z), main_loop).result()

    def do_decode():
        return my_generate(blocking_forward, ids, gen_length=gen, block_length=32,
                           threshold=0.5, mask_id=MASK_ID)

    x, nfe = await asyncio.to_thread(do_decode)
    gen_text = tok.decode(x[0, L:], skip_special_tokens=True)
    print(f"  engine decode (nfe={nfe}): {gen_text!r}")
    print(f"\n>>> {mode} export: logits cos={c:.6f} -> {'GATE PASS' if c >= 0.99 else 'GATE FAIL'} "
          f"(judge decode text for diffusion coherence)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int4hu", choices=["fp16", "int4lin", "int4hu", "int8lin", "int8hu"])
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--overwrite", action="store_true", help="re-export even if the bundle exists")
    ap.add_argument("--prompt", type=str,
                    default="Natalia sold clips to 48 friends in April, then half as many in May. "
                            "How many clips did she sell altogether? Reason step by step.")
    a = ap.parse_args()
    global OVERWRITE
    OVERWRITE = a.overwrite
    ART.mkdir(parents=True, exist_ok=True)
    asyncio.run(run(a.mode, a.seq, a.prompt))


if __name__ == "__main__":
    main()
