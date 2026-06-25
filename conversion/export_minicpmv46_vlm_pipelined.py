"""Export the MiniCPM-V-4.6 PIPELINED VLM decode bundle (input_ids -> logits + a static
`image_embeds` buffer) — the on-device VLM design (Qwen3-VL static-buffer hook, simplified:
NO deepstack, NO mrope/rope-shift; plain 1D positions; qwen3_5 hybrid core w/ 4 states).

In-graph per token:  embedding = ids < V ? embed_tokens[ids] : image_embeds[ids - V]
The host rewrites the 64 image-token positions to `V + slot` (slot 0..63 in patch order) and
binds `image_embeds` [64,1024] as a persistent static buffer (EngineOptions.staticInputBuffers).
With image_embeds zeroed and no V+slot ids, the graph is a plain qwen3_5 text decoder.

Run (GPU; _GPU_LOCK held):
    coreai-models/.venv/bin/python ../coreai-models-community/conversion/export_minicpmv46_vlm_pipelined.py [int8lin|fp16]
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.qwen3_5 import (
    DECODE_STATE_NAMES,
    Qwen3_5Config,
    Qwen3_5DecodeCore,
    build_decode_state,
)
from coreai_models.primitives.macos.cache import KVCache

DTYPE = torch.float16
TEXT_PREFIX = "model.language_model."
N_IMAGE_TOKENS = 64  # 448px slice → 8×8 merged visual tokens
SNAP = glob.glob("/Users/majimadaisuke/.cache/huggingface/hub/"
                 "models--openbmb--MiniCPM-V-4.6/snapshots/*")[0]


def build_config() -> Qwen3_5Config:
    lt = (["linear_attention"] * 3 + ["full_attention"]) * 6
    return Qwen3_5Config(
        hidden_size=1024, num_hidden_layers=24, vocab_size=248094, intermediate_size=3584,
        rms_norm_eps=1e-6, tie_word_embeddings=True, head_dim=256, num_attention_heads=8,
        num_key_value_heads=2, attn_output_gate=True, partial_rotary_factor=0.25, rope_theta=1e7,
        linear_num_key_heads=16, linear_num_value_heads=16, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, full_attention_interval=4, layer_types=lt)


class MiniCPMV46VLMPipelined(nn.Module):
    """input_ids[1,s] + position_ids + image_embeds[N,h] + 4 hybrid states -> logits.
    Reuses the validated Qwen3_5DecodeCore (inputs_embeds->hidden) behind the embed gather."""

    def __init__(self, cfg: Qwen3_5Config, n_image_tokens: int = N_IMAGE_TOKENS):
        super().__init__()
        self.cfg = cfg
        self.N = n_image_tokens
        self.core = Qwen3_5DecodeCore(cfg)            # has .model (embed_tokens + layers + norm)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.core.model.embed_tokens.weight  # tied

    def forward(self, input_ids, position_ids, image_embeds,
                k_cache, v_cache, conv_state, rec_state):
        V, N = self.cfg.vocab_size, self.N
        b, s = input_ids.shape
        is_img = input_ids >= V
        slot = (input_ids - V).clamp(0, N - 1)
        e_txt = self.core.model.embed_tokens(input_ids.clamp(0, V - 1))
        e_img = image_embeds.index_select(0, slot.reshape(-1)).reshape(b, s, -1)
        inputs_embeds = torch.where(is_img.unsqueeze(-1), e_img.to(e_txt.dtype), e_txt)
        hidden = self.core(inputs_embeds, position_ids, k_cache, v_cache, conv_state, rec_state)
        return self.lm_head(hidden)


def linear_quant_config() -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {"op_state_spec": {"weight": {
            "dtype": "int8", "qscheme": "symmetric_with_clipping",
            "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None},
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rope.RoPE": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormPlusOne": None,
            "coreai_models.primitives.macos.rms_norm.RMSNormGated": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None},
        "module_name_configs": {r".*lm_head$": None},
    }


def head_quant_spec() -> dict:
    """int8 lm_head: big-vocab heads are fat-tailed -> plain symmetric (absmax), per-block-32."""
    return {"op_state_spec": {"weight": {
        "dtype": "int8", "qscheme": "symmetric",
        "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
        "op_input_spec": None, "op_output_spec": None}


def main() -> None:
    ap = argparse.ArgumentParser()
    # int8hu = int8 body + UNTIED int8 head (head fp16 in int8lin = ~half the per-token read; quantizing
    # it is the proven decode lever — iPhone 17 Pro measured +48% decode vs int8lin).
    ap.add_argument("mode", nargs="?", default="int8hu", choices=["fp16", "int8lin", "int8hu"])
    ap.add_argument("--max-ctx", type=int, default=4096)
    args = ap.parse_args()
    name = f"minicpmv46_vlm_decode_{args.mode}"
    cfg = build_config()

    model = MiniCPMV46VLMPipelined(cfg).eval()
    sd = {}
    with safe_open(glob.glob(SNAP + "/model.safetensors")[0], framework="pt", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            if k.startswith(TEXT_PREFIX):
                sd["core.model." + k[len(TEXT_PREFIX):]] = f.get_tensor(k).to(DTYPE)
    model.load_state_dict(sd, strict=False, assign=True)
    model.lm_head.weight = model.core.model.embed_tokens.weight
    model.core.model.reset_buffers()
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unloaded: {meta[:6]}")
    print(f"[load] {len(sd)} tensors (embed+transformer in-graph, tied head)")

    n_lin = 0
    for layer in model.core.model.layers:
        if not layer.is_full:
            layer.linear_attn.use_loopfree_step = True
            n_lin += 1
    print(f"[loopfree] {n_lin} linear layers")

    trace_past = 64
    input_ids = torch.randint(1, cfg.vocab_size, (1, 1), dtype=torch.int32)
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    image_embeds = torch.zeros(N_IMAGE_TOKENS, cfg.hidden_size, dtype=DTYPE)
    state = build_decode_state(cfg, max_seq_len=TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "input_ids": input_ids, "position_ids": position_ids, "image_embeds": image_embeds,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
        "conv_state": state["conv_state"], "rec_state": state["rec_state"]}
    seq_pos = torch.export.Dim("seq_pos", min=2, max=args.max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=args.max_ctx)
    dynamic_shapes = {
        "input_ids": None, "position_ids": {1: seq_pos}, "image_embeds": None,
        "k_cache": {KVCache.seq_len_dim(): k_seq}, "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None, "rec_state": None}

    if args.mode in ("int8lin", "int8hu"):
        from coreai_models.export.compression import quantize_pytorch_model
        cfg_q = linear_quant_config()
        if args.mode == "int8hu":
            model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())  # untie
            cfg_q["module_name_configs"] = {r".*lm_head$": head_quant_spec()}
            print("[quant] int8 per-block-32 + UNTIED int8 head (block32 symmetric) ...")
        else:
            print("[quant] int8 per-block-32 (fp16 tied head) ...")
        model = quantize_pytorch_model(model, tuple(reference_inputs.values()),
                                       dynamic_shapes, cfg_q)

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print("[export] pipelined VLM -> Core AI dialect ...")
    prog = export_to_coreai(
        model, reference_inputs, dynamic_shapes=dynamic_shapes,
        input_names=("input_ids", "position_ids", "image_embeds"),
        output_names=("logits",), state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    prog.optimize()

    out_dir = Path("/Users/majimadaisuke/code/coreai/coreai-models/exports") / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    import coreai.runtime as rt
    aimodel = out_dir / f"{name}.aimodel"
    print(f"[save] {aimodel}")
    prog.save_asset(aimodel, rt.AIModelAssetMetadata())
    (out_dir / "metadata.json").write_text(json.dumps({
        "metadata_version": "0.2", "kind": "llm", "name": name,
        "assets": {"main": f"{name}.aimodel"},
        "language": {"tokenizer": "openbmb/MiniCPM-V-4.6", "vocab_size": cfg.vocab_size,
                     "max_context_length": args.max_ctx, "embedded_tokenizer": True,
                     "function_map": {"main": ["main"]}},
        "vlm": {"image_embeds": f"[{N_IMAGE_TOKENS},{cfg.hidden_size}] static buffer; image ids = V+slot"},
        "source": {"hf_model_id": "openbmb/MiniCPM-V-4.6"},
        "compilation": {"date": datetime.now(timezone.utc).isoformat()}}, indent=2))
    tdir = out_dir / "tokenizer"
    tdir.mkdir()
    for fn in ("tokenizer.json", "tokenizer_config.json"):
        src = Path(SNAP) / fn
        if src.exists():
            shutil.copy(src, tdir / fn)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
