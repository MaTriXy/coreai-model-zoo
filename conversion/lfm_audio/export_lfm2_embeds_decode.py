"""LFM2.5-Audio — Milestone A finish: the LFM2 backbone as an EMBEDS-INPUT decode
graph (one Core AI bundle for prefill+decode), driven end-to-end for on-device ASR.

Mirrors `export_lfm2_decode_pipelined.py` but the query arrives already embedded
(`inputs_embeds` [1,1,hidden]) instead of `input_ids` — LFM2-Audio assembles a mixed
text/audio prompt on the host, so the backbone cannot gather its own embeddings.
Uses `Lfm2EmbedsForCausalLMStateful` (overlay) + a `lfm.`-prefix weight loader that
pulls the LFM2-1.2B backbone out of the AUDIO checkpoint's `model.safetensors`
(arch == shipped LFM2.5-1.2B; weights differ = LFM2-1.2B base).

The graph is S=1 static-query (like the shipped decode bundle): prefill runs as S=1
chunked steps, decode as S=1 steps, KV grows in a fixed 2048-slot buffer, the conv
state is one fixed-shape extra state. Loop-free (no SSM while_loop).

Gate (both are prefill+greedy vs the liquid_audio oracle `asr_ref.npz`, 14 tokens):
  * source=oracle : prefill embeds = the oracle's own assembled `in_emb`
                    -> isolates the LFM2 embeds-decode GRAPH.
  * source=coreai : prefill embeds = lfm.embed_tokens(text_ids) at TEXT positions
                    + the Core AI encoder's audio_emb at AUDIO positions, keyed by
                    modality_flag -> the TRUE on-Core-AI ASR path.

This module is a LIBRARY (helpers only). Milestone-A Mac-GPU gate workflow (the plain
.aimodel JIT-thrashes the ANE compiler on Mac GPU, so gate through AOT):
  1. export:  coreai-models/.venv/bin/python export_worker.py fp16 artifacts/<name>
  2. AOT:     DEVELOPER_DIR=~/Downloads/Xcode-beta.app/Contents/Developer xcrun coreai-build compile \
                artifacts/<name>/<name>.aimodel --output artifacts/aot \
                --platform macOS --preferred-compute gpu --expect-frequent-reshapes --architecture h16c
  3. gate:    coreai-models/.venv/bin/python aot_gate.py artifacts/aot/<name>.h16c.aimodelc
`eager_greedy` (below) is the pure-torch reference; `device_greedy` drives the plain
.aimodel raw (kept for the iOS-AOT / fixed-OS track — DO NOT use it for the Mac GPU
gate, it re-JITs per position length and hangs the ANE compiler).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # conversion/
from _paths import hf_snapshot  # noqa: E402

import torch

# NOTE: numpy, asyncio and coreai.runtime are imported LAZILY (inside the
# functions that use them). Importing numpy/asyncio at module top — in the SAME
# module whose entrypoint calls export_to_coreai — makes the converter abort with
# "interleave must have rank (1) elements" (the shipped decode export, which
# works as a script, imports neither at top). coreai.runtime is likewise imported
# only after the converter has run.
from coreai_models.export._constants import TRACE_KV_CACHE_SEQ_LEN
from coreai_models.export.macos import _EXTERNALIZE_SPECS, export_to_coreai
from coreai_models.models.macos.lfm2 import (
    DECODE_STATE_NAMES,
    Lfm2EmbedsForCausalLMStateful,
    _MLP_KEY_MAP,
    build_decode_state,
    lfm2_config_from_dict,
)
from coreai_models.primitives.macos.cache import KVCache

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]                      # .../coreai
ORACLE = REPO / "_lfmaudio_oracle"          # parity refs from _lfmaudio_dump_asr.py
SNAP = hf_snapshot("LiquidAI/LFM2.5-Audio-1.5B")
CKPT = str(Path(SNAP) / "model.safetensors")
CFG_JSON = str(Path(SNAP) / "config.json")

DTYPE = torch.float16
TEXT, AUDIO_IN, EOS = 1, 2, 7               # modality enum + IM_END/eos (config)
_ATTN_RE = re.compile(r"\.self_attn\.(q_proj|k_proj|v_proj|out_proj)\.weight$")


# --------------------------------------------------------------------------- weights
def load_lfm_config():
    with open(CFG_JSON) as f:
        return lfm2_config_from_dict(json.load(f)["lfm"])


def load_backbone(cfg, target_dtype=DTYPE, fp32_attn_proj=True):
    """Load the LFM2 backbone from the audio checkpoint's `lfm.`-prefixed weights
    into `Lfm2EmbedsForCausalLMStateful`. Strip the `lfm.` prefix, apply the HF
    w1/w3/w2 -> gate/up/down MLP rename, keep the attention projections fp32 on an
    fp16 load (GPU-delegate exactness; see the lfm2 module header)."""
    from safetensors import safe_open

    with torch.device("meta"):
        model = Lfm2EmbedsForCausalLMStateful(cfg)
    model.to(dtype=target_dtype)

    sd: dict[str, torch.Tensor] = {}
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            if not key.startswith("lfm."):
                continue
            local = key[len("lfm."):]
            for old, new in _MLP_KEY_MAP.items():
                if old in local:
                    local = local.replace(old, new)
                    break
            t = f.get_tensor(key)
            dt = torch.float32 if (fp32_attn_proj and _ATTN_RE.search(local)) else target_dtype
            if t.dtype != dt:
                t = t.to(dt)
            sd["model." + local] = t

    model.load_state_dict(sd, assign=True, strict=False)
    if cfg.tie_embedding:
        model.lm_head.weight = model.model.embed_tokens.weight
    model.model.reset_buffers()
    meta = [n for n, p in model.named_parameters() if p.is_meta]
    if meta:
        raise RuntimeError(f"unloaded (meta) params: {meta[:6]}")
    model.eval()
    return model


def embed_table():
    """lfm.embed_tokens.weight [vocab, hidden] fp32 — host re-embedding of decode tokens."""
    from safetensors import safe_open
    with safe_open(CKPT, framework="pt", device="cpu") as f:
        return f.get_tensor("lfm.embed_tokens.weight").float()


# --------------------------------------------------------------------------- prefill embeds
def prefill_embeds(source: str) -> np.ndarray:
    """Assemble the [S, hidden] prefill embeds for the ASR prompt.

    oracle -> the oracle's own `in_emb` (isolates the LFM2 graph).
    coreai -> lfm.embed_tokens(text_ids) at TEXT slots + the Core AI encoder's
              audio_emb at AUDIO slots, keyed by modality_flag (true Core AI path).
    """
    import numpy as np
    d = np.load(ORACLE / "asr_pathref.npz")
    if source == "oracle":
        return d["in_emb"].astype(np.float32)               # [67, 2048]

    mflag = d["modality_flag"][0]                            # [67]
    text_ids = torch.from_numpy(d["text_ids"][0]).long()    # [19]
    text_emb = torch.nn.functional.embedding(text_ids, embed_table()).numpy()  # [19,2048]
    audio_emb = np.load(ORACLE / "coreai_audio_emb.npz")["emb"].astype(np.float32)  # [48,2048]
    out = np.zeros((mflag.shape[0], text_emb.shape[1]), np.float32)
    ti = ai = 0
    for p, m in enumerate(mflag):
        if m == AUDIO_IN:
            out[p] = audio_emb[ai]; ai += 1
        else:
            out[p] = text_emb[ti]; ti += 1
    assert ti == text_emb.shape[0] and ai == audio_emb.shape[0], (ti, ai)
    return out


# --------------------------------------------------------------------------- greedy drivers
def eager_greedy(model, pe, embed_w: torch.Tensor, n_gen: int) -> list[int]:
    """Batched prefill ([1,S,H]) + greedy decode, states mutated in place."""
    cfg = model.config
    st = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    k, v, conv = st["k_cache"], st["v_cache"], st["conv_state"]
    S = pe.shape[0]
    ew = embed_w.to(DTYPE)
    ids: list[int] = []
    with torch.no_grad():
        pos = torch.arange(S, dtype=torch.int32)[None]
        ie = torch.from_numpy(pe).to(DTYPE)[None]           # [1,S,H]
        logits = model(ie, pos, k, v, conv)
        t = int(logits[0, -1].argmax()); ids.append(t)
        for i in range(n_gen - 1):
            pos = torch.arange(S + i + 1, dtype=torch.int32)[None]
            ie = ew[t].view(1, 1, -1)
            logits = model(ie, pos, k, v, conv)
            t = int(logits[0, -1].argmax()); ids.append(t)
    return ids


def _nd(a):
    import numpy as np
    import coreai.runtime as rt
    return rt.NDArray(np.ascontiguousarray(a))


async def device_greedy(aimodel: Path, cfg, pe, embed_w: torch.Tensor,
                        n_gen: int, unit: str) -> list[int]:
    """S=1 chunked prefill + greedy decode on the engine (raw runtime, in-place states)."""
    import numpy as np
    import coreai.runtime as rt
    opts = (rt.SpecializationOptions.cpu_only() if unit == "cpu"
            else rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
    m = await rt.AIModel.load(str(aimodel), opts)
    fn = m.load_function("main")
    st = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    state = {n: _nd(t.numpy()) for n, t in
             zip(DECODE_STATE_NAMES, [st["k_cache"], st["v_cache"], st["conv_state"]])}
    S = pe.shape[0]
    pe16 = pe.astype(np.float16)
    ew16 = embed_w.to(DTYPE).numpy()

    async def step(emb_row, total_positions):
        ie = _nd(emb_row.reshape(1, 1, -1).astype(np.float16))
        pos = _nd(np.arange(total_positions, dtype=np.int32)[None])
        out = await fn(inputs={"inputs_embeds": ie, "position_ids": pos}, state=state)
        return out["logits"].numpy()

    logits = None
    for i in range(S):                                      # s=1 chunked prefill
        logits = await step(pe16[i], i + 1)
    t = int(logits[0, -1].argmax()); ids = [t]
    for i in range(n_gen - 1):                              # greedy decode
        logits = await step(ew16[t], S + i + 1)
        t = int(logits[0, -1].argmax()); ids.append(t)
    return ids


# --------------------------------------------------------------------------- quant (mirror shipped)
def linear_quant_config(dtype: str = "int8", block: int = 32) -> dict:
    def spec(d, b):
        return {"op_state_spec": {"weight": {
            "dtype": d, "qscheme": "symmetric_with_clipping",
            "granularity": {"type": "per_block", "block_size": b, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None}
    return {
        "execution_mode": "eager",
        "global_config": spec(dtype, block),
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "coreai_models.primitives.macos.rms_norm.RMSNorm": None,
            "torch.nn.modules.sparse.Embedding": None,
            "torch.nn.modules.conv.Conv1d": None,
        },
        "module_name_configs": {
            r".*lm_head$": None,
            r".*self_attn\.(q_proj|k_proj|v_proj|out_proj)$": None,
        },
    }


# --------------------------------------------------------------------------- export
def export_bundle(model, cfg, mode: str, max_ctx: int, out_dir: Path) -> Path:
    inputs_embeds = torch.randn(1, 1, cfg.hidden_size, dtype=DTYPE)
    trace_past = 64
    position_ids = torch.arange(trace_past + 1, dtype=torch.int32).unsqueeze(0)
    state = build_decode_state(cfg, TRACE_KV_CACHE_SEQ_LEN, dtype=DTYPE)
    reference_inputs = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "k_cache": state["k_cache"], "v_cache": state["v_cache"],
        "conv_state": state["conv_state"],
    }
    # S=1 static query; positions/KV dynamic (min=1 so the s=1 prefill step at
    # position 0 is in-range); conv state fixed-shape.
    seq_pos = torch.export.Dim("seq_pos", min=1, max=max_ctx - 1)
    k_seq = torch.export.Dim("k_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)
    v_seq = torch.export.Dim("v_seq", min=TRACE_KV_CACHE_SEQ_LEN, max=max_ctx)
    dynamic_shapes = {
        "inputs_embeds": None,
        "position_ids": {1: seq_pos},
        "k_cache": {KVCache.seq_len_dim(): k_seq},
        "v_cache": {KVCache.seq_len_dim(): v_seq},
        "conv_state": None,
    }

    if mode == "int8lin":
        from coreai_models.export.compression import quantize_pytorch_model
        print("quantizing (linear int8 per-block-32) ...", flush=True)
        model = quantize_pytorch_model(
            model, tuple(reference_inputs.values()), dynamic_shapes,
            linear_quant_config("int8", 32))

    specs = [s for s in _EXTERNALIZE_SPECS if s.composite_op_name != "gated_delta_update"]
    print(f"exporting embeds-decode graph ({mode}) to Core AI ...", flush=True)
    prog = export_to_coreai(
        model, reference_inputs, dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"), output_names=("logits",),
        state_names=DECODE_STATE_NAMES, externalize_modules=specs)
    prog.optimize()

    import coreai.runtime as rt  # lazy: only after the converter has run (see module top)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aimodel = out_dir / f"{out_dir.name}.aimodel"
    meta = rt.AIModelAssetMetadata()
    meta.license = "lfm-open-license-v1.0"
    prog.save_asset(aimodel, meta)
    sz = sum(f.stat().st_size for f in aimodel.rglob("*") if f.is_file()) / 1e6
    print(f"[save] {aimodel} ({sz:.1f} MB)")
    return aimodel


# The CLI orchestration (export + eager + on-engine gate) lives in
# run_lfm2_embeds_gate.py — running it from THIS module's own frame trips a
# coreai_torch converter bug ("interleave must have rank (1)"). Import these
# helpers from a thin driver module instead.
