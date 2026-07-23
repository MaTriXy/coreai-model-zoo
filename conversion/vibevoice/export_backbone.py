# Community port — NOT an Apple model.
"""Export the two VibeVoice Qwen2 backbones (main LM 4L norm=Identity, tts LM 20L norm=real) to
Core AI .aimodel decode (q=1) bundles + engine-gate vs the static-KV torch backbone (itself cos=1.0
vs the upstream LMs). Adapted from conversion/dots_tts/export.py.

Single q=1 decode graph per LM serves the whole loop: causal attention makes a q=W text window
numerically identical to W sequential q=1 steps, so text tokens are fed one at a time (no window
bucket). The tts graph is reused for both the positive and negative (CFG) KV streams.

  inputs_embeds[1,1,896] + pos[1] + KV state -> hidden[1,1,896]

  PYTHONPATH=. <coreai-venv>/bin/python export_backbone.py --mode int8 --cache-len 512
"""
from __future__ import annotations
import argparse, asyncio, glob, shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coreai_models.export.macos as _macos  # noqa: E402
from coreai_models.export.macos import export_to_coreai  # noqa: E402
from coreai_models.export.compression import quantize_pytorch_model  # noqa: E402
from backbone import build_kv_state, load_backbone  # noqa: E402

_DROP = {"scaled_dot_product_attention", "rope"}
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS if s.composite_op_name not in _DROP]
ART = HERE / "artifacts"
SNAP = "/Users/majimadaisuke/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4"
LMS = {"mainlm": ("model.language_model.", 4, False),
       "ttslm": ("model.tts_language_model.", 20, True)}


def quant_cfg(dtype: str) -> dict:
    return {
        "execution_mode": "eager",
        "global_config": {
            "op_state_spec": {"weight": {"dtype": dtype, "qscheme": "symmetric_with_clipping",
                                          "granularity": {"type": "per_block", "block_size": 32, "axis": 1}}},
            "op_input_spec": None, "op_output_spec": None,
        },
        "module_type_configs": {
            "coreai_models.primitives.macos.sdpa.SDPA": None,
            "backbone.RMSNorm": None,
        },
    }


def cos(a, b):
    a = torch.as_tensor(np.asarray(a), dtype=torch.float32).reshape(-1)
    b = torch.as_tensor(np.asarray(b), dtype=torch.float32).reshape(-1)
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def _du(p):
    return subprocess.run(["du", "-sh", str(p)], capture_output=True, text=True).stdout.split()[0]


def _save(prog, out_dir: Path) -> Path:
    import coreai.runtime as rt
    prog.optimize()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    aim = out_dir / f"{out_dir.name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    return aim


class DecodeWrap(nn.Module):
    def __init__(self, bb):
        super().__init__(); self.bb = bb

    def forward(self, inputs_embeds, pos, k_cache, v_cache):
        return self.bb.decode(inputs_embeds, pos, k_cache, v_cache)


async def gate_lm(name, DT, sd, cache_len, mode):
    import coreai.runtime as rt
    gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())
    prefix, nl, fn = LMS[name]
    bb = load_backbone(sd, prefix, nl, cache_len, final_norm=fn, dtype=DT)
    H, NKV, HD = bb.cfg.hidden_size, bb.cfg.num_key_value_heads, bb.cfg.head_dim
    kc, vc = build_kv_state(bb.cfg, cache_len, DT)
    dec_ref = {"inputs_embeds": torch.zeros(1, 1, H, dtype=DT), "pos": torch.tensor([0], dtype=torch.int32),
               "k_cache": kc.clone(), "v_cache": vc.clone()}
    model = DecodeWrap(bb).eval()
    if mode in ("int8", "int4"):
        model = quantize_pytorch_model(
            model, (dec_ref["inputs_embeds"], dec_ref["pos"], dec_ref["k_cache"], dec_ref["v_cache"]),
            {"inputs_embeds": None, "pos": None, "k_cache": None, "v_cache": None}, quant_cfg(mode))
    prog = export_to_coreai(model, dec_ref, dynamic_shapes=None,
                            input_names=("inputs_embeds", "pos"), output_names=("hidden",),
                            state_names=("keyCache", "valueCache"))
    ddir = ART / f"vibevoice_{name}_{mode}_decode_cl{cache_len}"
    aim = _save(prog, ddir)
    print(f"  [{name}] -> {ddir.name} ({_du(aim)})", flush=True)

    # gate: thread K decode steps, torch static-KV vs engine-state
    K = 8
    torch.manual_seed(0)
    embs = [torch.randn(1, 1, H, dtype=DT) for _ in range(K)]
    tk, tv = build_kv_state(bb.cfg, cache_len, DT)
    t_hid = []
    with torch.inference_mode():
        for i, e in enumerate(embs):
            t_hid.append(bb.decode(e, torch.tensor([i], dtype=torch.int32), tk, tv).reshape(-1).float())
    dfn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    state = {"keyCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16)),
             "valueCache": rt.NDArray(np.zeros((nl, 1, NKV, cache_len, HD), np.float16))}
    cs = []
    for i, e in enumerate(embs):
        r = await dfn(inputs={"inputs_embeds": rt.NDArray(np.ascontiguousarray(e.numpy().astype(np.float16))),
                              "pos": rt.NDArray(np.ascontiguousarray(np.array([i], np.int32)))}, state=state)
        cs.append(cos(r["hidden"].numpy(), t_hid[i]))
    print(f"  [{name}] decode engine-vs-torch min cos={min(cs):.6f}  {['%.4f' % c for c in cs]}")
    return min(cs)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="int8", choices=["fp16", "int8", "int4"])
    ap.add_argument("--cache-len", type=int, default=512)
    ap.add_argument("--which", default="both", choices=["both", "mainlm", "ttslm"])
    a = ap.parse_args()
    DT = torch.float16
    ART.mkdir(parents=True, exist_ok=True)
    sd = load_file(glob.glob(SNAP + "/*.safetensors")[0])
    names = ["mainlm", "ttslm"] if a.which == "both" else [a.which]
    res = {}
    for n in names:
        res[n] = await gate_lm(n, DT, sd, a.cache_len, a.mode)
    ok = all(c >= 0.99 for c in res.values())
    print(f"\n>>> backbone {a.mode} cl{a.cache_len}: " + "  ".join(f"{k}={v:.6f}" for k, v in res.items()) +
          f" -> {'GATE PASS' if ok else 'GATE FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
