# P1 oracle + R-SWA equivalence proof (CPU fp32).
#
# Goal: empirically confirm the P0 #3 claim that the Core AI export design
#   "constant-window gather"  (attend prefix[0,Lm) ∪ monotonic tail[p-W+1,p])
# is numerically identical to the stock model's R-SWA ring buffer
#   (scrambled in-place overwrite of W=128 slots, attend whole fixed cache).
#
# Strategy: drive a real OCR generation with the STOCK ring attention (= reference,
# = oracle). Inside each attention forward, additionally maintain an INDEPENDENT
# monotonic K,V history and compute the gather output, then compare ring-vs-gather
# cosine per (layer, decode-step). If cos ~ 1.0 across all layers/steps, the export
# design is validated at the torch level on real data with real weights.
import os, sys, time, contextlib, json
import numpy as np
import torch

DEV = "cpu"  # MPS = immediate-EOS bug; oracle must be CPU fp32
DTYPE = torch.float32
torch.manual_seed(0)

# --- monkeypatch CUDA-isms so infer() runs on Mac (same as _run_ocr.py) ---
torch.Tensor.cuda = lambda self, *a, **k: self.to(DEV)
if hasattr(torch.cuda, "is_available"):
    torch.cuda.is_available = lambda: True
_orig_autocast = torch.autocast
def _autocast(device_type="cuda", *a, **k):
    if device_type == "cuda":
        return contextlib.nullcontext()
    return _orig_autocast(device_type, *a, **k)
torch.autocast = _autocast

CKPT = os.path.join(os.path.dirname(__file__), "ckpt")
sys.path.insert(0, CKPT)
from transformers import AutoModel, AutoTokenizer

IMG = os.environ.get("OCR_IMG", "page_jp.png")
MAXLEN = int(os.environ.get("OCR_MAXLEN", "640"))   # total = prefill + decode; need decode > 128
CROP = os.environ.get("OCR_CROP", "0") == "1"        # 0 = Base mode (small prefill, fast)

tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
model = AutoModel.from_pretrained(
    CKPT, trust_remote_code=True, use_safetensors=True,
    torch_dtype=DTYPE, attn_implementation="eager",
).eval().to(DEV)
print("[loaded]", type(model).__name__, flush=True)

# ---------------------------------------------------------------------------
# Patch SlidingWindowLlamaAttention.forward: run stock ring (reference) and a
# side-channel monotonic gather, compare per step. Grab the class + helpers
# from the *loaded* (dynamically-imported) module — robust to transformers ver.
# ---------------------------------------------------------------------------
_attn_inst = next(m for m in model.modules()
                  if type(m).__name__ == "SlidingWindowLlamaAttention")
attn_cls = type(_attn_inst)
M = sys.modules[attn_cls.__module__]
_orig_forward = attn_cls.forward
_apply_rope = M._llama_apply_rotary_pos_emb
_repeat_kv = M._llama_repeat_kv

# per-(layer) accumulated cosine stats over steady-state decode steps
STATS = {}  # layer_idx -> list of (decode_step, cos, max_abs_diff, n_visible)
MONO_K, MONO_V, MONO_PREFIX = {}, {}, {}  # module-level monotonic K,V history per layer

def patched_forward(self, *args, **kwargs):
    import math
    def _get(name, idx, default=None):
        if name in kwargs: return kwargs[name]
        if len(args) > idx: return args[idx]
        return default
    hidden_states = _get('hidden_states', 0)
    position_ids = _get('position_ids', 2)
    past_kv = _get('past_key_value', 3) or kwargs.get('past_key_values', None)

    # Reference: the real stock ring attention (also advances past_kv ring + state).
    ref_out, _w, past_kv2 = _orig_forward(self, *args, **kwargs)

    # Side-channel: maintain an independent monotonic K,V history and gather.
    num_heads = self.config.num_attention_heads
    num_kv_heads = self.config.num_key_value_heads
    head_dim = self.head_dim
    num_kv_groups = self.num_key_value_groups
    W = getattr(self.config, '_ring_window', None)
    bsz, q_len, _ = hidden_states.size()
    li = self.layer_idx

    # Recompute Q,K,V + RoPE deterministically (same weights/inputs as orig).
    q = self.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    cos, sin = self.rotary_emb(v, position_ids)
    q, k = _apply_rope(q, k, cos, sin)

    if W is None:
        return ref_out, _w, past_kv2

    if q_len > 1:
        # (re)prefill: seed module-level monotonic history + prefix length Lm.
        # Module storage (NOT past_kv) because layer 0's prefill receives
        # past_kv=None in this model's legacy cache-init path.
        MONO_K[li], MONO_V[li], MONO_PREFIX[li] = k, v, q_len
        return ref_out, _w, past_kv2                   # prefill: no comparison

    if li not in MONO_PREFIX:
        return ref_out, _w, past_kv2                   # decode before prefill: skip

    # decode step (q_len == 1): append, then gather prefix ∪ tail
    MONO_K[li] = torch.cat([MONO_K[li], k], dim=2)
    MONO_V[li] = torch.cat([MONO_V[li], v], dim=2)
    mk, mv = MONO_K[li], MONO_V[li]
    Lm = MONO_PREFIX[li]
    T = mk.shape[2]                                   # total monotonic length so far
    p = T - 1                                         # abs index of current token
    tail_start = max(Lm, p - W + 1)                   # last W decode tokens
    # visible = prefix[0,Lm) ++ tail[tail_start, p]  (gap [Lm,tail_start) correctly evicted)
    gk = torch.cat([mk[:, :, :Lm, :], mk[:, :, tail_start:T, :]], dim=2)
    gv = torch.cat([mv[:, :, :Lm, :], mv[:, :, tail_start:T, :]], dim=2)
    gk = _repeat_kv(gk, num_kv_groups)
    gv = _repeat_kv(gv, num_kv_groups)
    aw = torch.matmul(q, gk.transpose(2, 3)) / math.sqrt(head_dim)
    aw = torch.nn.functional.softmax(aw, dim=-1, dtype=torch.float32).to(q.dtype)
    go = torch.matmul(aw, gv).transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
    go = self.o_proj(go)

    a = ref_out.flatten().float(); b = go.flatten().float()
    cos_v = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
    mad = (a - b).abs().max().item()
    n_visible = gk.shape[2]
    STATS.setdefault(li, []).append((p - Lm, cos_v, mad, n_visible))
    return ref_out, _w, past_kv2

attn_cls.forward = patched_forward

# --- oracle capture: actual generated token ids + reference logits @ sampled steps ---
SAVE_STEPS = {0, 1, 32, 64, 128, 129, 160, 256, 384}
_logits = {}                 # forward-call index -> last-pos logits (np)
_step = {"i": -1}
def _lm_hook(mod, inp, out):
    _step["i"] += 1
    if _step["i"] in SAVE_STEPS:
        _logits[_step["i"]] = out[:, -1, :].detach().float().cpu().numpy()
model.lm_head.register_forward_hook(_lm_hook)

_real_generate = model.generate
_gen = {}
def _cap_generate(*a, **k):
    o = _real_generate(*a, **k)
    _gen["ids"] = (o.sequences if hasattr(o, "sequences") else o).detach().cpu().numpy()
    return o
model.generate = _cap_generate

# capture the decoder INPUT (post-vision inputs_embeds) + position_ids at prefill, so the
# standalone Core AI decoder can be gated on the exact same input (decode steps reconstruct
# via embed(token_ids[prefill:])).
_dec = {}
def _layer0_prehook(mod, args, kwargs):
    hs = args[0] if args else kwargs.get("hidden_states")
    if hs is not None and hs.shape[1] > 1 and "embeds" not in _dec:
        _dec["embeds"] = hs.detach().float().cpu().numpy()
        pid = kwargs.get("position_ids")
        if pid is None and len(args) > 2:
            pid = args[2]
        _dec["position_ids"] = None if pid is None else pid.detach().cpu().numpy()
model.model.layers[0].register_forward_pre_hook(_layer0_prehook, with_kwargs=True)

# ---------------------------------------------------------------------------
t0 = time.time()
out = model.infer(
    tok, prompt="<image>document parsing.",
    image_file=os.path.join(os.path.dirname(__file__), IMG),
    output_path="out/_oracle", base_size=1024, image_size=640, crop_mode=CROP,
    max_length=MAXLEN, no_repeat_ngram_size=35, ngram_window=128,
    eval_mode=True, save_results=False,
)
dt = time.time() - t0
print(f"\n[gen] {dt:.1f}s, {len(out)} chars", flush=True)
print(out[:1200], flush=True)

# ---------------------------------------------------------------------------
# Report: ring-vs-gather equivalence across all layers / decode steps.
all_cos, all_mad, n_steps = [], [], 0
W = model.config._ring_window
steady = []  # steps where the gather actually evicted (tail_start>Lm => true ring regime)
for li, rows in sorted(STATS.items()):
    cs = [r[1] for r in rows]; ms = [r[2] for r in rows]
    all_cos += cs; all_mad += ms; n_steps = max(n_steps, len(rows))
    for (ds, c, m, nv) in rows:
        if ds + 1 > W:  # more than W decode tokens => steady-state eviction active
            steady.append((c, m))
print("\n" + "=" * 64)
print(f"R-SWA ring  vs  constant-window-gather  (W={W})")
print(f"layers={len(STATS)} keys={sorted(STATS.keys())}  decode-steps/layer={n_steps}")
if all_cos:
    print(f"ALL steps:    cos min={min(all_cos):.8f} mean={np.mean(all_cos):.8f}   "
          f"max|Δ|={max(all_mad):.2e}")
if steady:
    sc = [x[0] for x in steady]; sm = [x[1] for x in steady]
    print(f"STEADY (>{W}): cos min={min(sc):.8f} mean={np.mean(sc):.8f}   "
          f"max|Δ|={max(sm):.2e}   (n={len(steady)})")
else:
    print(f"STEADY: none reached — increase OCR_MAXLEN (need decode_len > {W}).")
verdict = "PASS" if (all_cos and min(all_cos) > 0.999) else "FAIL"
print(f"VERDICT: {verdict}  (threshold cos>0.999)")
print("=" * 64)

os.makedirs("out/_oracle", exist_ok=True)
with open("out/_oracle/equiv_report.json", "w") as f:
    json.dump({"W": W, "img": IMG, "maxlen": MAXLEN,
               "cos_min": min(all_cos) if all_cos else None,
               "cos_mean": float(np.mean(all_cos)) if all_cos else None,
               "max_abs_diff": max(all_mad) if all_mad else None,
               "steady_cos_min": min([x[0] for x in steady]) if steady else None,
               "n_layers": len(STATS), "verdict": verdict,
               "oracle_text": out}, f, ensure_ascii=False, indent=2)
print("[saved] out/_oracle/equiv_report.json", flush=True)

# --- save P2 oracle: greedy/actual token ids + reference logits at sampled steps ---
oracle_npz = {"token_ids": _gen.get("ids"),
              "dec_embeds": _dec.get("embeds"), "dec_position_ids": _dec.get("position_ids")}
for s, lg in _logits.items():
    oracle_npz[f"logits_step{s}"] = lg
np.savez("out/_oracle/oracle_tensors.npz", **{k: v for k, v in oracle_npz.items() if v is not None})
ids = _gen.get("ids")
print(f"[oracle] token_ids shape={None if ids is None else ids.shape}  "
      f"logits_steps={sorted(_logits.keys())}", flush=True)
print("[saved] out/_oracle/oracle_tensors.npz", flush=True)
