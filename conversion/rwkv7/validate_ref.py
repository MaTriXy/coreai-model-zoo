# Community port — NOT an Apple model.
"""S1b reference: a standalone pure-torch (NO fla / NO triton) M=1 decode of
RWKV7-Goose-World3-1.5B-HF. This is the exact per-token recurrence we will export to
Core AI (S2/S3) — token-shift + WKV7 delta-rule matrix-state update + sqrelu channel-mix.

RWKV-7 is a pure recurrence (no causal attention), so feeding the prompt one token at a
time through the decode path is mathematically identical to a parallel prefill — this file
therefore validates *exactly* the op sequence the engine will run. Gate = coherence of the
greedy continuation (an independent token-exact oracle comes in S3).

The math is transcribed 1:1 from flash-linear-attention (MIT):
  fla/ops/rwkv7/fused_recurrent.py  (decode kernel, lines: b_h update / write / read)
  fla/layers/rwkv7.py               (time-mix: lora/kk/fused_k/gate-correction)
  fla/models/rwkv7/modeling_rwkv7.py(block: norm_first pre-LN, channel-mix sqrelu)
  fla/layers/rwkv6.py::LoRA         (act(x@A^T)@B^T + b)

Run:
  cd ~/code/coreai && .venv/bin/python \
    coreai-models-community/conversion/rwkv7/validate_ref.py --new 40
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

REPO = "RWKV/RWKV7-Goose-World3-1.5B-HF"
W_DECAY = -0.6065306597126334  # -exp(-0.5): scales sigmoid(w_lora) -> log-decay


def load_world_tokenizer(snap: str):
    # Instantiate the custom World tokenizer directly — AutoTokenizer would pull in the
    # config's auto_map -> modeling_rwkv7 -> fla (triton), which we don't need / can't run.
    spec = importlib.util.spec_from_file_location(
        "hf_rwkv_tokenizer", os.path.join(snap, "hf_rwkv_tokenizer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.RwkvTokenizer(vocab_file=os.path.join(snap, "rwkv_vocab_v20230424.txt"))


def find_snapshot() -> str:
    root = os.path.expanduser(
        "~/.cache/huggingface/hub/models--RWKV--RWKV7-Goose-World3-1.5B-HF/snapshots")
    snaps = sorted(glob.glob(os.path.join(root, "*")))
    if not snaps:
        raise FileNotFoundError(f"no snapshot under {root}; download the model first")
    return snaps[-1]


def load_weights(snap: str):
    cfg = json.load(open(os.path.join(snap, "config.json")))
    shards = sorted(glob.glob(os.path.join(snap, "*.safetensors")))
    sd = {}
    for s in shards:
        sd.update(load_file(s))
    sd = {k: v.float() for k, v in sd.items()}

    # the published checkpoint may store the 6 token-shift mixers as one packed [6,1,1,H]
    # tensor `attn.x_x`; split it (mirrors RWKV7Model.load_state_dict migration hook).
    for i in range(cfg["num_hidden_layers"]):
        xx = sd.pop(f"model.layers.{i}.attn.x_x", None)
        if xx is not None:
            for j, n in enumerate(("x_r", "x_w", "x_k", "x_v", "x_a", "x_g")):
                sd[f"model.layers.{i}.attn.{n}"] = xx[j].reshape(-1)
    return cfg, sd


class RWKV7Ref:
    """Per-token decode with explicit recurrent + token-shift states (no seq dim)."""

    def __init__(self, cfg: dict, sd: dict):
        self.cfg = cfg
        self.H = cfg["hidden_size"]
        self.HD = cfg["head_dim"]
        self.NH = self.H // self.HD
        self.NL = cfg["num_hidden_layers"]
        self.eps = cfg["norm_eps"]
        self.gn_eps = self.HD * cfg["norm_eps"]   # GroupNorm eps = head_dim * norm_eps
        self.sd = sd
        self.reset()

    def reset(self):
        # WKV7 matrix state S [L, NH, K=HD, V=HD]; token-shift prev hidden for time-mix & ffn.
        self.S = torch.zeros(self.NL, self.NH, self.HD, self.HD)
        self.ts_attn = torch.zeros(self.NL, self.H)
        self.ts_ffn = torch.zeros(self.NL, self.H)

    def g(self, name):
        return self.sd[name]

    def _ln(self, x, prefix):
        return F.layer_norm(x, (self.H,), self.g(prefix + ".weight"),
                            self.g(prefix + ".bias"), self.eps)

    def _lora(self, x, p, act):
        # LoRA = act(x @ A^T) @ B^T + b ; A=lora.0.weight [low,in], B=lora.2.weight [out,low]
        y = x @ self.g(p + ".lora.0.weight").T
        if act == "tanh":
            y = torch.tanh(y)
        elif act == "sigmoid":
            y = torch.sigmoid(y)
        y = y @ self.g(p + ".lora.2.weight").T
        b = self.sd.get(p + ".lora.2.bias")
        if b is not None:
            y = y + b
        return y

    @torch.no_grad()
    def step(self, x_emb):
        """x_emb: [H] embedding of one token -> logits [vocab]."""
        x = x_emb
        v_first = None
        for i in range(self.NL):
            P = f"model.layers.{i}"
            if i == 0:                                  # RWKV ln0 (norm_first, layer 0 only)
                x = self._ln(x, P + ".pre_norm")
            residual = x

            # ---------------- time-mixing (WKV7) ----------------
            h = self._ln(x, P + ".attn_norm")
            A = P + ".attn"
            delta = self.ts_attn[i] - h                 # token_shift: prev - cur
            self.ts_attn[i] = h
            xr = h + delta * self.g(A + ".x_r").reshape(-1)
            xw = h + delta * self.g(A + ".x_w").reshape(-1)
            xk = h + delta * self.g(A + ".x_k").reshape(-1)
            xv = h + delta * self.g(A + ".x_v").reshape(-1)
            xa = h + delta * self.g(A + ".x_a").reshape(-1)
            xg = h + delta * self.g(A + ".x_g").reshape(-1)

            r = xr @ self.g(A + ".r_proj.weight").T
            w = W_DECAY * torch.sigmoid(self._lora(xw, A + ".w_lora", "tanh"))
            k = xk @ self.g(A + ".k_proj.weight").T
            v = xv @ self.g(A + ".v_proj.weight").T
            if i == 0:
                v_first = v
            else:
                v = torch.lerp(v, v_first, torch.sigmoid(self._lora(xv, A + ".v_lora", None)))
            a = torch.sigmoid(self._lora(xa, A + ".a_lora", None))
            gate = self._lora(xg, A + ".g_lora", "sigmoid")

            k_k = self.g(A + ".k_k")
            k_a = self.g(A + ".k_a")
            kk = F.normalize((k * k_k).view(self.NH, self.HD), dim=-1, p=2.0)  # from ORIGINAL k
            k = k * (1.0 + (a - 1.0) * k_a)                                    # fused_k (overwrite)

            rh = r.view(self.NH, self.HD)
            wh = w.view(self.NH, self.HD)
            kh = k.view(self.NH, self.HD)
            ah = a.view(self.NH, self.HD)
            vh = v.view(self.NH, self.HD)

            ew = torch.exp(wh)                  # [NH,K]  decay in (0,1)
            akk = -kk                           # a_op
            bb = kk * ah                        # b_op
            S = self.S[i]                       # [NH,K,V]
            tmp = (akk.unsqueeze(-1) * S).sum(1)               # [NH,V] = akk^T S
            S = ew.unsqueeze(-1) * S + bb.unsqueeze(-1) * tmp.unsqueeze(1)
            S = S + kh.unsqueeze(-1) * vh.unsqueeze(1)         # write k (x) v
            o = (S * rh.unsqueeze(-1)).sum(1)                  # [NH,V] = r^T S
            self.S[i] = S

            o = F.group_norm(o.reshape(1, self.H), self.NH,
                             self.g(A + ".g_norm.weight"), self.g(A + ".g_norm.bias"),
                             self.gn_eps).reshape(self.H)
            # gate output correction: (o + (sum_d(r*k*r_k) * v)) * g
            r_k = self.g(A + ".r_k").view(self.NH, self.HD)
            corr = ((rh * kh * r_k).sum(-1, keepdim=True) * vh).reshape(self.H)
            o = (o + corr) * gate
            o = o @ self.g(A + ".o_proj.weight").T
            x = residual + o

            # ---------------- channel-mixing (sqrelu MLP) ----------------
            residual = x
            h2 = self._ln(x, P + ".ffn_norm")
            delta2 = self.ts_ffn[i] - h2
            self.ts_ffn[i] = h2
            xk2 = h2 + delta2 * self.g(P + ".ffn.x_k").reshape(-1)
            inner = torch.relu(xk2 @ self.g(P + ".ffn.key.weight").T) ** 2     # sqrelu
            ffn = inner @ self.g(P + ".ffn.value.weight").T
            x = residual + ffn

        x = self._ln(x, "model.norm")
        logits = x @ self.g("lm_head.weight").T
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="Tell me about the moon.")
    ap.add_argument("--new", type=int, default=40)
    ap.add_argument("--raw", action="store_true",
                    help="feed prompt verbatim instead of the User/Assistant chat wrapper")
    args = ap.parse_args()

    snap = find_snapshot()
    print("snapshot:", snap, flush=True)
    cfg, sd = load_weights(snap)
    print(f"loaded {len(sd)} tensors; layers={cfg['num_hidden_layers']} hidden={cfg['hidden_size']}",
          flush=True)
    tok = load_world_tokenizer(snap)

    model = RWKV7Ref(cfg, sd)
    emb = sd["model.embeddings.weight"]
    if args.raw:
        ids = tok(args.prompt).input_ids
    else:                                   # World chat format: <eot> + "User: ...\n\nAssistant:"
        ids = [0] + tok(f"User: {args.prompt}\n\nAssistant:").input_ids
    print("prompt:", repr(args.prompt), "->", len(ids), "tokens", flush=True)

    with torch.no_grad():
        logits = None
        for t in ids:                       # prime state on prompt (M=1 loop == prefill)
            logits = model.step(emb[t])
        out = []
        for _ in range(args.new):
            nxt = int(logits.argmax())
            if nxt == 0:                    # EOS = token 0 (World tokenizer / gen config)
                break
            out.append(nxt)
            logits = model.step(emb[nxt])
    print("\nGENERATION:", repr(tok.decode(out)), flush=True)


if __name__ == "__main__":
    main()
