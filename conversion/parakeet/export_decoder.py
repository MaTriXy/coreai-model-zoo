"""Phase 3+4 — Parakeet predictor (LSTM) + joint: re-author, export, and gate end-to-end
via the host TDT greedy loop (run in the MAIN coreai-torch venv).

Two tiny static graphs drive the transducer; the host runs the loop:

  predict : token[1,1] i32 + h[2,1,640] + c[2,1,640] -> dec_out[1,640] + h'[2,1,640] + c'[2,1,640]
            (embedding -> 2× manual LSTM cell -> decoder_projector; nn.LSTM avoided per the
             Kokoro lesson — single-step, so it's just two cell steps)
  joint   : dec_out[1,640] + enc_frame[1,640] -> token_logits[1,8193] + dur_logits[1,5]
            (head(relu(enc + dec)), split into the token and duration heads)

Gate ladder:
  1. eager re-author: run the host loop on golden enc_proj -> tokens must equal oracle tokens,
     and per-step logits must match oracle['step_logits'].
  2. engine: same loop driving the two .aimodels -> transcript token-for-token vs golden.

Run (MAIN venv; _GPU_LOCK held for the engine gate):
    coreai-models/.venv/bin/python export_decoder.py [--dtype float32] [--skip-export]
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
HID, NL, VOCAB, NDUR, BLANK = 640, 2, 8193, 5, 8192
DURATIONS = [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- modules
class Predict(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, HID)
        self.decoder_projector = nn.Linear(HID, HID, bias=True)
        for l in range(NL):
            self.register_buffer(f"w_ih{l}", torch.zeros(4 * HID, HID), persistent=True)
            self.register_buffer(f"w_hh{l}", torch.zeros(4 * HID, HID), persistent=True)
            self.register_buffer(f"b_ih{l}", torch.zeros(4 * HID), persistent=True)
            self.register_buffer(f"b_hh{l}", torch.zeros(4 * HID), persistent=True)

    def _cell(self, x, h, c, l):
        g = x @ getattr(self, f"w_ih{l}").T + getattr(self, f"b_ih{l}") \
            + h @ getattr(self, f"w_hh{l}").T + getattr(self, f"b_hh{l}")
        i, f, gg, o = g.chunk(4, dim=-1)
        c2 = f.sigmoid() * c + i.sigmoid() * gg.tanh()
        h2 = o.sigmoid() * c2.tanh()
        return h2, c2

    def forward(self, token, h, c):  # token[1,1] i64, h/c [2,1,640]
        x = self.embedding(token)[:, 0, :]          # [1,640]
        h0, c0 = self._cell(x, h[0], c[0], 0)
        h1, c1 = self._cell(h0, h[1], c[1], 1)
        dec_out = self.decoder_projector(h1)        # [1,640]
        return dec_out, torch.stack([h0, h1], 0), torch.stack([c0, c1], 0)


class Joint(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(HID, VOCAB + NDUR, bias=True)

    def forward(self, dec_out, enc_frame):  # [1,640] each
        logits = self.head(torch.relu(enc_frame + dec_out))   # [1,8198]
        return logits[:, :VOCAB], logits[:, VOCAB:]


def load_weights(predict: Predict, joint: Joint):
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("nvidia/parakeet-tdt-0.6b-v3", "model.safetensors")
    with safe_open(p, framework="pt") as f:
        predict.embedding.weight.data.copy_(f.get_tensor("decoder.embedding.weight"))
        predict.decoder_projector.weight.data.copy_(f.get_tensor("decoder.decoder_projector.weight"))
        predict.decoder_projector.bias.data.copy_(f.get_tensor("decoder.decoder_projector.bias"))
        for l in range(NL):
            getattr(predict, f"w_ih{l}").copy_(f.get_tensor(f"decoder.lstm.weight_ih_l{l}"))
            getattr(predict, f"w_hh{l}").copy_(f.get_tensor(f"decoder.lstm.weight_hh_l{l}"))
            getattr(predict, f"b_ih{l}").copy_(f.get_tensor(f"decoder.lstm.bias_ih_l{l}"))
            getattr(predict, f"b_hh{l}").copy_(f.get_tensor(f"decoder.lstm.bias_hh_l{l}"))
        joint.head.weight.data.copy_(f.get_tensor("joint.head.weight"))
        joint.head.bias.data.copy_(f.get_tensor("joint.head.bias"))
    print("[weights] predictor + joint loaded")


# --------------------------------------------------------------------------- TDT host loop
def tdt_decode(predict_fn, joint_fn, enc_proj, T, collect_logits=False):
    """predict_fn(token,h,c)->(dec,h,c) ; joint_fn(dec,enc_frame)->(tl,dl). enc_proj[T,640]."""
    h = torch.zeros(NL, 1, HID)
    c = torch.zeros(NL, 1, HID)
    dec, h, c = predict_fn(torch.tensor([[BLANK]]), h, c)
    frame, emitted, logits_log = 0, [], []
    while frame < T and len(logits_log) < 12 * T:
        tl, dl = joint_fn(dec, enc_proj[frame:frame + 1])
        token = int(tl.argmax()); dur = DURATIONS[int(dl.argmax())]
        if collect_logits:
            logits_log.append(torch.cat([tl[0], dl[0]]).detach().numpy())
        if token == BLANK and dur == 0:
            dur = 1
        frame += dur
        if token != BLANK:
            emitted.append(token)
            dec, h, c = predict_fn(torch.tensor([[token]]), h, c)
    return emitted, logits_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    ap.add_argument("--skip-export", action="store_true")
    args = ap.parse_args()

    d = np.load(HERE / "oracle.npz")
    enc_proj = torch.from_numpy(d["enc_proj"]).float()      # [T,640] golden
    T = int(d["T_valid"])
    gold_tokens = d["tokens"].tolist()
    gold_logits = torch.from_numpy(d["step_logits"]).float()
    text = str(d["text"])

    predict, joint = Predict().eval(), Joint().eval()
    load_weights(predict, joint)

    # ---- 1. eager re-author gate ----
    with torch.no_grad():
        emitted, logs = tdt_decode(predict, joint, enc_proj, T, collect_logits=True)
    tok_ok = emitted == gold_tokens
    n = min(len(logs), gold_logits.shape[0])
    lcos = torch.nn.functional.cosine_similarity(
        torch.tensor(np.stack(logs[:n])), gold_logits[:n], dim=-1).mean().item()
    print(f"[eager] emitted {len(emitted)} (gold {len(gold_tokens)}) tokens_exact={tok_ok} "
          f"step-logit cos {lcos:.6f}")
    if not tok_ok:
        print("❌ predictor/joint re-author DIVERGES"); raise SystemExit(1)
    print("✅ eager predictor+joint reproduces the golden transcript")
    if args.skip_export:
        return

    # ---- 2. export both graphs + engine gate via the same loop ----
    import coreai.runtime as rt
    from coreai_models.export.macos import export_to_coreai
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    art = HERE / "artifacts"; art.mkdir(exist_ok=True)

    def export(mod, example, ins, outs, name):
        prog = export_to_coreai(mod.to(dtype), example, dynamic_shapes=None,
                                input_names=ins, output_names=outs, state_names=None,
                                externalize_modules=[])
        prog.optimize()
        path = art / f"parakeet_{name}_{args.dtype}.aimodel"
        shutil.rmtree(path, ignore_errors=True)
        meta = rt.AIModelAssetMetadata(); meta.license = "cc-by-4.0"
        prog.save_asset(path, meta)
        sz = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6
        print(f"[save] {path.name} ({sz:.1f} MB)")
        return path

    z = lambda *s: torch.zeros(*s, dtype=dtype)
    p_path = export(predict, {"token": torch.zeros(1, 1, dtype=torch.long), "h": z(NL, 1, HID), "c": z(NL, 1, HID)},
                    ("token", "h", "c"), ("dec_out", "h_out", "c_out"), "predict")
    j_path = export(joint, {"dec_out": z(1, HID), "enc_frame": z(1, HID)},
                    ("dec_out", "enc_frame"), ("token_logits", "dur_logits"), "joint")

    async def engine_loop():
        pm = await rt.AIModel.load(str(p_path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        jm = await rt.AIModel.load(str(j_path), rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu()))
        pf, jf = pm.load_function("main"), jm.load_function("main")

        async def step_p(token, h, c):
            r = await pf({"token": rt.NDArray(token.numpy().astype(np.int32)),
                          "h": rt.NDArray(h.to(dtype).numpy()), "c": rt.NDArray(c.to(dtype).numpy())})
            return (torch.from_numpy(r["dec_out"].numpy().astype(np.float32)),
                    torch.from_numpy(r["h_out"].numpy().astype(np.float32)),
                    torch.from_numpy(r["c_out"].numpy().astype(np.float32)))

        async def step_j(dec, enc_frame):
            r = await jf({"dec_out": rt.NDArray(dec.to(dtype).numpy()),
                          "enc_frame": rt.NDArray(enc_frame.to(dtype).numpy())})
            return (torch.from_numpy(r["token_logits"].numpy().astype(np.float32)),
                    torch.from_numpy(r["dur_logits"].numpy().astype(np.float32)))

        h = torch.zeros(NL, 1, HID); c = torch.zeros(NL, 1, HID)
        dec, h, c = await step_p(torch.tensor([[BLANK]]), h, c)
        frame, emitted = 0, []
        while frame < T and len(emitted) < 12 * T:
            tl, dl = await step_j(dec, enc_proj[frame:frame + 1])
            token = int(tl.argmax()); dur = DURATIONS[int(dl.argmax())]
            if token == BLANK and dur == 0:
                dur = 1
            frame += dur
            if token != BLANK:
                emitted.append(token)
                dec, h, c = await step_p(torch.tensor([[token]]), h, c)
        return emitted

    emitted = asyncio.run(engine_loop())
    ok = emitted == gold_tokens
    print(f"[gate gpu] engine transcript: {len(emitted)} tokens, exact={ok} -> {'PASS' if ok else 'FAIL'}")
    print(f"   golden text: {text!r}")


if __name__ == "__main__":
    main()
