# Community port — re-export all 3 Stable Audio bundles in fp16 (ship precision).
import json, os, sys, shutil, asyncio, numpy as np, torch
from pathlib import Path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "_ref", "stable-audio-tools"))
import coreai_models.export.macos as _macos
from coreai_models.export.macos import export_to_coreai
import coreai.runtime as rt
_macos._EXTERNALIZE_SPECS = [s for s in _macos._EXTERNALIZE_SPECS
                             if s.composite_op_name not in {"scaled_dot_product_attention", "rope"}]
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from safetensors.torch import load_file
from oobleck_vae import load_decoder

DT = torch.float16
ART = Path(HERE) / "artifacts"
cfg = json.load(open(os.path.join(HERE, "model_config.json")))
gpu = rt.SpecializationOptions.from_preferred_compute_unit_kind(rt.ComputeUnitKind.gpu())


def save(prog, name):
    prog.optimize()
    d = ART / name
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True)
    aim = d / f"{name}.aimodel"
    prog.save_asset(aim, rt.AIModelAssetMetadata())
    import subprocess
    sz = subprocess.run(["du", "-sh", str(aim)], capture_output=True, text=True).stdout.split()[0]
    print(f"  saved {name} ({sz})", flush=True)
    return aim


async def egate(aim, inputs, out_name, ref):
    fn = (await rt.AIModel.load(str(aim), gpu)).load_function("main")
    r = await fn(inputs=inputs)
    eng = torch.as_tensor(r[out_name].numpy().astype(np.float32))
    return torch.nn.functional.cosine_similarity(eng.reshape(-1), ref.reshape(-1), dim=0).item()


def ndf(a): return rt.NDArray(np.ascontiguousarray(a.detach().numpy().astype(np.float16)))  # fp16 bundles
def ndi(a): return rt.NDArray(np.ascontiguousarray(a.detach().numpy().astype(np.int32)))


async def main():
    model = create_model_from_config(cfg)
    model.load_state_dict(load_ckpt_state_dict(os.path.join(HERE, "model.safetensors")), strict=False)
    model = model.to("cpu").eval()

    # ---- DiT fp16 ----
    dit = next(m for m in model.modules() if type(m).__name__ == "DiffusionTransformer").half()
    io = torch.load(os.path.join(HERE, "dit_io.pt"))
    x, t, cac, ge, mk = (io["x"].to(DT), io["t"].to(DT), io["cross_attn_cond"].to(DT),
                         io["global_embed"].to(DT), io["cross_attn_cond_mask"].to(DT))

    class DiTW(torch.nn.Module):
        def __init__(s, d): super().__init__(); s.d = d
        def forward(s, x, t, cross_attn_cond, global_embed, cross_attn_cond_mask):
            return s.d(x, t, cross_attn_cond=cross_attn_cond, global_embed=global_embed,
                       cross_attn_cond_mask=cross_attn_cond_mask)
    prog = export_to_coreai(DiTW(dit).eval(), {"x": x, "t": t, "cross_attn_cond": cac, "global_embed": ge, "cross_attn_cond_mask": mk},
                            input_names=("x", "t", "cross_attn_cond", "global_embed", "cross_attn_cond_mask"),
                            output_names=("v",))
    aim = save(prog, "sa_dit_fp16")
    c = await egate(aim, {"x": ndf(x), "t": ndf(t), "cross_attn_cond": ndf(cac), "global_embed": ndf(ge),
                          "cross_attn_cond_mask": ndf(mk)}, "v", io["out"].float())
    print(f"  DiT engine cos={c:.6f}", flush=True)

    # ---- VAE fp16 ----
    dec = load_decoder(load_file(os.path.join(HERE, "model.safetensors")), DT)
    oracle = torch.load(os.path.join(HERE, "ref_oracle.pt"))
    lat = oracle["latent"].to(DT)
    prog = export_to_coreai(dec, {"latent": lat}, input_names=("latent",), output_names=("audio",))
    aim = save(prog, "sa_vae_fp16")
    c = await egate(aim, {"latent": ndf(lat)}, "audio", oracle["audio"].float())
    print(f"  VAE engine cos={c:.6f}", flush=True)

    # ---- Conditioner fp16 ----
    t5c = model.conditioner.conditioners["prompt"]; numc = model.conditioner.conditioners["seconds_total"]
    t5c.model.half(); numc.embedder.half()
    tok = t5c.tokenizer(["128 BPM tech house drum loop"], truncation=True, max_length=64, padding="max_length", return_tensors="pt")
    ids, attn = tok["input_ids"].int(), tok["attention_mask"].to(DT)
    sn = torch.tensor([11.0 / 256.0], dtype=DT)

    class CondW(torch.nn.Module):
        def __init__(s, t5, p, n): super().__init__(); s.t5 = t5; s.p = p; s.n = n
        def forward(s, input_ids, attention_mask, seconds_norm):
            e = s.t5(input_ids=input_ids, attention_mask=attention_mask)["last_hidden_state"]
            e = s.p(e) * attention_mask.unsqueeze(-1)
            num = s.n(seconds_norm).unsqueeze(1)
            ones = torch.ones(attention_mask.shape[0], 1, dtype=attention_mask.dtype)
            return torch.cat([e, num], dim=1), num.squeeze(1), torch.cat([attention_mask, ones], dim=1)
    with torch.inference_mode():
        ref_cross, ref_glob, _ = CondW(t5c.model, t5c.proj_out, numc.embedder).eval()(ids, attn, sn)
    prog = export_to_coreai(CondW(t5c.model, t5c.proj_out, numc.embedder).eval(),
                            {"input_ids": ids, "attention_mask": attn, "seconds_norm": sn},
                            input_names=("input_ids", "attention_mask", "seconds_norm"),
                            output_names=("cross_attn_cond", "global_embed", "cond_mask"))
    aim = save(prog, "sa_cond_fp16b")
    c = await egate(aim, {"input_ids": ndi(ids), "attention_mask": ndf(attn), "seconds_norm": ndf(sn)},
                    "cross_attn_cond", ref_cross.float())
    print(f"  Conditioner engine cos={c:.6f}", flush=True)
    print("[fp16] all 3 bundles re-exported")

asyncio.run(main())
