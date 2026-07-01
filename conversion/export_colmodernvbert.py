# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coreai-core==1.0.0b1",
#     "coreai-torch==0.4.0",
#     "colpali-engine>=0.3.13",
#     "transformers>=5.5",
#     "peft>=0.13",
#     "pillow",
# ]
#
# [tool.uv]
# index-url       = "https://pypi.org/simple"
# prerelease      = "allow"
# index-strategy  = "unsafe-best-match"
# ///
# Export ColModernVBERT (visual document retriever, ModernVBERT/colmodernvbert) as static
# Core AI graphs. zoo's FIRST visual-document retriever and FIRST late-interaction
# (ColBERT/MaxSim) multi-vector model. Completes the on-device RAG stack alongside text
# Qwen3-Embedding (text->text dense) and Qwen3-Reranker (cross-encoder).
#
# ColModernVBERT = a 250M VLM encoder (ModernBERT-150M bidirectional text encoder + SigLIP2
# vision, pixel-shuffle x4) with a custom_text_proj Linear(768->128) head producing a
# *per-token* L2-normalized 128-d multi-vector. Retrieval = late interaction (MaxSim):
# score = sum_q max_d <Eq, Ed>.
#
# Two static graphs (two encoders, shared backbone), selected by --phase:
#   query : (input_ids [1,Sq] int32, attention_mask [1,Sq] int32) -> query_embeddings [1,Sq,128]
#   doc   : (pixel_values [1,1,3,512,512], pixel_attention_mask [1,1,512,512])
#                                                            -> doc_embeddings [1,Sd,128]
#           the text template (CLS + image markers + 64 <image> placeholders + SEP) is a baked
#           CONSTANT (single 512x512 tile / "global image" layout), so the only runtime inputs
#           are the pixels. Host MaxSim scores query vs doc multi-vectors.
#
# Both encoders are bidirectional, single forward, no KV cache / no generation. Queries (<128
# tok) and the single-tile doc (89 tok) stay under ModernBERT's sliding-window(128), so all
# layers see full attention here; multi-tile high-res docs (windowing) are a later enhancement.
#
# Each wrapper is verified against ColModernVBert(**processor.process_*(...)) BEFORE export
# (per-token cosine), and reference multi-vectors are dumped to JSON for the engine/Swift gate.
import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from coreai.runtime import AIModelAssetMetadata
from coreai_torch import TorchConverter, get_decomp_table

MODEL_ID = "ModernVBERT/colmodernvbert"
PROJ_DIM = 128
IMAGE_TOKEN_ID = 50407

REFERENCE_QUERIES = {
    "q_revenue": "What was the total revenue in the third quarter?",
    "q_headcount": "How many employees does the company have?",
    "q_chart": "bar chart of quarterly sales by region",
    "q_duedate_ja": "請求書の支払い期日はいつですか？",
}


# ----------------------------------------------------------------------------- modules
class QueryEncoder(torch.nn.Module):
    # (input_ids, attention_mask) -> per-token L2-normalized 128-d multi-vector [1, S, 128].
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return _as_tensor(self.model(input_ids=input_ids, attention_mask=attention_mask))


class DocEncoder(torch.nn.Module):
    # (pixel_values, pixel_attention_mask) -> doc multi-vector [1, Sd, 128]. The text template
    # input_ids / attention_mask are baked constants (single-tile layout, image-token-id 50407).
    def __init__(self, model: torch.nn.Module, input_ids: torch.Tensor,
                 attention_mask: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("input_ids", input_ids)            # [1, Sd] int (not cast by .half)
        self.register_buffer("attention_mask", attention_mask)  # [1, Sd] int

    def forward(self, pixel_values: torch.Tensor, pixel_attention_mask: torch.Tensor):
        return _as_tensor(self.model(
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        ))


def _as_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    for attr in ("embeddings", "multi_vector", "last_hidden_state"):
        if hasattr(out, attr):
            return getattr(out, attr)
    raise TypeError(f"Unexpected ColModernVBert output type: {type(out)}")


def _make_static_merger(start, n):
    # Export-friendly replacement for ModernVBertModel.inputs_merger. The stock merger uses
    # data-dependent ops (num_image_tokens.sum(), a torch_compilable_check assert, a boolean
    # masked-assign) that introduce unbacked symints under torch.export. With a CONSTANT
    # input_ids (single-tile doc template) the image tokens are a CONTIGUOUS block [start,
    # start+n); a scatter/index_put lowers to mps.scatter_nd which the GPU delegate rejects on
    # rank-3 data, so we splice via cat instead: [text_before | image_embeds | text_after].
    # The k-th image token (sequence order) gets image_hidden_states[0, k] -- identical to the
    # stock merger (single image / single block: block_idx=0, local_idx = 0..n-1 in mask order).
    def merger(input_ids, inputs_embeds, image_hidden_states):
        flat = image_hidden_states.reshape(-1, image_hidden_states.shape[-1])  # [n, hidden]
        return torch.cat([
            inputs_embeds[:, :start, :],
            flat.unsqueeze(0).to(inputs_embeds.dtype),
            inputs_embeds[:, start + n:, :],
        ], dim=1)
    return merger


def _patch_static_merger(model, image_token_id, input_ids):
    inner = model.model  # ModernVBertModel (holds inputs_merger)
    pos = (input_ids[0] == image_token_id).nonzero(as_tuple=False).flatten().to(torch.long)
    start, n = int(pos[0].item()), len(pos)
    assert torch.equal(pos, torch.arange(start, start + n)), \
        "image tokens are not a contiguous block; cat-splice merger needs contiguity"
    inner.inputs_merger = _make_static_merger(start, n)
    return pos


def _patch_static_image_features(model):
    # Export-friendly replacement for ModernVBertModel.get_image_features. The stock version (a)
    # filters all-zero padding images via data-dependent boolean indexing and (b) derives the
    # vision patch-attention mask from pixel_attention_mask with aten.unfold (unsupported by
    # coreai_torch). With a single real 512x512 tile both are unnecessary: pass the one image
    # through, and compute the patch mask with a reshape (numerically == the unfold).
    inner = model.model

    def get_image_features(pixel_values, pixel_attention_mask=None, **kwargs):
        bsz, num_images, C, H, W = pixel_values.shape
        pv = pixel_values.to(dtype=inner.dtype).view(bsz * num_images, C, H, W)
        patch_size = inner.config.vision_config.patch_size
        if pixel_attention_mask is None:
            pam = torch.ones((pv.shape[0], H, W), dtype=torch.bool, device=pv.device)
        else:
            pam = pixel_attention_mask.view(bsz * num_images, H, W)
        ph, pw = H // patch_size, W // patch_size
        m = pam.view(pv.shape[0], ph, patch_size, pw, patch_size)
        patch_attention_mask = (m.sum(dim=(2, 4)) > 0).bool()                 # == stock unfold
        image_outputs = inner.vision_model(
            pixel_values=pv, patch_attention_mask=patch_attention_mask, return_dict=True)
        image_outputs.pooler_output = inner.connector(image_outputs.last_hidden_state)
        return image_outputs

    inner.get_image_features = get_image_features


# ----------------------------------------------------------------------------- helpers
def pad_to_grid(input_ids, attention_mask, seq_len, pad_id):
    cur = input_ids.shape[1]
    if cur > seq_len:
        return input_ids[:, :seq_len].contiguous(), attention_mask[:, :seq_len].contiguous()
    if cur < seq_len:
        pad_n = seq_len - cur
        ids_pad = torch.full((1, pad_n), pad_id, dtype=input_ids.dtype)
        mask_pad = torch.zeros((1, pad_n), dtype=attention_mask.dtype)
        input_ids = torch.cat([input_ids, ids_pad], dim=1)
        attention_mask = torch.cat([attention_mask, mask_pad], dim=1)
    return input_ids.contiguous(), attention_mask.contiguous()


def per_token_cos(a, b, n_real) -> float:
    worst = 1.0
    for t in range(n_real):
        worst = min(worst, float(F.cosine_similarity(a[t].float(), b[t].float(), dim=0)))
    return worst


def make_test_page():
    # Deterministic synthetic "page" (A4-ish) used as the doc-encoder gate fixture. Saved next
    # to the bundle so the engine gate reproduces identical pixel_values.
    from PIL import Image, ImageDraw
    page = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(page)
    d.rectangle([200, 300, 1040, 360], fill="black")          # title bar
    for i in range(6):
        y = 480 + i * 120
        d.rectangle([200, y, 1040, y + 40], fill=(60, 60, 60))  # text lines
    d.rectangle([200, 1300, 600, 1650], outline="black", width=8)  # a figure box
    return page


def load_model():
    from colpali_engine.models import ColModernVBert, ColModernVBertProcessor
    print("[INFO] Loading ColModernVBert (CPU, fp32; torch.export traces on CPU)...")
    processor = ColModernVBertProcessor.from_pretrained(MODEL_ID)
    try:
        model = ColModernVBert.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True,
            attn_implementation="eager")
    except (TypeError, ValueError) as e:
        print(f"[WARN] attn_implementation=eager not accepted ({e}); loading default.")
        model = ColModernVBert.from_pretrained(
            MODEL_ID, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    for cfg in _walk_configs(model.config):
        if hasattr(cfg, "reference_compile"):
            cfg.reference_compile = False
        if hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"
    return model, processor


def convert_and_save(module, example, input_names, output_names, model_path, description):
    print("[INFO] Running torch export with decompositions...")
    exported = torch.export.export(module, args=(), kwargs=example)
    exported = exported.run_decompositions(get_decomp_table())
    print("[INFO] Converting to Core AI...")
    converter = TorchConverter().add_exported_program(
        exported_program=exported, input_names=input_names, output_names=output_names)
    prog = converter.to_coreai()
    prog.optimize()
    print("[INFO] Model optimized.")
    if model_path.exists():
        shutil.rmtree(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    meta = AIModelAssetMetadata()
    meta.author = "ModernVBERT (Illuin Technology / ETH)"
    meta.license = "MIT"
    meta.model_description = description
    meta.creation_date = int(time.time())
    prog.save_asset(model_path, meta)
    print(f"[INFO] Saved {model_path}")


# ----------------------------------------------------------------------------- phases
def export_query(args, dtype):
    model, processor = load_model()
    tok = processor.tokenizer
    pad_id = int(tok.pad_token_id) if tok.pad_token_id is not None else 0
    print(f"[INFO] pad_token_id={pad_id}, proj_dim={PROJ_DIM}, grid={args.seq_len}")
    module = QueryEncoder(model).eval()

    oracle, mine, real_lens, proc_inputs = {}, {}, {}, {}
    with torch.no_grad():
        for key, q in REFERENCE_QUERIES.items():
            ti = processor.process_texts([q])
            ids0, mask0 = ti["input_ids"].to(torch.int32), ti["attention_mask"].to(torch.int32)
            real_lens[key] = int(mask0.sum().item())
            proc_inputs[key] = (ids0, mask0)
            oracle[key] = module(ids0, mask0)[0]
            ids, mask = pad_to_grid(ids0, mask0, args.seq_len, pad_id)
            mine[key] = module(ids, mask)[0]
    print("[CHECK] wrapper-vs-processor per-token worst cosine (real tokens):")
    worst = 1.0
    for key in REFERENCE_QUERIES:
        n = min(real_lens[key], args.seq_len)
        c = per_token_cos(oracle[key], mine[key], n)
        worst = min(worst, c)
        print(f"          {key:14s} n_real={real_lens[key]:3d}  worst_tok_cos={c:.6f}")
    print(f"[CHECK] worst per-token cosine across queries = {worst:.6f}")
    assert worst > 0.999, f"padded-grid wrapper diverges ({worst})"

    vectors = {k: mine[k].float().tolist() for k in REFERENCE_QUERIES}
    ids, mask = pad_to_grid(*proc_inputs["q_revenue"], args.seq_len, pad_id)
    if dtype == torch.float16:
        module.to(torch.float16)
        _keep_rotary_fp32(module)

    out_dir = Path(args.output_dir)
    model_path = out_dir / f"colmodernvbert-query_{args.dtype}_s{args.seq_len}_static.aimodel"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"{model_path} exists; pass --overwrite")
    convert_and_save(
        module, {"input_ids": ids.clone(), "attention_mask": mask.clone()},
        ["input_ids", "attention_mask"], ["query_embeddings"], model_path,
        "ColModernVBERT visual document retriever (query/text encoder): ModernBERT-150M "
        "bidirectional -> custom_text_proj(768->128) -> per-token L2-normalized 128-d multi-"
        "vector for ColBERT-style late-interaction (MaxSim). https://huggingface.co/ModernVBERT/colmodernvbert")

    (out_dir / "reference_query.json").write_text(json.dumps({
        "model": MODEL_ID, "phase": "query", "seq_len": args.seq_len, "dtype": args.dtype,
        "proj_dim": PROJ_DIM, "pad_token_id": pad_id, "padding_side": "right",
        "queries": REFERENCE_QUERIES, "real_lens": real_lens,
        "query_embeddings": vectors, "selfcheck_worst_token_cos": worst,
    }, indent=2, ensure_ascii=False))
    tok_dir = out_dir / "tokenizer"; tok_dir.mkdir(exist_ok=True); tok.save_pretrained(tok_dir)
    print("[DONE] Phase 1 (query) gate passed; bundle written.")


def export_doc(args, dtype):
    model, processor = load_model()
    # Single 512x512 tile ("global image") layout: text template is constant.
    processor.image_processor.do_image_splitting = False
    page = make_test_page()
    di = processor.process_images([page])
    input_ids = di["input_ids"].to(torch.int64)               # [1, Sd] constant template
    attention_mask = di["attention_mask"].to(torch.int64)     # [1, Sd] all ones
    pixel_values = di["pixel_values"].to(torch.float32)        # [1, 1, 3, 512, 512]
    pixel_attention_mask = di["pixel_attention_mask"].to(torch.int32)  # [1, 1, 512, 512]
    Sd = input_ids.shape[1]
    n_img = int((input_ids[0] == IMAGE_TOKEN_ID).sum().item())
    print(f"[INFO] doc layout: Sd={Sd}, image_tokens={n_img}, "
          f"pixel_values={tuple(pixel_values.shape)}, pixmask={tuple(pixel_attention_mask.shape)}")
    assert Sd < 128, f"single-tile doc seq {Sd} >= 128 (sliding-window would engage)"

    # Oracle uses the STOCK (data-dependent) inputs_merger; capture it before patching.
    with torch.no_grad():
        oracle = _as_tensor(model(input_ids=input_ids, attention_mask=attention_mask,
                                  pixel_values=pixel_values,
                                  pixel_attention_mask=pixel_attention_mask))[0]   # [Sd,128]
    # Swap in export-friendly static paths (no data-dependent ops / no aten.unfold) and verify
    # they reproduce the oracle exactly.
    pos = _patch_static_merger(model, IMAGE_TOKEN_ID, input_ids)
    _patch_static_image_features(model)
    print(f"[INFO] static merger: {len(pos)} image-token positions {pos[0].item()}..{pos[-1].item()}")
    module = DocEncoder(model, input_ids, attention_mask).eval()
    with torch.no_grad():
        mine = module(pixel_values, pixel_attention_mask)[0]
    worst = per_token_cos(oracle, mine, Sd)
    print(f"[CHECK] doc static-merger vs stock-merger per-token worst cosine = {worst:.6f}")
    assert worst > 0.999, f"static merger diverges from stock ({worst})"
    vectors = mine.float().tolist()

    pv = pixel_values.clone()
    pam = pixel_attention_mask.clone()
    if dtype == torch.float16:
        module.to(torch.float16)
        _keep_rotary_fp32(module)
        pv = pv.to(torch.float16)

    out_dir = Path(args.output_dir)
    model_path = out_dir / f"colmodernvbert-doc_{args.dtype}_s{Sd}_static.aimodel"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"{model_path} exists; pass --overwrite")
    convert_and_save(
        module, {"pixel_values": pv, "pixel_attention_mask": pam},
        ["pixel_values", "pixel_attention_mask"], ["doc_embeddings"], model_path,
        "ColModernVBERT visual document retriever (document/image encoder, single 512px tile): "
        "SigLIP2 vision + pixel-shuffle x4 -> ModernBERT-150M -> custom_text_proj(768->128) -> "
        "per-token L2-normalized 128-d multi-vector for ColBERT-style late-interaction (MaxSim). "
        "https://huggingface.co/ModernVBERT/colmodernvbert")

    page.save(out_dir / "test_doc.png")
    (out_dir / "reference_doc.json").write_text(json.dumps({
        "model": MODEL_ID, "phase": "doc", "seq_len": Sd, "dtype": args.dtype,
        "proj_dim": PROJ_DIM, "image_token_id": IMAGE_TOKEN_ID, "n_image_tokens": n_img,
        "input_ids": input_ids[0].tolist(), "test_image": "test_doc.png",
        "doc_embeddings": vectors, "selfcheck_worst_token_cos": worst,
    }, indent=2, ensure_ascii=False))
    print("[DONE] Phase 2 (doc) gate passed; bundle written.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["query", "doc"], default="query")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float32")
    parser.add_argument("--seq-len", type=int, default=32,
                        help="query grid (queries are short; keep <128 so sliding==full)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    if args.phase == "query":
        export_query(args, dtype)
    else:
        export_doc(args, dtype)


def _walk_configs(cfg):
    seen, stack = [], [cfg]
    while stack:
        c = stack.pop()
        if c is None or any(id(c) == id(s) for s in seen):
            continue
        seen.append(c)
        yield c
        for name in dir(c):
            if name.endswith("_config"):
                try:
                    sub = getattr(c, name)
                except Exception:
                    continue
                if hasattr(sub, "to_dict") or hasattr(sub, "__dict__"):
                    stack.append(sub)


def _keep_rotary_fp32(module):
    n = 0
    for m in module.modules():
        for bname, buf in list(m.named_buffers(recurse=False)):
            if buf is not None and ("inv_freq" in bname or "cos" in bname or "sin" in bname):
                setattr(m, bname, buf.float())
                n += 1
    print(f"[INFO] restored {n} rotary buffer(s) to fp32 after fp16 cast")


if __name__ == "__main__":
    main()
