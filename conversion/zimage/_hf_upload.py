"""Publish the Z-Image-Turbo Core AI bundles to the Hub.

Layout mirrors conversion/zimage/exports/ so a downloaded snapshot drops straight
into pipeline_engine.py. HF_HUB_DISABLE_XET=1 is required (xet panics on big shards).
"""
import os
import sys

from huggingface_hub import HfApi

REPO = "mlboydaisuke/Z-Image-Turbo-CoreAI"
HERE = os.path.dirname(os.path.abspath(__file__))
EXPORTS = os.path.join(HERE, "exports")

# The app-facing graphs expose fp32 boundaries (bf16 weights + bf16 compute inside):
# Swift cannot fill a bfloat16 NDArray, so the bf16-IO variants are host-hostile.
BUNDLES = [
    "zimage_dit_512_cap32_full_native_bf16_dyncap_dynimg_iofp32",
    "zimage_encoder_seq64_full_bf16_ids_iofp32",
    "zimage_vae_256_fp32",
    "zimage_vae_512_fp32",
    "zimage_vae_1024_fp32",
]

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True, private=False)
print(f"[hf] repo ready: {REPO}", flush=True)

api.upload_file(path_or_fileobj=os.path.join(HERE, "HF_CARD.md"),
                path_in_repo="README.md", repo_id=REPO, repo_type="model")
print("[hf] README.md uploaded", flush=True)

for b in BUNDLES:
    src = os.path.join(EXPORTS, b, f"{b}.aimodel")
    if not os.path.isdir(src):
        print(f"[hf] MISSING {src}", flush=True)
        sys.exit(1)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(src) for f in fs) / 1e9
    print(f"[hf] uploading {b}.aimodel ({size:.1f} GB) ...", flush=True)
    api.upload_folder(folder_path=src, path_in_repo=f"{b}.aimodel",
                      repo_id=REPO, repo_type="model",
                      commit_message=f"add {b}.aimodel")
    print(f"[hf] done {b}", flush=True)

# glue: RoPE tables + t_embedder graph, and the tokenizer the host needs.
for folder, dest in [("glue", "glue"), (None, None)]:
    if folder and os.path.isdir(os.path.join(HERE, folder)):
        print(f"[hf] uploading {folder}/ ...", flush=True)
        api.upload_folder(folder_path=os.path.join(HERE, folder), path_in_repo=dest,
                          repo_id=REPO, repo_type="model", commit_message=f"add {folder}")

import glob
snap = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image-Turbo/snapshots/*/tokenizer"))
if snap:
    print("[hf] uploading tokenizer/ ...", flush=True)
    api.upload_folder(folder_path=snap[0], path_in_repo="tokenizer",
                      repo_id=REPO, repo_type="model", commit_message="add tokenizer")

print("[hf] ALL UPLOADED", flush=True)
