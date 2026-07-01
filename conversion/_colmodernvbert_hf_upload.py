"""Upload the ColModernVBERT Core AI bundles (query + doc encoders, fp16/fp32) to HF. USER-GATED.

Run:  HF_XET_HIGH_PERFORMANCE=1 python conversion/_colmodernvbert_hf_upload.py
Staging dir holds the 4 *.aimodel bundles, tokenizer/, reference_*.json, test_doc.png, README.md.
"""
import os

from huggingface_hub import HfApi

REPO = "mlboydaisuke/ColModernVBERT-CoreAI"
STAGE = "/Users/majimadaisuke/code/ColModernVBERT-CoreAI/hf"

api = HfApi()
api.create_repo(REPO, repo_type="model", exist_ok=True)
api.upload_folder(
    folder_path=STAGE,
    repo_id=REPO,
    repo_type="model",
    commit_message="ColModernVBERT -> Core AI: query + doc encoders (fp16/fp32), "
                   "visual document retrieval (late-interaction / MaxSim)",
)
print("uploaded", REPO)
