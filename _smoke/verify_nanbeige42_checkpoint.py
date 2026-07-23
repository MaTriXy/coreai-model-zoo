#!/usr/bin/env python3
"""Pinned official-checkpoint parity gate for the Nanbeige4.2 Core AI overlay.

The official model runs in a separate interpreter, exits, and leaves only its
reference tensors behind. Run this script with the overlay interpreter and pass
an isolated vendor-compatible interpreter through ``--official-python``.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import tempfile
from pathlib import Path

HF_ID = "Nanbeige/Nanbeige4.2-3B"
REVISION = "5ff54fb7ed86ce8e216d78bff5417ab9981de3d4"
CONFIG_SHA256 = "f6cb15b22847664f3a6049dc4b58fdd10f1650d112ac99a1da3d051f17c2ca19"
PROMPT = "The capital of France is"
INPUT_IDS = [166100, 363, 5463, 290, 7914, 322]
GREEDY_TOKENS = 32
MAX_CONTEXT = 128


def verify_config(snapshot: Path) -> None:
    actual = hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest()
    if actual != CONFIG_SHA256:
        raise RuntimeError(f"config SHA-256 mismatch: expected {CONFIG_SHA256}, got {actual}")


def official_reference(output: Path, device_name: str) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    snapshot = Path(
        snapshot_download(
            HF_ID,
            revision=REVISION,
            allow_patterns=[
                "*.json",
                "*.model",
                "*.py",
                "*.safetensors",
                "*.safetensors.index.json",
            ],
        )
    )
    verify_config(snapshot)
    device = torch.device(device_name)
    model = AutoModelForCausalLM.from_pretrained(
        snapshot,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).to(device).eval()
    input_ids = torch.tensor([INPUT_IDS], dtype=torch.long, device=device)

    with torch.no_grad():
        full = model(input_ids=input_ids, use_cache=False).logits.cpu()
        past = None
        steps = []
        for index in range(input_ids.shape[1]):
            result = model(
                input_ids=input_ids[:, index : index + 1],
                past_key_values=past,
                use_cache=True,
            )
            past = result.past_key_values
            steps.append(result.logits.cpu())
        incremental = torch.cat(steps, dim=1)

        greedy = []
        logits = steps[-1].to(device)
        for _ in range(GREEDY_TOKENS):
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            greedy.append(int(next_id.item()))
            result = model(input_ids=next_id, past_key_values=past, use_cache=True)
            past = result.past_key_values
            logits = result.logits

    torch.save(
        {
            "input_ids": input_ids.cpu(),
            "full_logits": full,
            "incremental_logits": incremental,
            "greedy_ids": greedy,
        },
        output,
    )


def overlay_result(reference: dict) -> dict:
    import torch

    from coreai_models.models.macos.nanbeige import NanbeigeForCausalLM, create_cache_tensors

    model = NanbeigeForCausalLM.from_hf_memory_efficient(
        HF_ID,
        revision=REVISION,
        max_context_length=MAX_CONTEXT,
        target_dtype=torch.float32,
    ).eval()
    input_ids = reference["input_ids"].to(torch.int32)
    positions = torch.arange(input_ids.shape[1], dtype=torch.int32).unsqueeze(0)

    with torch.no_grad():
        full_k, full_v = create_cache_tensors(model.config)
        full = model(input_ids, positions, full_k, full_v)

        k_cache, v_cache = create_cache_tensors(model.config)
        steps = []
        for index in range(input_ids.shape[1]):
            steps.append(
                model(
                    input_ids[:, index : index + 1],
                    torch.arange(index + 1, dtype=torch.int32).unsqueeze(0),
                    k_cache,
                    v_cache,
                )
            )
        incremental = torch.cat(steps, dim=1)

        greedy = []
        logits = steps[-1]
        length = input_ids.shape[1]
        for _ in range(GREEDY_TOKENS):
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True).to(torch.int32)
            greedy.append(int(next_id.item()))
            length += 1
            logits = model(
                next_id,
                torch.arange(length, dtype=torch.int32).unsqueeze(0),
                k_cache,
                v_cache,
            )

    return {"full_logits": full, "incremental_logits": incremental, "greedy_ids": greedy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-python", help="isolated interpreter for pinned vendor code")
    parser.add_argument("--official-device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--official-worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.official_worker:
        official_reference(args.official_worker, args.official_device)
        return
    if not args.official_python:
        parser.error("--official-python is required")

    import torch

    with tempfile.TemporaryDirectory(prefix="nanbeige42-parity-") as temporary:
        reference_path = Path(temporary) / "official.pt"
        subprocess.run(
            [
                args.official_python,
                str(Path(__file__).resolve()),
                "--official-worker",
                str(reference_path),
                "--official-device",
                args.official_device,
            ],
            check=True,
        )
        reference = torch.load(reference_path, map_location="cpu", weights_only=True)
        candidate = overlay_result(reference)

    for key in ("full_logits", "incremental_logits"):
        torch.testing.assert_close(candidate[key], reference[key], rtol=1e-4, atol=1e-4)
        error = (candidate[key] - reference[key]).abs().max().item()
        print(f"{key}: PASS (max_abs={error:.8g})")
    if candidate["greedy_ids"] != reference["greedy_ids"]:
        raise AssertionError("32-token greedy continuation differs from the official checkpoint")
    print(f"prompt: {PROMPT!r} -> {INPUT_IDS}")
    print(f"greedy_ids: PASS ({GREEDY_TOKENS}/{GREEDY_TOKENS} exact)")
    print(f"config_sha256: PASS ({CONFIG_SHA256})")


if __name__ == "__main__":
    main()
