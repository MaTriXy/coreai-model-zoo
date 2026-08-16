# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Transformers ``AutoConfig`` shim for Muse-Glimmer checkpoints.

The pinned transformers (4.57.x) predates Muse-Glimmer (added in 5.15), so
``AutoConfig.from_pretrained("meta-models/Muse-Glimmer-30B")`` raises
``ValueError: model type 'muse_glimmer' not recognized``. The registry / CLI
export path reads the config through ``AutoConfig``, so without this it can't
load.

Same shape as the Gemma 4 shim: permissive ``PretrainedConfig`` subclasses that
carry the checkpoint's JSON fields as attributes, enough for the re-authored
decoder to consume. No modeling classes are registered — the decoder is our own
module. Drop this once the venv's transformers ships Muse-Glimmer natively.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class MuseGlimmerTextHFConfig(PretrainedConfig):
    model_type = "muse_glimmer_text"


class MuseGlimmerVisionHFConfig(PretrainedConfig):
    model_type = "muse_glimmer_vision"


class MuseGlimmerHFConfig(PretrainedConfig):
    model_type = "muse_glimmer"
    # Parse the nested per-modality dicts as typed sub-configs.
    sub_configs = {
        "text_config": MuseGlimmerTextHFConfig,
        "vision_config": MuseGlimmerVisionHFConfig,
    }

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        if isinstance(text_config, dict):
            text_config = MuseGlimmerTextHFConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = MuseGlimmerVisionHFConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        super().__init__(**kwargs)


def register_muse_glimmer_configs() -> None:
    """Register muse_glimmer* with ``AutoConfig`` (idempotent)."""
    for model_type, cls in (
        ("muse_glimmer_text", MuseGlimmerTextHFConfig),
        ("muse_glimmer_vision", MuseGlimmerVisionHFConfig),
        ("muse_glimmer", MuseGlimmerHFConfig),
    ):
        try:
            AutoConfig.register(model_type, cls)
        except ValueError:
            pass  # already registered in this interpreter


# Register on import so a module-level import in the registry makes
# AutoConfig.from_pretrained work before get_model_entry runs.
register_muse_glimmer_configs()
