# [arctic-platform] Vendored from SpecForge: specforge/export/checkpoint_io.py @ fdcc2edd (MIT License, sgl-project; see NOTICE).
# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Shared exporter plumbing: resolve a training checkpoint, materialize the draft."""

from __future__ import annotations

import json
import os

_DISABLE_LEGACY_ROPE_SCALING_ENV = "SPECFORGE_DISABLE_LEGACY_ROPE_SCALING"


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def apply_legacy_rope_scaling(output_dir: str) -> bool:
    """Keep modern and legacy RoPE scaling fields compatible in an export.

    Transformers 5 writes ``rope_parameters`` while older serving stacks read
    only ``rope_scaling`` and the top-level ``rope_theta``. Mirror non-default
    scaling representations and preserve the modern ``rope_theta`` for legacy
    readers. On by default; set
    ``SPECFORGE_DISABLE_LEGACY_ROPE_SCALING=1`` to skip. Returns whether the
    config was rewritten.
    """
    if _env_flag_enabled(_DISABLE_LEGACY_ROPE_SCALING_ENV):
        return False

    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)

    rope_parameters = config.get("rope_parameters")
    rope_scaling = config.get("rope_scaling")

    def rope_kind(payload):
        return (payload or {}).get("rope_type") or (payload or {}).get("type")

    changed = False
    if rope_parameters and "rope_theta" in rope_parameters:
        rope_theta = rope_parameters["rope_theta"]
        if config.get("rope_theta") != rope_theta:
            config["rope_theta"] = rope_theta
            changed = True

    if (
        rope_parameters
        and not rope_scaling
        and rope_kind(rope_parameters) not in (None, "default")
    ):
        config["rope_scaling"] = {
            key: value for key, value in rope_parameters.items() if key != "rope_theta"
        }
        changed = True
    elif (
        rope_scaling
        and not rope_parameters
        and rope_kind(rope_scaling) not in (None, "default")
    ):
        mirrored = dict(rope_scaling)
        if "rope_theta" in config:
            mirrored.setdefault("rope_theta", config["rope_theta"])
        config["rope_parameters"] = mirrored
        changed = True

    if not changed:
        return False

    temporary = f"{config_path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, config_path)
    return True
