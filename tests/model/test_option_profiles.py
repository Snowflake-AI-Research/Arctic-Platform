# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The model profiles match the loaders. CPU only.

Each option key maps to its loader-side name: a field path under ``ModelSpec`` (its ``Patches`` and its own
fields) for the ``huggingface`` loader, and a field path under the loader's options model for the custom loaders.
A profile ``supports`` a key exactly when the mapped field exists, and a key with no loader-side name is never
``supports``. Every registered loader has a profile named after it.
"""

from __future__ import annotations

import types
import typing

import pytest
from pydantic import BaseModel

from arctic_platform.common.option_registry import SUPPORTS
from arctic_platform.common.option_registry import options
from arctic_platform.common.option_registry import profiles
from arctic_platform.common.option_registry import settle
from arctic_platform.model import ModelSpec
from arctic_platform.model import loader as loader_mod

HUGGINGFACE = "huggingface"

# Field paths under ``ModelSpec``. None: the loader has no option for the key.
HUGGINGFACE_FIELDS: dict[str, str | None] = {
    "checkpointing.mode": None,
    "checkpointing.freq": "patches.gradient_checkpointing",
    "checkpointing.targets": None,
    "checkpointing.offload": "patches.activation_offload",
    "tiled_mlp.token_chunk_size": "patches.tiled_mlp.token_chunk_size",
    "lm_head.fp32": "patches.lm_head.fp32",
    "lm_head.token_chunk_size": "patches.lm_head.token_chunk_size",
    "lm_head.vocab_chunk_size": "patches.lm_head.vocab_chunk_size",
    "lm_head.cross_entropy": None,
    "liger": "patches.liger",
    "moe.grouped_mm": None,
    "moe.comm_backend": None,
    "moe.comm_sms": None,
    "moe.comm_token_chunk": None,
    "attention.backend": "attn_implementation",
    "attention.sparse_mla": None,
    "numerics.reduce_dtype": None,
    "compile.fullgraph": "patches.compile.fullgraph",
    "peft": "patches.peft",
    "zorro_train": "patches.zorro_train",
}

# Field paths under a custom loader's options model (``ModelSpec.loader_options``). None: no loader-side name.
LOADER_OPTION_FIELDS: dict[str, str | None] = {
    "checkpointing.mode": "ac_config.mode",
    "checkpointing.freq": "ac_config.freq",
    "checkpointing.targets": "ac_config.targets",
    "checkpointing.offload": "ac_config.offload_config",
    "tiled_mlp.token_chunk_size": "tiled_mlp_token_chunk_size",
    "lm_head.fp32": "fp32_lm_head",
    "lm_head.token_chunk_size": "fused_lm_head_token_chunk_size",
    "lm_head.vocab_chunk_size": None,
    "lm_head.cross_entropy": "fused_cross_entropy",
    "liger": None,
    "moe.grouped_mm": "moe_use_grouped_mm",
    "moe.comm_backend": "ep_comm_backend",
    "moe.comm_sms": "deepep_num_sms",
    "moe.comm_token_chunk": "deepep_token_chunk_size",
    "attention.backend": None,
    "attention.sparse_mla": "sparse_mla_backend",
    "numerics.reduce_dtype": "reduce_dtype",
    "compile.fullgraph": None,
    "peft": None,
    "zorro_train": None,
}


def _model_classes(annotation: typing.Any) -> list[type[BaseModel]]:
    """The pydantic models inside a field annotation such as ``LmHeadPatch | None``."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return [annotation]
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        return [cls for arg in typing.get_args(annotation) for cls in _model_classes(arg)]
    return []


def _field_exists(root: type[BaseModel] | None, path: str | None) -> bool:
    if root is None or path is None:
        return False
    models = [root]
    parts = path.split(".")
    for index, part in enumerate(parts):
        found = [model.model_fields[part] for model in models if part in model.model_fields]
        if len(found) == 0:
            return False
        if index < len(parts) - 1:
            models = [cls for info in found for cls in _model_classes(info.annotation)]
    return True


def _loader_side(name: str) -> tuple[type[BaseModel] | None, dict[str, str | None]]:
    if name == HUGGINGFACE:
        return ModelSpec, HUGGINGFACE_FIELDS
    return loader_mod._LOADERS[name].options, LOADER_OPTION_FIELDS


def test_mapping_tables_cover_every_key():
    keys = {entry.key for entry in options()}
    assert set(HUGGINGFACE_FIELDS) == keys
    assert set(LOADER_OPTION_FIELDS) == keys


def test_every_mapped_name_exists_somewhere():
    """Guards the tables against typos: each mapped name is a real field of at least one loader."""
    for key, path in HUGGINGFACE_FIELDS.items():
        if path is not None:
            assert _field_exists(ModelSpec, path), (key, path)
    custom_roots = [entry.options for name, entry in loader_mod._LOADERS.items() if name != HUGGINGFACE]
    for key, path in LOADER_OPTION_FIELDS.items():
        if path is not None:
            assert any(_field_exists(root, path) for root in custom_roots), (key, path)


def test_profile_names_equal_loader_names():
    assert {profile.name for profile in profiles()} == set(loader_mod._LOADERS)


@pytest.mark.parametrize("name", sorted(loader_mod._LOADERS))
def test_supports_exactly_when_loader_field_exists(name):
    root, fields = _loader_side(name)
    cells = settle(name)
    mismatches = []
    for key, path in fields.items():
        supported = cells[key].status == SUPPORTS
        if supported != _field_exists(root, path):
            mismatches.append(f"{key}: profile says {cells[key].status}, loader field {path!r} on {root}")
        if path is None:
            assert not supported, key
    assert mismatches == []
