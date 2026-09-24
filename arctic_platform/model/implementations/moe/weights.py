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
"""Safetensors state-dict helpers for the carved-out Qwen3.5 load path.

Slimmed from prime-rl's ``trainer/weights.py``: keeps only the key-scan, load,
and save helpers used by ``load_dcp_from_hf`` for the one-time HF->Prime
conversion cache. The distributed gather / LoRA helpers are dropped.
"""

import json
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME
from transformers.utils import SAFE_WEIGHTS_NAME
from transformers.utils import WEIGHTS_INDEX_NAME
from transformers.utils import WEIGHTS_NAME

from .logging_utils import get_logger


def get_max_layer_num(state_dict: dict[str, Tensor], layer_prefix: str = "model.layers.") -> int:
    """Get the maximum number of layers in the model."""
    max_num = -1
    for key in state_dict:
        if not key.startswith(layer_prefix):
            continue
        layer_num_str = key[len(layer_prefix) :].split(".")[0]
        if layer_num_str.isdigit():
            max_num = max(max_num, int(layer_num_str))
    return max_num + 1


def load_state_dict_keys(save_dir: Path) -> list[str]:
    """Load only the key names from safetensor files without reading tensor data."""
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
):
    """Save a state dict to a local directory in safetensors or torch format."""
    logger = get_logger()
    weights_name = SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                del state_dict[tensor]
            if save_format == "safetensors":
                save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, save_dir / shard_file)
        del state_dict

        # Written whether or not the weights needed splitting, and written last. Loaders accept a single-shard
        # index, and the weight-conversion cache uses the index's presence to tell a finished directory from an
        # interrupted one -- a state dict small enough to fit one file would otherwise convert and then be
        # rejected as incomplete on every subsequent load.
        index = {
            "metadata": {**state_dict_split.metadata},
            "weight_map": state_dict_split.tensor_to_filename,
        }
        save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
        save_index_file = save_dir / save_index_file
        with open(save_index_file, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)
    else:
        if save_format == "safetensors":
            save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_dir / weights_name)


def _strip_activation_checkpoint_segments(name: str) -> str:
    """Drop torch AC wrappers so PEFT/vLLM see the original module path."""
    return name.replace("._checkpoint_wrapped_module", "")


_FUSED_EXPERT_WEIGHT_NAMES = frozenset({"w1", "w2", "w3"})


def hf_export_param_name(name: str) -> str | None:
    """HF weight name, or None for LoRA tensors. Strips PEFT/AC wrappers."""
    if "lora_" in name:
        return None
    name = _strip_activation_checkpoint_segments(name)
    # Nested ParamWrappers share the '.' between `.base_layer` segments, so a
    # single ``str.replace(".base_layer.", ".")`` leaves a leftover
    # ``.base_layer`` (e.g. ``experts.base_layer.base_layer.base_layer.w1``
    # -> ``experts.base_layer.w1``). Drop every PEFT wrapper segment instead.
    parts = [part for part in name.split(".") if part != "base_layer"]
    if (
        len(parts) >= 2
        and parts[-1] == "weight"
        and parts[-2] in _FUSED_EXPERT_WEIGHT_NAMES
        and (len(parts) < 3 or parts[-3] == "experts")
    ):
        parts = parts[:-1]
    name = ".".join(parts)
    if name.startswith("base_model.model."):
        name = name[len("base_model.model.") :]
    return name
