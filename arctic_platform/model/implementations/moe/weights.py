"""Safetensors state-dict helpers for the carved-out Qwen3.5 load path.

Slimmed from prime-rl's ``trainer/weights.py``: keeps only the key-scan, load,
and save helpers used by ``load_dcp_from_hf`` for the one-time HF->Prime
conversion cache. The distributed gather / LoRA helpers are dropped.
"""

import inspect
import json
import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Literal

import torch
from huggingface_hub import get_torch_storage_id
from huggingface_hub import get_torch_storage_size
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.serialization._base import parse_size_to_int
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


def convert_safetensor_checkpoint(
    source_dir: Path,
    save_dir: Path,
    convert_fn: Callable[[dict[str, Tensor]], object],
    partition_fn: Callable[[tuple[str, ...]], Iterable[Iterable[str]]],
) -> None:
    """Convert complete dependency groups, retaining only one group in memory."""
    locations: dict[str, Path] = {}
    for shard_path in sorted(source_dir.glob("*.safetensors")):
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name in locations:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                locations[name] = shard_path
    if not locations:
        raise ValueError(f"No safetensors weights in {source_dir}")
    groups = [tuple(group) for group in partition_fn(tuple(locations))]
    names = [name for group in groups for name in group]
    if any(not group for group in groups) or len(names) != len(set(names)) or set(names) != set(locations):
        raise ValueError("Checkpoint conversion groups must cover every source tensor exactly once")

    save_dir.mkdir(parents=True, exist_ok=False)
    weight_map: dict[str, str] = {}
    total_size = 0
    for group_index, group in enumerate(groups):
        state: dict[str, Tensor] = {}
        by_shard: dict[Path, list[str]] = {}
        for name in group:
            by_shard.setdefault(locations[name], []).append(name)
        for shard_path, selected in by_shard.items():
            with safe_open(shard_path, framework="pt", device="cpu") as shard:
                for name in selected:
                    state[name] = shard.get_tensor(name)
        convert_fn(state)
        if not state or set(state).intersection(weight_map):
            raise ValueError("Checkpoint conversion produced empty or overlapping groups")
        split = split_torch_state_dict_into_shards(
            state, filename_pattern=f"model-group{group_index:05d}{{suffix}}.safetensors"
        )
        for filename, tensor_names in split.filename_to_tensors.items():
            output = {name: state.pop(name).contiguous() for name in tensor_names}
            save_file(output, save_dir / filename, metadata={"format": "pt"})
            del output
        weight_map.update(split.tensor_to_filename)
        total_size += split.metadata["total_size"]
        del state

    # The root index is the cache's completion marker; never publish a partial conversion.
    with (save_dir / SAFE_WEIGHTS_INDEX_NAME).open("x", encoding="utf-8") as stream:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": weight_map}, stream, indent=2, sort_keys=True)
        stream.write("\n")


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


class _StreamedShard:
    def __init__(self) -> None:
        self.index: int | None = None
        self.tensors: dict[str, Tensor] = {}
        self.names: list[str] = []
        self.size = 0
        self.storages: set = set()
        self.partial: Path | None = None


def save_state_dict_stream(weights: Iterable[tuple[str, Tensor]], save_dir: Path) -> None:
    """Write :func:`save_state_dict`'s sharded safetensors output while consuming ``weights`` once.

    Each tensor is copied to the CPU on arrival, as a caller collecting ``weights`` into a dict for ``save_state_dict``
    would, so storage sharing is the same. ``split_torch_state_dict_into_shards`` places every view in the shard of its
    storage's first tensor, so a closed shard is written only once its tensors tile every storage they view; until then
    a later view may still join it. Host memory holds the open shard and the closed shards still awaiting views, not
    the whole state.
    """
    max_shard_size = inspect.signature(split_torch_state_dict_into_shards).parameters["max_shard_size"].default
    if isinstance(max_shard_size, str):
        max_shard_size = parse_size_to_int(max_shard_size)
    filename_pattern = SAFE_WEIGHTS_NAME.replace(".safetensors", "{suffix}.safetensors")
    save_dir.mkdir(parents=True, exist_ok=True)

    shards: list[_StreamedShard] = []
    current = _StreamedShard()
    owners: dict = {}
    spans: dict = {}
    seen: set[str] = set()
    total_size = 0

    def tiled(storage_id) -> bool:
        storage_spans = spans[storage_id]
        nbytes = storage_id[-1]
        # A zero-byte storage id is shared by every empty tensor, so it never stops accepting views.
        if storage_spans is None or nbytes == 0 or sum(end - start for start, end in storage_spans) < nbytes:
            return False
        covered, reach = 0, 0
        for start, end in sorted(storage_spans):
            covered += max(0, end - max(start, reach))
            reach = max(reach, end)
        return covered == nbytes

    def write(shard: _StreamedShard) -> None:
        assert shard.index is not None
        shard.partial = save_dir / f".{filename_pattern.format(suffix=f'-{shard.index + 1:05d}')}.partial"
        tensors = {name: tensor.contiguous() for name, tensor in shard.tensors.items()}
        save_file(tensors, shard.partial, metadata={"format": "pt"})
        shard.tensors = {}
        for storage_id in shard.storages:
            del owners[storage_id]
            del spans[storage_id]

    def write_if_complete(shard: _StreamedShard) -> None:
        if shard.index is not None and shard.partial is None and all(tiled(storage) for storage in shard.storages):
            write(shard)

    def close(shard: _StreamedShard) -> None:
        shard.index = len(shards)
        shards.append(shard)
        write_if_complete(shard)

    for name, tensor in weights:
        if name in seen:
            raise ValueError(f"Duplicate exported tensor: {name}")
        seen.add(name)
        tensor = tensor.detach().cpu()
        storage_id = get_torch_storage_id(tensor)
        shared = storage_id in owners
        if not shared:
            tensor_size = get_torch_storage_size(tensor)
            total_size += tensor_size
            if tensor_size > max_shard_size:
                owners[storage_id] = _StreamedShard()
            else:
                if current.size + tensor_size > max_shard_size:
                    close(current)
                    current = _StreamedShard()
                current.size += tensor_size
                owners[storage_id] = current
            spans[storage_id] = []
        shard = owners[storage_id]
        shard.tensors[name] = tensor
        shard.names.append(name)
        shard.storages.add(storage_id)
        if spans[storage_id] is not None and tensor.is_contiguous():
            start = tensor.storage_offset() * tensor.element_size()
            spans[storage_id].append((start, start + tensor.numel() * tensor.element_size()))
        else:
            spans[storage_id] = None
        if not shared and shard is not current:
            close(shard)
        else:
            write_if_complete(shard)
        # A view pins its whole storage; holding it until the producer's next item would keep a written layer alive.
        del tensor
    if current.names:
        close(current)
    for shard in shards:
        if shard.partial is None:
            write(shard)

    weight_map: dict[str, str] = {}
    for shard in shards:
        suffix = "" if len(shards) == 1 else f"-{shard.index + 1:05d}-of-{len(shards):05d}"
        filename = filename_pattern.format(suffix=suffix)
        os.replace(shard.partial, save_dir / filename)
        weight_map.update((name, filename) for name in shard.names)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    with open(save_dir / SAFE_WEIGHTS_INDEX_NAME, "w", encoding="utf-8") as f:
        f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")


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
