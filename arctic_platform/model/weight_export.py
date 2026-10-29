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
"""Model checkpoint and sampler-weight export contracts."""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from typing import Literal
from weakref import WeakKeyDictionary

import torch
import torch.nn as nn

WeightFormat = Literal["hf", "vllm", "lora"]
WeightIterator = Iterator[tuple[str, torch.Tensor]]
WeightIteratorBuilder = Callable[[nn.Module], Callable[..., WeightIterator]]

PEFT_ADAPTER_DIRNAME = "default"

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightExportContract:
    """Family-owned builders for full HF and direct-vLLM weight streams."""

    hf: WeightIteratorBuilder | None = None
    vllm: WeightIteratorBuilder | None = None


_WEIGHT_EXPORTS: WeakKeyDictionary[nn.Module, WeightExportContract] = WeakKeyDictionary()


def register_weight_export(model: nn.Module, contract: WeightExportContract) -> None:
    _WEIGHT_EXPORTS[model] = contract


def transfer_weight_export(source: nn.Module, target: nn.Module) -> None:
    contract = _WEIGHT_EXPORTS.get(source)
    if contract is not None:
        _WEIGHT_EXPORTS[target] = contract


def weight_export_contract(model: nn.Module) -> WeightExportContract | None:
    return _WEIGHT_EXPORTS.get(model)


def supports_weight_format(model: nn.Module, weight_format: WeightFormat) -> bool:
    if weight_format == "lora":
        return is_peft_model(model)
    contract = weight_export_contract(model)
    if contract is not None:
        if getattr(contract, weight_format) is not None:
            return True
        if weight_format == "vllm":
            return False
    return not _has_ep_sharded_parameters(model)


def strip_activation_checkpoint_segments(name: str) -> str:
    return name.replace("._checkpoint_wrapped_module", "")


_FUSED_EXPERT_WEIGHT_NAMES = frozenset({"w1", "w2", "w3"})


def hf_export_parameter_name(name: str) -> str | None:
    """Return the unwrapped HF name, or ``None`` for adapter tensors."""
    if "lora_" in name:
        return None
    name = strip_activation_checkpoint_segments(name)
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


def hf_checkpoint_name_map(model: nn.Module) -> dict[str, str]:
    """Map exported parameter names onto the names ``save_pretrained`` writes.

    Transformers renames checkpoint keys on load (for example, Transformers 5
    drops Siglip's ``vision_model`` segment). Weight sync streams live
    parameters, so it must apply the same reverse renames that saving does.
    Only names that change are returned. Names consumed by a tensor converter
    are left alone because they do not map one-to-one.
    """
    from transformers import PreTrainedModel
    from transformers.core_model_loading import PrefixChange
    from transformers.core_model_loading import WeightConverter
    from transformers.core_model_loading import WeightRenaming
    from transformers.core_model_loading import rename_source_key

    base = pretrained_module_for_hf_save(model)
    if not isinstance(base, PreTrainedModel):
        return {}
    conversions = getattr(base, "_weight_conversions", None)
    if conversions is None:
        from transformers.conversion_mapping import get_model_conversion_mapping

        # Without load-time history Transformers cannot tell whether a removed
        # prefix existed, so ``save_pretrained`` drops prefix changes too.
        conversions = [
            conversion
            for conversion in get_model_conversion_mapping(base, add_legacy=False)
            if not isinstance(conversion, PrefixChange)
        ]
    if not conversions:
        return {}

    reverse = [conversion.reverse_transform() for conversion in conversions[::-1]]
    renamings = [transform for transform in reverse if isinstance(transform, WeightRenaming)]
    converters = [transform for transform in reverse if isinstance(transform, WeightConverter)]
    names: dict[str, str] = {}
    for name, _ in model.named_parameters():
        export_name = hf_export_parameter_name(name)
        if export_name is None:
            continue
        checkpoint_name, converter_pattern = rename_source_key(export_name, renamings, converters, reverse=True)
        if converter_pattern is None and checkpoint_name != export_name:
            names[export_name] = checkpoint_name
    return names


def read_checkpoint_weight_names(checkpoint_dir: str) -> set[str]:
    """Tensor names in an HF checkpoint directory, or empty if unreadable.

    Only the index or safetensors headers are read.
    """
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = os.path.join(checkpoint_dir, index_name)
        if os.path.exists(index_path):
            try:
                with open(index_path) as handle:
                    return set(json.load(handle)["weight_map"])
            except (OSError, ValueError, KeyError):
                return set()

    shards = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not shards:
        return set()
    try:
        from safetensors import safe_open

        names: set[str] = set()
        for shard in shards:
            with safe_open(shard, framework="pt") as handle:
                names.update(handle.keys())
        return names
    except Exception:  # noqa: BLE001 - best-effort probe
        return set()


_HF_SYNC_NAME_MAPS: WeakKeyDictionary[nn.Module, tuple[str | None, dict[str, str]]] = WeakKeyDictionary()


def hf_sync_name_map(model: nn.Module, checkpoint_dir: str | None = None) -> dict[str, str]:
    """Renames that put generic ``weight_format="hf"`` names in the checkpoint's layout.

    Starts from :func:`hf_checkpoint_name_map`. With ``checkpoint_dir``, a rename
    is kept only if the checkpoint has the target name, and a partial translation
    raises: the hf receiver drops unmatched names silently. Cached per model.
    """
    cached = _HF_SYNC_NAME_MAPS.get(model)
    if cached is not None and cached[0] == checkpoint_dir:
        return cached[1]

    names = hf_checkpoint_name_map(model)
    saved = read_checkpoint_weight_names(checkpoint_dir) if checkpoint_dir else set()
    if saved:
        renames: dict[str, str] = {}
        unmapped: list[str] = []
        for name, _ in model.named_parameters():
            export_name = hf_export_parameter_name(name)
            if export_name is None:
                continue
            target = names.get(export_name, export_name)
            if target in saved:
                if target != export_name:
                    renames[export_name] = target
            # Tied-embedding checkpoints omit lm_head.
            elif export_name not in saved and export_name != "lm_head.weight":
                unmapped.append(export_name)
        if renames and unmapped:
            raise RuntimeError(
                f"weight_format='hf' name translation is incomplete: {len(renames)} parameter(s) map onto "
                f"{checkpoint_dir} but {len(unmapped)} do not (e.g. {', '.join(unmapped[:10])}). Was the model "
                "loaded with from_pretrained?"
            )
        names = renames
    if names:
        logger.info(
            "weight sync: translating %d parameter name(s) into the HF checkpoint layout (e.g. %s -> %s)",
            len(names),
            *next(iter(names.items())),
        )
    _HF_SYNC_NAME_MAPS[model] = (checkpoint_dir, names)
    return names


def requires_hf_weight_sync(model: nn.Module, checkpoint_dir: str | None = None) -> bool:
    """Whether direct vLLM copy would drop tensors, so sync must use ``weight_format="hf"``."""
    if getattr(model, "_requires_hf_weight_sync", False):
        return True
    contract = weight_export_contract(model)
    if contract is not None and contract.vllm is not None:
        return False
    return bool(hf_sync_name_map(model, checkpoint_dir))


def is_peft_model(model: Any) -> bool:
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        return True
    try:
        from peft import PeftModel

        return isinstance(model, PeftModel)
    except ImportError:
        return False


def is_peft_adapter_tensor_name(name: str) -> bool:
    return "lora_" in name or "modules_to_save" in name


def pretrained_module_for_hf_save(model: Any) -> Any:
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def _family_model_for_export(model: nn.Module) -> nn.Module:
    """Return the model export builders were written against.

    HF and direct-vLLM builders read class methods such as ``convert_layer_to_hf``
    from ``type(model)``. Those methods live on the family model. A PEFT wrapper
    does not define them, and instance forwarding does not satisfy the lookup.
    """
    if not is_peft_model(model):
        return model
    return pretrained_module_for_hf_save(model)


def pretrained_config_of(model: Any) -> Any:
    candidates: list[Any] = []
    get_base = getattr(model, "get_base_model", None)
    if callable(get_base):
        candidates.append(get_base())
    inner = getattr(model, "base_model", None)
    if inner is not None:
        candidates.append(getattr(inner, "model", inner))
    candidates.append(model)
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is not None and hasattr(config, "save_pretrained") and not getattr(config, "peft_type", None):
            return config
    return getattr(model, "config", None)


def checkpoint_peft_adapter_dir(model_dir: str) -> str | None:
    if not model_dir:
        return None
    path = os.path.join(model_dir, PEFT_ADAPTER_DIRNAME)
    if os.path.isfile(os.path.join(path, "adapter_config.json")):
        return path
    return None


def _is_ep_sharded_parameter(parameter: Any) -> bool:
    return getattr(parameter, "group_name", None) is not None and getattr(parameter, "allreduce", True) is False


def _has_ep_sharded_parameters(model: nn.Module) -> bool:
    return any(_is_ep_sharded_parameter(parameter) for parameter in model.parameters())


def _expert_parallel_group(parameter: Any):
    import deepspeed.utils.groups as ds_groups

    return ds_groups._get_expert_parallel_group(parameter.group_name)


def _gather_ep_parameter(parameter: Any, *, dim: int) -> torch.Tensor:
    import torch.distributed as dist

    group = _expert_parallel_group(parameter)
    local = parameter.data.contiguous()
    shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=group))]
    dist.all_gather(shards, local, group=group)
    return torch.cat(shards, dim=dim)


def gather_peft_adapter_state_dict(model: nn.Module, *, rank: int, is_zero3: bool) -> dict[str, torch.Tensor]:
    """Gather a PEFT adapter pack onto rank zero."""
    adapter_parameters = [
        (name, parameter) for name, parameter in model.named_parameters() if is_peft_adapter_tensor_name(name)
    ]
    if is_zero3:
        ep_parameters = [parameter for _, parameter in adapter_parameters if _is_ep_sharded_parameter(parameter)]
        if ep_parameters:
            raise NotImplementedError(
                "LoRA adapter checkpoints do not support ZeRO-3 with expert-parallel sharded adapters."
            )
        import deepspeed

        parameters = [parameter for _, parameter in adapter_parameters]
        if not parameters:
            return {}
        with deepspeed.zero.GatheredParameters(parameters, modifier_rank=0):
            if rank != 0:
                return {}
            return {
                strip_activation_checkpoint_segments(name): parameter.detach().cpu().contiguous()
                for name, parameter in adapter_parameters
            }

    output: dict[str, torch.Tensor] = {}
    for name, parameter in adapter_parameters:
        export_name = strip_activation_checkpoint_segments(name)
        if _is_ep_sharded_parameter(parameter):
            import torch.distributed as dist

            if not dist.is_initialized():
                raise RuntimeError(f"EP-sharded LoRA parameter {name!r} needs a process group")
            gather_dim = 1 if "lora_B" in export_name.split(".") else 0
            gathered = _gather_ep_parameter(parameter, dim=gather_dim)
            if rank == 0:
                output[export_name] = gathered.detach().cpu().contiguous()
        elif rank == 0:
            output[export_name] = parameter.detach().cpu().contiguous()
    return output


def save_peft_adapters(model: nn.Module, checkpoint_dir: str, *, is_zero3: bool = False) -> None:
    """Collectively write a PEFT adapter pack under ``checkpoint_dir/default``."""
    if not is_peft_model(model):
        return
    import torch.distributed as dist

    rank = dist.get_rank() if dist.is_initialized() else 0
    state = gather_peft_adapter_state_dict(model, rank=rank, is_zero3=is_zero3)
    if rank != 0:
        return
    adapter_dir = os.path.join(checkpoint_dir, PEFT_ADAPTER_DIRNAME)
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir, state_dict=state, safe_serialization=True)


def save_hf_pretrained(model: nn.Module, output_dir: str) -> None:
    """Write an unmerged HF base checkpoint on the calling rank."""
    base = pretrained_module_for_hf_save(model)
    if not is_peft_model(model):
        base.save_pretrained(output_dir, safe_serialization=True, max_shard_size="4GB")
        return
    state = {
        export_name: parameter.detach().cpu().contiguous()
        for name, parameter in model.named_parameters()
        if (export_name := hf_export_parameter_name(name)) is not None
    }
    base.save_pretrained(
        output_dir,
        state_dict=state,
        safe_serialization=True,
        max_shard_size="4GB",
    )


def save_exported_state_dict(state_dict: dict[str, torch.Tensor], output_dir: str) -> None:
    """Write an exported HF state dict using AP's sharded safetensors layout."""
    from pathlib import Path

    from arctic_platform.model.implementations.moe.weights import save_state_dict

    save_state_dict(state_dict, Path(output_dir), save_sharded=True)


def validate_lora_sync_trainable_parameters(model: nn.Module) -> None:
    from arctic_platform.model.patches.peft import is_peft_lora_param

    unsupported = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and not is_peft_lora_param(name, parameter)
    ]
    if unsupported:
        raise NotImplementedError(
            "LoRA weight export supports only trainable lora_A/lora_B parameters. "
            f"Unsupported trainable parameters: {unsupported[:8]}"
        )


def _peft_lora_parameter_to_vllm_name(name: str) -> str:
    parts = strip_activation_checkpoint_segments(name).split(".")
    if len(parts) >= 3 and parts[-1] == "weight" and parts[-3] in ("lora_A", "lora_B"):
        parts = parts[:-2] + parts[-1:]
    return ".".join(parts)


_EXPERT_W_TO_VLLM_PROJ = {
    "w1": "gate_proj",
    "w2": "down_proj",
    "w3": "up_proj",
}


def _reshape_expert_lora_for_ep(local: torch.Tensor, kind: str, rank: int, *, label: str) -> torch.Tensor:
    if kind not in ("A", "B"):
        raise ValueError(f"kind must be 'A' or 'B', got {kind!r} ({label})")
    if local.ndim != 2:
        raise ValueError(f"expected expert LoRA to be 2-D flat, got {tuple(local.shape)} ({label})")
    if kind == "A":
        if local.shape[0] % rank:
            raise ValueError(f"lora_A dim0={local.shape[0]} not divisible by r={rank} ({label})")
        return local.view(local.shape[0] // rank, rank, local.shape[1])
    if local.shape[1] % rank:
        raise ValueError(f"lora_B dim1={local.shape[1]} not divisible by r={rank} ({label})")
    return local.view(local.shape[0], rank, local.shape[1] // rank).permute(2, 0, 1).contiguous()


def _expert_lora_vllm_key(experts_prefix: str, weight_name: str, kind: str) -> str:
    if weight_name not in _EXPERT_W_TO_VLLM_PROJ:
        raise ValueError(f"unknown expert LoRA parameter {weight_name!r}")
    if kind not in ("A", "B"):
        raise ValueError(f"kind must be 'A' or 'B', got {kind!r}")
    return f"{experts_prefix}.{weight_name}.lora_{kind}.weight"


def _peft_adapter_rank(model: nn.Module) -> int:
    peft_config = getattr(model, "peft_config", None)
    if isinstance(peft_config, dict) and peft_config:
        rank = getattr(next(iter(peft_config.values())), "r", None)
        if rank is not None and int(rank) > 0:
            return int(rank)
    raise RuntimeError("LoRA export requires model.peft_config[*].r")


def _iter_param_wrapper_expert_loras(model: nn.Module):
    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        return
    for module_name, module in model.named_modules():
        if not isinstance(module, ParamWrapper):
            continue
        weight_name = getattr(module, "parameter_name", None)
        if weight_name not in _EXPERT_W_TO_VLLM_PROJ:
            continue
        experts_prefix = module_name.replace(".base_layer", "")
        if not (experts_prefix.endswith(".experts") or experts_prefix == "experts"):
            continue
        for kind, bucket in (("A", module.lora_A), ("B", module.lora_B)):
            if bucket:
                weight = getattr(bucket[next(iter(bucket))], "weight", None)
                if weight is not None:
                    yield experts_prefix, weight_name, kind, weight


def iter_lora_weights(
    model: nn.Module,
    *,
    is_master: bool,
    is_zero3: bool,
) -> WeightIterator:
    """Yield vLLM LoRA tensors; every rank participates in EP gathers."""
    if is_zero3:
        raise NotImplementedError("weight_format='lora' does not support ZeRO-3")
    if not is_peft_model(model):
        raise ValueError("weight_format='lora' requires a PEFT model")
    validate_lora_sync_trainable_parameters(model)

    from arctic_platform.model.patches.peft import is_peft_lora_param

    lora_rank = _peft_adapter_rank(model)
    # Families whose module tree differs from the served checkpoint rename adapter tensors here.
    rename = getattr(type(_family_model_for_export(model)), "vllm_lora_name", None) or (lambda name: name)
    expert_parameter_ids: set[int] = set()
    for (
        experts_prefix,
        weight_name,
        kind,
        parameter,
    ) in _iter_param_wrapper_expert_loras(model):
        expert_parameter_ids.add(id(parameter))
        prefix = strip_activation_checkpoint_segments(experts_prefix).replace(".base_layer", "")
        label = f"{prefix}.{weight_name}.lora_{kind}"
        local = _reshape_expert_lora_for_ep(parameter.data.contiguous(), kind, lora_rank, label=label)
        if _is_ep_sharded_parameter(parameter):
            import torch.distributed as dist

            group = _expert_parallel_group(parameter)
            assert not is_master or dist.get_rank(group=group) == 0
            shards = [torch.empty_like(local) for _ in range(dist.get_world_size(group=group))]
            dist.all_gather(shards, local, group=group)
            if is_master:
                yield (
                    rename(_expert_lora_vllm_key(prefix, weight_name, kind)),
                    torch.cat(shards, dim=0).contiguous(),
                )
        elif is_master:
            yield rename(_expert_lora_vllm_key(prefix, weight_name, kind)), local

    for name, parameter in model.named_parameters():
        if not is_peft_lora_param(name, parameter) or id(parameter) in expert_parameter_ids:
            continue
        export_name = rename(_peft_lora_parameter_to_vllm_name(name))
        if _is_ep_sharded_parameter(parameter):
            import torch.distributed as dist

            group = _expert_parallel_group(parameter)
            assert not is_master or dist.get_rank(group=group) == 0
            shards = [torch.empty_like(parameter.data) for _ in range(dist.get_world_size(group=group))]
            dist.all_gather(shards, parameter.data.contiguous(), group=group)
            if is_master:
                yield export_name, torch.cat(shards, dim=0)
        elif is_master:
            yield export_name, parameter.data


def _iter_generic_weights(
    model: nn.Module,
    weight_format: WeightFormat,
    *,
    is_master: bool,
    is_zero3: bool,
    checkpoint_dir: str | None,
) -> WeightIterator:
    if _has_ep_sharded_parameters(model):
        raise RuntimeError("EP-sharded models require an AP weight-export contract")
    names = hf_sync_name_map(model, checkpoint_dir) if weight_format == "hf" else {}
    for name, parameter in model.named_parameters():
        if weight_format == "hf":
            export_name = hf_export_parameter_name(name)
            if export_name is None:
                continue
            export_name = names.get(export_name, export_name)
        else:
            export_name = strip_activation_checkpoint_segments(name)

        if is_zero3 and hasattr(parameter, "ds_id"):
            import deepspeed

            with deepspeed.zero.GatheredParameters(parameter):
                tensor = parameter.data.clone() if is_master else None
        else:
            tensor = parameter.data
        if is_master:
            assert tensor is not None
            yield export_name, tensor


def iter_model_weights(
    model: nn.Module,
    weight_format: WeightFormat,
    *,
    is_master: bool,
    is_zero3: bool,
    include_skipped: bool = False,
    checkpoint_dir: str | None = None,
) -> WeightIterator:
    """Yield model weights on ``is_master``; every rank must iterate in lockstep for the gathers.

    Models without a family contract stream ``named_parameters()``: unchanged for
    ``"vllm"`` (direct copy), and in the checkpoint's layout for ``"hf"`` (see
    :func:`hf_sync_name_map`, which uses ``checkpoint_dir`` when given).
    """
    if weight_format not in ("hf", "vllm", "lora"):
        raise ValueError(f"weight_format must be 'hf', 'vllm', or 'lora'; got {weight_format!r}")
    if include_skipped and weight_format != "hf":
        raise ValueError("include_skipped is only supported for weight_format='hf'")
    if weight_format == "lora":
        yield from iter_lora_weights(model, is_master=is_master, is_zero3=is_zero3)
        return

    contract = weight_export_contract(model)
    builder = getattr(contract, weight_format) if contract is not None else None
    if builder is not None:
        if is_zero3:
            raise NotImplementedError(f"weight_format={weight_format!r} does not support ZeRO-3")
        iterator = builder(_family_model_for_export(model))
        yield from (iterator(include_skipped=True) if weight_format == "hf" and include_skipped else iterator())
        return

    if weight_format == "vllm" and contract is not None:
        raise ValueError("this model does not provide direct vLLM weight export")
    yield from _iter_generic_weights(
        model,
        weight_format,
        is_master=is_master,
        is_zero3=is_zero3,
        checkpoint_dir=checkpoint_dir,
    )
