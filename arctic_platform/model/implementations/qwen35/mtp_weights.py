"""Preserve inactive Qwen prediction weights for full HF checkpoint export."""

from __future__ import annotations

import json
import os
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import nn

_STATE_ATTRIBUTE = "_ap_preserved_mtp_weights"
_BASE_NAMES = (
    "fc.weight",
    "norm.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
)
_LAYER_NAMES = (
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate.weight",
    "mlp.experts.gate_up_proj",
    "mlp.experts.down_proj",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
    "mlp.shared_expert_gate.weight",
)
_QWEN4_BASE_NAMES = (
    "fc_embedding.weight",
    "fc_hidden.weight",
    "pre_fc_norm_embedding.weight",
    "pre_fc_norm_hidden.weight",
    "hyper_connection_mixer.hc_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight",
)
_QWEN4_LAYER_NAMES = (
    "attn_hyper_connection.block_inject_weight.weight",
    "attn_hyper_connection.hc_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight",
    "mlp_hyper_connection.hc_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight",
    "self_attn.indexer.index_qk_proj.weight",
    "self_attn.indexer.k_layernorm.weight",
    "self_attn.indexer.q_layernorm.weight",
    *_LAYER_NAMES[2:],
)


@dataclass(frozen=True)
class _PreservedWeight:
    name: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype


def _configured_names(model: nn.Module) -> set[str]:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    model_type = getattr(text_config, "model_type", None)
    if model_type == "qwen3_5_moe_text":
        base_names, layer_names = _BASE_NAMES, _LAYER_NAMES
    elif model_type == "qwen4_exp_text":
        base_names, layer_names = _QWEN4_BASE_NAMES, _QWEN4_LAYER_NAMES
    else:
        return set()
    count = getattr(text_config, "mtp_num_hidden_layers", 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError("Qwen mtp_num_hidden_layers must be a nonnegative integer")
    if count == 0:
        return set()
    return {f"mtp.{name}" for name in base_names} | {
        f"mtp.layers.{index}.{name}" for index in range(count) for name in layer_names
    }


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint index key: {key}")
        result[key] = value
    return result


def _capture_original_weights(
    source_path: Path, expected: set[str], *, prefixes: tuple[str, ...] = ("mtp.",)
) -> list[_PreservedWeight]:
    """The resolved original checkpoint must stay immutable through ordinary DCP loading."""
    preserved: list[_PreservedWeight] = []
    held: list[tuple[Path, int, os.stat_result]] = []
    with ExitStack() as stack:

        def hold(path: Path) -> int:
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            stack.callback(os.close, fd)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"original checkpoint source is not a regular file: {path}")
            if _file_identity(os.stat(path)) != _file_identity(metadata):
                raise ValueError(f"original checkpoint source changed while opening: {path}")
            held.append((path, fd, metadata))
            return fd

        index_path = source_path / "model.safetensors.index.json"
        indexed: dict[Path, set[str]] | None = None
        if os.path.lexists(index_path):
            index_fd = hold(index_path)
            with os.fdopen(os.dup(index_fd), encoding="utf-8") as handle:
                index = json.load(handle, object_pairs_hook=_unique_json_object)
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            if not isinstance(weight_map, dict) or not all(isinstance(name, str) for name in weight_map):
                raise ValueError("original checkpoint index requires a weight_map object")
            mtp_names = {name for name in weight_map if name.startswith(prefixes)}
            if mtp_names != expected:
                raise ValueError(
                    "original Qwen3.5 MTP index mismatch: "
                    f"missing={sorted(expected - mtp_names)}, unexpected={sorted(mtp_names - expected)}"
                )
            indexed = {}
            for name in sorted(mtp_names):
                shard_name = weight_map[name]
                if (
                    not isinstance(shard_name, str)
                    or not shard_name
                    or "\\" in shard_name
                    or Path(shard_name).is_absolute()
                    or ".." in Path(shard_name).parts
                    or Path(shard_name).suffix != ".safetensors"
                ):
                    raise ValueError(f"unsafe original MTP shard path for {name}: {shard_name!r}")
                indexed.setdefault(source_path / shard_name, set()).add(name)

        shards = sorted(indexed if indexed is not None else source_path.glob("*.safetensors"))
        names: set[str] = set()
        for shard in shards:
            shard_fd = hold(shard)
            # Read the held file generation, not a separately reopened mutable pathname.
            with safe_open(f"/proc/self/fd/{shard_fd}", framework="pt", device="cpu") as reader:
                shard_names = {name for name in reader.keys() if name.startswith(prefixes)}
                if indexed is not None and shard_names != indexed[shard]:
                    raise ValueError(f"original MTP shard does not match checkpoint index: {shard}")
                for name in sorted(shard_names):
                    if name in names:
                        raise ValueError(f"duplicate original MTP tensor: {name}")
                    names.add(name)
                    source = reader.get_tensor(name)
                    tensor = source.detach().clone().contiguous()
                    preserved.append(_PreservedWeight(name, tensor, tuple(source.shape), source.dtype))
        if names != expected:
            raise ValueError(
                "original Qwen3.5 MTP inventory mismatch: "
                f"missing={sorted(expected - names)}, unexpected={sorted(names - expected)}"
            )
        for path, fd, metadata in held:
            if _file_identity(os.fstat(fd)) != _file_identity(metadata) or _file_identity(
                os.stat(path)
            ) != _file_identity(metadata):
                raise ValueError(f"original checkpoint source changed during MTP capture: {path}")
    return preserved


def capture_mtp_weights(model: nn.Module, source_path: Path) -> None:
    """Capture only rank 0's original CPU tensors, before selecting a converted cache."""
    model.__dict__.pop(_STATE_ATTRIBUTE, None)
    expected = _configured_names(model)
    if not expected:
        return

    distributed = dist.is_initialized()
    is_master = not distributed or dist.get_rank() == 0
    preserved: list[_PreservedWeight] = []
    failure: list[str | None] = [None]
    cause = None
    if is_master:
        try:
            preserved = _capture_original_weights(source_path, expected)
        except Exception as exc:
            cause = exc
            failure[0] = f"{type(exc).__name__}: {exc}"

    # A source failure on the serializer rank must not strand peers in the following DCP barrier.
    if distributed:
        dist.broadcast_object_list(failure, src=0)
    if failure[0] is not None:
        raise RuntimeError(f"Cannot preserve original Qwen3.5 MTP weights: {failure[0]}") from cause
    if is_master:
        setattr(model, _STATE_ATTRIBUTE, tuple(sorted(preserved, key=lambda weight: weight.name)))


def iter_preserved_mtp_weights(model: nn.Module):
    """Yield unchanged original weights without registering parameters or buffers."""
    expected = _configured_names(model)
    if not expected:
        return
    preserved = getattr(model, _STATE_ATTRIBUTE, ())
    names = [weight.name for weight in preserved]
    if len(names) != len(set(names)) or set(names) != expected:
        raise RuntimeError("Full Qwen3.5 HF export requires captured original MTP weights")
    for weight in preserved:
        tensor = weight.tensor
        if (
            tuple(tensor.shape) != weight.shape
            or tensor.dtype != weight.dtype
            or tensor.device.type != "cpu"
            or not tensor.is_contiguous()
            or tensor.requires_grad
        ):
            raise RuntimeError(f"Preserved original MTP tensor changed layout or dtype: {weight.name}")
        yield weight.name, tensor
