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

"""Opt-in reduced-fixture authentication and router replay observations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import ExitStack
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from pathlib import Path
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

_MODEL_TYPES = {"qwen3_5_moe_text", "qwen4_exp_text", "glm5_next_text"}


class FixtureFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        path = Path(value)
        if (
            not value
            or "\\" in value
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != value
            or value == "."
            or any(part in {".cache", ".git"} for part in path.parts)
            or path.name == "_neutrino_manifest.json"
        ):
            raise ValueError("fixture manifest requires canonical relative file paths")
        return value


class ReducedFixtureMTP(BaseModel):
    """An explicit load-only authorization, not proof of the fixture's generation."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1]
    config_id: str = Field(min_length=1)
    fixture_locator: str = Field(min_length=1)
    reference_path: str = Field(min_length=1)
    reviewed_hash: str = Field(pattern=r"^[0-9a-f]{16}$")
    fixture_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: list[FixtureFile] = Field(min_length=1)
    source_checkpoint: str = Field(min_length=1)
    num_hidden_layers: int = Field(gt=0)
    vision_depth: int | None = Field(gt=0)
    materialization: Literal["pretrained-prefix-reduction"]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("fixture schema version must be an integer")
        return value

    @model_validator(mode="after")
    def _manifest(self):
        names = [item.path for item in self.files]
        if names != sorted(set(names)):
            raise ValueError("fixture manifest must be sorted with unique paths")
        if "config.json" not in names or "model.safetensors.index.json" not in names:
            raise ValueError("reduced MTP authorization requires an indexed fixture and config")
        if self.source_checkpoint == self.reference_path or not Path(self.reference_path).is_absolute():
            raise ValueError("reduced MTP authorization requires a separate absolute reference path")
        if _manifest_sha(self.files) != self.fixture_sha256:
            raise ValueError("fixture manifest digest differs")
        return self


def _manifest_sha(files) -> str:
    rows = [item.model_dump() if isinstance(item, FixtureFile) else item for item in files]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate fixture JSON key: {key}")
        result[key] = value
    return result


def _identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _file_sha(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    for chunk in iter(lambda: os.read(fd, 1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _json_from_fd(fd: int):
    if os.fstat(fd).st_size > 16 * 1024 * 1024:
        raise ValueError("fixture JSON exceeds the metadata limit")
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(os.dup(fd), encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=_unique_object)


@contextmanager
def _held_files(root: Path, files: list[FixtureFile]):
    held = []
    with ExitStack() as stack:
        for item in files:
            path = root / item.path
            fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
            stack.callback(os.close, fd)
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode) or _identity(os.stat(path)) != _identity(metadata):
                raise ValueError("fixture file generation differs")
            held.append((path, fd, metadata))
            if metadata.st_size != item.size_bytes or _file_sha(fd) != item.sha256:
                raise ValueError(f"fixture file bytes differ: {item.path}")
        yield {item.path: row[1] for item, row in zip(files, held)}
        for path, fd, metadata in held:
            if _identity(os.fstat(fd)) != _identity(metadata) or _identity(os.stat(path)) != _identity(metadata):
                raise ValueError("fixture file changed during authentication")


def _fixture_names(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name != "_neutrino_manifest.json"
        and not any(part in {".cache", ".git"} for part in path.relative_to(root).parts)
    )


def _text_config(config):
    if not isinstance(config, dict) or not isinstance(config.get("text_config", config), dict):
        raise ValueError("fixture requires a text configuration object")
    return config.get("text_config", config)


def _validate_depths(config, request: ReducedFixtureMTP) -> None:
    text = _text_config(config)
    vision = config.get("vision_config")
    depth = vision.get("depth") if isinstance(vision, dict) else None
    if (
        text.get("model_type") not in _MODEL_TYPES
        or type(text.get("num_hidden_layers")) is not int
        or text["num_hidden_layers"] != request.num_hidden_layers
        or depth != request.vision_depth
        or (depth is not None and type(depth) is not int)
    ):
        raise ValueError("fixture family or reduced depths differ")


def _authenticate_fixture(model, root: Path, config, request: ReducedFixtureMTP) -> bool:
    from safetensors import safe_open

    from arctic_platform.model.implementations.glm53.auxiliary_weights import _layer_prefixes
    from arctic_platform.model.implementations.qwen35.mtp_weights import _configured_names

    if config.name != request.fixture_locator:
        raise ValueError("native model locator differs from the fixture authorization")
    root_metadata = root.stat()
    if not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("fixture snapshot is not a directory")
    names = [item.path for item in request.files]
    if _fixture_names(root) != names:
        raise ValueError("fixture snapshot inventory differs")
    runtime_text = getattr(model.config, "text_config", model.config)
    glm = getattr(runtime_text, "model_type", None) == "glm5_next_text"
    prefixes = _layer_prefixes(model.config) if glm else ("mtp.",)
    expected = (
        {prefix + suffix for prefix in prefixes for suffix in ("eh_proj.weight", "enorm.weight", "hnorm.weight")}
        if glm
        else _configured_names(model)
    )
    if not expected:
        raise ValueError("fixture authorization requires configured prediction weights")
    with _held_files(root, request.files) as held:
        source_config = _json_from_fd(held["config.json"])
        _validate_depths(source_config, request)
        source_text = _text_config(source_config)
        count_field = "num_nextn_predict_layers" if glm else "mtp_num_hidden_layers"
        if (
            getattr(runtime_text, "model_type", None) != source_text.get("model_type")
            or type(getattr(runtime_text, "num_hidden_layers", None)) is not int
            or getattr(runtime_text, "num_hidden_layers", None) != request.num_hidden_layers
            or type(source_text.get(count_field)) is not int
            or type(getattr(runtime_text, count_field, None)) is not int
            or getattr(runtime_text, count_field, None) != source_text[count_field]
        ):
            raise ValueError("runtime model differs from the reduced fixture configuration")
        index = _json_from_fd(held["model.safetensors.index.json"])
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("fixture index requires a nonempty weight_map")
        indexed: dict[str, set[str]] = {}
        for name, shard in weight_map.items():
            if not isinstance(name, str) or not isinstance(shard, str) or not shard.endswith(".safetensors"):
                raise ValueError("fixture index has an invalid tensor or shard")
            if shard not in held:
                raise ValueError("fixture index references an unauthenticated shard")
            indexed.setdefault(shard, set()).add(name)
        shards = {name for name in held if name.endswith(".safetensors")}
        if shards != set(indexed):
            raise ValueError("fixture shard inventory differs from its index")
        for shard, tensor_names in indexed.items():
            with safe_open(f"/proc/self/fd/{held[shard]}", framework="pt", device="cpu") as reader:
                if set(reader.keys()) != tensor_names:
                    raise ValueError("fixture tensor headers differ from its index")
        mtp_names = {name for name in weight_map if name.startswith(prefixes)}
        if glm:
            layer_prefix = "model.language_model.layers."
            inactive = {
                name
                for name in weight_map
                if name.startswith(layer_prefix)
                and name[len(layer_prefix) :].partition(".")[0].isdigit()
                and int(name[len(layer_prefix) :].partition(".")[0]) >= request.num_hidden_layers
            }
            if inactive != mtp_names or (mtp_names and not expected <= mtp_names):
                raise ValueError("fixture has partial or unexpected GLM prediction weights")
        elif mtp_names and mtp_names != expected:
            raise ValueError("fixture has partial or unexpected MTP weights")
        if _fixture_names(root) != names or _identity(root.stat()) != _identity(root_metadata):
            raise ValueError("fixture snapshot changed during authentication")
    return not mtp_names


def maybe_capture_reduced_fixture_mtp(model, source_path: Path, config) -> None:
    """Authenticate each rank's snapshot; the source remains immutable through later DCP loading."""
    from arctic_platform.model.implementations.glm53 import auxiliary_weights
    from arctic_platform.model.implementations.qwen35 import mtp_weights

    model.__dict__.pop("_ap_preserved_mtp_weights", None)
    model.__dict__.pop(auxiliary_weights.STATE_ATTRIBUTE, None)
    supplied = getattr(config.debug, "reduced_fixture_mtp", None)
    if supplied is None:
        mtp_weights.capture_mtp_weights(model, source_path)
        auxiliary_weights.capture_auxiliary_weights(model, source_path)
        return
    import torch.distributed as dist

    distributed = dist.is_initialized()
    result: list[Any] = [None, False]
    cause = None
    try:
        request = ReducedFixtureMTP.model_validate(supplied)
        result[1] = _authenticate_fixture(model, Path(source_path), config, request)
    except Exception as exc:
        cause = exc
        result[0] = f"{type(exc).__name__}: {exc}"
    results: list[Any] = [result]
    if distributed:
        results = [None] * dist.get_world_size()
        dist.all_gather_object(results, result)
    failures = [row[0] for row in results if row[0] is not None]
    if failures:
        raise RuntimeError(f"Cannot authenticate reduced prediction fixture: {failures}") from cause
    if any(row[1] != result[1] for row in results):
        raise RuntimeError("Reduced fixture prediction-weight decisions differ across ranks")
    if not result[1]:
        mtp_weights.capture_mtp_weights(model, source_path)
        auxiliary_weights.capture_auxiliary_weights(model, source_path)


TRAINING_OPTION = "arctic_platform_router_replay_trace"
TRACE_SCHEMA = "arctic-platform-router-replay-call-v2"
_LAYER_INDEX = re.compile(r"(?:^|\.)layers\.(\d+)\.")
_active: RouterReplayTrace | None = None
_publication: dict | None = None


def maybe_configure_router_replay_sampling(sampling, *, verify):
    """Enable capture only for the replay check, preserving the configured cache limits."""
    if not verify:
        return sampling
    inference = deepcopy(sampling.arctic_inference_config or {})
    inference.setdefault("router_replay", {})["enabled"] = True
    return sampling.model_copy(update={"arctic_inference_config": inference}, deep=True)


def router_replay_model_inventory(config: dict) -> dict:
    """Bind required layers to resolved model config, never to discovered router modules."""
    if not isinstance(config, dict):
        raise NotImplementedError("router-replay architectural coverage requires a model config mapping")
    text = config
    if "text_config" in config:
        if config.get("model_type") != "qwen3_5_moe" or not isinstance(config["text_config"], dict):
            raise NotImplementedError("unverified composite router-replay model config")
        text = config["text_config"]
    family = text.get("model_type")
    depth = text.get("num_hidden_layers")
    top_k = text.get("num_experts_per_tok")
    if type(depth) is not int or depth <= 0 or type(top_k) is not int or top_k <= 0:
        raise ValueError("router-replay config requires positive integer layer count and routing top-k")
    if family in ("qwen3_moe", "qwen3_5_moe_text"):
        experts = text.get("num_experts")
    elif family in ("glm4_moe", "glm_moe_dsa"):
        experts = text.get("n_routed_experts")
    else:
        raise NotImplementedError(f"unverified router-replay layer predicate for model_type={family!r}")
    if type(experts) is not int or experts <= 0 or top_k > experts:
        raise ValueError("router-replay config requires a positive expert count at least as large as routing top-k")
    if family == "qwen3_moe":
        dense = text.get("mlp_only_layers")
        step = text.get("decoder_sparse_step")
        if (
            not isinstance(dense, list)
            or any(type(layer) is not int or layer < 0 or layer >= depth for layer in dense)
            or len(set(dense)) != len(dense)
            or type(step) is not int
            or step <= 0
        ):
            raise ValueError("invalid Qwen3-MoE dense-layer list or decoder_sparse_step")
        layers = [layer for layer in range(depth) if layer not in dense and (layer + 1) % step == 0]
    elif family in ("glm4_moe", "glm_moe_dsa"):
        prefix = text.get("first_k_dense_replace")
        if type(prefix) is not int or prefix < 0 or prefix > depth:
            raise ValueError("invalid GLM MoE dense-layer prefix")
        layers = list(range(prefix, depth))
    else:
        layers = list(range(depth))
    if not layers:
        raise NotImplementedError("router-replay config contains no routed decoder layers")
    resolved = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return {
        "model_type": family,
        "num_hidden_layers": depth,
        "num_experts_per_tok": top_k,
        "num_experts": experts,
        "expected_layers": layers,
        "resolved_model_config_sha256": hashlib.sha256(resolved).hexdigest(),
    }


class RouterReplayTrace:
    def __init__(self, model: Any) -> None:
        self.received: dict[str, list] = {}
        self.calls: list[dict] = []
        self.unattributed: list[dict] = []
        self._call_open = False
        self._hook_handles: list[Any] = []
        config = getattr(model, "config", None)
        if config is None:
            raise NotImplementedError("router-replay execution model has no independently bound config")
        config = config if isinstance(config, dict) else config.to_dict()
        self.model_inventory = router_replay_model_inventory(config)
        inventory = {}
        for name, module in model.named_modules():
            if type(module).__name__ != "TokenChoiceTopKRouter":
                continue
            match = _LAYER_INDEX.search(name)
            if match is None:
                raise ValueError(f"router-replay inventory has an unknown decoder layer index: {name}")
            layer = int(match.group(1))
            if layer in inventory:
                raise ValueError(f"router-replay inventory has a duplicate decoder layer index: {layer}")
            inventory[layer] = module
        if len(inventory) == 0:
            raise ValueError("router-replay inventory contains no TokenChoiceTopKRouter modules")
        if sorted(inventory) != self.model_inventory["expected_layers"]:
            raise ValueError("installed router modules differ from the model-config architectural roster")
        self.expected_layers = tuple(self.model_inventory["expected_layers"])
        self._router_layers = {id(module): layer for layer, module in inventory.items()}
        try:
            for module in inventory.values():
                self._hook_handles.append(module.register_forward_hook(self._record_router, with_kwargs=True))
        except BaseException:
            self.close()
            raise

    def _record_router(self, router: Any, args: tuple, kwargs: dict, output: Any) -> None:
        routed = kwargs.get("routed_experts")
        if routed is None and len(args) >= 3:
            routed = args[2]
        if not isinstance(output, tuple) or len(output) < 2:
            raise TypeError("TokenChoiceTopKRouter must return selected expert indices as output 1")
        self.record_router(router, output[1], replayed=routed is not None)

    def close(self) -> None:
        handles, self._hook_handles = self._hook_handles, []
        failure = None
        for handle in handles:
            try:
                handle.remove()
            except Exception as exc:
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    def record_received(self, sample_ids, routed_experts_list) -> None:
        import torch

        for sample_id, routed in zip(sample_ids, routed_experts_list, strict=True):
            key = str(sample_id)
            if key in self.received:
                raise ValueError(f"duplicate router-replay RX observation for {key}")
            self.received[key] = routed.detach().to(device="cpu", dtype=torch.int64).tolist()

    def begin_call(self, model_kwargs: dict, *, padding: bool) -> None:
        import torch

        routed = model_kwargs.get("routed_experts")
        self.calls.append(
            {
                "padding": bool(padding),
                "input_ids": model_kwargs["input_ids"].detach().reshape(-1).cpu().tolist(),
                "position_ids": model_kwargs["position_ids"].detach().reshape(-1).cpu().tolist(),
                "routed_experts": (
                    None
                    if routed is None
                    else routed.detach().reshape(-1, *routed.shape[-2:]).to(device="cpu", dtype=torch.int64).tolist()
                ),
                "router": [],
            }
        )
        self._call_open = True

    def record_router(self, router: Any, indices: Any, *, replayed: bool) -> None:
        import torch

        layer = self._router_layers.get(id(router))
        if layer is None or layer not in self.expected_layers:
            raise ValueError("router-replay observation has an unknown inventory layer")
        entry = {
            "layer": layer,
            "phase": "recompute" if torch._C._current_graph_task_id() != -1 else "forward",
            "replayed": bool(replayed),
            "indices": indices.detach().to(device="cpu", dtype=torch.int64).tolist(),
        }
        if self._call_open:
            self.calls[-1]["router"].append(entry)
        else:
            self.unattributed.append(entry)
            if _publication is not None:
                raise RuntimeError("router-replay router ran outside an open model call")

    def export(self, rank: int) -> dict:
        return {
            "rank": int(rank),
            "expected_layers": list(self.expected_layers),
            "model_inventory": deepcopy(self.model_inventory),
            "received": self.received,
            "calls": self.calls,
            "unattributed_router_calls": self.unattributed,
        }


def start(model: Any) -> RouterReplayTrace:
    global _active
    if _active is not None:
        raise RuntimeError("router-replay trace already active")
    _active = RouterReplayTrace(model)
    return _active


def active() -> RouterReplayTrace | None:
    return _active


def maybe_finish_router_replay_trace(rank: int) -> None:
    """Release only this process's owned trace; never publish an unfinished call."""
    global _active, _publication
    if _publication is not None and (type(rank) is not int or _publication["rank"] != rank):
        raise RuntimeError("router-replay cleanup rank differs from the active publication rank")
    trace, _active = _active, None
    _publication = None
    if trace is not None:
        trace.close()


def maybe_cleanup_router_replay_on_failure(function=None, *, runtime: bool = False):
    """Select an owned failure boundary without changing the native return or exception.

    The ordinary selector returns the original callable when no publication is active.
    Import-time decorators must use ``runtime=True``: engine initialization can enable
    publication only after that decorated callable has already been entered.
    """
    if function is None:
        return lambda native: maybe_cleanup_router_replay_on_failure(native, runtime=runtime)
    if not runtime and _publication is None:
        return function

    @wraps(function)
    def observed(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BaseException:
            publication = _publication
            if publication is not None:
                try:
                    maybe_finish_router_replay_trace(publication["rank"])
                except BaseException:
                    pass
            raise

    return observed


def maybe_finish_router_replay_after_destroy(function):
    """Run native teardown before closing observers; preserve native teardown failures."""
    native = maybe_cleanup_router_replay_on_failure(function, runtime=True)

    @wraps(function)
    def observed(*args, **kwargs):
        result = native(*args, **kwargs)
        publication = _publication
        if publication is not None:
            maybe_finish_router_replay_trace(publication["rank"])
        return result

    return observed


def finish(rank: int) -> dict | None:
    global _active, _publication
    trace, _active = _active, None
    _publication = None
    try:
        return None if trace is None else trace.export(rank)
    finally:
        if trace is not None:
            trace.close()


def record_router(router: Any, indices: Any, *, replayed: bool) -> None:
    if _active is not None:
        _active.record_router(router, indices, replayed=replayed)


def maybe_configure_router_replay_trace(ds_config: dict, model: Any, rank: int) -> dict:
    """Consume a request-scoped option before passing configuration to DeepSpeed."""
    global _publication
    if TRAINING_OPTION not in ds_config:
        return ds_config
    option = ds_config[TRAINING_OPTION]
    if not isinstance(option, dict) or set(option) != {"request_id", "output_dir"}:
        raise ValueError(f"{TRAINING_OPTION} requires request_id and output_dir")
    request_id = option["request_id"]
    if not isinstance(request_id, str) or re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", request_id) is None:
        raise ValueError("router-replay request_id must be a nonempty safe identifier")
    root = Path(option["output_dir"]).resolve()
    if not root.is_relative_to("/data-fast"):
        raise ValueError("router-replay evidence must stay under /data-fast")
    if type(rank) is not int or rank < 0:
        raise ValueError("router-replay rank must be a nonnegative integer")
    start(model)
    _publication = {
        "request_id": request_id,
        "output_dir": root / request_id,
        "rank": rank,
        "completed": 0,
    }
    return {key: value for key, value in ds_config.items() if key != TRAINING_OPTION}


def maybe_record_router_replay_received(sample_id: str, routed: Any) -> None:
    """Observe the value actually removed from the trainer's RX cache."""
    if _publication is not None:
        trace = active()
        if trace is None:
            raise RuntimeError("router-replay publication has no active trace")
        trace.record_received([sample_id], [routed])


def maybe_begin_router_replay_call(engine: Any, model_kwargs: dict, *, backward: bool | str) -> None:
    """Capture every training model call, including duplicated padding schedule calls."""
    if _publication is None or backward is False:
        return
    trace = active()
    if trace is None:
        raise RuntimeError("router-replay publication has no active trace")
    if getattr(engine, "global_rank", None) != _publication["rank"]:
        raise RuntimeError("router-replay trace rank differs from the executing engine")
    if len(trace.calls) != _publication["completed"]:
        raise RuntimeError("previous router-replay model call has no successful backward")
    trace.begin_call(model_kwargs, padding=False)


def maybe_complete_router_replay_call(engine: Any) -> None:
    """Publish one immutable call only after its actual backward returned successfully."""
    if _publication is None:
        return
    trace = active()
    if trace is None:
        raise RuntimeError("router-replay publication has no active trace")
    rank = _publication["rank"]
    if getattr(engine, "global_rank", None) != rank:
        raise RuntimeError("router-replay completion rank differs from the executing engine")
    index = _publication["completed"]
    if len(trace.calls) != index + 1:
        raise RuntimeError("router-replay backward has no unique preceding model call")
    root = _publication["output_dir"]
    root.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": TRACE_SCHEMA,
        "request_id": _publication["request_id"],
        "rank": rank,
        "call_index": index,
        "expected_layers": list(trace.expected_layers),
        "model_inventory": deepcopy(trace.model_inventory),
        "received": trace.received,
        "calls": [trace.calls[index]],
        "unattributed_router_calls": trace.unattributed,
    }
    with (root / f"rank-{rank}-call-{index}.json").open("x") as stream:
        json.dump(record, stream, separators=(",", ":"))
        stream.write("\n")
    _publication["completed"] += 1
    trace._call_open = False
