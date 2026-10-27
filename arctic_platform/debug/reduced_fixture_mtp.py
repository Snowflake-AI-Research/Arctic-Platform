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

"""Authorize MTP-free loading only for an exact reviewed correctness fixture."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from contextlib import ExitStack
from contextlib import contextmanager
from copy import deepcopy
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


def maybe_bind_reduced_fixture_mtp(payload, spec, published_fixture) -> None:
    """Bind only an existing validated pretrained reduced publication."""
    training = payload["training_config"]
    prime = training.get("prime_rl") or {}
    debug = prime.get("debug") or {}
    if "reduced_fixture_mtp" in debug:
        raise ValueError("reduced fixture MTP descriptor must not be supplied or overwritten")
    if (
        published_fixture is None
        or training.get("model_provider") != "prime_rl"
        or type(spec.model.seed) is not int
        or spec.model.seed != 0
    ):
        return
    config_row = next(
        (row for row in published_fixture["files"] if row["path"] == "config.json"),
        None,
    )
    if config_row is None:
        raise ValueError("published fixture has no config binding")
    with _held_files(Path(spec.model.cache_path), [FixtureFile.model_validate(config_row)]) as held:
        config = _json_from_fd(held["config.json"])
        if _text_config(config).get("model_type") not in _MODEL_TYPES:
            return
    if (
        published_fixture.get("config_id") != spec.config_id
        or published_fixture.get("reference_path") != spec.model.cache_path
        or published_fixture.get("reviewed_hash") != spec.model.content_hash
        or published_fixture.get("model_name") != payload["model_name"]
    ):
        raise ValueError("reduced fixture does not match the reviewed spec and native model")
    request = ReducedFixtureMTP(
        schema_version=1,
        config_id=spec.config_id,
        fixture_locator=payload["model_name"],
        reference_path=spec.model.cache_path,
        reviewed_hash=spec.model.content_hash,
        fixture_sha256=published_fixture["fixture_sha256"],
        files=published_fixture["files"],
        source_checkpoint=spec.model.source_checkpoint,
        num_hidden_layers=spec.model.num_hidden_layers,
        vision_depth=spec.model.vision_depth,
        materialization="pretrained-prefix-reduction",
    )
    _validate_depths(config, request)
    updated = deepcopy(prime)
    updated["debug"] = deepcopy(debug)
    updated["debug"]["reduced_fixture_mtp"] = request.model_dump()
    training["prime_rl"] = updated


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
