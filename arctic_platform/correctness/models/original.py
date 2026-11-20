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

"""Original-checkpoint identity captured by onboarding; identity does not confer test applicability."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from types import SimpleNamespace
from typing import Mapping

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.harness import storage
from arctic_platform.correctness.harness.storage import require_data_fast_path

from ..harness.config import LoadedConfig
from ..harness.config import TestSpec
from ..harness.config import config_checksum


@dataclass(frozen=True)
class OriginalModelMetadata:
    model_name: str
    config_id: str | None
    source_checkpoints: tuple[str, ...]
    revision: str
    config_sha256: str
    index_sha256: str | None
    layer_pattern: str
    tensor_count: int | None
    shard_count: int | None
    payload_bytes: int | None
    vocabulary_shape: tuple[int, ...] | None
    schema_sha256: str | None
    dtype_bytes: Mapping[str, int]
    additional_config_ids: tuple[str, ...] = ()
    weights_sha256: str | None = None


ORIGINAL_MODELS: Mapping[str, OriginalModelMetadata] = MappingProxyType(
    {
        "Qwen/Qwen3-0.6B": OriginalModelMetadata(
            model_name="Qwen/Qwen3-0.6B",
            config_id=None,
            source_checkpoints=("Qwen/Qwen3-0.6B",),
            revision="c1899de289a04d12100db370d81485cdf75e47ca",
            config_sha256="660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
            index_sha256=None,
            layer_pattern=r"model\.layers\.(\d+)\.",
            tensor_count=311,
            shard_count=1,
            payload_bytes=1503264768,
            vocabulary_shape=(151936, 1024),
            schema_sha256="99ea7fbe3cd557925fcba2ad843352e9c612b0484560f25b4eb6d07fe1ca6703",
            dtype_bytes=MappingProxyType({"BF16": 2}),
            weights_sha256="f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
        ),
        "Qwen/Qwen3-8B": OriginalModelMetadata(
            model_name="Qwen/Qwen3-8B",
            config_id=None,
            source_checkpoints=("Qwen/Qwen3-8B",),
            revision="b968826d9c46dd6066d109eabc6255188de91218",
            config_sha256="f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
            index_sha256=None,
            layer_pattern=r"model\.layers\.(\d+)\.",
            tensor_count=None,
            shard_count=None,
            payload_bytes=None,
            vocabulary_shape=None,
            schema_sha256=None,
            dtype_bytes=MappingProxyType({}),
        ),
        "Qwen/Qwen3.8-27B": OriginalModelMetadata(
            model_name="Qwen/Qwen3.8-27B",
            config_id="qwen3.8-27b-b200-train-sft-8gpus-2k",
            source_checkpoints=("Qwen/Qwen3.8-27B", "/data-fast/base-models/Qwen/Qwen3.8-27B"),
            revision="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
            config_sha256="191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab",
            index_sha256="77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df",
            layer_pattern=r"model\.language_model\.layers\.(\d+)\.",
            tensor_count=1199,
            shard_count=18,
            payload_bytes=55562855904,
            vocabulary_shape=(248320, 5120),
            schema_sha256="9c0eb904ed294a62882adb287b81e36fdb48e5fb8e006ee25b084341ac78fd55",
            dtype_bytes=MappingProxyType({"BF16": 2, "F32": 4}),
        ),
        "Qwen/Qwen3.8-Flash-Next": OriginalModelMetadata(
            model_name="Qwen/Qwen3.8-Flash-Next",
            config_id="qwen3.8-flash-next-h200-train-sft-full-16gpus-sp16-256k",
            source_checkpoints=(
                "Qwen/Qwen3.8-Flash-Next",
                (
                    "/checkpoint/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/"
                    "snapshots/de4b8e4d43b917e7706784d8bb445c9af86a3540"
                ),
            ),
            revision="de4b8e4d43b917e7706784d8bb445c9af86a3540",
            config_sha256="889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b",
            index_sha256="99e815241ef03325536b0aaa4441deea45174c17fae31e10f0bb456410c590de",
            layer_pattern=r"model\.language_model\.layers\.(\d+)\.",
            tensor_count=1658,
            shard_count=131,
            payload_bytes=359999963128,
            vocabulary_shape=(248320, 2560),
            schema_sha256="1d55a6d3f64b1b2b004302864c0e74e11871ddf9f3481b10fc21fc8b0ec53d26",
            dtype_bytes=MappingProxyType({"BF16": 2, "I64": 8}),
            additional_config_ids=("qwen3.8-flash-next-h200-train-sft-full-8gpus-sp8-256k",),
        ),
    }
)


def _original_model(cfg: LoadedConfig, spec: TestSpec) -> OriginalModelMetadata:
    metadata = ORIGINAL_MODELS.get(cfg.model_name)
    if (
        metadata is None
        or (
            metadata.config_id is not None
            and cfg.config_id not in (metadata.config_id, *metadata.additional_config_ids)
        )
        or spec.model.source_checkpoint not in metadata.source_checkpoints
    ):
        raise ValueError(
            f"local full model mode requires an onboarded original identity for {cfg.model_name!r}; "
            "run full-model onboarding to capture it automatically; a reduced fixture is not a full-model substitute"
        )
    return metadata


def _admission_path(cfg: LoadedConfig, records_dir: Path | None) -> Path:
    ctx = SimpleNamespace(cfg=cfg, reference_records_dir=records_dir or storage.DEFAULT_RECORDS_DIR)
    return (
        storage._config_dir(ctx, {"config_id": cfg.config_id, "check_id": "e2e-sft-train-validate"})
        / "original-model.json"
    )


def _full_model_environment(backend: CortexConfig) -> Path:
    from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway

    require_local_cortex_gateway(backend)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("DSS_ALLOW_REMOTE_MODELS") != "1":
        raise ValueError("local full model requires offline HF cache and DSS_ALLOW_REMOTE_MODELS=1")
    return require_data_fast_path(os.environ.get("HF_HUB_CACHE", ""), label="offline full model cache")


def _source_root(cfg: LoadedConfig, cache: Path, revision: str | None) -> tuple[Path, Path, str | None]:
    source = Path(cfg.model_name).expanduser()
    if source.is_dir():
        root = require_data_fast_path(source, label="original full model")
        return root, root, None
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(cfg.model_name, cache_dir=str(cache), revision=revision, local_files_only=True))
    canonical = require_data_fast_path(root, label="offline full model snapshot")
    if (
        not canonical.is_relative_to(cache)
        or canonical.parent.name != "snapshots"
        or canonical.parent.parent.name != "models--" + cfg.model_name.replace("/", "--")
        or re.fullmatch(r"[0-9a-f]{40}", canonical.name) is None
        or revision is not None
        and canonical.name != revision
    ):
        raise ValueError("offline full model snapshot does not match the original model and revision")
    return canonical, cache, canonical.name


def _checkpoint_identity(root: Path, boundary: Path, reviewed: OriginalModelMetadata | None = None) -> dict:
    """Stream file digests and inspect tensor headers without loading tensor payloads into memory."""
    from safetensors import safe_open

    def asset(relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {".", ".."} for part in path.parts):
            raise ValueError("original full model has an unsafe asset path")
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(boundary) or not resolved.is_file() or not resolved.stat().st_size:
            raise ValueError(f"original full model asset is missing, empty or outside its source: {relative}")
        return resolved

    def fingerprint(path: Path) -> tuple:
        status = path.stat()
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns, status.st_ctime_ns)

    def filenames() -> list[str]:
        return sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file()
            and (
                path.suffix in {".safetensors", ".json"}
                or path.name in {"tokenizer.model", "spiece.model", "vocab.txt", "merges.txt"}
            )
        )

    names = filenames()
    paths = {name: asset(name) for name in names}
    before = {name: fingerprint(path) for name, path in paths.items()}
    config = storage._strict_record_json(asset("config.json").read_bytes())
    if not isinstance(config, dict) or not isinstance(config.get("text_config", config), dict):
        raise ValueError("original full model requires a model configuration object")
    text = config.get("text_config", config)
    layers = text.get("num_hidden_layers")
    if layers is None:
        from transformers import AutoConfig

        normalized = dict(text)
        model_type = normalized.pop("model_type", config.get("model_type"))
        layers = getattr(AutoConfig.for_model(model_type, **normalized), "num_hidden_layers", None)
    if type(layers) is not int or layers <= 0:
        raise ValueError("original full model requires a positive layer count")
    asset("tokenizer_config.json")
    if not set(names).intersection({"tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "vocab.txt"}):
        raise ValueError("original full model tokenizer assets are missing")
    if "model.safetensors.index.json" in names:
        index = storage._strict_record_json(paths["model.safetensors.index.json"].read_bytes())
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if (
            not isinstance(mapping, dict)
            or not mapping
            or not all(
                isinstance(name, str) and name and isinstance(shard, str) and shard.endswith(".safetensors")
                for name, shard in mapping.items()
            )
        ):
            raise ValueError("original full model requires a nonempty safetensors weight index")
        shards = sorted(set(mapping.values()))
    else:
        mapping = None
        shards = ["model.safetensors"]
    if set(shards) != {name for name in names if name.endswith(".safetensors")}:
        raise ValueError("original full model checkpoint has missing or unindexed shards")
    schema = {}
    for shard in shards:
        with safe_open(asset(shard), framework="pt", device="cpu") as stream:
            keys = set(stream.keys())
            if (
                not keys
                or schema.keys() & keys
                or (mapping is not None and keys != {name for name, filename in mapping.items() if filename == shard})
            ):
                raise ValueError("original full model tensor keys do not match its complete index")
            for name in keys:
                view = stream.get_slice(name)
                schema[name] = [view.get_dtype(), view.get_shape()]
    schema_bytes = json.dumps(schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    schema_digest = hashlib.sha256(schema_bytes).hexdigest()
    if reviewed is not None and reviewed.schema_sha256 is not None:
        if schema_digest != reviewed.schema_sha256:
            raise ValueError("original full model schema differs from its reviewed pin")
    else:
        _validate_checkpoint_schema(config, schema)
    files = {}
    for name, path in paths.items():
        with path.open("rb") as stream:
            files[name] = {"size": before[name][2], "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    if names != filenames() or any(
        asset(name) != path or fingerprint(path) != before[name] for name, path in paths.items()
    ):
        raise ValueError("original full model assets changed during identity verification")
    return {
        "files": files,
        "tensor_schema_sha256": schema_digest,
        "tensor_count": len(schema),
        "shard_count": len(shards),
        "num_hidden_layers": layers,
    }


def _validate_checkpoint_schema(config: dict, schema: dict) -> None:
    import torch
    from transformers import AutoConfig
    from transformers import AutoModelForCausalLM

    from ..onboarding.synth_model import _meta_model

    normalized = dict(config)
    model_type = normalized.pop("model_type", None)
    model_config = AutoConfig.for_model(model_type, **normalized)
    model_config._attn_implementation = "eager"
    with torch.device("meta"):
        model = (
            _meta_model(model_config) if model_config.architectures else AutoModelForCausalLM.from_config(model_config)
        )
    expected = model.state_dict()
    if set(schema) - expected.keys():
        raise ValueError("original full model has tensors outside its configured architecture")
    # Safetensors may omit one name of tied weights, but must retain the shared parameter.
    aliases = dict(model.named_parameters(remove_duplicate=False))
    aliases.update(model.named_buffers(remove_duplicate=False))
    retained_ids = {id(aliases[name]) for name in schema if name in aliases}
    missing = {
        name
        for name in expected.keys() - schema.keys()
        if name not in aliases or id(aliases[name]) not in retained_ids
    }
    if missing:
        raise ValueError(f"original full model is missing configured tensors: {sorted(missing)[:8]}")
    for name, (_, shape) in schema.items():
        if list(expected[name].shape) != shape:
            raise ValueError(f"original full model tensor shape differs from its configuration: {name}")


def _read_admission(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"original model identity is missing or unsafe: {path}; run full-model onboarding")
    with path.open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise ValueError("original model identity exceeds the compact record size limit")
    try:
        record = storage._strict_record_json(raw)
        if (
            not isinstance(record, dict)
            or set(record) != {"schema", "binding", "revision", "checkpoint"}
            or record["schema"] != "original-model-v1"
            or not isinstance(record["binding"], dict)
            or not isinstance(record["checkpoint"], dict)
            or record["revision"] is not None
            and re.fullmatch(r"[0-9a-f]{40}", record["revision"]) is None
        ):
            raise ValueError("invalid identity fields")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"original model identity is corrupt: {path}; restore its verified record") from exc
    return record


def _binding(cfg: LoadedConfig) -> dict:
    return {"config_id": cfg.config_id, "config_checksum": config_checksum(cfg), "model_name": cfg.model_name}


def _verify_admission(cfg: LoadedConfig, record: dict, cache: Path) -> Path:
    if record["binding"] != _binding(cfg):
        raise ValueError("original model identity does not match the configured recipe; use a distinct config variant")
    root, boundary, revision = _source_root(cfg, cache, record["revision"])
    if revision != record["revision"] or (
        _checkpoint_identity(root, boundary, ORIGINAL_MODELS.get(cfg.model_name)) != record["checkpoint"]
    ):
        raise ValueError("original full model checkpoint differs from its onboarded identity")
    return root


def onboard_original_model(backend: CortexConfig, cfg: LoadedConfig, *, records_dir: Path | None = None) -> Path:
    """Capture the configured original, or verify an existing identity without silently replacing it."""
    cache = _full_model_environment(backend)
    records_dir = storage.require_git_records_dir(records_dir or storage.DEFAULT_RECORDS_DIR)
    path = _admission_path(cfg, records_dir)
    path.parent.parent.mkdir(parents=True, exist_ok=True)
    with storage._publication_lock(path.parent, exclusive=True):
        if path.exists() or path.is_symlink():
            _verify_admission(cfg, _read_admission(path), cache)
            return path
        reviewed = ORIGINAL_MODELS.get(cfg.model_name)
        revision = reviewed.revision if reviewed is not None else None
        root, boundary, revision = _source_root(cfg, cache, revision)
        checkpoint = _checkpoint_identity(root, boundary, reviewed)
        if reviewed is not None:
            if checkpoint["files"]["config.json"]["sha256"] != reviewed.config_sha256:
                raise ValueError("original full model config differs from its reviewed pin")
            if reviewed.index_sha256 is not None and (
                checkpoint["files"].get("model.safetensors.index.json", {}).get("sha256") != reviewed.index_sha256
            ):
                raise ValueError("original full model index differs from its reviewed pin")
            if reviewed.schema_sha256 is not None and checkpoint["tensor_schema_sha256"] != reviewed.schema_sha256:
                raise ValueError("original full model schema differs from its reviewed pin")
            if reviewed.weights_sha256 is not None and (
                checkpoint["files"].get("model.safetensors", {}).get("sha256") != reviewed.weights_sha256
            ):
                raise ValueError("original full model weights differ from their reviewed pin")
        record = {
            "schema": "original-model-v1",
            "binding": _binding(cfg),
            "revision": revision,
            "checkpoint": checkpoint,
        }
        raw = (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        if len(raw) > 1 << 20:
            raise ValueError("original model identity exceeds the compact record size limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".original-model-", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
    print(f"Prepared original model identity: {path}", flush=True)
    return path


def validate_local_full_model(
    backend: CortexConfig, cfg: LoadedConfig, spec: TestSpec, *, records_dir: Path | None = None
) -> Path:
    """Verify a reviewed original or the config-bound identity saved during onboarding; never publish."""
    cache = _full_model_environment(backend)
    path = _admission_path(cfg, records_dir)
    if path.parent.parent.exists():
        with storage._publication_lock(path.parent, exclusive=False):
            if path.exists() or path.is_symlink():
                record = _read_admission(path)
                root = _verify_admission(cfg, record, cache)
                source = spec.model.source_checkpoint
                if source != cfg.model_name and Path(source).expanduser().resolve() != root:
                    raise ValueError(
                        "original full model specification names a different source; not a full-model substitute"
                    )
                return root
    return _validate_reviewed_local_full_model(backend, cfg, spec)


def _validate_reviewed_local_full_model(backend: CortexConfig, cfg: LoadedConfig, spec: TestSpec) -> Path:
    """Resolve the explicitly supported original model offline, without inventing a reduced binding."""
    from safetensors import safe_open
    from transformers.utils.hub import cached_file

    from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway

    require_local_cortex_gateway(backend)
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("DSS_ALLOW_REMOTE_MODELS") != "1":
        raise ValueError("local full model requires offline HF cache and DSS_ALLOW_REMOTE_MODELS=1")
    metadata = _original_model(cfg, spec)
    revision = metadata.revision
    config_sha256 = metadata.config_sha256
    index_sha256 = metadata.index_sha256
    layer_pattern = metadata.layer_pattern
    tensor_count, shard_count, payload_bytes = metadata.tensor_count, metadata.shard_count, metadata.payload_bytes
    vocabulary_shape = list(metadata.vocabulary_shape) if metadata.vocabulary_shape is not None else None
    schema_sha256 = metadata.schema_sha256
    dtype_bytes = metadata.dtype_bytes
    cache = require_data_fast_path(os.environ.get("HF_HUB_CACHE", ""), label="offline full model cache")
    config_file = cached_file(
        cfg.model_name, "config.json", cache_dir=str(cache), revision=revision, local_files_only=True
    )
    if not isinstance(config_file, str):
        raise ValueError("original full model config is not present in the offline cache")
    root = require_data_fast_path(Path(config_file).parent, label="offline full model snapshot")
    if root != cache / ("models--" + cfg.model_name.replace("/", "--")) / "snapshots" / revision:
        raise ValueError("offline full model snapshot does not match the pinned original revision")

    def readable(relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
            raise ValueError("offline full model has an unsafe relative asset path")
        resolved = require_data_fast_path(root / path, label="offline full model asset")
        if not resolved.is_relative_to(cache) or not resolved.is_file() or resolved.stat().st_size == 0:
            raise ValueError("offline full model asset is missing, empty or outside its cache")
        return resolved

    config_bytes = readable("config.json").read_bytes()
    if hashlib.sha256(config_bytes).hexdigest() != config_sha256:
        raise ValueError("offline full model config does not match the pinned original config")
    config = json.loads(config_bytes)
    readable("tokenizer_config.json")
    readable("tokenizer.json")
    if metadata.weights_sha256 is not None:
        weights = readable("model.safetensors")
        with weights.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != metadata.weights_sha256:
            raise ValueError("offline full model weights do not match the pinned original weights")
        tensor_schema = {}
        tensor_bytes = 0
        with safe_open(weights, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                view = stream.get_slice(name)
                shape, dtype = view.get_shape(), view.get_dtype()
                if dtype not in dtype_bytes:
                    raise ValueError("offline full model tensor dtype differs from the original source")
                elements = 1
                for size in shape:
                    elements *= size
                tensor_bytes += elements * dtype_bytes[dtype]
                tensor_schema[name] = [dtype, shape]
        schema_bytes = json.dumps(tensor_schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        layers = {
            int(match.group(1)) for name in tensor_schema if (match := re.match(layer_pattern, name)) is not None
        }
        if (
            len(tensor_schema) != tensor_count
            or tensor_bytes != payload_bytes
            or layers != set(range(config.get("text_config", config)["num_hidden_layers"]))
            or hashlib.sha256(schema_bytes).hexdigest() != schema_sha256
        ):
            raise ValueError("offline full model tensor schema differs from the pinned original source")
        return root
    index_bytes = readable("model.safetensors.index.json").read_bytes()
    if index_sha256 is not None and hashlib.sha256(index_bytes).hexdigest() != index_sha256:
        raise ValueError("offline full model index does not match the pinned original index")
    index = json.loads(index_bytes)
    mapping = index.get("weight_map") if isinstance(index, dict) else None
    if (
        not isinstance(mapping, dict)
        or not mapping
        or not all(
            isinstance(name, str) and name and isinstance(shard, str) and shard.endswith(".safetensors")
            for name, shard in mapping.items()
        )
    ):
        raise ValueError("offline full model requires an indexed safetensors checkpoint")
    if index_sha256 is not None and (
        len(mapping) != tensor_count
        or index.get("metadata", {}).get("total_size") != payload_bytes
        or set(mapping.values())
        != {f"model-{i:05d}-of-{shard_count:05d}.safetensors" for i in range(1, shard_count + 1)}
    ):
        raise ValueError("offline full model index is missing original tensors or shards")
    keys = set()
    vocabulary_shapes = {}
    tensor_bytes = 0
    tensor_schema = {}
    for shard in sorted(set(mapping.values())):
        with safe_open(readable(shard), framework="pt", device="cpu") as stream:
            actual = set(stream.keys())
            if actual != {name for name, filename in mapping.items() if filename == shard} or keys & actual:
                raise ValueError("offline full model safetensors keys do not match the complete weight index")
            keys.update(actual)
            if index_sha256 is not None:
                # Read tensor headers, not the full weights or GPU tensors.
                for name in actual:
                    view = stream.get_slice(name)
                    shape = view.get_shape()
                    dtype = view.get_dtype()
                    tensor_schema[name] = [dtype, shape]
                    if dtype not in dtype_bytes:
                        raise ValueError("offline full model tensor dtype differs from the original source")
                    elements = 1
                    for size in shape:
                        elements *= size
                    tensor_bytes += elements * dtype_bytes[dtype]
                    if name in {"lm_head.weight", "model.language_model.embed_tokens.weight"}:
                        vocabulary_shapes[name] = shape
    layers = {int(match.group(1)) for name in keys if (match := re.match(layer_pattern, name)) is not None}
    text_config = config.get("text_config", config)
    if layers != set(range(text_config["num_hidden_layers"])):
        raise ValueError("offline full model checkpoint is missing original transformer layers")
    if index_sha256 is not None:
        vision_layers = {
            int(match.group(1))
            for name in keys
            if (match := re.match(r"model\.visual\.blocks\.(\d+)\.", name)) is not None
        }
        if (
            vision_layers != set(range(config["vision_config"]["depth"]))
            or tensor_bytes != payload_bytes
            or vocabulary_shapes
            != {
                "lm_head.weight": vocabulary_shape,
                "model.language_model.embed_tokens.weight": vocabulary_shape,
            }
        ):
            raise ValueError("offline full model checkpoint is missing original vision or vocabulary tensors")
        schema_bytes = json.dumps(tensor_schema, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        if hashlib.sha256(schema_bytes).hexdigest() != schema_sha256:
            raise ValueError("offline full model tensor shapes or dtypes differ from the pinned original source")
    return root
