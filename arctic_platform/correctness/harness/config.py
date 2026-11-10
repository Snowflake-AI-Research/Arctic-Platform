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


"Reviewed model configurations, test specifications, and placement arms."

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from decimal import ROUND_CEILING
from decimal import Decimal
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional

from pydantic import Field
from pydantic import model_serializer

from arctic_platform.client.config import ArcticClientConfig
from arctic_platform.client.config import SamplingConfig
from arctic_platform.client.config import TrainingConfig

DEFAULT_REFERENCE_CALIBRATION_RUNS = 8
MAX_LOCAL_REFERENCE_CALIBRATION_WORKERS = 8


def reference_calibration_worker_count(
    workers: int | None = None, *, runs: int = DEFAULT_REFERENCE_CALIBRATION_RUNS
) -> int:
    """Select independent HF workers on the current node, not the target GPU group."""
    if runs < 2:
        raise ValueError("norm onboarding requires at least two calibration runs")
    if workers is not None:
        if not 1 <= workers <= MAX_LOCAL_REFERENCE_CALIBRATION_WORKERS:
            raise ValueError("reference calibration requires 1 to 8 HF workers on one node")
        return min(workers, runs)

    import torch

    # TODO: Use all available GPUs within one allocation, including multi-node
    # launches, node-local model/input staging, unique devices and result/failure collection.
    return max(1, min(runs, MAX_LOCAL_REFERENCE_CALIBRATION_WORKERS, torch.cuda.device_count()))


class ReviewedTrainingConfig(TrainingConfig):
    weight_format: Literal["vllm", "hf", "lora"] | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        result = handler(self)
        if self.weight_format is None:
            result.pop("weight_format", None)
        return result


class ReviewedSamplingConfig(SamplingConfig):
    weight_format: Literal["vllm", "hf", "lora"] | None = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        result = handler(self)
        if self.weight_format is None:
            result.pop("weight_format", None)
        return result


class ReviewedClientConfig(ArcticClientConfig):
    """Correctness-only sub-job fields; public client defaults remain unchanged."""

    training: ReviewedTrainingConfig = Field(default_factory=ReviewedTrainingConfig)
    sampling: ReviewedSamplingConfig = Field(default_factory=ReviewedSamplingConfig)

    def _cortex_sub_job(self, job_type: str, extra: dict[str, Any]) -> dict[str, Any]:
        sub = super()._cortex_sub_job(job_type, extra)
        settings = self.training if job_type == "training" else self.sampling
        if settings.weight_format is not None:
            sub["weight_format"] = settings.weight_format
        return sub

    def to_onprem(self, job_type: str) -> dict[str, Any]:
        settings = self.training if job_type == "training" else self.sampling
        if (
            settings.weight_format is not None
            or (self.sampling.arctic_inference_config or {}).get("num_replicas") is not None
        ):
            raise ValueError("reviewed weight formats and explicit replicas require the Cortex correctness route")
        return super().to_onprem(job_type)


@dataclass
class ArmSpec:
    """One global batch size, with the shape onboarding measured for it."""

    name: str
    global_batch_size: int
    max_seq_len: int
    total_tokens: int
    active_tokens: int
    ap_microbatches: int
    pad_fraction: float

    @property
    def exercises_accumulation(self) -> bool:
        return self.ap_microbatches > 1


@dataclass
class ModelSpec:
    """The synthetic model recipe. Regenerable from the seed, so the path is a cache and not an input."""

    source_checkpoint: str
    num_hidden_layers: int
    vision_depth: int
    seed: int
    param_count: int
    cache_path: str
    content_hash: str
    sized_against_gib: float
    measured_reference_peak_gib: Optional[float] = None


@dataclass
class TestTolerance:
    """A frozen regression gate and the reference measurement that selected it."""

    absolute: float
    calibration_runs: Optional[int] = None
    raw_max_same_tensor_range: Optional[float] = None
    multiplier: Optional[float] = None
    computed_gate: Optional[float] = None
    rounding_quantum: float = 1e-3
    minimum_gate: float = 1e-3
    worst_tensor: Optional[str] = None
    status: str = "uncalibrated"

    def __post_init__(self) -> None:
        if self.absolute <= 0:
            raise ValueError(f"test tolerance must be positive, got {self.absolute}")
        if self.rounding_quantum <= 0 or self.minimum_gate <= 0:
            raise ValueError("test tolerance rounding quantum and minimum gate must be positive")
        if self.computed_gate is None:
            if self.absolute < self.minimum_gate:
                raise ValueError(f"selected tolerance {self.absolute} is below minimum gate {self.minimum_gate}")
            return
        quantum = Decimal(str(self.rounding_quantum))
        computed = Decimal(str(self.computed_gate))
        minimum = Decimal(str(self.minimum_gate))
        expected = max(
            minimum,
            (computed / quantum).to_integral_value(rounding=ROUND_CEILING) * quantum,
        )
        if Decimal(str(self.absolute)) != expected:
            raise ValueError(
                f"selected tolerance {self.absolute} does not equal upward-rounded calibrated gate {expected}"
            )


@dataclass
class TestSpec:
    config_id: str
    config_path: str
    model: ModelSpec
    arms: List[ArmSpec]
    attn_implementations: List[str]
    applicable_tests: List[str]
    config_checksum: str = ""
    test_tolerances: Dict[str, TestTolerance] = field(default_factory=dict)
    test_settings: Dict[str, dict] = field(default_factory=dict)
    inapplicable: Dict[str, str] = field(default_factory=dict)
    param_name_map: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def tolerance_for(self, test_id: str) -> float:
        try:
            return self.test_tolerances[test_id].absolute
        except KeyError as exc:
            raise ValueError(
                f"test {test_id!r} has no frozen tolerance in config spec {self.config_id!r}; "
                "run config onboarding first"
            ) from exc

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json())

    @staticmethod
    def read(path: Path) -> "TestSpec":
        raw = json.loads(Path(path).read_text())
        # Previously onboarded specifications retain their original bytes.
        for arm in raw["arms"]:
            if "dss_microbatches" in arm:
                legacy = arm.pop("dss_microbatches")
                if "ap_microbatches" in arm and arm["ap_microbatches"] != legacy:
                    raise ValueError("specification has conflicting AP and legacy microbatch counts")
                arm.setdefault("ap_microbatches", legacy)
        return TestSpec(
            config_id=raw["config_id"],
            config_path=raw["config_path"],
            model=ModelSpec(**raw["model"]),
            arms=[ArmSpec(**a) for a in raw["arms"]],
            attn_implementations=raw["attn_implementations"],
            applicable_tests=raw["applicable_tests"],
            config_checksum=raw.get("config_checksum", ""),
            test_tolerances={
                test_id: TestTolerance(**tolerance) for test_id, tolerance in raw.get("test_tolerances", {}).items()
            },
            test_settings=raw.get("test_settings", {}),
            inapplicable=raw.get("inapplicable", {}),
            param_name_map=raw.get("param_name_map", {}),
            notes=raw.get("notes", []),
        )


NON_MODEL_PROCESSOR_SIDECARS = frozenset({"preprocessor_config.json", "video_preprocessor_config.json"})


def hash_directory(path: Path, suffixes=(".safetensors", ".json"), *, exclude_names=()) -> str:
    """Hash model and tokenizer assets while ignoring processor metadata that does not affect text execution."""
    h = hashlib.sha256()
    for f in sorted(Path(path).rglob("*")):
        if (
            f.is_file()
            and f.suffix in suffixes
            and f.name not in NON_MODEL_PROCESSOR_SIDECARS
            and f.name not in exclude_names
        ):
            h.update(f.name.encode())
            h.update(str(f.stat().st_size).encode())
    return h.hexdigest()[:16]


# Scratch probes retain the historical gate. Product tests read their frozen gate from TestSpec.
STATED_CRITERION_ABS = 1e-3


def tolerance_text(value: float) -> str:
    """``1e-03`` rather than ``1.000e-03``, without rounding a value that needs its digits."""
    mantissa, exponent = f"{value:.3e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{exponent}"


GPU_ATTENTION_IMPLEMENTATIONS = {
    "h200": "flash_attention_3",
    "b200": "flash_attention_4",
    "b300": "flash_attention_4",
}
MODEL_ATTENTION_IMPLEMENTATIONS = {
    "Qwen/Qwen3.8-Flash-Next": "qsa_flex",
    "zai-org/GLM-5.3-Flash": "sparse_mla",
    "zai-org/GLM-5.3-Flash-BF16": "sparse_mla",
}
MODEL_REFERENCE_ATTENTION_IMPLEMENTATIONS = {
    "zai-org/GLM-5.3-Flash": "sdpa",
    "zai-org/GLM-5.3-Flash-BF16": "sdpa",
}


@dataclass
class LoadedConfig:
    config_id: str
    path: Path
    sub_job: Dict[str, Any]
    # Every sub-job in the file, training included. The correctness tests execute the training sub-job
    # alone, but the surrounding job shape is what says whether that training step is part of an RL loop.
    sub_jobs: List[Dict[str, Any]] = field(default_factory=list)
    native: ArcticClientConfig | None = None
    hardware: Literal["H200", "B200", "B300"] | None = None
    placement_base_gpus: int | None = None

    @property
    def training(self) -> Dict[str, Any]:
        if not self.has_training:
            raise ValueError(
                f"{self.config_id}: correctness checks require a training sub-job; this is inference-only"
            )
        return self.sub_job["training_config"]

    @property
    def has_training(self) -> bool:
        return "training_config" in self.sub_job and self.sub_job.get("job_type", "training") == "training"

    @property
    def is_rl(self) -> bool:
        """Whether this job trains from rollouts it generates itself.

        Both sub-job types have to be present. A sampling sub-job on its own serves a policy for
        generation and trains nothing, and a training sub-job on its own consumes a fixed dataset; it is
        the pair that makes the job reinforcement learning. The training sub-job cannot be read for this,
        because a supervised job may still select ``model_provider: prime_rl`` and the PrimeRL loss
        options beneath it.
        """
        if self.native is not None:
            return self.native.training_gpus > 0 and self.native.sampling_gpus > 0
        job_types = {sub.get("job_type") for sub in self.sub_jobs}
        return "training" in job_types and "sampling" in job_types

    @property
    def model_name(self) -> str:
        value = self.sub_job.get("model_name")
        if not isinstance(value, str) or not value:
            raise ValueError(f"{self.config_id}: training sub-job requires a non-empty model_name")
        return value

    @property
    def effective_training(self) -> Dict[str, Any]:
        """Training values after applying the nested PrimeRL overrides used by the runtime."""
        prime_rl = self.training.get("prime_rl")
        if isinstance(prime_rl, dict):
            return {**self.training, **prime_rl}
        return self.training

    @property
    def fused_cross_entropy(self) -> bool | str:
        """Resolve the loss backend the selected product model builder applies."""
        training = self.effective_training
        if training.get("model_provider") == "prime_rl":
            return training.get("fused_cross_entropy", "liger")
        return training.get("fused_cross_entropy", False)

    @property
    def lm_head_token_chunk_size(self) -> int | None:
        """Resolve an integer chunk size; runtime sentinels such as ``disabled`` mean no chunking."""
        options = self.training.get("model_options") or {}
        lm_head = options.get("lm_head") or {}
        if "lm_head.token_chunk_size" in options:
            value = options["lm_head.token_chunk_size"]
        elif "token_chunk_size" in lm_head:
            value = lm_head["token_chunk_size"]
        else:
            value = self.effective_training.get("fused_lm_head_token_chunk_size")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        return None

    @property
    def optimizer_dtype(self) -> str:
        """Resolve the DeepSpeed Adam master/state precision selected by this config."""
        import os

        dtype = str(self.sub_job.get("dtype") or self.training.get("dtype") or "bfloat16").lower()
        default_bf16 = dtype in ("bfloat16", "bf16") and not os.environ.get("DSS_FP32_MASTERS")
        ds_config = self.training.get("ds_config") or {}
        bf16 = ds_config.get("bf16") if "bf16" in ds_config else None
        master_bf16 = default_bf16 if bf16 is None else bool(bf16.get("bf16_master_weights_and_grads", False))
        states_bf16 = default_bf16 if bf16 is None else bool(bf16.get("bf16_optimizer_states", False))
        if master_bf16 != states_bf16:
            raise ValueError(
                f"{self.config_id}: correctness requires matching Adam master and state precision; "
                f"got bf16_master_weights_and_grads={master_bf16} and bf16_optimizer_states={states_bf16}"
            )
        return "bfloat16" if master_bf16 else "float32"

    @property
    def n_gpus(self) -> int:
        if self.native is not None:
            return self.native.training_gpus
        return int(self.training["n_gpus"])

    def at_gpu_width(self, gpus: int) -> "LoadedConfig":
        """This config placed on ``gpus`` GPUs, for measuring the same work at a wider topology.

        The extra GPUs widen data parallelism, so the global batch grows by the same multiple as
        ``n_gpus`` while ``sp_size`` stays as written, everywhere the config states that batch:
        ``ds_config.train_batch_size`` and the training config's own ``train_batch_size``. DeepSpeed
        requires ``train_batch_size == train_micro_batch_size_per_gpu * gradient_accumulation_steps *
        dp_size``, and the data plane requires at least one row per data-parallel shard, so a batch left
        at its declared value fails at engine initialization or at the first dispatch on any width but
        the declared one. A config that states no ``train_batch_size`` in a given place has none written
        for it there.

        Everything else -- the identity, the model, the sequence lengths, the seed -- stays as the file
        declares it. The config's checksum and the cases its reviewed spec froze are properties of the file
        as written and are read from the loaded config before any width is derived, never from the config
        returned here.

        The DeepSpeed identity is applied at every width, including the declared one. A file whose written
        batch does not equal ``micro * gas * dp_size`` still places: the in-memory copies become that
        product and the file is not written.
        """
        multiple, remainder = divmod(gpus, self.n_gpus)
        if remainder or multiple < 1:
            raise ValueError(
                f"{self.config_id}: a width of {gpus} is not a whole multiple of the declared n_gpus={self.n_gpus}"
            )
        sub_job = deepcopy(self.sub_job)
        training = sub_job["training_config"]
        training["n_gpus"] = gpus
        required = deepspeed_train_batch_size(training, gpus)
        for holder in (training, training.get("ds_config")):
            if not isinstance(holder, dict) or "train_batch_size" not in holder:
                continue
            if required is not None:
                holder["train_batch_size"] = required
            elif multiple != 1:
                holder["train_batch_size"] = int(holder["train_batch_size"]) * multiple
        sub_jobs = [sub_job if sub is self.sub_job else sub for sub in self.sub_jobs]
        native = self.native
        if native is not None:
            native_training = native.training.model_copy(
                update={
                    "ds_config": deepcopy(training.get("ds_config") or {}),
                    "ds_worker_config": {
                        key: deepcopy(value)
                        for key, value in training.items()
                        if key
                        not in {"n_gpus", "max_seq_len", "ds_config", "train_batch_size", "optimizer", "peft_config"}
                    },
                },
                deep=True,
            )
            native = native.model_copy(
                update={"training_gpus": gpus, "training": native_training},
                deep=True,
            )
        return replace(
            self,
            sub_job=sub_job,
            sub_jobs=sub_jobs,
            native=native,
            placement_base_gpus=self.placement_base_gpus or self.n_gpus,
        )

    @property
    def sp_size(self) -> int:
        return int(self.training.get("sp_size", 1))

    @property
    def dp_size(self) -> int:
        return self.n_gpus // self.sp_size

    @property
    def gpu_type(self) -> str | None:
        """The explicit service hardware, otherwise the accelerator named by the config directory.

        A recipe imported outside the original catalog layout retains its explicit hardware identity.
        """
        if self.hardware is not None:
            return self.hardware.lower()
        return _path_gpu_type(self.path)

    @property
    def attention_implementation(self) -> str:
        """Use the model's required backend, otherwise the config's accelerator-family backend."""
        configured = self.training.get("attn_implementation")
        model_expected = MODEL_ATTENTION_IMPLEMENTATIONS.get(self.model_name)
        expected = model_expected or expected_attention_implementation(self.gpu_type)
        if configured is not None and expected is not None and configured != expected:
            requirement = self.model_name if model_expected else self.gpu_type
            raise ValueError(
                f"{self.config_id}: {requirement} requires attn_implementation={expected!r}, "
                f"but the config selects {configured!r}"
            )
        if configured is not None:
            return str(configured)
        if expected is not None:
            return expected
        raise ValueError(
            f"{self.config_id}: attn_implementation is required when the GPU type is not one of "
            f"{', '.join(sorted(GPU_ATTENTION_IMPLEMENTATIONS))}"
        )

    @property
    def reference_attention_implementation(self) -> str:
        """Use upstream GLM's top-k SDPA mask without changing AP's sparse-MLA target."""
        target = self.attention_implementation
        return MODEL_REFERENCE_ATTENTION_IMPLEMENTATIONS.get(self.model_name, target)

    @property
    def max_seq_len(self) -> int:
        """The padded row width the config declares, or the one the training engine would supply for it.

        ``max_seq_len`` is optional to Arctic Platform: a training config that omits it is served with the field default
        from ``TrainingConfig``, so a harness that refused the config would refuse something the engine runs.
        The default is read from that model rather than copied here, so the width measured is the width the
        engine would use. It is a per-sequence length, and ``mb_spec.max_tokens_per_mb`` does not replace it:
        the microbatch budget is divided by this width to decide how many rows a model call carries.
        """
        if self.native is not None:
            return self.native.max_seq_len
        if "max_seq_len" in self.training:
            return int(self.training["max_seq_len"])
        return int(ArcticClientConfig.model_fields["max_seq_len"].default)

    @property
    def max_tokens_per_mb(self) -> int:
        """The microbatch token budget this job runs with, resolved by the runtime that owns the default."""
        from arctic_platform.rl.processors.microbatch import DEFAULT_MAX_TOKENS_PER_MB

        return int((self.training.get("mb_spec") or {}).get("max_tokens_per_mb") or DEFAULT_MAX_TOKENS_PER_MB)


def normalize_ints(obj: Any) -> Any:
    """Integer-valued floats reach DeepSpeed directly, where ``stage: 2.0`` is not ``stage: 2``."""
    if isinstance(obj, dict):
        return {k: normalize_ints(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_ints(v) for v in obj]
    if isinstance(obj, float) and obj.is_integer() and abs(obj) < 2**53:
        return int(obj)
    return obj


CONFIG_ROOT = "configs"


def expected_attention_implementation(gpu_type: str | None) -> str | None:
    """Return the backend supported by a known accelerator family."""
    if gpu_type is None:
        return None
    return GPU_ATTENTION_IMPLEMENTATIONS.get(gpu_type.lower())


def config_id(path: Path) -> str:
    """Identify new recipes by their relative path and retain migrated recipes' frozen IDs."""
    path = resolve_config_path(path).resolve()
    root = _config_root(path)
    if root is None:
        return path.stem
    relative = path.relative_to(root)
    entry = _layout_entries(root).get(relative.as_posix())
    return entry["config_id"] if entry is not None else "-".join(relative.with_suffix("").parts)


def local_config_path(path: str | Path, config_dir: Path | None = None) -> Path:
    """Import recipe bytes at their hierarchical placement without overwriting another recipe."""
    source = resolve_config_path(path).resolve()
    root = (config_dir if config_dir is not None else Path(__file__).resolve().parents[1] / "configs").resolve()
    if source.suffix not in {".config", ".json"} or not source.is_file():
        raise ValueError(f"expected an existing .config or .json configuration: {source}")
    if source.is_relative_to(root):
        return source
    destination = root / config_import_subpath(load_config(source))
    if not destination.resolve().is_relative_to(root):
        raise ValueError(f"configuration destination must stay inside the config root: {destination}")
    identifier = config_id(destination)
    for candidate in config_paths(root):
        if candidate != destination and config_id(candidate) == identifier:
            raise ValueError(f"configuration ID {identifier!r} already exists at {candidate}")
    contents = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as imported:
            imported.write(contents)
    except FileExistsError:
        if destination.read_bytes() != contents:
            raise ValueError(
                f"configuration already exists with different contents: {destination}; "
                "use an explicit distinct recipe variant"
            ) from None
    return destination


def local_gpu_type() -> str | None:
    """The host's GPU as the layout spells it: ``NVIDIA H200`` -> ``h200``. ``None`` when there is no GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0).lower()
    for token in name.replace("-", " ").split():
        # Model designations are a letter followed by digits: h100, h200, b200, b300, a100, l40.
        if len(token) >= 3 and token[0].isalpha() and token[1:].isdigit():
            return token
    return name


def _optimizer_from_ds(value: Any) -> Dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = value.get("type")
    params = value.get("params")
    if not isinstance(name, str) or not isinstance(params, dict):
        return None
    return {"name": name, **deepcopy(params)}


def _legacy_views(native: ArcticClientConfig) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    ds = deepcopy(native.training.ds_config or {})
    worker = deepcopy(native.training.ds_worker_config or {})
    training = {**worker, "n_gpus": native.training_gpus, "max_seq_len": native.max_seq_len, "ds_config": ds}
    for key in ("train_batch_size", "gradient_clipping"):
        if key in ds:
            training[key] = ds[key]
    optimizer = _optimizer_from_ds(ds.get("optimizer"))
    if optimizer is not None:
        training["optimizer"] = optimizer
    if native.training.peft is not None:
        training["peft_config"] = deepcopy(native.training.peft)
    training_sub = {
        "job_type": "training",
        "model_name": native.model_name,
        "dtype": native.dtype,
        "seed": native.seed,
        "training_config": training,
    }
    training_weight_format = getattr(native.training, "weight_format", None)
    if training_weight_format is not None:
        training_sub["weight_format"] = training_weight_format
    sub_jobs = [training_sub]
    if native.sampling_gpus > 0:
        inference = {
            "n_gpus": native.sampling_gpus,
            "max_seq_len": native.max_seq_len,
            "vllm_config": deepcopy(native.sampling.vllm),
        }
        if native.sampling.arctic_inference_config:
            inference.update(deepcopy(native.sampling.arctic_inference_config))
        if native.training.peft is not None:
            inference["peft_config"] = deepcopy(native.training.peft)
        sub_jobs.append(
            {
                "job_type": "sampling",
                "model_name": native.model_name,
                "dtype": native.dtype,
                "seed": native.seed,
                "inference_config": inference,
            }
        )
        sampling_weight_format = getattr(native.sampling, "weight_format", None)
        if sampling_weight_format is not None:
            sub_jobs[-1]["weight_format"] = sampling_weight_format
    return training_sub, sub_jobs


def _load_cortex_recipe(path: Path, raw: dict) -> LoadedConfig:
    """Read the service recipe without losing engine fields in a client-schema conversion."""
    unsupported = set(raw) - {"sub_job_configs", "hardware"}
    if unsupported:
        raise ValueError(f"{path}: unsupported Cortex recipe fields: {sorted(unsupported)}")
    hardware = raw.get("hardware")
    if "hardware" in raw and (not isinstance(hardware, str) or hardware not in {"H200", "B200", "B300"}):
        raise ValueError(f"{path}: hardware must be one of H200, B200, B300")
    sub_jobs = raw["sub_job_configs"]
    if not isinstance(sub_jobs, list) or not sub_jobs:
        raise ValueError(f"{path}: sub_job_configs must be a non-empty list")
    roles = set()
    for sub in sub_jobs:
        if not isinstance(sub, dict):
            raise ValueError(f"{path}: each sub_job_configs entry must be an object")
        role = sub.get("job_type")
        if not isinstance(role, str) or role not in {"training", "sampling", "log_probability"} or role in roles:
            raise ValueError(f"{path}: unsupported or duplicate sub-job type: {role!r}")
        roles.add(role)
        if not isinstance(sub.get("model_name"), str) or not sub["model_name"].strip():
            raise ValueError(f"{path}: {role} sub-job requires a non-empty model_name")
        key = "training_config" if role == "training" else "inference_config"
        settings = sub.get(key)
        if not isinstance(settings, dict):
            raise ValueError(f"{path}: {role} sub-job requires an object {key}")
        for name in ("n_gpus", "max_seq_len"):
            value = settings.get(name)
            if name == "max_seq_len" and name not in settings:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{path}: {role}.{name} must be a positive integer")
        if role == "training":
            ReviewedTrainingConfig.model_validate(
                {
                    "ds_config": settings.get("ds_config"),
                    "ds_worker_config": settings,
                    "weight_format": sub.get("weight_format"),
                }
            )
        else:
            ReviewedSamplingConfig.model_validate({"weight_format": sub.get("weight_format")})
    selected = next((sub for sub in sub_jobs if sub["job_type"] == "training"), sub_jobs[0])
    return LoadedConfig(config_id=config_id(path), path=path, sub_job=selected, sub_jobs=sub_jobs, hardware=hardware)


def load_config(path: str | Path) -> LoadedConfig:
    path = resolve_config_path(path)
    raw = normalize_ints(json.loads(path.read_text()))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: configuration must be a JSON object")
    if "sub_job_configs" in raw:
        return _load_cortex_recipe(path, raw)
    backend = raw.get("backend")
    if isinstance(backend, dict) and backend.get("type") == "onprem" and backend.get("protocol") == "ray":
        # Retained reviewed recipes predate the public client's single on-prem protocol.
        backend.pop("protocol")
    native = ReviewedClientConfig.model_validate(raw)
    training, sub_jobs = _legacy_views(native)
    if native.training_gpus <= 0:
        sub_jobs = [sub for sub in sub_jobs if sub["job_type"] != "training"]
        if not sub_jobs:
            raise ValueError(f"{path}: correctness config requires a training or sampling sub-job")
        training = sub_jobs[0]
    return LoadedConfig(config_id=config_id(path), path=path, sub_job=training, sub_jobs=sub_jobs, native=native)


def config_checksum(config: LoadedConfig) -> str:
    """Hash the normalized native fields that define the training job."""
    if config.native is None:
        projection = config.sub_job
        if config.hardware is not None:
            projection = {"hardware": config.hardware, "sub_job": config.sub_job}
        canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    else:
        raw = config.native.model_dump(mode="json", exclude_none=False)
        projection = {
            key: raw[key] for key in ("model_name", "seed", "dtype", "max_seq_len", "training_gpus", "training")
        }
        if (
            getattr(config.native.sampling, "weight_format", None) is not None
            or (config.native.sampling.arctic_inference_config or {}).get("num_replicas") is not None
        ):
            projection.update({key: raw[key] for key in ("sampling_gpus", "sampling")})
        canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_first_two_model_dtype(config: LoadedConfig) -> None:
    """Admit only BF16 model parameters, independently of accumulator and optimizer-state precision."""
    # The declared model dtype and the first-two payload's training override are separate inputs.
    for location, dtype in (
        ("sub_job.dtype", config.sub_job.get("dtype")),
        ("training_config.dtype", config.training.get("dtype")),
    ):
        if dtype is None:
            continue
        if not isinstance(dtype, str) or dtype.lower() not in {"bf16", "bfloat16"}:
            raise ValueError(
                f"{config.config_id}: first-two correctness onboarding, reference preparation and regression "
                f"support only BF16 model dtype ('bfloat16' or 'bf16'); got {location}={dtype!r}. "
                "Non-BF16 model execution is not admitted; the target and HF reference use BF16. "
                "FP32 gradient accumulation is a separate setting and remains supported."
            )


def _sibling_for(path: Path, gpu_type: str) -> str:
    """Name the corresponding accelerator recipe, whether or not that recipe exists."""
    parts = list(path.parts)
    if CONFIG_ROOT in parts:
        i = parts.index(CONFIG_ROOT)
        if len(parts) - i >= 4:
            if parts[i + 2] in {"standard", "long"}:
                parts[-1] = re.sub(r"^[^-]+-", f"{gpu_type}-", parts[-1], count=1)
                return str(Path(*parts))
            parts[i + 2] = gpu_type
            return str(Path(*parts))
    return str(Path(CONFIG_ROOT, "<model>", "standard", f"{gpu_type}-training-full.json"))


# A config is also measured at a multiple of the width it declares, never wider than this many times it.
GPU_WIDTH_MULTIPLE_CAP = 4


def deepspeed_train_batch_size(training: Dict[str, Any], gpus: int) -> Optional[int]:
    """The global batch DeepSpeed requires at ``gpus`` ranks: ``micro * gas * dp_size``.

    ``None`` when the config states no microbatch size or no accumulation depth, so a placement cannot
    invent a batch the file never wrote.
    """
    ds_config = training.get("ds_config")
    if not isinstance(ds_config, dict):
        return None
    micro = ds_config.get("train_micro_batch_size_per_gpu")
    gas = ds_config.get("gradient_accumulation_steps")
    if micro is None or gas is None:
        return None
    sp_size = int(training.get("sp_size", 1))
    return int(micro) * int(gas) * (gpus // sp_size)


def placement_widths(declared_gpus: int, pool_gpus: int) -> List[int]:
    """The GPU widths one config is measured at, given a pool of ``pool_gpus`` to place jobs on.

    Agreement with the single-GPU reference is a claim about the configuration, not about one topology, so a
    pool with room for several copies of the declared width earns a second, wider measurement of the same
    config. The multiple is how many times the declared width fits in the pool, capped at
    ``GPU_WIDTH_MULTIPLE_CAP``; a pool narrower than the declared width offers no width at all, because the
    config is run as written and is never narrowed to fit.

    The declared width always comes first, so a run that stops early has measured the reviewed topology.
    """
    if declared_gpus < 1:
        raise ValueError(f"declared gpu count must be positive, got {declared_gpus}")
    if pool_gpus < declared_gpus:
        return []
    multiple = min(pool_gpus // declared_gpus, GPU_WIDTH_MULTIPLE_CAP)
    widths = [declared_gpus]
    if multiple > 1:
        widths.append(declared_gpus * multiple)
    return widths


def validate_against_host(cfg: LoadedConfig, available_gpus: int, *, check_gpu_type: bool = True) -> None:
    """Refuse a config the host cannot run, naming the specific constraint rather than failing later.

    The GPU type is part of the config's identity, not a detail: a config tuned for one accelerator carries
    token budgets and offload choices sized for its memory and bandwidth, so running it elsewhere measures a
    configuration nobody ships. ``check_gpu_type=False`` is for deliberately running one anyway.
    """
    _ = cfg.attention_implementation
    if check_gpu_type:
        wanted, have = cfg.gpu_type, local_gpu_type()
        if wanted and have and wanted != have:
            raise ValueError(
                f"{cfg.config_id}: config is for {wanted} but this host has {have}. Run a config written "
                f"for {have}, at {_sibling_for(cfg.path, have)}, or pass --any-gpu to run this one here "
                "anyway."
            )
    if cfg.n_gpus > available_gpus:
        raise ValueError(
            f"{cfg.config_id}: a job from this process can be placed on {available_gpus} GPU(s), and this "
            f"config declares n_gpus={cfg.n_gpus}. The config is run as written and is never reshaped, so a "
            "wider topology has no equivalent in a smaller pool. A gateway the harness starts itself serves "
            "one node; to use a wider allocation, start a gateway on the allocation's hostfile and name it "
            "with DSS_GATEWAY_URL and DSS_GATEWAY_HOSTFILE."
        )
    if cfg.n_gpus % cfg.sp_size != 0:
        raise ValueError(f"{cfg.config_id}: sp_size={cfg.sp_size} does not divide n_gpus={cfg.n_gpus}")


GAS1_TOKEN_CAP = 65_536


@dataclass(frozen=True)
class ArmDefinition:
    name: str
    global_batch_size: int
    max_seq_len: int

    @property
    def total_tokens(self) -> int:
        return self.global_batch_size * self.max_seq_len

    def microbatches(self, max_tokens_per_mb: int, dp_size: int = 1) -> int:
        """Maximum per-DP-shard model calls after distributing the global rows.

        A row is never split, so an arm whose single row exceeds the budget is still one model call on the
        shard that receives it. The global batch is balanced over data-parallel shards before packing.
        """
        rows_per_mb = max(1, max_tokens_per_mb // self.max_seq_len)
        global_calls = math.ceil(self.global_batch_size / rows_per_mb)
        return math.ceil(global_calls / dp_size)


# Both cases use one padded width so batch-size effects are not confounded with sequence-length effects.
# Active row lengths vary below that width, preserving sequence boundaries and padding coverage. GAS1 is the
# control; GAS4 changes only the number of rows.
GAS_VALUES = (1, 4)


def max_sequence_tokens(config_max_seq_len: int, gpu_count: int) -> int:
    """Largest padded row width that keeps the whole GAS1 batch within 64K slots."""
    if gpu_count < 1:
        raise ValueError(f"gpu_count must be positive, got {gpu_count}")
    if config_max_seq_len < 1:
        raise ValueError(f"max sequence length must be positive, got {config_max_seq_len}")
    return min(config_max_seq_len, GAS1_TOKEN_CAP // gpu_count)


def correctness_microbatch_tokens(configured_tokens: int) -> int:
    """Training configs may lower the correctness budget but cannot raise it above 64K."""
    if configured_tokens < 1:
        raise ValueError(f"microbatch token budget must be positive, got {configured_tokens}")
    return min(configured_tokens, GAS1_TOKEN_CAP)


def scaled_for_placement(arms: List[ArmDefinition], multiple: int) -> List[ArmDefinition]:
    """The reviewed cases with proportionally more rows, for a config placed on a multiple of its GPUs.

    A wider placement widens data parallelism, and a data-parallel shard that receives no row has nothing
    to reduce, so the row count has to grow with the width. Only the row count grows: the padded width of
    each sequence, the case names, and the seed are the reviewed ones, so the wider run differs from the
    declared-width run in how many sequences it carries and in nothing else.

    Each width therefore has its own single-GPU reference, because the reference executes the rows of the
    case it is compared against. A wider run costs one extra reference pass.
    """
    if multiple < 1:
        raise ValueError(f"placement multiple must be at least 1, got {multiple}")
    if multiple == 1:
        return list(arms)
    return [replace(arm, global_batch_size=arm.global_batch_size * multiple) for arm in arms]


def arms_for(max_row_tokens: int, gpu_count: int) -> List[ArmDefinition]:
    """One and four GAS groups, each with one variable-length row per configured GPU."""
    row_tokens = max_sequence_tokens(max_row_tokens, gpu_count)
    return [ArmDefinition(f"gas{gas}", gpu_count * gas, row_tokens) for gas in GAS_VALUES]


def _config_root(path: Path) -> Path | None:
    parents = path.resolve().parents
    # Catalog variants can contain these directory names; keep the outer recipe root.
    root = next((parent for parent in reversed(parents) if parent.name == CONFIG_ROOT), None)
    if root is not None:
        return root
    # Hierarchical recipes also support an explicitly supplied root not named "configs".
    return next((parent.parent.parent for parent in reversed(parents) if parent.name in {"standard", "long"}), None)


def _layout_entries(root: Path) -> dict:
    manifest = root.parent / "config-layout.json"
    return json.loads(manifest.read_text()) if manifest.is_file() else {}


def resolve_config_path(path: str | Path) -> Path:
    """Resolve a pre-migration config locator without changing frozen specification bytes."""
    path = Path(path).expanduser()
    root = _config_root(path)
    if not path.exists() and root is not None:
        relative = path.resolve().relative_to(root).as_posix()
        for destination, entry in _layout_entries(root).items():
            if entry["previous_path"] == relative:
                return root / destination
    return path


def config_paths(root: Path) -> List[Path]:
    """Discover recipe JSON and legacy configs, excluding metadata and archived evidence."""
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in {".config", ".json"}
        and not any(part.startswith(".") or part in {"archive", "specs"} for part in path.relative_to(root).parts)
        and path.name != "model.json"
    )


def reference_config_subpath(path: str | Path | None, identifier: str) -> Path:
    """Mirror config placement, keeping numerical identity independent of filesystem placement."""
    if path is not None:
        source = resolve_config_path(path).resolve()
        root = _config_root(source)
        if source.is_file() and root is not None:
            return source.relative_to(root).with_suffix("")
    root = Path(__file__).resolve().parents[1] / CONFIG_ROOT
    for relative, entry in _layout_entries(root).items():
        if entry["config_id"] == identifier:
            return Path(relative).with_suffix("")
    return Path(identifier)


def _path_gpu_type(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = re.search(r"(?:^|[-_])(h\d+|b\d+|a\d+|l\d+)(?:[-_.]|$)", part.lower())
        if match:
            return match[1]
    return None


def _model_basename(name: str) -> str:
    parts = Path(name.rstrip("/")).parts
    if not parts:
        raise ValueError(f"model_name does not have a usable model basename: {name!r}")
    cached = next((part for part in parts if part.startswith("models--")), None)
    basename = cached.split("--", 2)[-1] if cached is not None else parts[-1]
    basename = basename.lower()
    if re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", basename) is None:
        raise ValueError(f"model_name does not have a usable model basename: {name!r}")
    return basename


def config_import_subpath(config: LoadedConfig) -> Path:
    """Catalog recipes retain their path; other recipes use model/sequence/hardware-mode-tuning."""
    source = config.path.resolve()
    for parent in source.parents:
        if (
            parent.name == "models"
            and parent.parent.name == "model-catalog"
            and parent.parent.parent.name == "cortex-training"
        ):
            relative = source.relative_to(parent)
            if len(relative.parts) < 3 or relative.name == "model.json":
                raise ValueError(f"expected a model-catalog recipe, not model metadata: {source}")
            return relative
    settings = config.training if config.has_training else config.sub_job["inference_config"]
    budget = (settings.get("mb_spec") or {}).get("max_tokens_per_mb")
    if budget is not None and (type(budget) is not int or budget <= 0):
        raise ValueError(f"{source}: mb_spec.max_tokens_per_mb must be a positive integer")
    sequence = "long" if budget is not None and budget >= 262144 else "standard"
    gpu = config.gpu_type or local_gpu_type()
    if gpu is None:
        raise ValueError(
            f"{source}: config placement requires explicit hardware, a GPU in the source path, or a GPU host"
        )
    mode = "training" if config.has_training else "inference"
    tuning = "lora" if settings.get("peft_config") else "full"
    variant = source.stem.partition("--")[2]
    if variant and re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", variant) is None:
        raise ValueError(
            f"{source}: recipe variant must contain lowercase letters, digits, dots, dashes or underscores"
        )
    suffix = f"--{variant}" if variant else ""
    return Path(_model_basename(config.model_name), sequence, f"{gpu}-{mode}-{tuning}{suffix}.json")
