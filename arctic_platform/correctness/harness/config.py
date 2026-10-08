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

"""Intake for an operator-supplied Arctic Platform job config.

The config arrives already sized for one node, so this validates and refuses rather than reshaping. A config
that does not fit produces one message naming the constraint it missed.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from arctic_platform.client.config import ArcticClientConfig

GPU_ATTENTION_IMPLEMENTATIONS = {
    "h200": "flash_attention_3",
    "b200": "flash_attention_4",
    "b300": "flash_attention_4",
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

    @property
    def training(self) -> Dict[str, Any]:
        return self.sub_job["training_config"]

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
        return replace(self, sub_job=sub_job, sub_jobs=sub_jobs, native=native)

    @property
    def sp_size(self) -> int:
        return int(self.training.get("sp_size", 1))

    @property
    def dp_size(self) -> int:
        return self.n_gpus // self.sp_size

    @property
    def gpu_type(self) -> str | None:
        """The GPU type this config was written for, read from its directory: ``h200``, ``b200``, ``b300``.

        ``None`` for a config outside the ``configs/`` tree, where the layout carries no such claim.
        """
        parts = list(self.path.resolve().with_suffix("").parts)
        if CONFIG_ROOT not in parts:
            return None
        below = parts[parts.index(CONFIG_ROOT) + 1 :]
        return below[1].lower() if len(below) >= 3 else None

    @property
    def attention_implementation(self) -> str:
        """Attention backend required by the accelerator named in the config path."""
        configured = self.training.get("attn_implementation")
        expected = expected_attention_implementation(self.gpu_type)
        if configured is not None and expected is not None and configured != expected:
            raise ValueError(
                f"{self.config_id}: {self.gpu_type} requires attn_implementation={expected!r}, "
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
    """``configs/qwen3.8-27b/h200/train-sft-8gpus-2k.config`` -> ``qwen3.8-27b-h200-train-sft-8gpus-2k``.

    The layout is model / GPU type / workload, and a filename is unique only within one model and GPU
    type, so the identifier names the whole path to keep reports and spec files distinct. A config outside
    the tree keeps its filename.
    """
    parts = list(path.resolve().with_suffix("").parts)
    if CONFIG_ROOT in parts:
        parts = parts[parts.index(CONFIG_ROOT) + 1 :]
    else:
        parts = parts[-1:]
    return "-".join(parts)


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
    sub_jobs = [training_sub]
    if native.sampling_gpus > 0:
        inference = {
            "n_gpus": native.sampling_gpus,
            "max_seq_len": native.max_seq_len,
            "vllm_config": deepcopy(native.sampling.vllm),
        }
        if native.sampling.arctic_inference_config:
            inference.update(deepcopy(native.sampling.arctic_inference_config))
        sub_jobs.append(
            {
                "job_type": "sampling",
                "model_name": native.model_name,
                "dtype": native.dtype,
                "seed": native.seed,
                "inference_config": inference,
            }
        )
    return training_sub, sub_jobs


def load_config(path: str | Path) -> LoadedConfig:
    path = Path(path)
    raw = normalize_ints(json.loads(path.read_text()))
    native = ArcticClientConfig.model_validate(raw)
    training, sub_jobs = _legacy_views(native)
    return LoadedConfig(config_id=config_id(path), path=path, sub_job=training, sub_jobs=sub_jobs, native=native)


def config_checksum(config: LoadedConfig) -> str:
    """Hash the normalized native fields that define the training job."""
    if config.native is None:
        canonical = json.dumps(config.sub_job, sort_keys=True, separators=(",", ":"))
    else:
        raw = config.native.model_dump(mode="json", exclude_none=False)
        projection = {
            key: raw[key] for key in ("model_name", "seed", "dtype", "max_seq_len", "training_gpus", "training")
        }
        canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _sibling_for(path: Path, gpu_type: str) -> str:
    """The same workload under another accelerator: ``<model>/h200/<name>`` -> ``<model>/b200/<name>``.

    Named whether or not it exists, because the point is to say where the config for this host belongs.
    """
    parts = list(path.parts)
    if CONFIG_ROOT in parts:
        i = parts.index(CONFIG_ROOT)
        if len(parts) - i >= 4:
            parts[i + 2] = gpu_type
            return str(Path(*parts))
    return str(Path(CONFIG_ROOT, "<model>", gpu_type, path.name))


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
