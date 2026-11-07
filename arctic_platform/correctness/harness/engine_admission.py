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

"""Resolve omitted accumulation only against the automatically owned local DSS runtime."""

from __future__ import annotations

import importlib

from .config import validate_first_two_model_dtype


def _owned_local_engine_config(config, *, automatically_owned_local: bool) -> dict:
    """Use the actual runtime helper and actual submitted payload, without adding overrides."""
    if not automatically_owned_local:
        raise ValueError("runtime-default admission requires the automatically owned local DSS gateway")

    import torch

    from arctic_platform.client import CortexConfig

    from .gateway import _allocation_hostfile
    from .gateway import training_client_config

    try:
        JobConfig = importlib.import_module("dss.ray_dss.models").JobConfig

        capacity = torch.cuda.device_count()
        if config.n_gpus > capacity:
            raise ValueError("runtime-default admission cannot verify a nonlocal DSS worker installation")
        hostfile = _allocation_hostfile()
        if hostfile is not None:
            LocalPlacementBackend = importlib.import_module("dss.ray_dss.placement.local").LocalPlacementBackend
            is_local = importlib.import_module("dss.ray_dss.utils").is_local

            # An owned gateway starts a fresh placement pool. Use its real
            # selection, not the on-prem gateway's unrelated peer decision.
            placement = LocalPlacementBackend(hostfile=str(hostfile))
            layout = placement._pick_hosts(config.n_gpus)
            if any(not is_local(node.host) or any(gpu >= capacity for gpu in node.gpu_ids) for node in layout):
                raise ValueError("runtime-default admission cannot verify a nonlocal DSS worker installation")
        # Without an allocation hostfile, the owned gateway writes this host
        # and its visible device count into its launch-only hostfile.
        runtime = importlib.import_module("dss.ray_dss.jobs.gpu.training")
        builder = getattr(runtime, "_build_ds_config", None)
        if not callable(builder):
            raise ValueError("installed DSS runtime does not expose _build_ds_config")
        client = training_client_config(config, CortexConfig(base_url="http://127.0.0.1:1"))
        payload = client.to_cortex()[0]
        cfg = JobConfig.model_validate(payload).to_engine_dict("correctness-admission")
        return builder(cfg, config.n_gpus, config.sp_size)
    except Exception as exc:
        raise ValueError(f"cannot resolve installed DSS runtime accumulation: {exc}") from exc


def validate_reference_accumulation(config, *, automatically_owned_local: bool = False) -> None:
    """Preserve explicit FP32 support; infer defaults only from the owned executing runtime."""
    validate_first_two_model_dtype(config)
    if config.is_rl:
        return
    data_types = (config.training.get("ds_config") or {}).get("data_types") or {}
    accumulation = data_types.get("grad_accum_dtype") if isinstance(data_types, dict) else "unsupported"
    if accumulation == "fp32":
        return
    if accumulation is not None:
        raise ValueError(
            f"FP32 HF reference accumulation requires grad_accum_dtype='fp32'; got {accumulation!r}. "
            'Set data_types.grad_accum_dtype to "fp32" in a supported supplied configuration; '
            "admission does not change this setting."
        )
    if not automatically_owned_local:
        raise ValueError("omitted accumulation requires an automatically owned local DSS runtime; unverified defaults")

    import torch
    from deepspeed.runtime.config import DtypeEnum
    from deepspeed.runtime.config import get_data_types_params
    from deepspeed.runtime.config import get_torch_autocast_enabled
    from deepspeed.runtime.constants import GRAD_ACCUM_DTYPE_DEFAULT
    from deepspeed.runtime.precision_config import get_bfloat16_config
    from deepspeed.runtime.precision_config import get_float16_config
    from deepspeed.runtime.zero.config import get_zero_config

    resolved = _owned_local_engine_config(config, automatically_owned_local=True)
    # These are the runtime's typed parsers, without initializing distributed
    # groups or validating batch sizes against an uninitialized process group.
    get_zero_config(resolved)
    if (
        not get_bfloat16_config(resolved).enabled
        or get_float16_config(resolved).enabled
        or get_torch_autocast_enabled(resolved)
    ):
        raise ValueError("runtime-default reference admission requires BF16 without FP16 or autocast")
    data_types = get_data_types_params(resolved)
    param_dtype = data_types.get("param_dtype")
    if param_dtype is not None and DtypeEnum(param_dtype).value != torch.bfloat16:
        raise ValueError("runtime-default reference admission requires BF16 model parameters")
    dtype = data_types.get("grad_accum_dtype", GRAD_ACCUM_DTYPE_DEFAULT)
    # DeepSpeedEngine.get_data_types falls back to the BF16 model dtype on None.
    accumulation = torch.bfloat16 if dtype is None else DtypeEnum(dtype).value
    if accumulation != torch.float32:
        raise ValueError(
            f"installed DSS shared engine resolves {accumulation} gradient accumulation, not FP32; "
            "the unchanged recipe is unsupported by this runtime"
        )
