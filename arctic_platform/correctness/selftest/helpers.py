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

"""Static correctness selftests: helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

# Config factory.


def _config_factory_native_config(
    training: dict,
    *,
    model_name: str = "Qwen/Qwen3-8B",
    dtype: str = "bfloat16",
    seed: int | None = 42,
    sampling: dict | None = None,
) -> dict:
    training = deepcopy(training)
    n_gpus = int(training.pop("n_gpus"))
    max_seq_len = int(training.pop("max_seq_len", 8192))
    ds = deepcopy(training.pop("ds_config", {}))
    for key in ("train_batch_size", "gradient_clipping"):
        value = training.pop(key, None)
        if value is not None:
            ds.setdefault(key, value)
    optimizer = training.pop("optimizer", None)
    if optimizer is not None:
        optimizer = deepcopy(optimizer)
        name = optimizer.pop("name", "AdamW")
        ds.setdefault("optimizer", {"type": name, "params": optimizer})
    peft = training.pop("peft_config", None)
    result = {
        "model_name": model_name,
        "dtype": dtype,
        "seed": seed,
        "max_seq_len": max_seq_len,
        "training_gpus": n_gpus,
        "sampling_gpus": 0,
        "log_prob_gpus": 0,
        "training": {"ds_config": ds, "ds_worker_config": training},
        "sampling": {},
        "backend": {"type": "onprem", "protocol": "ray"},
    }
    if peft is not None:
        result["training"]["peft"] = peft
    if sampling is not None:
        sampling = deepcopy(sampling)
        result["sampling_gpus"] = int(sampling.pop("n_gpus"))
        sample_max = int(sampling.pop("max_seq_len", max_seq_len))
        if sample_max != max_seq_len:
            raise ValueError("training and sampling max_seq_len must match")
        result["sampling"] = {"vllm": sampling.pop("vllm_config", {}), "arctic_inference_config": sampling or None}
    return result


def _synthetic_config_spec(root: Path, *, lora: bool = False):
    """Small native config/spec inputs; no bundled catalog or model assets."""
    from arctic_platform.correctness.harness.config import ArmSpec
    from arctic_platform.correctness.harness.config import TestTolerance
    from arctic_platform.correctness.harness.config import arms_for
    from arctic_platform.correctness.harness.config import config_checksum
    from arctic_platform.correctness.harness.config import load_config

    from .check_catalog_configuration import _applicable_tests_spec

    kind = "lora" if lora else "full"
    path = root / "configs" / "qwen3-8b" / "h200" / f"train-sft-{kind}-4gpus-64k.config"
    path.parent.mkdir(parents=True, exist_ok=True)
    training = {
        "n_gpus": 4,
        "max_seq_len": 65536,
        "sp_size": 1,
        "ep_size": 1,
        "model_provider": "huggingface",
        "attn_implementation": "flash_attention_3",
        "mb_spec": {"max_tokens": 65536},
        "ds_config": {
            "gradient_accumulation_steps": 1,
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 4,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
        },
        "optimizer": {"name": "AdamW", "lr": 0.0001, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
    }
    if lora:
        training["peft_config"] = {
            "peft_type": "Lora",
            "r": 2,
            "lora_alpha": 4,
            "lora_dropout": 0.0,
            "target_modules": ["q_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }
    path.write_text(json.dumps(_config_factory_native_config(training)))
    cfg = load_config(path)
    spec = _applicable_tests_spec(["single-step-grads", "single-step-optimizer"])
    spec.config_id = cfg.config_id
    spec.config_path = str(path)
    spec.config_checksum = config_checksum(cfg)
    spec.test_tolerances["single-step-grads"] = TestTolerance(absolute=0.005)
    spec.arms = [
        ArmSpec(
            name=arm.name,
            global_batch_size=arm.global_batch_size,
            max_seq_len=arm.max_seq_len,
            total_tokens=arm.total_tokens,
            active_tokens=arm.total_tokens - arm.global_batch_size,
            dss_microbatches=arm.microbatches(cfg.max_tokens_per_mb, cfg.dp_size),
            pad_fraction=0.0,
        )
        for arm in arms_for(cfg.max_seq_len, cfg.n_gpus)
    ]
    spec.model.source_checkpoint = cfg.model_name
    spec.model.num_hidden_layers = 4
    model = root / "reviewed-model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"num_hidden_layers":4,"vocab_size":128}')
    spec.model.cache_path = str(model)
    spec.test_settings["single-step-optimizer"] = {
        "reference_token_budget": 10240,
        "model_cache_path": str(model),
    }
    spec_path = root / "specs" / f"{cfg.config_id}.json"
    spec_path.parent.mkdir(exist_ok=True)
    spec.write(spec_path)
    return path, spec_path
