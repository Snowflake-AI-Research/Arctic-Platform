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

"""Native ArcticClientConfig fixtures for correctness selftests."""

from __future__ import annotations

from copy import deepcopy


def native_config(
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
        result["sampling"] = {
            "vllm": sampling.pop("vllm_config", {}),
            "arctic_inference_config": sampling or None,
        }
    return result
