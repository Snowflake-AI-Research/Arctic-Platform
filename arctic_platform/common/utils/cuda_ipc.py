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

"""CUDA IPC handle merging for colocated multi-GPU weight sync."""

from __future__ import annotations

import base64
import pickle


def merge_cuda_ipc_payloads(results: list[dict]) -> dict:
    """Merge per-rank CUDA IPC handle payloads into one payload.

    Each training rank returns ``gather_cuda_ipc_handles`` output with the same
    parameter names (deterministic ``named_parameters()`` order) and a list of
    per-parameter ``{gpu_uuid: handle}`` dicts for its own physical GPU. We merge
    them element-wise so every parameter's handle dict spans all GPUs, letting a
    colocated inference replica on any GPU find a handle for itself.
    """
    payloads = [r for r in results if r]
    if not payloads:
        return {
            "names": [],
            "dtype_names": [],
            "shapes": [],
            "ipc_handles_pickled": base64.b64encode(pickle.dumps([])).decode("utf-8"),
            "num_params": 0,
        }

    base = payloads[0]
    names = base["names"]
    merged_handles: list[dict] = [dict() for _ in names]
    for r in payloads:
        rank_handles = pickle.loads(base64.b64decode(r["ipc_handles_pickled"]))
        for i, hd in enumerate(rank_handles):
            merged_handles[i].update(hd)

    return {
        "names": names,
        "dtype_names": base["dtype_names"],
        "shapes": base["shapes"],
        "ipc_handles_pickled": base64.b64encode(pickle.dumps(merged_handles)).decode("utf-8"),
        "num_params": base["num_params"],
    }
