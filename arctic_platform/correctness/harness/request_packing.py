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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit microbatch request packing, separate from the ordinary unsplit request."""

from __future__ import annotations

from copy import deepcopy

from .dss_driver import pack


def pack_microbatches(
    batch,
    microbatches: int,
    *,
    model_provider: str = "huggingface",
    loss_fn: str = "sft",
    processing_config=None,
) -> dict:
    """Pack an explicitly requested number of microbatches without changing the input config."""
    body = pack(batch, model_provider=model_provider)
    body["processing"] = {"loss_fn": loss_fn}
    if processing_config is not None:
        body["processing"]["config"] = deepcopy(processing_config)
    if microbatches <= 1:
        return body
    if batch.rows < microbatches:
        raise ValueError(f"cannot split {batch.rows} row(s) into {microbatches} microbatches")
    base_rows, extra_rows = divmod(batch.rows, microbatches)
    tensors = body["batch"]
    splits = []
    start = 0
    for index in range(microbatches):
        end = start + base_rows + (index < extra_rows)
        splits.append({key: value[start:end].contiguous() for key, value in tensors.items()})
        start = end
    body["batch"] = splits
    return body
