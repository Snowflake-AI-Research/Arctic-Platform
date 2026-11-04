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

"""The ``numerics.near_certain_logprobs`` switch, shared by the loaders, the LM heads and the RL processors.

On (the default), every trainer log-prob path keeps a near-certain target's log-probability and gradient: the
chunked LM heads leave each row's target out of the softmax sum, the grouped heads normalize against the largest
bucket, grouped distillation sums the bucket gradients to zero in float64, and ``compute_logprobs`` leaves the argmax
out of its sum. Off, each path computes ``x[t] - logsumexp(x)``, which in float32 rounds a target whose probability is
within about 6e-8 of 1 to a log-probability and gradient of exactly 0.

The loader marks the model with :func:`mark_model`. The heads read the mark at forward time, and the RL pipeline
copies it into the post-processors' ``meta`` under :data:`META_KEY`, where ``compute_logprobs`` and grouped
distillation read it. A model no loader marked, and a ``meta`` without the key, count as on.
"""

from __future__ import annotations

MODEL_ATTR = "_arctic_near_certain_logprobs"
META_KEY = "near_certain_logprobs"

LOADER_OPTION_DESCRIPTION = (
    "Keep a near-certain target's log-probability and gradient in the trainer: the chunked LM head leaves each "
    "row's target out of the softmax sum, the grouped heads and grouped distillation keep near-certain buckets, and "
    "compute_logprobs leaves the argmax out of its sum. On by default. False computes x[t] - logsumexp(x), which in "
    "float32 rounds a target within about 6e-8 of probability 1 to a log-probability and gradient of exactly 0."
)


def mark_model(model, enabled: bool) -> None:
    """Record on ``model`` whether its log-prob paths keep near-certain targets."""
    setattr(model, MODEL_ATTR, bool(enabled))


def model_enabled(model) -> bool:
    """Whether ``model``, or the module a DeepSpeed engine wraps, keeps near-certain targets. Unmarked means on."""
    for candidate in (model, getattr(model, "module", None)):
        enabled = getattr(candidate, MODEL_ATTR, None) if candidate is not None else None
        if enabled is not None:
            return bool(enabled)
    return True


def meta_enabled(meta) -> bool:
    """Whether the RL pipeline asked for near-certain targets in ``meta``. A missing key means on."""
    return bool(meta.get(META_KEY, True)) if meta else True
