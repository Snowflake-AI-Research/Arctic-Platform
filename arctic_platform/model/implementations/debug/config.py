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
"""Provider-neutral training diagnostic configuration."""

from __future__ import annotations

from typing import Any
from typing import Mapping


def training_debug_config(training_config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return ``training_config.debug`` with fallback to the legacy Prime-RL location."""
    debug_config = training_config.get("debug")
    source = "training_config.debug"
    if debug_config is None:
        prime_rl_config = training_config.get("prime_rl") or {}
        if not isinstance(prime_rl_config, Mapping):
            raise TypeError(f"training_config.prime_rl must be a mapping, got {type(prime_rl_config).__name__}")
        debug_config = prime_rl_config.get("debug")
        source = "training_config.prime_rl.debug"

    if debug_config is None:
        return {}
    if not isinstance(debug_config, Mapping):
        raise TypeError(f"{source} must be a mapping, got {type(debug_config).__name__}")
    return debug_config


def gradient_sample_max_numel(training_config: Mapping[str, Any]) -> int:
    """Return the validated gradient-sampling element limit."""
    max_numel = int(training_debug_config(training_config).get("gradient_sample_max_numel", 0))
    if max_numel < 0:
        raise ValueError(
            f"gradient_sample_max_numel in the training debug config must be non-negative, got {max_numel}"
        )
    return max_numel


def gradient_norms_per_param(training_config: Mapping[str, Any]) -> bool:
    """Return whether ``step`` should report one gradient norm per parameter.

    A single global ``grad_norm`` averages a localized discrepancy away: a gradient that differs only in the
    lm_head, or only in one layer's attention projections, moves it by a hair that reads as precision noise. One
    norm per parameter localizes such a difference to a module in a single run, which is the difference between
    bisecting a model and reading an answer. It gathers every parameter's full gradient, so it is a debugging
    tool, not something to leave on in production.
    """
    value = training_debug_config(training_config).get("gradient_norms_per_param", False)
    if not isinstance(value, bool):
        raise TypeError(
            f"gradient_norms_per_param in the training debug config must be a bool, got {type(value).__name__}"
        )
    return value


def router_tokens_per_expert(training_config: Mapping[str, Any]) -> bool:
    """Return whether ``step`` should report how many tokens the router sent to each global expert.

    Gradient norms answer "did this tensor's gradient move"; they cannot answer "did the router send the same
    work here". Those are different questions whenever the expert-parallel degree changes, because a comparison
    of expert gradients across two degrees rests on the assumption that top-k routing is degree-independent, and
    that assumption is checkable only by counting. The count is per MoE layer, indexed by global expert, summed
    over the whole world, so two topologies produce the same-length vector indexed the same way.
    """
    value = training_debug_config(training_config).get("router_tokens_per_expert", False)
    if not isinstance(value, bool):
        raise TypeError(
            f"router_tokens_per_expert in the training debug config must be a bool, got {type(value).__name__}"
        )
    return value
