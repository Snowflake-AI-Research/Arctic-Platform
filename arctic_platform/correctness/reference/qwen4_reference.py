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

"""Independent upstream QSA with the target's frozen PLE and indexer parameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Qwen4ParameterPolicy:
    cpu_parameter_ids: frozenset[int]
    frozen_parameter_ids: frozenset[int]


def qwen4_parameter_policy(model) -> Qwen4ParameterPolicy:
    """Discover the reference's structural placement/freezing policy without changing the model."""
    from torch import nn

    registered = {id(parameter) for parameter in model.parameters()}
    tables = set()
    indexers = set()
    try:
        layers = model.get_decoder().layers
        for layer in layers:
            ple = getattr(layer, "ple", None)
            if ple is not None:
                table = ple.ple_embedding.ngram_embedding.weight
                if not isinstance(table, nn.Parameter):
                    raise ValueError("qwen4_exp PLE table must be a registered Parameter")
                tables.add(id(table))
            attention = getattr(layer, "self_attn", None)
            if attention is not None:
                indexer = attention.indexer
                if not isinstance(indexer, nn.Module):
                    raise ValueError("qwen4_exp QSA indexer must be a registered module")
                parameters = tuple(indexer.parameters())
                if not parameters:
                    raise ValueError("qwen4_exp QSA indexer must contain registered parameters")
                indexers.update(id(parameter) for parameter in parameters)
    except (AttributeError, TypeError) as exc:
        raise ValueError("qwen4_exp reference requires a complete PLE/QSA layer period") from exc
    if not tables or not indexers:
        raise ValueError("qwen4_exp reference requires a complete PLE/QSA layer period")
    frozen = tables | indexers
    if not frozen <= registered:
        raise ValueError("qwen4_exp policy parameters must be registered on the model")
    return Qwen4ParameterPolicy(frozenset(tables), frozenset(frozen))


def load_qwen4_reference(model_class, model_path, *, dtype, attn_implementation, device):
    """Keep only the frozen n-gram table on CPU, without accelerator reload hooks."""
    if attn_implementation == "qsa_flex":
        # Upstream SDPA consumes its own QSA-selected mask, not dense causal attention.
        attn_implementation = "sdpa"
    if attn_implementation not in {"sdpa", "eager"}:
        raise ValueError("independent qwen4_exp reference requires upstream QSA through sdpa or eager")

    model, loading_info = model_class.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        output_loading_info=True,
    )
    for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs"):
        if loading_info.get(key):
            raise ValueError(f"qwen4_exp reference checkpoint {key}: {loading_info[key]}")
    if any(parameter.device.type != "cpu" for parameter in model.parameters()):
        raise ValueError("qwen4_exp reference must load on CPU before selective placement")

    policy = qwen4_parameter_policy(model)
    for parameter in model.parameters():
        if id(parameter) in policy.frozen_parameter_ids:
            parameter.requires_grad_(False)

    # Module.to uses _apply too, but would also allocate the 95 GiB frozen table on CUDA.
    model._apply(lambda tensor: tensor if id(tensor) in policy.cpu_parameter_ids else tensor.to(device=device))
    return model
