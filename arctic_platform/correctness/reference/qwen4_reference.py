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

    tables = []
    indexers = []
    for layer in model.get_decoder().layers:
        ple = getattr(layer, "ple", None)
        if ple is not None:
            tables.append(ple.ple_embedding.ngram_embedding.weight)
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            indexers.append(attention.indexer)
    if not tables or not indexers:
        raise ValueError("qwen4_exp reference requires a complete PLE/QSA layer period")
    for table in tables:
        table.requires_grad_(False)
    for indexer in indexers:
        indexer.requires_grad_(False)

    table_ids = {id(table) for table in tables}
    # Module.to uses _apply too, but would also allocate the 95 GiB frozen table on CUDA.
    model._apply(lambda tensor: tensor if id(tensor) in table_ids else tensor.to(device=device))
    return model
