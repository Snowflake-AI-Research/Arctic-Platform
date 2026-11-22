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

"""GLM-5.3 (glm5next) sparse-attention selections ride the routed-experts capture, for the trainer to replay.

With ``additional_config={"arctic_router_replay_indexer": True}`` (for the Arctic inference server, in
``extra_engine_kwargs``) each sparse-attention layer's selected pools follow the decoder layers' routing rows, in the
layout of ``arctic_platform.common.glm5next_indexer_replay``.
"""

import re
import sys

from arctic_platform.common.glm5next_indexer_replay import capture_pools
from arctic_platform.common.glm5next_indexer_replay import pool_rows
from arctic_platform.common.glm5next_indexer_replay import sparse_layers

INDEXER_REPLAY_KEY = "arctic_router_replay_indexer"
UINT16_MAX = 65535


def _layout(vllm_config) -> tuple[int, int] | None:
    if not (vllm_config.additional_config or {}).get(INDEXER_REPLAY_KEY):
        return None
    config = vllm_config.model_config.hf_text_config
    rows = pool_rows(config) if getattr(config, "index_kpool", None) else None
    if rows is None:
        raise ValueError(f"{INDEXER_REPLAY_KEY} requires a glm5next indexer whose pools fill whole routed-expert rows")
    # The final token has no returned capture.
    max_pool_code = (vllm_config.model_config.max_model_len - 1) // config.index_kpool
    if max_pool_code > UINT16_MAX:
        raise ValueError(f"{INDEXER_REPLAY_KEY} requires (max_model_len - 1) // index_kpool <= {UINT16_MAX}")
    return rows, max_pool_code


def ensure_glm5next_indexer_replay() -> None:
    from vllm.model_executor.layers.fused_moe import routed_experts_capturer

    if getattr(routed_experts_capturer.bind_routed_experts_capturer, "_arctic_indexer_replay", False):
        return
    shape, bind = routed_experts_capturer._get_routed_experts_shape, routed_experts_capturer.bind_routed_experts_capturer

    def widened_shape(vllm_config):
        num_layers, num_experts, top_k = shape(vllm_config)
        layout = _layout(vllm_config)
        if layout is None:
            return num_layers, num_experts, top_k
        sparse = len(sparse_layers(vllm_config.model_config.hf_text_config))
        rows, max_pool_code = layout
        return num_layers + sparse * rows, max(num_experts, max_pool_code + 1), top_k

    def bind_with_indexers(model, capturer):
        bind(model, capturer)
        for module in model.modules():
            vllm_config = getattr(module, "vllm_config", None)
            if getattr(module, "indexer_op", None) is None or vllm_config is None:
                continue
            layout = _layout(vllm_config)
            if layout is None:
                continue
            layer = int(re.search(r"layers\.(\d+)\.", module.prefix).group(1))
            num_layers = vllm_config.model_config.get_total_num_hidden_layers()
            if layer >= num_layers:  # MTP layers sit past the capture
                continue
            op = module.indexer_op
            rows, _ = layout
            first_row = num_layers + sparse_layers(module.config).index(layer) * rows

            def forward(hidden_states, *args, _row=first_row, _forward=op._forward_method, _module=module, **kwargs):
                topk_indices = _forward(hidden_states, *args, **kwargs)
                capture_pools(
                    capturer.capture,
                    _row,
                    topk_indices[: hidden_states.shape[0]],
                    _module.topk_tokens,
                    _module.index_kpool,
                    _module.config.num_experts_per_tok,
                )
                return topk_indices

            op._forward_method = forward  # CustomOp dispatches through the method it bound at construction

    bind_with_indexers._arctic_indexer_replay = True  # type: ignore[attr-defined]
    routed_experts_capturer._get_routed_experts_shape = widened_shape
    routed_experts_capturer.bind_routed_experts_capturer = bind_with_indexers
    # The aux-output worker imports the binder by name; patch its copy only if it is already loaded.
    aux_worker = sys.modules.get("vllm.distributed.aux_output_connector.worker")
    if aux_worker is not None:
        setattr(aux_worker, "bind_routed_experts_capturer", bind_with_indexers)
