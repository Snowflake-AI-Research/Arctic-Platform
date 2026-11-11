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
``extra_engine_kwargs``) a glm5next model's routed-experts capture grows past its decoder layers by
``index_topk // index_kpool // num_experts_per_tok`` rows of ``num_experts_per_tok`` ids per sparse-attention layer,
in layer order: the pool ids that layer's indexer selected for the token, ``-1`` (65535 once vLLM narrows the capture
to uint16) where it selected none. The prefix-cache store, the replay cache and the transport carry the rows
unchanged; ``glm53.modeling_glm5_next`` hands them to the trainer's sparse attention.
"""

import re
import sys

import torch

INDEXER_REPLAY_KEY = "arctic_router_replay_indexer"
UINT16_SENTINEL = 65535  # where -1 lands once the capture is narrowed to uint16
SPARSE_LAYER_TYPES = ("deepseek_sparse_attention", "indexed_attention")


def pool_rows(text_config) -> int | None:
    """Capture rows per sparse-attention layer; None when the selected pools do not fill whole rows."""
    pools = text_config.index_topk // text_config.index_kpool
    rows, remainder = divmod(pools, text_config.num_experts_per_tok)
    return None if remainder else rows


def sparse_layers(text_config) -> list[int]:
    return [i for i, kind in enumerate(text_config.layer_types) if kind in SPARSE_LAYER_TYPES]


def capture_pools(capture, first_row: int, topk_indices: torch.Tensor, topk_tokens: int, kpool: int, top_k: int):
    """Writes one layer's selected pools, read off its expanded token indices (``pool * kpool + offset`` per pool,
    then the tail), into ``topk_tokens // kpool // top_k`` capture rows from ``first_row``."""
    pools = torch.div(topk_indices[:, :topk_tokens:kpool], kpool, rounding_mode="floor")
    for row in range(pools.shape[1] // top_k):
        capture(first_row + row, pools[:, row * top_k : (row + 1) * top_k])


def _rows(vllm_config) -> int | None:
    """Capture rows per sparse layer when the replay is on, else None; refuses a model it cannot carry."""
    if not (vllm_config.additional_config or {}).get(INDEXER_REPLAY_KEY):
        return None
    config = vllm_config.model_config.hf_text_config
    rows = pool_rows(config) if getattr(config, "index_kpool", None) else None
    # Pool ids must stay below the sentinel; the capture stops one token short of the row.
    if rows is None or (vllm_config.model_config.max_model_len - 1) // config.index_kpool > UINT16_SENTINEL:
        raise ValueError(
            f"{INDEXER_REPLAY_KEY} needs a glm5next model whose index_topk // index_kpool fills rows of "
            f"num_experts_per_tok ids, and (max_model_len - 1) // index_kpool <= {UINT16_SENTINEL}"
        )
    return rows


def ensure_glm5next_indexer_replay() -> None:
    from vllm.model_executor.layers.fused_moe import routed_experts_capturer

    if getattr(routed_experts_capturer.bind_routed_experts_capturer, "_arctic_indexer_replay", False):
        return
    shape, bind = routed_experts_capturer._get_routed_experts_shape, routed_experts_capturer.bind_routed_experts_capturer

    def widened_shape(vllm_config):
        num_layers, num_experts, top_k = shape(vllm_config)
        rows = _rows(vllm_config)
        if rows is None:
            return num_layers, num_experts, top_k
        sparse = len(sparse_layers(vllm_config.model_config.hf_text_config))
        # More than 256 experts makes vLLM narrow the capture to uint16, which pool ids need.
        return num_layers + sparse * rows, max(num_experts, 257), top_k

    def bind_with_indexers(model, capturer):
        for module in model.modules():
            vllm_config = getattr(module, "vllm_config", None)
            if getattr(module, "indexer_op", None) is None or vllm_config is None or _rows(vllm_config) is None:
                continue
            layer = int(re.search(r"layers\.(\d+)\.", module.prefix).group(1))
            num_layers = vllm_config.model_config.get_total_num_hidden_layers()
            if layer >= num_layers:  # MTP layers sit past the capture
                continue
            op = module.indexer_op
            # layer_id and capture_fn make the op a RoutedExpertsCaptureSource, which bind() binds.
            op.layer_id = num_layers + sparse_layers(module.config).index(layer) * _rows(vllm_config)
            op.capture_fn = None

            def forward(hidden_states, *args, _op=op, _forward=op._forward_method, _module=module, **kwargs):
                topk_indices = _forward(hidden_states, *args, **kwargs)
                capture_pools(
                    _op.capture_fn.func,
                    _op.layer_id,
                    topk_indices[: hidden_states.shape[0]],
                    _module.topk_tokens,
                    _module.index_kpool,
                    _module.config.num_experts_per_tok,
                )
                return topk_indices

            op._forward_method = forward  # CustomOp dispatches through the method it bound at construction
        bind(model, capturer)

    bind_with_indexers._arctic_indexer_replay = True  # type: ignore[attr-defined]
    routed_experts_capturer._get_routed_experts_shape = widened_shape
    routed_experts_capturer.bind_routed_experts_capturer = bind_with_indexers
    # The aux-output worker imports the binder by name; patch its copy only if it is already loaded.
    aux_worker = sys.modules.get("vllm.distributed.aux_output_connector.worker")
    if aux_worker is not None:
        setattr(aux_worker, "bind_routed_experts_capturer", bind_with_indexers)
