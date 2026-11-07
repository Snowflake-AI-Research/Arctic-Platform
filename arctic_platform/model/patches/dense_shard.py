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
"""``patches.dense_shard``: shard a Hugging Face model's trainable weights over the GPUs of a node, for full FT.

A full fine-tune on the huggingface route keeps every weight whole on every GPU: Gemma 4 31B holds 61 GB of BF16
weights per GPU, Qwen3.8-27B 54 GB, beside ZeRO-2's share of the FP32 masters and moments. This patch applies the MoE
loaders' node sharding (``implementations/moe/dense_shard.py``, which documents the layout, the gradients and the state
dict) to each decoder layer of the text decoder and to the input embedding: each GPU of a node keeps ``1 / g`` of
their trainable weights, gathered over the node around each call (and again in the recompute under activation
checkpointing). A parameter that another module also holds stays whole, so an LM head tied to the embedding keeps the
embedding whole too. The LM head itself always stays whole: Liger's fused linear cross-entropy and the chunked LM head
read ``lm_head.weight`` from the model's forward, not by calling the head.

It runs last, after the patches that wrap or replace modules (Liger, gradient checkpointing, keep-attention,
activation offload), before the optimizer and DeepSpeed are built, and once the weights are loaded. It refuses PEFT
(the frozen base is not sharded here) and the tiled MLP (which reads the MLP's weights again in its backward, after
the layer released them).

The model has no family export contract, so its weight export (a weights-only checkpoint, the weight sync) is the
generic one, which gathers the shards onto rank 0 one sharded module at a time and yields the unsharded model's
tensors under their names (``weight_export.iter_model_weights``). Its resumable checkpoints take DeepSpeed's non-MoE
path inside ``full_state_dict`` (dss's checkpointer for ``shard_dense``).
"""

from __future__ import annotations

import torch.distributed as dist
import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


def text_decoder(model: nn.Module) -> nn.Module:
    """The model's text decoder: the module holding the decoder layers (``layers``) and the input embedding.

    ``get_decoder()`` when it returns one (a multimodal model returns its language model), else the one module with
    both a ``layers`` ModuleList and ``embed_tokens``.
    """
    get_decoder = getattr(model, "get_decoder", None)
    decoder = get_decoder() if callable(get_decoder) else None
    if decoder is not None and isinstance(getattr(decoder, "layers", None), nn.ModuleList):
        return decoder
    found = [
        module
        for module in model.modules()
        if isinstance(getattr(module, "layers", None), nn.ModuleList) and hasattr(module, "embed_tokens")
    ]
    if len(found) != 1:
        raise ValueError(
            f"dense_shard: found {len(found)} modules with decoder layers and an input embedding; it shards the text "
            "decoder of a causal LM"
        )
    return found[0]


@register_patch("dense_shard")
def apply_dense_shard(model: nn.Module, ctx: LoaderContext) -> None:
    from arctic_platform.model.implementations.moe.deepspeed_integration import patch_deepspeed_moe_detection
    from arctic_platform.model.implementations.moe.dense_shard import install_dense_shards
    from arctic_platform.model.implementations.moe.logging_utils import get_logger

    if not dist.is_initialized():
        raise RuntimeError("dense_shard needs torch.distributed: it shards over the GPUs of each node")
    value = ctx.spec.patches.dense_shard
    group_size = None if value is True else int(value)
    decoder = text_decoder(model)
    layers = list(decoder.layers)
    modules: list[nn.Module] = [getattr(layer, "_checkpoint_wrapped_module", layer) for layer in layers]
    embedding = model.get_input_embeddings()
    if isinstance(embedding, nn.Module):
        modules.append(embedding)
    sharded = install_dense_shards(model, modules, group_size)
    # ZeRO reduces the shards over their expert-data-parallel group only for a model it knows to have MoE layers.
    patch_deepspeed_moe_detection()
    if sharded:
        full = sum(s.full_bytes() for s in sharded)
        kept = sum(s.shard_bytes() for s in sharded)
        get_logger().info(
            f"dense_shard: trainable weights of {len(sharded)} modules sharded over "
            f"{dist.get_world_size(sharded[0].group)} GPUs of the node: {full / 2**30:.2f} GiB -> "
            f"{kept / 2**30:.2f} GiB per GPU"
        )
