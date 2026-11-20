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
"""Expert-weight gradients of a chunked dispatch are summed over the chunks in fp32, as one call over all tokens is."""

import dataclasses

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from arctic_platform.model.implementations.moe.distributed import expert_parallel
from arctic_platform.model.implementations.moe.distributed.token_permute import permute_tokens
from arctic_platform.model.implementations.moe.distributed.token_permute import unpermute_tokens
from arctic_platform.model.implementations.moe.layers import moe
from arctic_platform.model.implementations.moe.layers.checkpointing import checkpoint_method
from arctic_platform.model.implementations.moe.routing_weights import apply_routing_weights

CHUNKS, TOKENS, SMALL = 8, 64, 2.0**-10


@dataclasses.dataclass
class _State:
    permuted_scores: torch.Tensor | None
    permuted_indices: torch.Tensor
    num_recv_tokens: int


class _OneRankDispatch:
    """Dispatch and combine with every expert on this rank: only the cross-rank exchange is replaced."""

    @staticmethod
    def dispatch_tokens_async(hidden, indices, scores, num_experts, group, *, score_before_experts):
        return hidden, indices, scores.float(), num_experts

    @staticmethod
    def finalize_dispatch_tokens(pending):
        hidden, indices, scores, num_experts = pending
        rows, row_scores, row_tokens, counts = permute_tokens(hidden, indices, scores, num_experts)
        return rows, counts, _State(row_scores, row_tokens, len(hidden))

    @staticmethod
    def combine_tokens(rows, state):
        if state.permuted_scores is not None:
            rows = apply_routing_weights(rows, state.permuted_scores)
        return unpermute_tokens(rows, state.permuted_indices, state.num_recv_tokens)

    @staticmethod
    def sync_combine():
        pass


@pytest.mark.parametrize("recompute", ["none", "selective", "dispatch_chunks", "full"])
def test_expert_gradients_sum_over_dispatch_chunks_in_fp32(monkeypatch, recompute):
    """Eight chunks of the same tokens, the last weighted 1 and the others 2**-10 in the output gradient. Summed in
    fp32 (one dispatch over all tokens, or the chunk sum), each expert-weight gradient grows by about 7 * 2**-10, and
    the chunked gradient equals the single dispatch's to bf16 rounding; a bf16 running sum meets the last chunk first
    in backward and drops the rest. Holds under selective (routed experts), per-chunk and full-layer recompute."""
    histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda values, *args, **kwargs: histc(values.float(), *args, **kwargs))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(expert_parallel, "get_ep_group", lambda experts: None)
    monkeypatch.setattr(moe, "get_ep_comm_module", lambda backend: _OneRankDispatch())
    torch.manual_seed(0)
    args = moe.MoEArgs(
        num_experts=4, num_shared_experts=0, top_k=2, score_before_experts=False, load_balance_coeff=None
    )
    layer = moe.MoE(args, dim=256, hidden_dim=128).to(torch.bfloat16)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.5)
    layer.set_ep_comm_backend("deepep")
    layer.checkpoint_dispatch_chunks = recompute == "dispatch_chunks"
    if recompute == "selective":
        checkpoint_method(layer, "_run_local_routed_experts")
    forward = (lambda x: checkpoint(layer, x, use_reentrant=False)) if recompute == "full" else layer
    tokens = torch.randn(1, TOKENS, 256).to(torch.bfloat16).repeat(1, CHUNKS, 1)
    last_chunk = (torch.arange(CHUNKS * TOKENS) >= (CHUNKS - 1) * TOKENS)[None, :, None]

    def expert_grads(chunk_size: int | None, small: float) -> list[torch.Tensor]:
        layer.set_deepep_token_chunk_size(chunk_size)
        layer.zero_grad(set_to_none=True)
        output = forward(tokens)
        output.backward(torch.where(last_chunk, 1.0, small).expand_as(output).to(output.dtype).contiguous())
        return [getattr(layer.experts, name).grad for name in ("w1", "w2", "w3")]

    for whole, chunked in zip(expert_grads(None, SMALL), expert_grads(TOKENS, SMALL)):
        # Chunk and single-dispatch GEMMs round differently; near-cancelling elements see it at the gradient's scale.
        torch.testing.assert_close(chunked, whole, rtol=1.6e-2, atol=2**-10 * whole.abs().max().item())
    growth = {}
    for chunk_size in (None, TOKENS):
        big, both = (torch.stack([g.float().norm() for g in expert_grads(chunk_size, s)]) for s in (0.0, SMALL))
        growth[chunk_size] = both / big - 1
    assert (growth[None] > 0.5 * (CHUNKS - 1) * SMALL).all(), growth
    torch.testing.assert_close(growth[TOKENS], growth[None], rtol=0.1, atol=0)
