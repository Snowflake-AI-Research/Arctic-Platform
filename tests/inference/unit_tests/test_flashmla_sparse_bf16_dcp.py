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
import os
from types import SimpleNamespace

import pytest
import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.config import DeviceConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.mla import flashmla_sparse
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="vLLM's top-k slot conversion is a Triton kernel: run on a GPU or with TRITON_INTERPRET=1",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEADS, LATENT, TOPK, BLOCK, DCP = 64, 512, 128, 64, 4
SEQ_LENS = (300, 517)
# A prefill chunk of request 0 and a decode token of request 1.
QUERY_POSITIONS = ((0, range(292, 300)), (1, range(516, 517)))


def _sparse_attention(q, kv, indices, sm_scale, d_v=512, attn_sink=None, topk_length=None, out=None):
    """flash_mla_sparse_fwd's documented result, for hosts without its kernel. A row with no valid index is
    undefined: a NaN output and a finite log-sum-exp here."""
    selected = indices[:, 0].long()
    valid = (selected >= 0) & (selected < kv.shape[0])
    if topk_length is not None:
        valid &= torch.arange(selected.shape[1], device=selected.device) < topk_length[:, None]
    keys = kv[selected.clamp(min=0), 0].float()
    scores = torch.einsum("qhd,qtd->qht", q.float(), keys) * sm_scale
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    output = torch.einsum("qht,qtd->qhd", torch.exp(scores - lse[..., None]), keys[..., :d_v])
    lse[~valid.any(dim=-1)] = 0.0
    return output.to(q.dtype), scores.amax(-1), lse


def _dcp_merge(outs, lses):
    """vLLM's DCP merge (CorrectAttnCPOutKernel, then the reduce-scatter's sum): NaN and +inf log-sum-exps count as
    -inf, and each rank's output is scaled by exp(its log-sum-exp - the merged one)."""
    lses = torch.where(lses.isnan() | (lses == float("inf")), float("-inf"), lses)
    merged = torch.logsumexp(lses, dim=0)
    return sum(out.float() * torch.exp(lse - merged)[..., None] for out, lse in zip(outs, lses))


def _selections(generator):
    """Per query token, up to TOPK causal positions; the last row holds only positions on DCP rank 0."""
    rows = []
    for request, positions in QUERY_POSITIONS:
        for position in positions:
            picked = torch.randperm(position + 1, generator=generator)[:TOPK]
            rows.append((request, picked))
    request, _ = rows[-1]
    rows[-1] = (request, torch.arange(0, SEQ_LENS[request], DCP)[:TOPK])
    req_id = torch.tensor([request for request, _ in rows], dtype=torch.int32)
    topk = torch.full((len(rows), TOPK), -1, dtype=torch.int32)
    for row, (_, picked) in enumerate(rows):
        topk[row, : len(picked)] = picked.to(torch.int32)
    return req_id, topk


def _paged_cache(latents, dcp_rank, dcp_size):
    """This rank's cache and block table, slots as vLLM's DCP slot mapping places them (interleave 1)."""
    blocks_per_request = -(-max(SEQ_LENS) // BLOCK)
    cache = torch.zeros(len(SEQ_LENS) * blocks_per_request, BLOCK, LATENT, dtype=torch.bfloat16)
    block_table = torch.arange(cache.shape[0], dtype=torch.int32).view(len(SEQ_LENS), blocks_per_request)
    for request, request_latents in enumerate(latents):
        positions = torch.arange(len(request_latents))
        virtual_block, offset = positions // (BLOCK * dcp_size), positions % (BLOCK * dcp_size)
        local = virtual_block * BLOCK + offset // dcp_size
        mine = offset % dcp_size == dcp_rank
        slots = block_table[request, local[mine] // BLOCK].long() * BLOCK + local[mine] % BLOCK
        cache.view(-1, LATENT)[slots] = request_latents[mine]
    return cache.to(DEVICE), block_table.to(DEVICE)


def _impl(monkeypatch, topk_buffer, dcp_size, dcp_rank):
    """FlashMLASparseImpl as an engine with this DCP group would construct it."""
    group = SimpleNamespace(world_size=dcp_size, rank_in_group=dcp_rank)
    monkeypatch.setattr(parallel_state, "_DCP", group if dcp_size > 1 else None)
    monkeypatch.setattr(parallel_state, "_TP", group)
    config = VllmConfig(
        device_config=DeviceConfig(device=DEVICE),
        parallel_config=ParallelConfig(tensor_parallel_size=dcp_size, decode_context_parallel_size=dcp_size),
    )
    with set_current_vllm_config(config):
        return flashmla_sparse.FlashMLASparseImpl(
            HEADS // dcp_size, LATENT, LATENT**-0.5, 1, None, None, "auto", None, "decoder", None,
            topk_indices_buffer=topk_buffer, indexer=None, q_lora_rank=1536, kv_lora_rank=LATENT,
            qk_nope_head_dim=256, qk_rope_head_dim=0, qk_head_dim=256, v_head_dim=256, kv_b_proj=None,
        )


@pytest.fixture
def workspace():
    init_workspace_manager(torch.device(DEVICE))
    yield
    reset_workspace_manager()


def test_bf16_sparse_mla_dcp_ranks_merge_to_the_single_rank_output(monkeypatch, workspace):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    arctic_inference_plugin()
    if not (torch.cuda.is_available() and is_flashmla_sparse_supported()[0]):
        monkeypatch.setattr(flashmla_sparse, "flash_mla_sparse_fwd", _sparse_attention)

    generator = torch.Generator().manual_seed(0)
    latents = [torch.randn(length, LATENT, generator=generator).to(torch.bfloat16) for length in SEQ_LENS]
    req_id, topk = _selections(generator)
    q = torch.randn(len(req_id), HEADS, LATENT, generator=generator).to(torch.bfloat16).to(DEVICE)
    topk_buffer = topk.to(DEVICE)

    def forward(dcp_size, dcp_rank):
        impl = _impl(monkeypatch, topk_buffer, dcp_size, dcp_rank)
        cache, block_table = _paged_cache(latents, dcp_rank, dcp_size)
        metadata = SimpleNamespace(
            req_id_per_token=req_id.to(DEVICE), block_table=block_table, block_size=BLOCK,
            cp_kv_cache_interleave_size=1, num_decode_tokens=0, num_actual_tokens=len(req_id),
        )
        return impl.forward_mqa(q, cache, metadata, layer=None)

    expected, _ = forward(1, 0)
    outs, lses = zip(*(forward(DCP, rank) for rank in range(DCP)))
    lses = torch.stack(lses)
    merged = _dcp_merge(outs, lses)

    torch.testing.assert_close(merged, expected.float(), atol=2e-2, rtol=2e-2)
    assert torch.isinf(lses[1:, -1]).all(), "the last token's selection lives on rank 0 only"
