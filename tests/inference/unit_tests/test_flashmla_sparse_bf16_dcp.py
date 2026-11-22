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
from vllm.distributed import parallel_state
from vllm.config import DeviceConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.v1.attention.backends.mla import flashmla_sparse
from vllm.v1.attention.ops.dcp import _lse_weighted_combine
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="vLLM's top-k slot conversion is a Triton kernel: run on a GPU or with TRITON_INTERPRET=1",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HEADS, LATENT, TOPK, BLOCK, DCP, INTERLEAVE = 64, 512, 128, 64, 4, 2
SEQ_LENS = (517, 300)


def _sparse_attention(q, kv, indices, sm_scale, *, topk_length, out):
    """FlashMLA's sink-free bf16 forward. An empty row returns zero output and positive-infinite LSE."""
    selected = indices[:, 0].long()
    valid = (selected >= 0) & (selected < kv.shape[0])
    if topk_length is not None:
        valid &= torch.arange(selected.shape[1], device=selected.device) < topk_length[:, None]
    keys = kv[selected.masked_fill(~valid, 0), 0].float()
    scores = torch.einsum("qhd,qtd->qht", q.float(), keys) * sm_scale
    scores = scores.masked_fill(~valid[:, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1).masked_fill(~valid.any(dim=-1, keepdim=True), float("inf"))
    output = torch.einsum("qht,qtd->qhd", torch.exp(scores - lse[..., None]), keys[..., :LATENT])
    return output.to(q.dtype), scores.amax(-1), lse


def _selections(generator):
    """A decode selecting only rank zero, a prefill chunk spanning ranks, and an empty graph-padding row."""
    prefill_positions = range(SEQ_LENS[1] - 8, SEQ_LENS[1])
    req_id = torch.tensor([0] + [1] * len(prefill_positions) + [0], dtype=torch.int32)
    topk = torch.full((len(req_id), TOPK), -1, dtype=torch.int32)
    positions = torch.arange(SEQ_LENS[0])
    topk[0] = positions[(positions // INTERLEAVE) % DCP == 0][:TOPK]
    for row, position in enumerate(prefill_positions, start=1):
        topk[row] = torch.randperm(position + 1, generator=generator)[:TOPK]
    return req_id, topk


def _paged_cache(latents, dcp_rank, dcp_size):
    """Pack each request's owned tokens in order, with another layer's pages between physical blocks."""
    blocks_per_request = -(-max(SEQ_LENS) // BLOCK)
    cache = torch.zeros(len(latents), blocks_per_request, BLOCK, LATENT, dtype=torch.bfloat16)
    for pages, request_latents in zip(cache, latents, strict=True):
        owner = (torch.arange(len(request_latents)) // INTERLEAVE) % dcp_size
        local = request_latents[owner == dcp_rank]
        pages.view(-1, LATENT)[: len(local)] = local
    pages = cache.flatten(0, 1)
    block_table = torch.arange(len(pages), dtype=torch.int32).view(len(latents), blocks_per_request)
    return torch.stack((pages, torch.zeros_like(pages)), dim=1).to(DEVICE)[:, 0], block_table.to(DEVICE)


def _impl(monkeypatch, topk_buffer, dcp_size, dcp_rank):
    """FlashMLASparseImpl as an engine with this DCP group would construct it."""
    group = SimpleNamespace(world_size=dcp_size, rank_in_group=dcp_rank)
    monkeypatch.setattr(parallel_state, "_DCP", group if dcp_size > 1 else None)
    monkeypatch.setattr(parallel_state, "_TP", group)
    config = VllmConfig(
        device_config=DeviceConfig(device=DEVICE),
        parallel_config=ParallelConfig(
            tensor_parallel_size=dcp_size, decode_context_parallel_size=dcp_size, cp_kv_cache_interleave_size=INTERLEAVE
        ),
    )
    with set_current_vllm_config(config):
        return flashmla_sparse.FlashMLASparseImpl(
            HEADS // dcp_size,
            LATENT,
            LATENT**-0.5,
            1,
            None,
            None,
            "auto",
            None,
            "decoder",
            None,
            topk_indices_buffer=topk_buffer,
            indexer=None,
            q_lora_rank=None,
            kv_lora_rank=LATENT,
            qk_nope_head_dim=256,
            qk_rope_head_dim=0,
            qk_head_dim=256,
            v_head_dim=256,
            kv_b_proj=None,
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
            req_id_per_token=req_id.to(DEVICE),
            block_table=block_table,
            block_size=BLOCK,
            cp_kv_cache_interleave_size=INTERLEAVE,
            num_decode_tokens=1,
            num_actual_tokens=len(req_id),
        )
        return impl.forward_mqa(q, cache, metadata, layer=None)

    expected, _ = forward(1, 0)
    outs, lses = zip(*(forward(DCP, rank) for rank in range(DCP)), strict=True)
    merged = _lse_weighted_combine(torch.stack(outs), torch.stack(lses))
    torch.testing.assert_close(merged, expected.float(), atol=2e-2, rtol=2e-2)
