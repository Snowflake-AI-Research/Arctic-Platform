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
from unittest.mock import Mock

import pytest
import torch
from vllm.config import DeviceConfig, ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import parallel_state
from vllm.model_executor.layers.attention import sparse_mla_attention
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import flashmla_sparse
from vllm.v1.attention.ops.dcp import MLADCPManager, _lse_weighted_combine
from vllm.v1.attention.ops.flashmla import is_flashmla_sparse_supported
from vllm.v1.kv_cache_interface import MLAAttentionSpec
from vllm.v1.worker.utils import AttentionGroup
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from arctic_platform.inference.vllm.plugin import arctic_inference_plugin

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="vLLM's top-k slot conversion is a Triton kernel: run on a GPU or with TRITON_INTERPRET=1",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LATENT, TOPK, BLOCK, INTERLEAVE = 512, 128, 64, 2
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


def _selections(generator, dcp_size):
    """A decode selecting only rank zero, a prefill chunk spanning ranks, and an empty graph-padding row."""
    prefill_positions = range(SEQ_LENS[1] - 8, SEQ_LENS[1])
    topk = torch.full((2 + len(prefill_positions), TOPK), -1, dtype=torch.int32)
    positions = torch.arange(SEQ_LENS[0])
    owned = positions[(positions // INTERLEAVE) % dcp_size == 0][:TOPK]
    topk[0, : len(owned)] = owned
    for row, position in enumerate(prefill_positions, start=1):
        topk[row] = torch.randperm(position + 1, generator=generator)[:TOPK]
    return topk


def _paged_cache(latents, dcp_rank, dcp_size):
    """Pack each request's owned tokens in order, with another layer's pages between physical blocks."""
    blocks_per_request = -(-max(SEQ_LENS) // BLOCK)
    cache = torch.zeros(len(latents), blocks_per_request, BLOCK, latents[0].shape[-1], dtype=torch.bfloat16)
    for pages, request_latents in zip(cache, latents, strict=True):
        owner = (torch.arange(len(request_latents)) // INTERLEAVE) % dcp_size
        local = request_latents[owner == dcp_rank]
        pages.flatten(0, 1)[: len(local)] = local
    pages = cache.flatten(0, 1)
    block_table = torch.arange(len(pages), dtype=torch.int32).view(len(latents), blocks_per_request)
    return torch.stack((pages, torch.zeros_like(pages)), dim=1).to(DEVICE)[:, 0], block_table.to(DEVICE)


def _forward(monkeypatch, q, topk_buffer, latents, dcp_size, dcp_rank, cache_dtype):
    backend = flashmla_sparse.FlashMLASparseBackend
    heads, width = q.shape[1:]
    rope_dim = width - LATENT
    process_group = SimpleNamespace(world_size=dcp_size, rank_in_group=dcp_rank)
    monkeypatch.setattr(parallel_state, "_DCP", process_group if dcp_size > 1 else None)
    monkeypatch.setattr(parallel_state, "_TP", process_group)
    config = VllmConfig(
        device_config=DeviceConfig(device=DEVICE),
        parallel_config=ParallelConfig(
            tensor_parallel_size=dcp_size,
            decode_context_parallel_size=dcp_size,
            cp_kv_cache_interleave_size=INTERLEAVE,
            dcp_comm_backend="ag_rs",
            distributed_executor_backend="mp",  # The test runs the logical ranks sequentially on one device.
        ),
    )
    config.cache_config.block_size = BLOCK
    config.cache_config.cache_dtype = cache_dtype
    spec = MLAAttentionSpec(
        block_size=BLOCK, num_kv_heads=1, head_size=width, dtype=q.dtype, cache_dtype_str=cache_dtype
    )
    dims = dict(kv_lora_rank=LATENT, qk_nope_head_dim=256, qk_rope_head_dim=rope_dim, v_head_dim=256)
    config.model_config = SimpleNamespace(
        hf_text_config=SimpleNamespace(index_topk=TOPK, **dims),
        model_arch_config=SimpleNamespace(total_num_attention_heads=heads),
        dtype=torch.bfloat16,
        max_model_len=max(SEQ_LENS),
        get_num_attention_heads=lambda parallel: heads // parallel.tensor_parallel_size,
    )
    config.compilation_config.static_forward_context["layer"] = SimpleNamespace(
        prefill_backend=None,
        dcp_manager=Mock(spec=MLADCPManager),
    )
    with set_current_vllm_config(config):
        group = AttentionGroup(backend, ["layer"], spec, kv_cache_group_id=0)
        group.create_metadata_builders(config, q.device)
        (builder,) = group.metadata_builders
        impl = backend.get_impl_cls()(
            heads // dcp_size,
            width,
            width**-0.5,
            1,
            None,
            None,
            cache_dtype,
            None,
            "decoder",
            None,
            topk_indices_buffer=topk_buffer,
            indexer=None,
            q_lora_rank=None,
            qk_head_dim=256 + rope_dim,
            kv_b_proj=None,
            **dims,
        )

    cache, block_table = _paged_cache(latents, dcp_rank, dcp_size)
    starts = torch.tensor([0, 1, len(q) - 1], dtype=torch.int32)
    metadata = builder.build(
        0,
        CommonAttentionMetadata(
            query_start_loc=starts.to(DEVICE),
            query_start_loc_cpu=starts,
            seq_lens=torch.tensor(SEQ_LENS, device=DEVICE, dtype=torch.int32),
            num_reqs=len(SEQ_LENS),
            num_actual_tokens=len(q),
            max_query_len=len(q) - 2,
            max_seq_len=max(SEQ_LENS),
            block_table_tensor=block_table,
            slot_mapping=torch.full((len(q),), -1, device=DEVICE, dtype=torch.int64),
        ),
    )
    out, lse = impl.forward_mqa(q, cache, metadata, layer=None)
    if dcp_size > 1:
        assert out.is_contiguous()  # The collective consumes packed heads even when the bf16 kernel pads them.
    return out, lse


@pytest.fixture
def workspace():
    init_workspace_manager(torch.device(DEVICE))
    yield
    reset_workspace_manager()


@pytest.mark.parametrize(
    ("heads", "dcp_size", "width", "cache_dtype"),
    [(64, 8, 512, "auto"), (32, 4, 512, "bfloat16"), (64, 8, 576, "bfloat16")],
)
def test_bf16_sparse_mla_dcp_ranks_merge_to_the_single_rank_output(
    monkeypatch, workspace, heads, dcp_size, width, cache_dtype
):
    monkeypatch.setenv("ARCTIC_INFERENCE_ENABLED", "0")
    arctic_inference_plugin()
    if DEVICE == "cpu":
        monkeypatch.setattr(flashmla_sparse, "num_compute_units", lambda _: 1)
        monkeypatch.setattr(sparse_mla_attention, "PIN_MEMORY", False)
        monkeypatch.setattr(flashmla_sparse, "flash_mla_sparse_fwd", _sparse_attention)
    elif not current_platform.is_cuda() or torch.cuda.get_device_capability()[0] not in (9, 10):
        pytest.skip("FlashMLA sparse requires Hopper or Blackwell")
    else:
        supported, reason = is_flashmla_sparse_supported()
        assert supported, reason

    generator = torch.Generator().manual_seed(0)
    latents = [torch.randn(length, width, generator=generator).to(torch.bfloat16) for length in SEQ_LENS]
    topk = _selections(generator, dcp_size)
    q = torch.randn(len(topk), heads, width, generator=generator).to(torch.bfloat16).to(DEVICE)
    topk_buffer = topk.to(DEVICE)

    expected, _ = _forward(monkeypatch, q, topk_buffer, latents, 1, 0, "auto")
    outs, lses = zip(
        *(_forward(monkeypatch, q, topk_buffer, latents, dcp_size, rank, cache_dtype) for rank in range(dcp_size)),
        strict=True,
    )
    merged = _lse_weighted_combine(torch.stack(outs), torch.stack(lses))
    torch.testing.assert_close(merged, expected.float(), atol=2e-2, rtol=2e-2)
