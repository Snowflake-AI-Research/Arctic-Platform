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

"""NCCL expert-parallel logits, loss, and backward parity."""

from __future__ import annotations

from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from arctic_platform.model import ActivationCheckpointConfig
from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload
from arctic_platform.model.implementations.moe.distributed import ncclep as nccl_ep
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


def batch(rank, dtype, layout, device):
    tokens = 0 if layout == "empty" and rank == 1 else 5 + rank
    values = torch.arange(tokens * 32, device=device).reshape(tokens, 32)
    hidden = ((values.float() % 13 - 6) * 0.1).to(dtype).requires_grad_()
    ids = torch.arange(tokens, device=device)
    indices = torch.stack(((ids + rank) % 4, (ids + rank + 1) % 4), dim=-1)
    if layout == "hot":
        indices = torch.tensor([0, 1], device=device).expand(tokens, -1).clone()
    scores = torch.tensor([0.4, 0.6], device=device).expand(tokens, -1).clone()
    if layout == "masked" and tokens:
        scores[0, 0] = 0
    return hidden, indices, scores.requires_grad_()


def reference(rows, weights, dtype, before):
    outputs = []
    for hidden, indices, scores in rows:
        owner_outputs = []
        for owner in range(2):
            expert_outputs = []
            for slot in range(2):
                ids = indices[:, slot]
                selected = (ids // 2 == owner) & (scores[:, slot] != 0)
                weight = weights[ids]
                inputs = hidden
                if before:
                    inputs = (inputs.float() * scores[:, slot, None]).to(dtype)
                out = torch.bmm(weight, inputs.unsqueeze(-1)).squeeze(-1)
                if not before:
                    out = (out.float() * scores[:, slot, None]).to(dtype)
                expert_outputs.append(out.masked_fill(~selected[:, None], 0).float())
            owner_outputs.append(sum(expert_outputs).to(dtype).float())
        outputs.append(sum(owner_outputs).to(dtype))
    return outputs


class Routed(torch.nn.Module):
    def __init__(self, weight, before):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.before = before

    def forward(self, hidden, indices, scores):
        pending = nccl_ep.dispatch_tokens_async(
            hidden, indices, scores, 4, dist.group.WORLD, score_before_experts=self.before
        )
        received, counts, state = nccl_ep.finalize_dispatch_tokens(pending)
        parts = []
        offset = 0
        for expert, count in enumerate(counts.tolist()):
            parts.append(F.linear(received[offset : offset + count], self.weight[expert]))
            offset += count
        routed = torch.cat(parts)
        if routed.shape[0] == 0:
            routed = routed.detach()
        return nccl_ep.combine_tokens(routed, state)


class Checkpointed(torch.nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, hidden, indices, scores):
        return checkpoint(self.inner, hidden, indices, scores, use_reentrant=False, early_stop=False)


def worker(rank, port):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for layout in ("normal", "hot", "empty", "masked"):
                for before in (False, True):
                    for checkpointed in (False, True):
                        set_seed(947)
                        full_weights = (torch.randn(4, 32, 32, device="cuda") * 0.1).to(dtype).requires_grad_()
                        rows = [batch(r, dtype, layout, "cuda") for r in range(2)]
                        expected = reference(rows, full_weights, dtype, before)
                        denominator = sum(len(x[0]) for x in rows) * 32
                        expected_loss = sum(x.float().square().sum() for x in expected) / denominator
                        expected_loss.backward()
                        hidden, indices, scores = batch(rank, dtype, layout, "cuda")
                        local_weights = full_weights.detach()[rank * 2 : rank * 2 + 2].clone().requires_grad_()
                        routed_model = Routed(local_weights, before)
                        model = routed_model
                        if checkpointed:
                            model = Checkpointed(model)
                            install_activation_offload(
                                model,
                                config=ActivationCheckpointConfig(
                                    offload_config=dict(
                                        enabled=True, keep_last_n=0, tensor_size_threshold=0, use_streams=False
                                    )
                                ).offload_config,
                            )
                        output = model(hidden, indices, scores)
                        loss = output.float().square().sum() / denominator
                        loss.backward()
                        reduced_loss = loss.detach().clone()
                        dist.all_reduce(reduced_loss)
                        atol = 1e-6 if dtype == torch.float32 else 1e-3
                        torch_assert_close(output, expected[rank], rtol=0, atol=atol)
                        torch_assert_close(reduced_loss, expected_loss, rtol=0, atol=atol)
                        torch_assert_close(hidden.grad, rows[rank][0].grad, rtol=0, atol=atol)
                        torch_assert_close(scores.grad, rows[rank][2].grad, rtol=0, atol=atol)
                        actual_grad = (
                            routed_model.weight.grad
                            if routed_model.weight.grad is not None
                            else torch.zeros_like(local_weights)
                        )
                        torch_assert_close(actual_grad, full_weights.grad[rank * 2 : rank * 2 + 2], rtol=0, atol=atol)
                        norm_sq = actual_grad.float().square().sum()
                        dist.all_reduce(norm_sq)
                        torch_assert_close(norm_sq.sqrt(), full_weights.grad.float().norm(), rtol=0, atol=atol)
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(300)
class TestNcclExpertParallel(TestCasePlus):
    @require_torch_gpu
    def test_dispatch_combine_and_gradients(self):
        if torch.cuda.device_count() < 2:
            self.skipTest("requires two GPUs")
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        torch.multiprocessing.spawn(worker, args=(port,), nprocs=2, join=True)
