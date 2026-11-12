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

"""Multi-GPU Inkling SP parity, including convolution shard boundaries."""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from arctic_platform.model.implementations.inkling.attention import enable_flex_attention
from arctic_platform.model.implementations.inkling.modeling_inkling import InklingMoE
from arctic_platform.model.implementations.inkling.sequence_parallel import InklingSequenceParallelConvolution
from arctic_platform.model.implementations.inkling.sequence_parallel import apply_sequence_parallelism
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from tests.model.test_inkling_training import initialized_prime_model
from tests.model.test_inkling_training import tiny_config


def _worker(rank, size, port):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size, timeout=timedelta(seconds=300)
    )
    try:
        _convolution_parity(rank, size)
        set_seed(947)
        config = tiny_config()
        config.head_dim = config.swa_head_dim = 16
        config.num_attention_heads = config.swa_num_attention_heads = 4
        config.inkling_mlp_token_chunk_size = 128
        config.log_scaling_n_floor = 128
        config.log_scaling_alpha = 0.1
        length = 520
        ids = torch.randint(0, 60, (2, length), device="cuda")
        labels = torch.cat((ids[:, 1:], ids[:, :1]), dim=1)
        labels[:, -1] = -100
        for kv_heads, dtype, lora, checkpointed, layout in (
            (1, torch.float32, False, False, "plain"),
            (4, torch.float32, False, True, "packed"),
            (1, torch.bfloat16, False, True, "padding"),
            (1, torch.bfloat16, True, True, "packed"),
        ):
            config.num_key_value_heads = config.swa_num_key_value_heads = kv_heads
            set_seed(947)
            reference = initialized_prime_model(config).cuda().to(dtype)
            for module in reference.modules():
                if isinstance(module, InklingMoE):
                    module.ep_comm_backend = "local"
                if hasattr(module, "conv1d"):
                    module.float()
            enable_flex_attention(reference, query_chunk_size=128)
            if lora:
                reference = apply_peft(
                    reference,
                    dict(
                        peft_type="LORA", task_type="CAUSAL_LM", r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]
                    ),
                )
                # Exercise the backward through both adapter factors.
                for name, parameter in reference.named_parameters():
                    if "lora_B" in name:
                        torch.nn.init.normal_(parameter, std=0.01)
            actual = copy.deepcopy(reference)
            base = actual.base_model.model if lora else actual
            apply_sequence_parallelism(base, size, dist.group.WORLD)
            if checkpointed:
                base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            positions = torch.arange(length, device="cuda")[None].expand_as(ids).clone()
            mask = torch.ones_like(ids, dtype=torch.bool)
            targets = labels.clone()
            if layout == "packed":
                positions[:, length // size - 1 :] -= length // size - 1
                targets[:, length // size - 2] = -100
            elif layout == "padding":
                mask[0, :3] = False
                mask[1, -3:] = False
                targets[~mask] = -100
            shard = slice(rank * (length // size), (rank + 1) * (length // size))
            expected = reference(input_ids=ids, attention_mask=mask, position_ids=positions).logits
            count = (targets != -100).sum()
            expected_loss = F.cross_entropy(expected.float().flatten(0, 1), targets.flatten(), reduction="sum") / count
            expected_loss.backward()
            output = actual(
                input_ids=ids[:, shard], attention_mask=mask[:, shard], position_ids=positions[:, shard]
            ).logits
            loss = F.cross_entropy(output.float().flatten(0, 1), targets[:, shard].flatten(), reduction="sum") / count
            loss.backward()
            dist.all_reduce(loss.detach())
            atol = 1e-6 if dtype == torch.float32 else 1e-3
            torch_assert_close(output, expected[:, shard], atol=atol, rtol=0)
            torch_assert_close(loss, expected_loss, atol=atol, rtol=0)
            gradients, expected_gradients = [], []
            for (name, parameter), baseline in zip(actual.named_parameters(), reference.parameters(), strict=True):
                if parameter.requires_grad:
                    # The worker sums contributions from the token shards.
                    grad = parameter.grad.float()
                    dist.all_reduce(grad)
                    torch_assert_close(grad, baseline.grad.float(), atol=atol, rtol=0, msg=name)
                    gradients.append(grad)
                    expected_gradients.append(baseline.grad.float())
            norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
            expected_norm = torch.stack([g.square().sum() for g in expected_gradients]).sum().sqrt()
            torch_assert_close(norm, expected_norm, atol=atol, rtol=0)
            if rank == 0:
                print(
                    f"SP{size} parity passed: kv_heads={kv_heads}, dtype={dtype}, lora={lora},"
                    f" checkpoint={checkpointed}, {layout}",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


def _convolution_parity(rank, size):
    from transformers.models.inkling.modeling_inkling import InklingShortConvolution

    for local_length in (1, 2, 5):
        set_seed(947)
        reference = InklingShortConvolution(16, 4, 0, 0).cuda().float()
        actual = copy.deepcopy(reference)
        actual.__class__ = InklingSequenceParallelConvolution
        actual._sp_group = dist.group.WORLD
        hidden = torch.randn(2, local_length * size, 16, device="cuda", requires_grad=True)
        mask = torch.ones(hidden.shape[:2], device="cuda", dtype=torch.bool)
        mask[1, local_length - 1] = False
        shard = slice(rank * local_length, (rank + 1) * local_length)
        local = hidden[:, shard].detach().clone().requires_grad_(True)
        expected = reference(hidden, conv_mask=mask)
        (expected.square().sum() / hidden.numel()).backward()
        output = actual(local, conv_mask=mask[:, shard])
        (output.square().sum() / hidden.numel()).backward()
        dist.all_reduce(actual.conv1d.weight.grad)
        torch_assert_close(output, expected[:, shard], atol=1e-6, rtol=0)
        torch_assert_close(local.grad, hidden.grad[:, shard], atol=1e-6, rtol=0)
        torch_assert_close(actual.conv1d.weight.grad, reference.conv1d.weight.grad, atol=1e-6, rtol=0)


@pytest.mark.timeout(900)
class TestInklingSequenceParallel(TestCasePlus):
    @require_torch_gpu
    def test_sp2_parity(self):
        self._run(2)

    @require_torch_gpu
    def test_sp4_parity(self):
        self._run(4)

    def _run(self, size):
        if torch.cuda.device_count() < size:
            self.skipTest(f"requires {size} GPUs")
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        torch.multiprocessing.spawn(_worker, args=(size, port), nprocs=size, join=True)
