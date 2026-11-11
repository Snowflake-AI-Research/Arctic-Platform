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

"""Inkling2 sequence parallelism matches the unsharded model."""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from arctic_platform.model.implementations.inkling2.deepspeed_integration import apply_sequence_parallelism
from arctic_platform.model.implementations.inkling2.modeling_inkling2 import Inkling2Batch
from arctic_platform.model.implementations.inkling2.modeling_inkling2 import short_conv
from arctic_platform.model.patches.peft import apply_peft
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from tests.model.test_inkling2_training import LORA
from tests.model.test_inkling2_training import tiny_model


def _worker(rank, size, port):
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=size, timeout=timedelta(seconds=300)
    )
    try:
        _convolution_parity(rank, size)
        _model_parity(rank, size)
    finally:
        dist.destroy_process_group()


def _convolution_parity(rank, size):
    for local_length in (1, 2, 5):
        set_seed(947)
        weights = [torch.randn(16, device="cuda", requires_grad=True) for _ in range(4)]
        hidden = torch.randn(2, local_length * size, 16, device="cuda", requires_grad=True)
        valid = torch.ones(2, local_length * size, 1, device="cuda")
        valid[1, local_length] = 0
        masks = [valid] * 3
        shard = slice(rank * local_length, (rank + 1) * local_length)
        local = hidden[:, shard].detach().clone().requires_grad_(True)
        expected = short_conv(hidden, weights, Inkling2Batch(None, masks, None, None, 16), 0.5)
        (expected.square().sum() / hidden.numel()).backward()
        expected_grads = [w.grad.clone() for w in weights]
        for w in weights:
            w.grad = None
        batch = Inkling2Batch(None, [mask[:, shard] for mask in masks], None, None, 16, dist.group.WORLD)
        output = short_conv(local, weights, batch, 0.5)
        (output.square().sum() / hidden.numel()).backward()
        torch_assert_close(output, expected[:, shard], atol=1e-6, rtol=0)
        torch_assert_close(local.grad, hidden.grad[:, shard], atol=1e-6, rtol=0)
        for w, baseline in zip(weights, expected_grads, strict=True):
            dist.all_reduce(w.grad)
            torch_assert_close(w.grad, baseline, atol=1e-6, rtol=0)


def _model_parity(rank, size):
    from arctic_platform.model import ActivationCheckpointConfig
    from arctic_platform.model.implementations.qwen35.model_builder import apply_ac

    length = 96
    local = length // size
    shard = slice(rank * local, (rank + 1) * local)
    for dtype, lora, checkpointed, layout in (
        (torch.float32, False, False, "plain"),
        (torch.float32, False, True, "packed"),
        (torch.float32, False, False, "padding"),
        (torch.bfloat16, True, True, "packed"),
    ):
        set_seed(947)
        # The default tiny initialization is chaotic in BF16: rounding flips routes and saturates logits.
        init = {} if dtype == torch.float32 else dict(initializer_range=0.02)
        reference = tiny_model(swa_num_attention_heads=4, **init).cuda().to(dtype)
        for name, parameter in reference.named_parameters():
            if reference.keeps_fp32_parameter(name):
                parameter.data = parameter.data.float()
        if lora:
            reference = apply_peft(reference, LORA)
            for name, parameter in reference.named_parameters():
                if "lora_B" in name:
                    torch.nn.init.normal_(parameter, std=0.01)
        actual = copy.deepcopy(reference)
        base = actual.base_model.model if lora else actual
        apply_sequence_parallelism(base, size, dist.group.WORLD)
        if checkpointed:
            apply_ac(base, ActivationCheckpointConfig(router_replay_recompute=False))
        ids = torch.randint(0, 61, (2, length), device="cuda")
        targets = torch.cat((ids[:, 1:], torch.full_like(ids[:, :1], -100)), dim=1)
        inputs = dict(input_ids=ids)
        if layout == "packed":
            # The second sequence starts one token before a shard boundary, inside the next rank's halo.
            positions = torch.arange(length, device="cuda")[None].expand_as(ids).clone()
            positions[:, local - 1 :] -= local - 1
            targets[:, local - 2] = -100
            inputs["position_ids"] = positions
        elif layout == "padding":
            mask = torch.ones_like(ids, dtype=torch.bool)
            mask[0, :3] = False
            mask[1, -5:] = False
            targets[~mask] = -100
            positions = (mask.cumsum(-1) - 1).clamp(min=0)
            inputs.update(attention_mask=mask, position_ids=positions)
        count = (targets != -100).sum()
        expected = reference(**inputs).logits
        expected_loss = F.cross_entropy(expected.float().flatten(0, 1), targets.flatten(), reduction="sum") / count
        expected_loss.backward()
        local_inputs = {name: value[:, shard] for name, value in inputs.items()}
        output = actual(**local_inputs).logits
        loss = actual(**local_inputs, labels=ids[:, shard], shift_labels=targets[:, shard], num_items_in_batch=count)
        loss = loss.loss
        loss.backward()
        loss = loss.detach().clone()
        dist.all_reduce(loss)
        atol = 2e-5 if dtype == torch.float32 else 4e-3
        failures = []

        def check(value, baseline, name, rtol=0):
            try:
                torch_assert_close(value, baseline, atol=atol, rtol=rtol, msg=name)
            except AssertionError:
                diff = (value.float() - baseline.float()).abs().max().item()
                failures.append(f"rank {rank} {layout} {dtype}: {name} max abs diff {diff:.3g}")

        check(output, expected[:, shard], "logits")
        check(loss, expected_loss, "loss")
        for (name, parameter), baseline in zip(actual.named_parameters(), reference.parameters(), strict=True):
            if parameter.requires_grad:
                grad = parameter.grad.float()
                dist.all_reduce(grad)
                check(grad, baseline.grad.float(), name, rtol=1e-3)
        # Fail on every rank together so no rank waits in a collective.
        failed = torch.tensor([len(failures)], device="cuda")
        dist.all_reduce(failed)
        if failed.item():
            raise AssertionError("\n".join(failures) or f"rank {rank}: another rank failed")
        try:
            actual(**local_inputs, labels=ids[:, shard])
        except ValueError as error:
            assert "shift_labels" in str(error)
        else:
            raise AssertionError("SP accepted unshifted labels")
        if rank == 0:
            print(f"SP{size} parity passed: dtype={dtype}, lora={lora}, checkpoint={checkpointed}, {layout}")


@pytest.mark.timeout(900)
class TestInkling2SequenceParallel(TestCasePlus):
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
