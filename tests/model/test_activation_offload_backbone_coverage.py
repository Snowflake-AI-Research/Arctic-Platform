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

"""A caller that invokes the backbone directly must still get the activation-offload hooks.

The hooks are installed by a context manager wrapped around ``model.forward``, so they are dynamically scoped:
they reach only activations saved by code running inside that call. Any path that rebinds ``model.forward`` and
then calls the backbone itself -- the chunked LM head does, to skip the full-vocab projection -- would otherwise
save every per-block boundary activation on the device with no pack hook registered. Wrapping the backbone as
well closes that, provided both wrappers share one manager and only the outermost entry drops stale slots.
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from arctic_platform.model.implementations.gpu.activation_offload import ActivationOffloadManager
from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="activation CPU-offload requires a GPU")

# Each boundary is TOKENS x HIDDEN x 2 bytes; at 4096 x 1024 that is 8 MiB, well over the 1 MiB offload
# threshold, so every block boundary is eligible and a covered path must report several of them.
TOKENS, HIDDEN, NUM_BLOCKS = 4096, 1024, 6


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, x):
        return x + torch.relu(self.lin(x))


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(checkpoint_wrapper(_Block(), preserve_rng_state=False) for _ in range(NUM_BLOCKS))

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class _CausalLM(nn.Module):
    """A causal-LM shaped host: ``base_model`` names the backbone, as it does on Hugging Face models."""

    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.head = nn.Linear(HIDDEN, HIDDEN)

    @property
    def base_model(self):
        return self.model

    def forward(self, x):
        return self.head(self.model(x))


def _install(model: nn.Module) -> ActivationOffloadManager:
    manager = install_activation_offload(model, keep_last_n=1)
    install_activation_offload(model.base_model, keep_last_n=1, manager=manager)
    return manager


def _offloaded_in_one_step(manager: ActivationOffloadManager, call) -> int:
    manager.stats.reset()
    call().float().pow(2).sum().backward()
    return manager.stats.offloaded_tensors


def test_wrapping_installs_one_shared_manager_on_the_model_and_its_backbone():
    model = _CausalLM()
    manager = _install(model)

    assert model.model._activation_offload_manager is manager
    assert model.forward.__wrapped__ is not None
    assert model.model.forward.__wrapped__ is not None


def test_nested_wrappers_drop_stale_slots_once_per_forward():
    # reset_pending clears every live slot, so an inner wrapper that called it would discard what the outer
    # wrapper had already packed and leave backward unpacking in the wrong order.
    model = _CausalLM()
    manager = _install(model)
    calls = 0
    original = manager.reset_pending

    def counting_reset():
        nonlocal calls
        calls += 1
        original()

    manager.reset_pending = counting_reset
    model(torch.randn(2, HIDDEN))

    assert calls == 1
    assert manager._active_depth == 0


def test_sharing_a_manager_is_required_for_a_second_module():
    model = _CausalLM()
    manager = ActivationOffloadManager()
    returned = install_activation_offload(model.model, manager=manager)

    assert returned is manager
    assert model.model._activation_offload_manager is manager


def test_reinstall_rejects_a_manager_different_from_the_one_captured_by_the_wrapper():
    """The advertised manager and the one execution enters must never diverge."""
    model = _CausalLM()
    installed = ActivationOffloadManager()
    conflicting = ActivationOffloadManager()
    install_activation_offload(model, manager=installed)

    entered = []
    original_step_hooks = installed.step_hooks

    def recording_step_hooks():
        entered.append(installed)
        return original_step_hooks()

    installed.step_hooks = recording_step_hooks

    with pytest.raises(ValueError, match="different activation-offload manager"):
        install_activation_offload(model, manager=conflicting)

    assert model._activation_offload_manager is installed
    model(torch.randn(2, HIDDEN))
    assert entered == [installed]


@pytest.mark.integration
@requires_cuda
def test_a_forward_that_calls_the_backbone_directly_still_offloads():
    model = _CausalLM().cuda()
    manager = _install(model)
    x = torch.randn(TOKENS, HIDDEN, device="cuda")

    through_the_top = _offloaded_in_one_step(manager, lambda: model(x))

    # What a rebound forward does when it skips the head: call the backbone itself. Before the backbone is
    # wrapped this saves every boundary with no pack hook registered and offloads nothing.
    model.forward = types.MethodType(lambda self, inputs: self.model(inputs), model)
    bypassing_the_top = _offloaded_in_one_step(manager, lambda: model(x))

    assert through_the_top >= NUM_BLOCKS - manager.keep_last_n
    assert bypassing_the_top >= NUM_BLOCKS - manager.keep_last_n


@pytest.mark.integration
@requires_cuda
def test_offload_through_nested_wrappers_is_numerically_transparent():
    torch.manual_seed(0)
    reference = _CausalLM().cuda()
    x = torch.randn(TOKENS, HIDDEN, device="cuda")
    expected = reference(x).detach().clone()
    expected.pow(2).sum()
    reference(x).float().pow(2).sum().backward()
    expected_grads = [p.grad.detach().clone() for p in reference.parameters()]

    for p in reference.parameters():
        p.grad = None
    _install(reference)
    output = reference(x)
    output.float().pow(2).sum().backward()

    assert torch.allclose(output, expected, atol=1e-5, rtol=1e-5)
    for grad, expected_grad in zip((p.grad for p in reference.parameters()), expected_grads):
        assert torch.allclose(grad, expected_grad, atol=1e-4, rtol=1e-4)
