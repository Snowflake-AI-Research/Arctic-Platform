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

"""``patches.checkpoint_keep_attention``: the backward recompute reuses the attention kernel's outputs.

A two-layer Qwen3 on SDPA (CPU, FP32; on CUDA FlashAttention 3 when it is installed) trains to the same loss and
gradients as plain whole-layer checkpointing and as no checkpointing, while the attention kernel runs once per layer
instead of twice. The kernel calls are counted below the checkpoint's own dispatch modes, so a call the selective
policy serves from its cache is not counted.
"""

from __future__ import annotations

import copy
import types

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from arctic_platform.model import LoaderContext
from arctic_platform.model.config import Patches
from arctic_platform.model.patches.checkpoint_keep_attention import apply_checkpoint_keep_attention
from arctic_platform.model.patches.checkpoint_keep_attention import attention_kernel_ops
from arctic_platform.model.patches.gradient_checkpointing import apply_gradient_checkpointing

transformers = pytest.importorskip("transformers")

LAYERS = 2


def _ctx(gradient_checkpointing) -> LoaderContext:
    return LoaderContext(
        spec=types.SimpleNamespace(
            patches=types.SimpleNamespace(gradient_checkpointing=gradient_checkpointing),
            parallelism=types.SimpleNamespace(sequence_parallel=1),
        )
    )


def _model(device, dtype, attn_implementation):
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    config._attn_implementation = attn_implementation
    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(config).to(device=device, dtype=dtype).train()


class _CountKernels(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.kernels = attention_kernel_ops()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in self.kernels:
            self.calls += 1
        return func(*args, **(kwargs or {}))


def _train_step(model, input_ids):
    with _CountKernels() as counter:
        loss = model(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
    # The periodic path wraps layers in a CheckpointWrapper, which adds a level to their parameter names.
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): param.grad.detach().clone()
        for name, param in model.named_parameters()
    }
    return loss.detach(), grads, counter.calls


def _cases():
    yield pytest.param("cpu", torch.float32, "sdpa", id="cpu-sdpa")
    if torch.cuda.is_available():
        try:
            import flash_attn_interface  # noqa: F401
        except ImportError:
            pass
        else:
            yield pytest.param("cuda", torch.bfloat16, "flash_attention_3", id="cuda-fa3")


@pytest.mark.parametrize("frequency", [True, 2, [1]], ids=["every-layer", "every-2nd-layer", "layer-list"])
@pytest.mark.parametrize("device,dtype,attn_implementation", list(_cases()))
def test_keep_attention_matches_whole_layer_recompute(device, dtype, attn_implementation, frequency):
    base = _model(device, dtype, attn_implementation)
    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (1, 96), device=device)

    plain = copy.deepcopy(base)
    apply_gradient_checkpointing(plain, _ctx(frequency))
    kept = copy.deepcopy(base)
    apply_gradient_checkpointing(kept, _ctx(frequency))
    apply_checkpoint_keep_attention(kept, _ctx(frequency))

    reference_loss, reference_grads, reference_calls = _train_step(copy.deepcopy(base), input_ids)
    plain_loss, plain_grads, plain_calls = _train_step(plain, input_ids)
    kept_loss, kept_grads, kept_calls = _train_step(kept, input_ids)

    if isinstance(frequency, list):
        checkpointed = len(frequency)
    else:
        checkpointed = LAYERS if frequency is True else (LAYERS + frequency - 1) // frequency
    assert reference_calls == LAYERS
    assert plain_calls == LAYERS + checkpointed
    assert kept_calls == LAYERS
    # The kept outputs are the forward's own, so the backward is the plain recompute's to the bit.
    torch.testing.assert_close(kept_loss, plain_loss, rtol=0, atol=0)
    for name, grad in plain_grads.items():
        torch.testing.assert_close(kept_grads[name], grad, rtol=0, atol=0, msg=name)
    torch.testing.assert_close(kept_loss, reference_loss, rtol=0, atol=0)
    for name, grad in reference_grads.items():
        torch.testing.assert_close(kept_grads[name], grad, rtol=0, atol=0, msg=name)


def test_keep_attention_without_checkpointed_layers_is_rejected():
    model = _model("cpu", torch.float32, "sdpa")
    with pytest.raises(ValueError, match="no checkpointed layer"):
        apply_checkpoint_keep_attention(model, _ctx(True))


def test_keep_attention_requires_gradient_checkpointing():
    with pytest.raises(ValueError, match="checkpoint_keep_attention requires gradient_checkpointing"):
        Patches(checkpoint_keep_attention=True)
    assert Patches(gradient_checkpointing=True, checkpoint_keep_attention=True).checkpoint_keep_attention


def test_layer_list_checkpoints_exactly_those_layers():
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    model = _model("cpu", torch.float32, "sdpa")
    apply_gradient_checkpointing(model, _ctx([1]))
    wrapped = [isinstance(layer, CheckpointWrapper) for layer in model.model.layers]
    assert wrapped == [False, True]
    with pytest.raises(ValueError, match=r"lists layers \[5\], but the model has 2"):
        apply_gradient_checkpointing(_model("cpu", torch.float32, "sdpa"), _ctx([0, 5]))
    with pytest.raises(ValueError, match="lists a layer twice"):
        Patches(gradient_checkpointing=[1, 1])
