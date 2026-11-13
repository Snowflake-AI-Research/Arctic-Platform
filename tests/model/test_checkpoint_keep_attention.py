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
from arctic_platform.model.patches.checkpoint_keep_attention import keep_attention_checkpoint_fn
from arctic_platform.model.patches.checkpoint_keep_attention import register_attention_kernel_op
from arctic_platform.model.patches.gradient_checkpointing import apply_gradient_checkpointing

transformers = pytest.importorskip("transformers")

LAYERS = 2


def _ctx(gradient_checkpointing, keep=True) -> LoaderContext:
    return LoaderContext(
        spec=types.SimpleNamespace(
            patches=types.SimpleNamespace(
                gradient_checkpointing=gradient_checkpointing, checkpoint_keep_attention=keep
            ),
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


_CUSTOM_KERNEL_CALLS = []


@torch.library.custom_op("arctic_test_keep_attention::scaled_scores", mutates_args=())
def _scaled_scores(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    _CUSTOM_KERNEL_CALLS.append(1)
    return torch.softmax(q @ k.transpose(-1, -2), dim=-1)


def _scaled_scores_backward(ctx, grad):
    q, k, p = ctx.saved_tensors
    ds = p * (grad - (grad * p).sum(-1, keepdim=True))
    return ds @ k, ds.transpose(-1, -2) @ q


def _scaled_scores_setup(ctx, inputs, output):
    ctx.save_for_backward(*inputs, output)


_scaled_scores.register_autograd(_scaled_scores_backward, setup_context=_scaled_scores_setup)


def test_a_registered_kernel_op_is_kept():
    # A model's own attention op, registered by its module: the recompute reuses its output instead of rerunning it.
    torch.manual_seed(2)
    q = torch.randn(4, 8, requires_grad=True)
    k = torch.randn(4, 8, requires_grad=True)

    def layer(q, k):
        return torch.ops.arctic_test_keep_attention.scaled_scores(torch.tanh(q), k).sin().sum()

    def run():
        _CUSTOM_KERNEL_CALLS.clear()
        out = keep_attention_checkpoint_fn()(layer, q, k)
        grads = torch.autograd.grad(out, (q, k))
        return len(_CUSTOM_KERNEL_CALLS), grads

    unregistered_calls, unregistered_grads = run()
    register_attention_kernel_op(torch.ops.arctic_test_keep_attention.scaled_scores)
    assert torch.ops.arctic_test_keep_attention.scaled_scores.default in attention_kernel_ops()
    registered_calls, registered_grads = run()
    assert unregistered_calls == 2 and registered_calls == 1
    for a, b in zip(registered_grads, unregistered_grads):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    with pytest.raises(TypeError, match="OpOverload"):
        register_attention_kernel_op(layer)


def test_a_named_selection_keeps_only_those_kernels():
    # checkpoint_keep_attention as a list keeps the named kernels and recomputes every other one.
    base = _model("cpu", torch.float32, "sdpa")
    torch.manual_seed(1)
    input_ids = torch.randint(0, 128, (1, 96))
    plain = copy.deepcopy(base)
    apply_gradient_checkpointing(plain, _ctx(True))
    _, plain_grads, plain_calls = _train_step(plain, input_ids)

    named = copy.deepcopy(base)
    apply_gradient_checkpointing(named, _ctx(True))
    apply_checkpoint_keep_attention(named, _ctx(True, ["aten::_scaled_dot_product_flash_attention_for_cpu"]))
    _, named_grads, named_calls = _train_step(named, input_ids)

    other = copy.deepcopy(base)
    apply_gradient_checkpointing(other, _ctx(True))
    apply_checkpoint_keep_attention(other, _ctx(True, ["arctic_test_keep_attention::scaled_scores.default"]))
    _, other_grads, other_calls = _train_step(other, input_ids)

    assert plain_calls == 2 * LAYERS and named_calls == LAYERS and other_calls == 2 * LAYERS
    for name, grad in plain_grads.items():
        torch.testing.assert_close(named_grads[name], grad, rtol=0, atol=0, msg=name)
        torch.testing.assert_close(other_grads[name], grad, rtol=0, atol=0, msg=name)


def test_a_named_selection_must_resolve():
    model = _model("cpu", torch.float32, "sdpa")
    apply_gradient_checkpointing(model, _ctx(True))
    for bad in (["flash_attn_3::no_such_kernel"], ["not-an-op-name"], ["aten::mm.no_such_overload"]):
        with pytest.raises(ValueError, match="checkpoint_keep_attention"):
            apply_checkpoint_keep_attention(model, _ctx(True, bad))
    with pytest.raises(ValueError, match="lists no op"):
        Patches(gradient_checkpointing=True, checkpoint_keep_attention=[])
    assert Patches(gradient_checkpointing=True, checkpoint_keep_attention=["aten::mm"]).checkpoint_keep_attention == [
        "aten::mm"
    ]
