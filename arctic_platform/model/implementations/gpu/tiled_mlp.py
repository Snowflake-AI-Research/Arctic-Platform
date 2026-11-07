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
"""Tiled MLP (ALST): shard a dense FFN's forward over the token dimension to cut its activation high-water.

A dense FFN (e.g. ``w2(silu(w1(x)) * w3(x))``) materializes two ``[n_tokens, intermediate]`` tensors at
once, so its activation memory grows with the number of tokens. DeepSpeed's ``TiledMLP`` computes the FFN
shard-by-shard over the token dim (recomputing in backward) for a ``~1/shards`` intermediate, and defers ZeRO
grad reduction to the last shard via ``ds_grad_is_ready`` (ZeRO stage 1/2) so gradients stay correct.

Autograd adds each shard's weight gradient into ``param.grad`` in the weight's dtype, so a BF16 weight would
round once per shard (a BF16 running sum stops moving once the sum outgrows the increments). Outside ZeRO-3 the
shards therefore go through ``_TiledMLPFp32Grads``, which sums the shards' weight gradients in FP32 and hands
autograd one gradient per weight, rounded once. ``compute_params`` has to list every trainable weight the FFN reads.
ZeRO-3 keeps DeepSpeed's ``TiledMLP``: its per-shard gather and reduction schedule is not reproduced here.

Model-agnostic: the caller supplies ``is_target`` (which submodules to tile), ``mlp_forward`` (un-tiled FFN
on a shard) and ``compute_params`` (weights in the reduction). Shard count is derived per forward as
``ceil(n_tokens / token_chunk_size)``. Callers omit the wrap when tiling is unset; a non-positive chunk
size or a wrap that matches no modules is rejected.

Each shard's recompute runs under the generator state its forward used, so an FFN that draws is differentiated
against the sample it actually produced. RNG replay is unconditional because PEFT can add dropout after tiling
is installed; inspecting the module during installation would silently miss that production order.
"""

from __future__ import annotations

import contextlib
import types
import weakref
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.distributed as dist
from deepspeed.runtime.sequence_parallel.ulysses_sp import TiledMLP

# ``mlp_forward(module, hidden_states) -> output``: the un-tiled FFN compute, run on each token shard.
MlpForward = Callable[..., torch.Tensor]
# ``compute_params(module) -> [weights]``: params whose ZeRO grad reduction TiledMLP defers to the last shard.
ComputeParams = Callable[[torch.nn.Module], List[torch.Tensor]]
# A shard's generator state: the CPU state, and the state of its accelerator device when it has one.
RngState = Tuple[torch.Tensor, Optional[torch.Tensor]]
_PEFT_PARAM_WRAPPERS = "_tiled_mlp_peft_param_wrappers"


def _rng_state(device: torch.device) -> RngState:
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda_state


@contextlib.contextmanager
def _rng_state_replayed(state: RngState, device: torch.device):
    """Run the body under ``state``, then put the generators back where the caller had them.

    Restoring afterwards keeps the replay local to one shard's recompute: backward is not the only consumer of
    these generators, and rewinding them for the rest of the step would change every draw that follows.
    """
    saved_cpu, saved_cuda = _rng_state(device)
    replay_cpu, replay_cuda = state
    torch.set_rng_state(replay_cpu)
    if replay_cuda is not None:
        torch.cuda.set_rng_state(replay_cuda, device)
    try:
        yield
    finally:
        torch.set_rng_state(saved_cpu)
        if saved_cuda is not None:
            torch.cuda.set_rng_state(saved_cuda, device)


@contextlib.contextmanager
def _peft_parameter_wrappers_reentered(module: torch.nn.Module):
    """Restore PEFT parameter LoRA while a tiled shard is recomputed.

    ``ParamWrapper.forward`` temporarily parameterizes its base module, but that context has exited by the time
    ``TiledMLP.backward`` recomputes a shard. Wrapper references are registered after PEFT installation so the
    recompute can restore every nested target-parameter adapter without recursively invoking tiled forward.
    """
    references = getattr(module, _PEFT_PARAM_WRAPPERS, ())
    with contextlib.ExitStack() as stack:
        for reference in references:
            wrapper = reference()
            if wrapper is not None and not wrapper.disable_adapters and not wrapper.merged:
                stack.enter_context(wrapper._activate_lora(wrapper.active_adapters))
        yield


def _shard_forward_replaying_rng(mlp_forward: MlpForward) -> MlpForward:
    """Wrap ``mlp_forward`` so a shard's recompute draws the numbers its own forward drew.

    ``TiledMLP`` computes each shard in the forward and computes it a second time in the backward, to rebuild
    the intermediates it chose not to keep, and it carries no RNG state between the two. An FFN that draws is
    then differentiated against a sample that never produced its output -- dropout on the projections, which
    is what a LoRA adapter installs, and which draws under ``no_grad`` like anywhere else. The gradient that
    comes back belongs to a network that never ran.

    The two passes are told apart by grad mode, which is how ``TiledMLP`` runs them: the forward shards under
    ``no_grad`` (recorded here, in shard order) and the recompute shards under ``enable_grad`` (replayed in
    the same order). A recompute call with no recorded state behind it -- an unexpected extra backward --
    falls through to an ordinary call rather than failing, since that case is nondeterministic either way.

    One wrapper is built per tiled forward call, so its record covers exactly that call's shards. Activation
    checkpointing, which reruns the whole tiled forward, therefore records afresh and stays consistent.
    """
    recorded: List[RngState] = []
    replays = 0

    def shard_forward(module: torch.nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        nonlocal replays
        device = hidden_states.device
        if not torch.is_grad_enabled():
            recorded.append(_rng_state(device))
            return mlp_forward(module, hidden_states)
        index = replays
        replays += 1
        with _peft_parameter_wrappers_reentered(module):
            if index >= len(recorded):
                return mlp_forward(module, hidden_states)
            with _rng_state_replayed(recorded[index], device):
                return mlp_forward(module, hidden_states)

    return shard_forward


def _unlisted_trainable_leaves(output: torch.Tensor, listed: set[int]) -> list[torch.Tensor]:
    """Trainable leaves the graph of ``output`` reaches that are not in ``listed`` (by ``id``)."""
    found: list[torch.Tensor] = []
    seen = set()
    stack = [output.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        variable = getattr(node, "variable", None)
        if variable is not None and id(variable) not in listed:
            found.append(variable)
        stack.extend(next_node for next_node, _ in node.next_functions)
    return found


class _TiledMLPFp32Grads(torch.autograd.Function):
    """``deepspeed``'s ``TiledMLP`` with the weight gradients summed over shards in FP32.

    ``params`` are the weights the shards read. They are inputs of the function so that autograd delivers
    each one's gradient, rounded to the weight dtype once, after the last shard, which also makes the ZeRO
    post-accumulate hook fire once per weight without ``ds_grad_is_ready`` bookkeeping.
    """

    @staticmethod
    def forward(ctx, fn, module, x, shards, *params):
        ctx.fn = fn
        ctx.module = module
        ctx.shards = shards
        ctx.params = params
        ctx.save_for_backward(x)
        with torch.no_grad():
            outputs = [fn(module, shard) for shard in torch.chunk(x, chunks=shards, dim=-2)]
        return torch.cat(outputs, dim=-2)

    @staticmethod
    def backward(ctx, *grads):
        (x,) = ctx.saved_tensors
        x_requires_grad = ctx.needs_input_grad[2]
        trainable = [index for index, param in enumerate(ctx.params) if ctx.needs_input_grad[4 + index]]
        params = [ctx.params[index] for index in trainable]

        x_shape = x.shape
        x = x.detach().reshape(-1, x_shape[-1])
        incoming = grads[0].reshape(-1, x_shape[-1])
        x_grad = torch.zeros_like(x) if x_requires_grad else None
        sums: list[torch.Tensor | None] = [None] * len(params)
        offset = 0
        for index, shard in enumerate(torch.chunk(x, chunks=ctx.shards, dim=0)):
            rows = shard.shape[0]
            shard.requires_grad_(x_requires_grad)
            with torch.enable_grad():
                output = ctx.fn(ctx.module, shard)
                inputs = ([shard] if x_requires_grad else []) + params
                if index == 0:
                    stray = _unlisted_trainable_leaves(output, {id(t) for t in inputs})
                    if stray:
                        raise RuntimeError(
                            f"tiled MLP reads {len(stray)} trainable tensor(s) that compute_params does not list, "
                            f"whose gradients would be dropped (shapes {[tuple(t.shape) for t in stray[:4]]})"
                        )
                computed = (
                    torch.autograd.grad(output, inputs, incoming.narrow(0, offset, rows), allow_unused=True)
                    if inputs
                    else ()
                )
            if x_requires_grad:
                x_grad.narrow(0, offset, rows).copy_(computed[0])
                computed = computed[1:]
            for slot, (param, grad) in enumerate(zip(params, computed)):
                if grad is None:
                    continue
                if sums[slot] is None:
                    sums[slot] = grad.to(torch.promote_types(param.dtype, torch.float32))
                else:
                    sums[slot].add_(grad)
            offset += rows

        param_grads: list[torch.Tensor | None] = [None] * len(ctx.params)
        for slot, index in enumerate(trainable):
            if sums[slot] is not None:
                param_grads[index] = sums[slot].to(ctx.params[index].dtype)
        x_grad = x_grad.view(x_shape) if x_requires_grad else None
        return (None, None, x_grad, None, *param_grads)


def make_tiled_forward(
    mlp_forward: MlpForward,
    compute_params: ComputeParams,
    token_chunk_size: int,
):
    def forward(
        self: torch.nn.Module,
        hidden_states: torch.Tensor,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        def call_mlp(module, shard):
            return mlp_forward(module, shard, *args, **kwargs)

        # DeepSpeed flattens every leading dimension before it shards in backward. Flatten here too, so forward
        # and backward partition the same token stream for both [T,H] and the documented [B,S,H] input.
        original_shape = hidden_states.shape
        hidden_size = hidden_states.shape[-1]
        num_tokens = hidden_states.numel() // hidden_size
        local_shards = min((num_tokens + token_chunk_size - 1) // token_chunk_size, num_tokens)
        num_shards = local_shards
        uses_zero3 = any(hasattr(parameter, "ds_id") for parameter in self.parameters())
        if uses_zero3 and dist.is_available() and dist.is_initialized():
            # Each tile can trigger a ZeRO-3 parameter all-gather. Every rank therefore executes the maximum
            # tile count needed by any peer, matching DeepSpeed's own TiledMLP example and collective order.
            shard_count = torch.tensor(local_shards, dtype=torch.int64, device=hidden_states.device)
            dist.all_reduce(shard_count, op=dist.ReduceOp.MAX)
            num_shards = int(shard_count.item())
        if num_shards <= 1:
            return call_mlp(self, hidden_states)
        hidden_states = hidden_states.contiguous().view(num_tokens, hidden_size)
        # torch.chunk may return fewer chunks than requested when the dimension is not divisible by the
        # requested count (for example, five rows split four ways yields three chunks). Pad to an exact
        # multiple so every rank executes precisely the synchronized number of ZeRO collective schedules.
        padded_tokens = max(num_shards, ((num_tokens + num_shards - 1) // num_shards) * num_shards)
        if padded_tokens != num_tokens:
            padding = hidden_states.new_zeros((padded_tokens - num_tokens, hidden_size))
            hidden_states = torch.cat((hidden_states, padding), dim=0)
        shard_forward = _shard_forward_replaying_rng(call_mlp)
        if uses_zero3:
            output = TiledMLP.apply(shard_forward, self, hidden_states, num_shards, compute_params(self))
        else:
            weights = [weight for weight in compute_params(self) if weight.requires_grad]
            output = _TiledMLPFp32Grads.apply(shard_forward, self, hidden_states, num_shards, *weights)
        output = output[:num_tokens]
        return output.view(*original_shape[:-1], output.shape[-1])

    return forward


def apply_tiled_mlp(
    model: torch.nn.Module,
    *,
    is_target: Callable[[torch.nn.Module], bool],
    mlp_forward: MlpForward,
    compute_params: ComputeParams,
    token_chunk_size: int,
    logger=None,
) -> int:
    """Wrap every matching submodule with a token-tiled forward.

    ``token_chunk_size`` is the target tokens-per-shard. It must be a positive integer. Returns the number of
    modules patched. A wrap that matches nothing is rejected so a configured setting cannot silently do no
    work. Safe before or after activation-checkpoint wrapping.
    """
    if isinstance(token_chunk_size, bool) or not isinstance(token_chunk_size, int) or token_chunk_size <= 0:
        raise ValueError(f"token_chunk_size must be a positive integer, got {token_chunk_size!r}")

    patched = 0
    for module in model.modules():
        if is_target(module):
            # PEFT is installed after dense tiling, so a dropout-free module here may become stochastic before
            # its first forward. Replay RNG unconditionally; fullgraph plus tiling is rejected by the schema.
            tiled_forward = make_tiled_forward(mlp_forward, compute_params, token_chunk_size)
            module.forward = types.MethodType(tiled_forward, module)
            setattr(module, _PEFT_PARAM_WRAPPERS, [])
            patched += 1

    if patched == 0:
        raise ValueError("tiled_mlp_token_chunk_size was configured but matched zero modules")
    if logger is not None:
        logger.info(
            "Applied Tiled MLP to %d module(s) (token_chunk_size=%d, shards derived per step as ceil(tokens/chunk))",
            patched,
            token_chunk_size,
        )
    return patched


def register_tiled_mlp_peft_parameter_wrappers(model: torch.nn.Module) -> int:
    """Associate PEFT target-parameter wrappers with tiled base modules after adapter installation."""
    try:
        from peft.tuners.lora.layer import ParamWrapper
    except ImportError:
        return 0

    registered = 0
    for wrapper in model.modules():
        if not isinstance(wrapper, ParamWrapper):
            continue
        base_layer = wrapper.get_base_layer()
        references = getattr(base_layer, _PEFT_PARAM_WRAPPERS, None)
        if references is None:
            continue
        references.append(weakref.ref(wrapper))
        registered += 1
    return registered


def trainable_parameters(module: torch.nn.Module) -> List[torch.Tensor]:
    """Return parameters whose ZeRO reduction must wait until the final tile.

    Resolve the list at forward time because PEFT can add trainable adapters after tiling is installed. PEFT
    target-parameter adapters wrap the tiled module from above, so include their parameters through the wrapper
    references registered after adapter installation.
    """
    parameters = list(module.parameters())
    for reference in getattr(module, _PEFT_PARAM_WRAPPERS, ()):
        wrapper = reference()
        if wrapper is not None:
            parameters.extend(wrapper.parameters())

    unique = []
    seen = set()
    for parameter in parameters:
        if parameter.requires_grad and id(parameter) not in seen:
            unique.append(parameter)
            seen.add(id(parameter))
    return unique


def enable_tiled_mlp(
    model: torch.nn.Module,
    *,
    is_target: Callable[[torch.nn.Module], bool],
    mlp_forward: MlpForward,
    token_chunk_size: int | None,
    logger=None,
) -> int:
    """Tile matching modules when a chunk size is set; otherwise leave the model unchanged."""
    if token_chunk_size is None:
        return 0
    return apply_tiled_mlp(
        model,
        is_target=is_target,
        mlp_forward=mlp_forward,
        compute_params=trainable_parameters,
        token_chunk_size=token_chunk_size,
        logger=logger,
    )


def apply_dense_tiled_mlp(
    model: torch.nn.Module,
    *,
    token_chunk_size: int | None,
    logger=None,
) -> int:
    """Tile dense SwiGLU MLPs while leaving rank-dependent routed experts untouched."""
    if token_chunk_size is None:
        return 0
    routed_experts = {module for module_name, module in model.named_modules() if "experts" in module_name.split(".")}

    def is_target(module: torch.nn.Module) -> bool:
        separate_gate_up = hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        fused_gate_up = hasattr(module, "gate_up_proj")
        return module not in routed_experts and hasattr(module, "down_proj") and (separate_gate_up or fused_gate_up)

    # Liger binds a replacement forward to each MLP instance. Capture those callables before replacing the
    # instance attributes so every tile keeps using the selected provider implementation.
    bound_forwards = {module: module.forward for module in model.modules() if is_target(module)}

    def mlp_forward(module, hidden_states, *args, **kwargs):
        return bound_forwards[module](hidden_states, *args, **kwargs)

    return apply_tiled_mlp(
        model,
        is_target=is_target,
        mlp_forward=mlp_forward,
        compute_params=trainable_parameters,
        token_chunk_size=token_chunk_size,
        logger=logger,
    )
