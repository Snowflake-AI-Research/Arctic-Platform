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

"""Token-tiled dense MLP forward with deterministic recomputation."""

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

MlpForward = Callable[..., torch.Tensor]
ComputeParams = Callable[[torch.nn.Module], List[torch.Tensor]]
RngState = Tuple[torch.Tensor, Optional[torch.Tensor]]
_PEFT_PARAM_WRAPPERS = "_tiled_mlp_peft_param_wrappers"


def _rng_state(device: torch.device) -> RngState:
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda_state


@contextlib.contextmanager
def _rng_state_replayed(state: RngState, device: torch.device):
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
    references = getattr(module, _PEFT_PARAM_WRAPPERS, ())
    with contextlib.ExitStack() as stack:
        for reference in references:
            wrapper = reference()
            if wrapper is not None and not wrapper.disable_adapters and not wrapper.merged:
                stack.enter_context(wrapper._activate_lora(wrapper.active_adapters))
        yield


def _shard_forward_replaying_rng(mlp_forward: MlpForward) -> MlpForward:
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

        original_shape = hidden_states.shape
        hidden_size = hidden_states.shape[-1]
        num_tokens = hidden_states.numel() // hidden_size
        local_shards = min(
            (num_tokens + token_chunk_size - 1) // token_chunk_size,
            num_tokens,
        )
        num_shards = local_shards
        uses_zero3 = any(hasattr(parameter, "ds_id") for parameter in self.parameters())
        if uses_zero3 and dist.is_available() and dist.is_initialized():
            shard_count = torch.tensor(
                local_shards,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            dist.all_reduce(shard_count, op=dist.ReduceOp.MAX)
            num_shards = int(shard_count.item())
        if num_shards <= 1:
            return call_mlp(self, hidden_states)

        hidden_states = hidden_states.contiguous().view(num_tokens, hidden_size)
        padded_tokens = max(
            num_shards,
            ((num_tokens + num_shards - 1) // num_shards) * num_shards,
        )
        if padded_tokens != num_tokens:
            padding = hidden_states.new_zeros((padded_tokens - num_tokens, hidden_size))
            hidden_states = torch.cat((hidden_states, padding), dim=0)
        shard_forward = _shard_forward_replaying_rng(call_mlp)
        output = TiledMLP.apply(
            shard_forward,
            self,
            hidden_states,
            num_shards,
            compute_params(self),
        )
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
) -> int:
    if isinstance(token_chunk_size, bool) or not isinstance(token_chunk_size, int) or token_chunk_size <= 0:
        raise ValueError(f"token_chunk_size must be a positive integer, got {token_chunk_size!r}")

    patched = 0
    for module in model.modules():
        if is_target(module):
            tiled_forward = make_tiled_forward(
                mlp_forward,
                compute_params,
                token_chunk_size,
            )
            module.forward = types.MethodType(tiled_forward, module)
            setattr(module, _PEFT_PARAM_WRAPPERS, [])
            patched += 1

    if patched == 0:
        raise ValueError("tiled_mlp_token_chunk_size was configured but matched zero modules")
    return patched


def register_tiled_mlp_peft_parameter_wrappers(model: torch.nn.Module) -> int:
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


def apply_dense_tiled_mlp(
    model: torch.nn.Module,
    *,
    token_chunk_size: int | None,
) -> int:
    if token_chunk_size is None:
        return 0
    routed_experts = {module for module_name, module in model.named_modules() if "experts" in module_name.split(".")}

    def is_target(module: torch.nn.Module) -> bool:
        separate_gate_up = hasattr(module, "gate_proj") and hasattr(module, "up_proj")
        fused_gate_up = hasattr(module, "gate_up_proj")
        return module not in routed_experts and hasattr(module, "down_proj") and (separate_gate_up or fused_gate_up)

    bound_forwards = {module: module.forward for module in model.modules() if is_target(module)}

    def mlp_forward(module, hidden_states, *args, **kwargs):
        return bound_forwards[module](hidden_states, *args, **kwargs)

    return apply_tiled_mlp(
        model,
        is_target=is_target,
        mlp_forward=mlp_forward,
        compute_params=trainable_parameters,
        token_chunk_size=token_chunk_size,
    )
