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

"""Debug only: record an exact per-token digest of every GLM-5.3-Flash operator's output, to find what is batch-dependent.

With ``ARCTIC_LAYER_DIGEST_DIR`` set, tensor-parallel rank 0 writes one file per model forward (up to
``ARCTIC_LAYER_DIGEST_MAX_STEPS``) holding the step's input ids and positions and, for every decoder layer's
hyper-connection ops, mixer, MLP and output in execution order, an int64 per token: the token's row of the output,
bits read as integers, weighted by column. Two runs computed a token identically when its digests are equal.
"""

from __future__ import annotations

import os

import torch

DIR_ENV = "ARCTIC_LAYER_DIGEST_DIR"
STEPS_ENV = "ARCTIC_LAYER_DIGEST_MAX_STEPS"
_LAYER_OPS = ("hc_pre", "hc_fused_post_pre", "hc_post")
_APPLIED = False
_state: dict = {"step": 0, "digests": None}


def row_digest(tensor: torch.Tensor, rows: int) -> torch.Tensor | None:
    """An int64 per row of ``tensor`` (``[rows, ...]``), equal for two rows only if their bits are (collisions aside)."""
    if not torch.is_tensor(tensor) or tensor.dim() == 0 or tensor.shape[0] != rows or tensor.numel() == 0:
        return None
    flat = tensor.detach().contiguous().view(rows, -1)
    if flat.element_size() == 2:
        bits = flat.view(torch.int16).long()
    elif flat.element_size() == 4:
        bits = flat.view(torch.int32).long()
    elif flat.element_size() == 1:
        bits = flat.view(torch.int8).long()
    else:
        bits = flat.view(torch.int64)
    weights = torch.arange(1, bits.shape[1] + 1, device=bits.device, dtype=torch.long)
    return (bits * weights).sum(-1).cpu()


def _record(name: str, output) -> None:
    digests = _state["digests"]
    if digests is None:
        return
    rows = _state["rows"]
    outputs = output if isinstance(output, (tuple, list)) else (output,)
    for i, value in enumerate(outputs):
        digest = row_digest(value, rows) if torch.is_tensor(value) else None
        if digest is not None:
            digests[f"{name}.{i}"] = digest


def _wrap(cls, method: str, name_of, inputs: bool = False):
    original = getattr(cls, method)

    def wrapped(self, *args, **kwargs):
        if inputs:
            _record(f"{name_of(self)}.in", [*args, *kwargs.values()])
        out = original(self, *args, **kwargs)
        _record(name_of(self), out)
        return out

    setattr(cls, method, wrapped)


def ensure_layer_digest() -> None:
    global _APPLIED
    directory = os.environ.get(DIR_ENV)
    if _APPLIED or not directory:
        return
    max_steps = int(os.environ.get(STEPS_ENV, "200"))
    from vllm.distributed import get_tensor_model_parallel_rank
    from vllm.models.glm5next.common import model as glm5

    original_forward = glm5.Glm5NextModel.forward

    def forward(self, input_ids, positions, *args, **kwargs):
        record = _state["step"] < max_steps and get_tensor_model_parallel_rank() == 0
        if record:
            _state["digests"], _state["rows"] = {}, int(positions.shape[-1])
        try:
            out = original_forward(self, input_ids, positions, *args, **kwargs)
        finally:
            if record:
                os.makedirs(directory, exist_ok=True)
                torch.save(
                    {
                        "input_ids": None if input_ids is None else input_ids.detach().cpu(),
                        "positions": positions.detach().cpu(),
                        "digests": _state["digests"],
                    },
                    os.path.join(directory, f"step{_state['step']:05d}.pt"),
                )
                _state["step"] += 1
                _state["digests"] = None
        return out

    glm5.Glm5NextModel.forward = forward
    layer = lambda self: f"L{self.layer_idx:02d}"  # noqa: E731
    _wrap(glm5.Glm5NextDecoderLayer, "forward", lambda self: f"{layer(self)}.out")
    for op in _LAYER_OPS:
        _wrap(glm5.Glm5NextDecoderLayer, op, lambda self, op=op: f"{layer(self)}.{op}", inputs=True)
    for cls in (glm5.Glm5NextMoE, glm5.Glm5NextMLP):
        _wrap(cls, "forward", lambda self, cls=cls: f"{cls.__name__}@{id(self):x}")
    from vllm.models.glm5next.common.attention import Glm5NextMLAAttention
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    for cls in (Glm5NextMLAAttention, Glm5NextLinearAttention):
        _wrap(cls, "forward", lambda self, cls=cls: f"{cls.__name__}@{id(self):x}")
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    _wrap(GateLinear, "forward", lambda self: f"gate@{id(self):x}")
    _wrap(FusedMoERouter, "select_experts", lambda self: f"select_experts@{id(self):x}")
    _wrap(MoERunner, "_apply_quant_method", lambda self: f"moe_apply@{id(self):x}")
    _APPLIED = True
