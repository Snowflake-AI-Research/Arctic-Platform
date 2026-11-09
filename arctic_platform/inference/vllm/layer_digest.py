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


def record_numbered(prefix: str, tensor: torch.Tensor) -> None:
    """Record ``tensor`` (one row per token of the step) as ``prefix#n``, n counting this step's calls."""
    digests = _state["digests"]
    if digests is None:
        return
    counts = _state.setdefault("counts", {})
    n = counts.get(prefix, 0)
    counts[prefix] = n + 1
    digest = row_digest(tensor, _state["rows"])
    if digest is not None:
        digests[f"{prefix}#{n}"] = digest
    dump = _state.get("dump")
    if dump is not None and _dump_selected(prefix):
        dump[f"{prefix}#{n}"] = {"args": [], "out": tensor.detach().cpu()}


def kernel_inventory(model: torch.nn.Module) -> dict[str, dict[str, str]]:
    """For every submodule that dispatches to a kernel, the class of what it dispatches to (debug)."""
    inventory: dict[str, dict[str, str]] = {}
    for name, module in model.named_modules():
        entry: dict[str, str] = {"module": type(module).__name__}
        for attr in ("quant_method", "_quant_method", "impl", "attn_backend", "kernel", "act_fn"):
            value = getattr(module, attr, None)
            if value is not None and not isinstance(value, (str, int, float, bool)):
                entry[attr] = value.__name__ if isinstance(value, type) else type(value).__name__
        quant = getattr(module, "quant_method", None) or getattr(module, "_quant_method", None)
        for attr in ("kernel", "fp8_linear", "moe_kernel", "fused_experts"):
            value = getattr(quant, attr, None)
            if value is not None:
                entry[f"quant.{attr}"] = type(value).__name__
                experts = getattr(value, "fused_experts", None)
                if experts is not None:
                    entry[f"quant.{attr}.fused_experts"] = type(experts).__name__
                    config = getattr(experts, "activation_config", None)
                    if config is not None:
                        entry["activation_config"] = repr(config)
        if len(entry) > 1:
            inventory[name] = entry
    return inventory


def _write_inventory(model: torch.nn.Module, directory: str) -> None:
    import json

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "inventory.json"), "w") as f:
        json.dump(kernel_inventory(model), f, indent=1, default=str)


DUMP_ROWS_ENV = "ARCTIC_LAYER_DUMP_ROWS"
DUMP_MATCH_ENV = "ARCTIC_LAYER_DUMP_MATCH"


def _dump_selected(name: str) -> bool:
    """Whether ``name`` contains ``ARCTIC_LAYER_DUMP_MATCH`` (every name when unset)."""
    return os.environ.get(DUMP_MATCH_ENV, "") in name


def _dump(name: str, args, out) -> None:
    """Debug: keep the full input and output tensors of a call in steps of at most ``ARCTIC_LAYER_DUMP_ROWS`` rows."""
    dump = _state.get("dump")
    if dump is None or not _dump_selected(name):
        return
    counts = _state.setdefault("dump_counts", {})
    base = f"L{_state.get('layer', -1):02d}.{name}"
    name = f"{base}#{counts.get(base, 0)}"
    counts[base] = counts.get(base, 0) + 1

    def cpu(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, (tuple, list)):
            return type(value)(cpu(v) for v in value)
        return None

    dump[name] = {"args": cpu(list(args)), "out": cpu(out)}


def _wrap(cls, method: str, name_of, inputs: bool = False, dump: bool = False):
    original = getattr(cls, method)

    def wrapped(self, *args, **kwargs):
        if inputs:
            _record(f"{name_of(self)}.in", [*args, *kwargs.values()])
        out = original(self, *args, **kwargs)
        _record(name_of(self), out)
        if dump:
            _dump(name_of(self), [*args, *kwargs.values()], out)
        return out

    setattr(cls, method, wrapped)


def _wrap_kpool_decode_update() -> None:
    """Record each decode row's tail ring before the kpool update and the pool entry it writes (``kpool.*``)."""
    from vllm.models.glm5next.nvidia.ops import kpool_compress

    original = kpool_compress.kpool_decode_update_and_maybe_write_cache_batched

    def update(kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape, slot_mapping, positions, *args,
               **kwargs):  # fmt: skip
        rows = key.shape[0] * key.shape[1]
        if _state["digests"] is None or rows != _state["rows"]:
            return original(kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape, slot_mapping,
                            positions, *args, **kwargs)  # fmt: skip
        ring = tail_kv_cache.shape[2]
        blocks = tail_slot_mapping.reshape(-1).clamp(min=0).long() // ring
        record_numbered("kpool.tail_ring", tail_kv_cache[blocks].reshape(rows, -1))
        record_numbered("kpool.key", key.reshape(rows, -1))
        out = original(kv_cache, tail_kv_cache, tail_slot_mapping, key, slot_score, ape, slot_mapping, positions,
                       *args, **kwargs)  # fmt: skip
        head_dim, page_size = key.shape[-1], kv_cache.shape[1]
        pages = kv_cache.as_strided((kv_cache.shape[0], kv_cache.stride(0)), (kv_cache.stride(0), 1))
        loc = slot_mapping.reshape(-1).long()
        safe = loc.clamp(min=0)
        cols = (safe % page_size)[:, None] * head_dim + torch.arange(head_dim, device=loc.device)
        entry = pages[(safe // page_size)[:, None], cols]
        record_numbered("kpool.pool_entry", torch.where(loc[:, None] >= 0, entry, torch.zeros_like(entry)))
        return out

    kpool_compress.kpool_decode_update_and_maybe_write_cache_batched = update


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
        if record and not _state.get("inventory"):
            _state["inventory"] = True
            _write_inventory(self, directory)
        if record:
            _state["digests"], _state["rows"], _state["counts"] = {}, int(positions.shape[-1]), {}
            dump_rows = int(os.environ.get(DUMP_ROWS_ENV, "0"))
            _state["dump"] = {} if 0 < _state["rows"] <= dump_rows else None
            _state["dump_counts"] = {}
        try:
            out = original_forward(self, input_ids, positions, *args, **kwargs)
            _record("model_out", out)
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
                if _state.get("dump"):
                    torch.save(
                        {"input_ids": None if input_ids is None else input_ids.detach().cpu(),
                         "positions": positions.detach().cpu(), "calls": _state["dump"]},
                        os.path.join(directory, f"dump{_state['step']:05d}.pt"),
                    )  # fmt: skip
                _state["dump"] = None
                _state["step"] += 1
                _state["digests"] = None
        return out

    glm5.Glm5NextModel.forward = forward
    original_logits = glm5.Glm5NextForCausalLM.compute_logits

    def compute_logits(self, hidden_states):
        logits = original_logits(self, hidden_states)
        if _state["step"] <= max_steps and get_tensor_model_parallel_rank() == 0 and torch.is_tensor(logits):
            # One row per sampled request, in the step's request order; the forward's step file is already written.
            rows = int(logits.shape[0])
            torch.save({"logits": row_digest(logits, rows), "hidden": row_digest(hidden_states, rows)},
                       os.path.join(directory, f"logits{_state['step'] - 1:05d}.pt"))  # fmt: skip
        return logits

    glm5.Glm5NextForCausalLM.compute_logits = compute_logits
    layer = lambda self: f"L{self.layer_idx:02d}"  # noqa: E731
    original_layer_forward = glm5.Glm5NextDecoderLayer.forward

    def layer_forward(self, *args, **kwargs):
        _state["layer"] = self.layer_idx
        return original_layer_forward(self, *args, **kwargs)

    glm5.Glm5NextDecoderLayer.forward = layer_forward
    _wrap(glm5.Glm5NextDecoderLayer, "forward", lambda self: f"{layer(self)}.out")
    for op in _LAYER_OPS:
        _wrap(glm5.Glm5NextDecoderLayer, op, lambda self, op=op: f"{layer(self)}.{op}", inputs=True)
    for cls in (glm5.Glm5NextMoE, glm5.Glm5NextMLP):
        _wrap(cls, "forward", lambda self, cls=cls: f"{cls.__name__}@{id(self):x}")
        _wrap(cls, "forward", lambda self, cls=cls: cls.__name__, dump=True)
    from vllm.models.glm5next.common.attention import Glm5NextMLAAttention
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention

    for cls in (Glm5NextMLAAttention, Glm5NextLinearAttention):
        _wrap(cls, "forward", lambda self, cls=cls: f"{cls.__name__}@{id(self):x}")
        _wrap(cls, "forward", lambda self, cls=cls: cls.__name__, dump=True)
    # KDA sub-steps, so a trainer-side mismatch can be located stage by stage.
    _wrap(Glm5NextLinearAttention, "_flashkda_prefill", lambda self: "kda.flashkda", dump=True)
    from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated

    _wrap(FusedRMSNormGated, "forward", lambda self: "kda.o_norm", dump=True)
    from vllm.models.glm5next.common import kda as kda_module

    original_conv = kda_module.causal_conv1d_fn

    def causal_conv1d_fn(*args, **kwargs):
        out = original_conv(*args, **kwargs)
        _dump("kda.conv", [*args, *(v for k, v in kwargs.items() if k in ("conv_states", "query_start_loc"))], out)
        return out

    kda_module.causal_conv1d_fn = causal_conv1d_fn
    from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    from vllm.models.glm5next.sparse_indexer import SparseAttnIndexerKpool
    from vllm.v1.attention.backends.mla.flashattn_mla_sparse import FlashAttnMLASparseImpl

    original_indexer = SparseAttnIndexerKpool.forward_cuda

    def indexer_forward(self, *args, **kwargs):
        out = original_indexer(self, *args, **kwargs)
        if _state["digests"] is not None and torch.is_tensor(out):
            _record(f"indexer_topk@{id(self):x}", out[: _state["rows"]])
        return out

    SparseAttnIndexerKpool.forward_cuda = indexer_forward
    _wrap_kpool_decode_update()
    _wrap(FlashAttnMLASparseImpl, "forward_mqa", lambda self: f"sparse_mla@{id(self):x}")
    _wrap(GateLinear, "forward", lambda self: f"gate@{id(self):x}")
    _wrap(FusedMoERouter, "select_experts", lambda self: f"select_experts@{id(self):x}")
    _wrap(MoERunner, "_apply_quant_method", lambda self: f"moe_apply@{id(self):x}")
    _APPLIED = True
