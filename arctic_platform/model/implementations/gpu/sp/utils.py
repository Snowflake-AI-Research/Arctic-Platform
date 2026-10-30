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

"""Token-local model execution with a replicated logical-row GRPO objective.

Sequence parallelism (SP) partitions one logical replica's packed tokens in
actual process-group order. Logprobs, not vocabulary logits, are gathered before
the unchanged loss runs on the full logical replica. Workers select this adapter
only for the supported GRPO route and native ZeRO2 group settings. Installed
source admission and constructor capacity remain the operational caller's work.
"""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from numbers import Real

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

server_models = importlib.import_module("arctic_platform.common.utils.server_models")

# Fields are prediction aligned except input_ids and position_ids. Unknown
# tensors cannot be silently replicated, dropped, or interpreted as token fields.
TOKEN_FIELDS = frozenset(
    {
        "input_ids",
        "position_ids",
        "labels",
        "advantages",
        "loss_mask",
        "old_log_probs_shifted",
        "prox_logp_shifted",
        "ref_log_probs_shifted",
        "teacher_log_probs_shifted",
        "versions",
        "rollout_is_weights",
        "routed_experts",
    }
)


@dataclass(frozen=True)
class TokenPartition:
    rank: int
    world_size: int
    ranks: tuple[int, ...]
    local_rank: int
    token_count: int
    row_shape: tuple[int, int]

    @property
    def width(self) -> int:
        return self.token_count // len(self.ranks)

    @property
    def start(self) -> int:
        return self.local_rank * self.width

    @property
    def stop(self) -> int:
        return self.start + self.width


def _membership(info: dict) -> tuple[int, int, tuple[int, ...], int]:
    if not isinstance(info, dict):
        raise ValueError("SP membership must be an actual worker record")
    world = server_models.resolve_parallelism_degree(info.get("world_size"), "world_size")
    size = server_models.resolve_parallelism_degree(info.get("sp_size"), "sp_size")
    rank, local = info.get("rank"), info.get("sp_rank")
    ranks = info.get("sp_ranks")
    if (
        world % size
        or type(rank) is not int
        or not 0 <= rank < world
        or type(local) is not int
        or not 0 <= local < size
        or not isinstance(ranks, list)
        or len(ranks) != size
        or any(type(peer) is not int or not 0 <= peer < world for peer in ranks)
        or len(set(ranks)) != size
        or ranks[local] != rank
    ):
        raise ValueError("inconsistent ordered SP membership")
    return rank, world, tuple(ranks), local


def prepare_token_partition(
    batch: dict, *, sp_info: dict
) -> tuple[dict, dict, torch.Tensor, torch.Tensor, TokenPartition]:
    """Return local tensors, full packed tensors, row mask, boundaries and slice.

    Input tensors have padded shape ``[rows, columns, ...]``. Right padding is
    removed without changing any valid token or supplied position. Next-token
    labels are constructed per full row *before* packing/slicing; an explicit
    ``labels`` tensor is already prediction aligned and is preserved. Active
    loss positions must have valid next-token targets. No SP alignment padding
    or token dropping is invented: nondivisible packed lengths fail pre-model.
    """
    rank, world, ranks, local = _membership(sp_info)
    if not isinstance(batch, dict) or set(batch) - TOKEN_FIELDS - {"attention_mask"}:
        raise ValueError("undeclared SP token fields")
    ids, attention = batch.get("input_ids"), batch.get("attention_mask")
    if (
        not isinstance(ids, torch.Tensor)
        or ids.ndim != 2
        or ids.dtype not in (torch.int32, torch.int64)
        or not isinstance(attention, torch.Tensor)
        or attention.shape != ids.shape
        or attention.device != ids.device
        or ids.numel() == 0
    ):
        raise ValueError("SP requires nonempty input_ids and same-device 2D attention_mask")
    if not bool(((attention == 0) | (attention == 1)).all().item()):
        raise ValueError("SP attention_mask must be binary")
    mask = attention.bool()
    if bool((mask[:, 1:] & ~mask[:, :-1]).any().item()) or not bool(mask.any(dim=1).all().item()):
        raise ValueError("SP requires nonempty right-padded rows")
    token_count = int(mask.sum().item())
    if token_count % len(ranks):
        raise ValueError("packed token count must divide evenly across SP ranks; no padding recipe admitted")
    tensors = dict(batch)
    tensors.pop("attention_mask")
    if "position_ids" not in tensors:
        tensors["position_ids"] = torch.arange(ids.shape[1], device=ids.device).expand_as(ids)
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[:2] != ids.shape:
            raise ValueError(f"SP token field {name!r} must have leading input_ids shape")
        if value.device != ids.device:
            raise ValueError(f"SP token field {name!r} must be on the input_ids device")
    for name in ("labels", "position_ids"):
        if name not in tensors:
            continue
        value = tensors[name]
        if value.shape != ids.shape or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"SP {name} must be a 2D integer tensor")
    if bool((tensors["position_ids"][mask] < 0).any().item()):
        raise ValueError("SP valid-token positions must be nonnegative")
    if "labels" not in tensors:
        labels = torch.full_like(ids, -100)
        labels[:, :-1] = ids[:, 1:]
        next_valid = torch.zeros_like(mask)
        # A position reset begins another document, even inside a padded row.
        next_valid[:, :-1] = mask[:, 1:] & (tensors["position_ids"][:, 1:] != 0)
        tensors["labels"] = labels.masked_fill(~next_valid, -100)
    loss_mask = tensors.get("loss_mask")
    if loss_mask is not None:
        if loss_mask.shape != ids.shape or not bool(torch.isfinite(loss_mask).all().item()):
            raise ValueError("SP loss_mask must be finite and prediction aligned")
        if bool(((loss_mask != 0) & (~mask | (tensors["labels"] == -100))).any().item()):
            raise ValueError("SP active loss positions need valid next-token targets")
    packed = {name: value[mask].unsqueeze(0) for name, value in tensors.items()}
    # Include supplied packed-document resets, not only original padded rows.
    positions = packed["position_ids"][0]
    starts = torch.nonzero(positions == 0, as_tuple=False).reshape(-1)
    if starts.numel() == 0 or int(starts[0].item()) != 0:
        raise ValueError("SP packed positions must begin at a document boundary")
    cu = torch.cat([starts, starts.new_tensor([token_count])]).to(torch.int32)
    partition = TokenPartition(rank, world, ranks, local, token_count, tuple(ids.shape))
    local_batch = {name: value[:, partition.start : partition.stop].contiguous() for name, value in packed.items()}
    return local_batch, packed, mask, cu, partition


def restore_token_rows(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Restore gathered ``[1, valid_tokens, ...]`` to original padded rows."""
    if value.ndim < 2 or value.shape[:2] != (1, int(mask.sum().item())):
        raise ValueError("gathered SP tensor does not cover the full logical rows")
    restored = value.new_zeros((*mask.shape, *value.shape[2:]))
    restored[mask] = value.squeeze(0)
    return restored


def _check_group(partition: TokenPartition, group) -> None:
    if (
        group is None
        or not dist.is_initialized()
        or dist.get_rank() != partition.rank
        or dist.get_world_size() != partition.world_size
        or tuple(dist.get_process_group_ranks(group)) != partition.ranks
        or dist.get_rank(group) != partition.local_rank
    ):
        raise ValueError("SP supplied slice does not match the live process group")


class _TokenLocalEngine:
    def __init__(self, engine, partition, group):
        self.engine, self.partition, self.group = engine, partition, group
        self.global_rank = partition.rank

    def eval(self):
        return self.engine.eval()

    def train(self):
        return self.engine.train()

    def __call__(self, **kwargs):
        processors = importlib.import_module("arctic_platform.rl.processors")

        part = self.partition
        _check_group(part, self.group)
        local_kwargs = {}
        for name, value in kwargs.items():
            if name in TOKEN_FIELDS:
                if not torch.is_tensor(value) or value.shape[:2] != (1, part.token_count):
                    raise ValueError(f"model token field {name!r} is not full packed SP input")
                local_kwargs[name] = value[:, part.start : part.stop].contiguous()
            elif name in {"use_cache", "temperature", "calculate_entropy"}:
                if torch.is_tensor(value) and value.numel() != 1:
                    raise ValueError(f"SP model scalar {name!r} cannot contain token data")
                local_kwargs[name] = value
            else:
                raise ValueError(f"SP model input {name!r} has no selected token-local contract")
        outputs = self.engine(**local_kwargs)
        if not isinstance(outputs, dict):
            raise ValueError("selected PrimeRL SP model must return its native output dict")
        if set(outputs) - {"logits", "logprobs", "entropy", "loss"} or outputs.get("loss") is not None:
            raise ValueError("selected SP GRPO model returned an unsupported output domain")
        local_logprobs = processors.compute_logprobs_post(outputs, local_kwargs, {}, "unused")["logprobs"]
        if local_logprobs.shape != (1, part.width):
            raise ValueError("SP native logprobs do not cover exactly the local token slice")
        _check_group(part, self.group)
        # all_gather's backward SUMs the replicated loss derivatives. Divide
        # that replication factor in backward(), not the reported loss.
        gathered = dist_nn.all_gather(local_logprobs.contiguous(), group=self.group)
        if len(gathered) != len(part.ranks) or any(value.shape != local_logprobs.shape for value in gathered):
            raise ValueError("SP gather did not return one equal token slice per member")
        return {"logprobs": torch.cat(gathered, dim=1)}

    def backward(self, loss, *, scale_wrt_gas):
        _check_group(self.partition, self.group)
        return self.engine.backward(loss / len(self.partition.ranks), scale_wrt_gas=scale_wrt_gas)


def validate_sequence_parallel_grpo(meta: dict, processing: dict, *, backward: bool) -> None:
    """Reject unsupported objectives before dispatching any token-local model."""
    config = processing.get("config") or {}
    if (
        type(backward) is not bool
        or processing.get("loss_fn") != "grpo"
        or processing.get("post") != ["compute_logprobs"]
        or config.get("loss_agg_mode", "token-mean") != "token-mean"
        or meta.get("zorro_train_enable")
        or meta.get("calculate_entropy")
        or any(name in meta for name in TOKEN_FIELDS | {"cu_seqlens", "fwd_meta_keys", "action_masks"})
    ):
        raise ValueError("SP adapter supports only the selected compute_logprobs/token-mean grpo route")


def verify_sequence_parallel_reduction(engine, *, sp_info: dict, sp_group) -> None:
    """Bind the selected ZeRO2 sequence sum / logical-DP average group settings.

    This checks live objects, not installed source equality. The owner must admit
    the selected engine, reducer and group implementations before a native run.
    """
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    rank, world, ranks, local = _membership(sp_info)
    _check_group(TokenPartition(rank, world, ranks, local, len(ranks), (1, len(ranks))), sp_group)
    optimizer = getattr(engine, "optimizer", None)
    if (
        type(optimizer) is not DeepSpeedZeroOptimizer
        or engine.zero_optimization_stage() != 2
        or engine.sequence_parallel_size != len(ranks)
        or optimizer.sequence_parallel_size != len(ranks)
        or not optimizer.partition_gradients
        or not optimizer.reduce_scatter
        or not optimizer.postscale_gradients
        or optimizer.gradient_predivide_factor != 1.0
        or optimizer.dp_process_group is not engine.seq_data_parallel_group
        or set(dist.get_process_group_ranks(optimizer.dp_process_group)) != set(range(world))
        or dist.get_world_size(optimizer.dp_process_group) != world
    ):
        raise ValueError("SP backward requires the selected native ZeRO2 sequence-sum/logical-DP-average reducer")


def collect_sequence_parallel_results(results: list[dict], *, worker_sp_info: list[dict], sp_size: int) -> list[dict]:
    """Keep one replicated logical result per actual SP group, in shard order."""
    batch_utils = importlib.import_module("arctic_platform.common.utils.batch")

    groups = batch_utils.sequence_parallel_worker_groups(worker_sp_info, len(worker_sp_info), sp_size)
    if len(results) != len(worker_sp_info) or any(
        not isinstance(result, dict) or result.get("_sequence_parallel_info") != info
        for result, info in zip(results, worker_sp_info)
    ):
        raise ValueError("SP result membership differs from its dispatched worker")
    batch_utils.sequence_parallel_worker_groups(
        [result["_sequence_parallel_info"] for result in results], len(results), sp_size
    )
    # Each worker gathered the same complete logprobs and evaluated the full
    # logical objective. Summing these replicas would multiply loss/counts by SP.
    return [
        {key: value for key, value in results[group[0]].items() if key != "_sequence_parallel_info"}
        for group in groups
    ]


def run_sequence_parallel_grpo(
    engine,
    batch: dict,
    meta: dict,
    processing: dict,
    device: str,
    *,
    sp_info: dict,
    sp_group,
    backward: bool = True,
    loss_object=None,
    loss_scale: float = 1.0,
) -> dict:
    """Run the selected token-mean GRPO route without changing its loss recipe.

    Consume one full logical-DP microbatch already assigned to this actual SP
    group. Only native model inputs are token local; the existing run_pipeline
    sees gathered logprobs and the complete packed logical loss context. Requires
    a selected SP engine whose dense/expert gradient reduction sums sequence
    contributions and averages logical DP replicas. This source adapter is not
    installed-runtime, native-capacity, router-application or numerical proof.
    """
    # Keep the optional RL runtime outside the model library's import/typecheck boundary.
    processors = importlib.import_module("arctic_platform.rl.processors")

    validate_sequence_parallel_grpo(meta, processing, backward=backward)
    if (
        isinstance(loss_scale, bool)
        or not isinstance(loss_scale, Real)
        or not math.isfinite(loss_scale)
        or loss_scale < 0
    ):
        raise ValueError("SP packed loss scale must be finite and nonnegative")
    _, packed, mask, cu, part = prepare_token_partition(batch, sp_info=sp_info)
    _check_group(part, sp_group)
    logical_dp = part.world_size // len(part.ranks)
    configured_dp = (processing.get("config") or {}).get("dp_size")
    meta_dp = meta.get("dp_size", logical_dp)
    if (
        type(meta_dp) is not int
        or meta_dp != logical_dp
        or (configured_dp is not None and (type(configured_dp) is not int or configured_dp != logical_dp))
    ):
        raise ValueError("SP meta dp_size does not match actual logical-DP membership")
    loss_object = processors.resolve_loss("grpo") if loss_object is None else loss_object
    adapter = _TokenLocalEngine(engine, part, sp_group)
    result = processors.run_pipeline(
        adapter,
        (),
        packed,
        {**meta, "cu_seqlens": cu, "dp_size": logical_dp},
        processing,
        device,
        backward="loss_only" if backward else False,
        pack=False,
        return_tensors=True,
        loss_object=loss_object,
    )
    if backward:
        processors.apply_packed_loss_reduction(adapter, result.pop("loss_tensor"), loss_scale, backward=True)
    _check_group(part, sp_group)
    result["batch"] = {name: restore_token_rows(value, mask) for name, value in result.get("batch", {}).items()}
    return result
