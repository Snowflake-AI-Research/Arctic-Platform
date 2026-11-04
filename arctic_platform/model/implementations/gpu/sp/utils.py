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

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from numbers import Real
from typing import Callable

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
        return (self.token_count + len(self.ranks) - 1) // len(self.ranks)

    @property
    def start(self) -> int:
        return self.local_rank * self.width

    @property
    def stop(self) -> int:
        return self.start + self.width


def _local_token_slice(name: str, value: torch.Tensor, partition: TokenPartition) -> torch.Tensor:
    local = value[:, partition.start : partition.stop]
    padding = partition.width - local.shape[1]
    if padding:
        tail = value.new_full((1, padding, *value.shape[2:]), -100 if name == "labels" else 0)
        if name == "position_ids":
            start = max(0, partition.start - partition.token_count)
            tail[0] = torch.arange(start, start + padding, device=value.device)
        local = torch.cat((local, tail), dim=1)
    return local.contiguous()


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
    loss positions must have valid next-token targets. Equal local slices add
    at most SP-1 ignored tokens in a separate document. Full packed loss tensors
    and boundaries exclude that alignment tail, preserving every denominator.
    """
    rank, world, ranks, local = _membership(sp_info)
    if not isinstance(batch, dict):
        raise ValueError("SP batch must be a dictionary")
    unsupported = set(batch) - TOKEN_FIELDS - {"attention_mask"}
    if unsupported:
        raise ValueError(
            f"undeclared SP token fields: {sorted(unsupported)}; "
            "optional GRPO masks, KL weights and prompt-group/count fields are not supported by this SP route"
        )
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
    if bool((tensors["position_ids"][:, 0] != 0).any().item()):
        raise ValueError("SP every row must begin at a document boundary (position_ids=0)")
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
    local_batch = {name: _local_token_slice(name, value, partition) for name, value in packed.items()}
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
                local_kwargs[name] = _local_token_slice(name, value, part)
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
        return {"logprobs": torch.cat(gathered, dim=1)[:, : part.token_count]}

    def backward(self, loss, *, scale_wrt_gas):
        _check_group(self.partition, self.group)
        return self.engine.backward(loss / len(self.partition.ranks), scale_wrt_gas=scale_wrt_gas)


def validate_sequence_parallel_grpo(meta: dict, processing: dict, *, backward: bool) -> None:
    """Reject unsupported objectives before dispatching any token-local model."""
    config = processing.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("SP GRPO config must be a dictionary")
    # The logical rows are already replicated after the logprob gather. Sequence-level
    # loss collectives would reduce that replication a second time in autograd.
    allowed = {
        "eps_clip",
        "eps_clip_higher",
        "c_clip",
        "behav_imp_weight_cap",
        "current_version",
        "prox_logp_method",
        "use_sapo_loss",
        "sapo_tau_pos",
        "sapo_tau_neg",
        "use_decoupled_loss",
        "use_cispo_loss",
        "is_weight_clip_max",
        "teacher_tau",
        "teacher_clip",
        "teacher_clip_negative",
        "loss_agg_mode",
        "dp_size",
        "batch_num_tokens",
        "global_batch_size",
        "entropy_coeff",
        "use_kl_loss",
        "kl_loss_coef",
        "kl_loss_type",
        "sampler_kl_coef",
        "sampler_kl_type",
        "sampler_kl_ratio_cap",
        "importance_sampling_level",
    }
    unsupported = set(config) - allowed
    if unsupported or config.get("importance_sampling_level", "token") != "token":
        raise ValueError(f"SP GRPO supports token-level loss config only; unsupported keys: {sorted(unsupported)}")
    for value in (config.get("dp_size"), meta.get("dp_size")):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError("SP dp_size must be a positive integer")
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


def _request_signature(microbatches, meta, processing, loss_scales, device):
    """Compare accidental input drift with bounded device-side byte moments, not a host tensor copy."""
    checksum = hashlib.sha256()
    checksum.update(
        json.dumps(
            dict(meta=meta, processing=processing, loss_scales=loss_scales), sort_keys=True, allow_nan=False
        ).encode()
    )
    moments = torch.zeros(3, dtype=torch.int64, device=device)
    offset = 0
    for index, microbatch in enumerate(microbatches):
        for name in sorted(TOKEN_FIELDS & microbatch.keys() | {"attention_mask"}):
            tensor = microbatch[name].detach().contiguous()
            checksum.update(f"{index}:{name}:{tensor.dtype}:{tuple(tensor.shape)}:".encode())
            payload = tensor.reshape(-1).view(torch.uint8)
            for start in range(0, payload.numel(), 1 << 20):
                values = payload[start : start + (1 << 20)].to(dtype=torch.int64)
                positions = torch.arange(
                    offset + 1, offset + 1 + values.numel(), dtype=torch.int64, device=tensor.device
                )
                moments += torch.stack((values.sum(), (values * positions).sum(), (values * positions.square()).sum()))
                offset += values.numel()
    metadata = torch.tensor(list(checksum.digest()), dtype=torch.uint8, device=device)
    return torch.cat((metadata, moments.view(torch.uint8)))


def validate_sequence_parallel_microbatches(
    microbatches: list[dict],
    *,
    sp_info: dict | Callable[[], dict],
    synchronize: bool = False,
    device=None,
    meta: dict | None = None,
    processing: dict | None = None,
    backward: bool = True,
    ep_group=None,
    loss_scales=None,
    request_check: Callable[[], None] | None = None,
    sp_group=None,
) -> dict:
    """Validate the entire request before any model call or gradient-boundary mutation."""
    failure = None
    index = None
    signature = None
    try:
        membership = sp_info() if callable(sp_info) else sp_info
        _, world, ranks, _ = _membership(membership)
        if ep_group is not None:
            raise ValueError("SP GRPO currently requires ep_size=1; EP>1 gradient parity is not established")
        if request_check is not None:
            request_check()
        if meta is not None and processing is not None:
            validate_sequence_parallel_grpo(meta, processing, backward=backward)
            logical_dp = world // len(ranks)
            configured_dp = (processing.get("config") or {}).get("dp_size")
            if meta.get("dp_size", logical_dp) != logical_dp or (
                configured_dp is not None and configured_dp != logical_dp
            ):
                raise ValueError("SP meta dp_size does not match actual logical-DP membership")
        loss_scales = loss_scales() if callable(loss_scales) else loss_scales
        if loss_scales is not None and len(loss_scales) != len(microbatches):
            raise ValueError("SP packed loss scale count must match the microbatch count")
        if loss_scales is not None and any(
            isinstance(scale, bool) or not isinstance(scale, Real) or not math.isfinite(scale) or scale < 0
            for scale in loss_scales
        ):
            raise ValueError("SP packed loss scale must be finite and nonnegative")
        for index, microbatch in enumerate(microbatches):
            prepare_token_partition(microbatch, sp_info=membership)
        index = None
        if sp_group is not None:
            signature = _request_signature(microbatches, meta, processing, loss_scales, device)
    except Exception as error:
        context = "request" if index is None else f"microbatch {index}"
        failure_type = ValueError if isinstance(error, ValueError) else RuntimeError
        failure = failure_type(f"SP {context}: {error}")
    if synchronize:
        flag = torch.tensor(int(failure is not None), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item() and failure is None:
            failure = ValueError("SP request preflight failed on another logical-DP shard")
    if failure is not None:
        raise failure
    if synchronize and signature is not None:
        signatures = [torch.empty_like(signature) for _ in ranks]
        dist.all_gather(signatures, signature, group=sp_group)
        mismatch = any(not torch.equal(signature, peer) for peer in signatures)
        flag = torch.tensor(int(mismatch), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise ValueError("SP members must receive identical token rows and loss inputs")
    return membership


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
        or not optimizer.gradient_average
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

    Equal model slices include an ignored alignment tail; loss inputs and returned rows exclude it.
    Preflight failures are collective, but errors inside native
    model/backward collectives (including CUDA OOM) require the launcher to abort the whole process group. Peers
    can otherwise wait until their process-group timeout, and earlier microbatch gradients must not be reused.
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
