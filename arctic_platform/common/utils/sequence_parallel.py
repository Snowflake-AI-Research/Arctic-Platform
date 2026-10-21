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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Token-local model execution with a replicated logical-row GRPO objective.

Sequence parallelism (SP) partitions one logical replica's packed tokens in
actual process-group order. Logprobs, not vocabulary logits, are gathered before
the unchanged loss runs on the full logical replica. Native dispatch does not
select this adapter automatically; the caller must admit the selected engine's
SP gradient sum / logical data-parallel average and constructor capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn

from .server_models import resolve_parallelism_degree

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
    world = resolve_parallelism_degree(info.get("world_size"), "world_size")
    size = resolve_parallelism_degree(info.get("sp_size"), "sp_size")
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
        not torch.is_tensor(ids)
        or ids.ndim != 2
        or ids.dtype not in (torch.int32, torch.int64)
        or not torch.is_tensor(attention)
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
    if "labels" not in tensors:
        labels = torch.full_like(ids, -100)
        labels[:, :-1] = ids[:, 1:]
        next_valid = torch.zeros_like(mask)
        next_valid[:, :-1] = mask[:, 1:]
        tensors["labels"] = labels.masked_fill(~next_valid, -100)
    for name, value in tensors.items():
        if not torch.is_tensor(value) or value.ndim < 2 or value.shape[:2] != ids.shape:
            raise ValueError(f"SP token field {name!r} must have leading input_ids shape")
        if value.device != ids.device:
            raise ValueError(f"SP token field {name!r} must be on the input_ids device")
    for name in ("labels", "position_ids"):
        value = tensors[name]
        if value.shape != ids.shape or value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"SP {name} must be a 2D integer tensor")
    if bool((tensors["position_ids"][mask] < 0).any().item()):
        raise ValueError("SP valid-token positions must be nonnegative")
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


def assemble_token_tensor(values: list[torch.Tensor], partitions: list[TokenPartition]) -> torch.Tensor:
    """Concatenate exactly one ``[1, local_tokens, ...]`` result per SP member."""
    if not partitions or len(values) != len(partitions):
        raise ValueError("SP assembly needs one tensor and slice per member")
    first = partitions[0]
    expected = set(first.ranks)
    if len(partitions) != len(expected) or {part.rank for part in partitions} != expected:
        raise ValueError("SP assembly has missing or duplicate ranks")
    pairs = sorted(zip(partitions, values), key=lambda pair: pair[0].local_rank)
    exemplar = pairs[0][1]
    for index, (part, value) in enumerate(pairs):
        if (
            part.ranks != first.ranks
            or part.world_size != first.world_size
            or part.row_shape != first.row_shape
            or part.token_count != first.token_count
            or part.local_rank != index
            or part.rank != first.ranks[index]
            or not torch.is_tensor(value)
            or value.ndim < 2
            or value.shape[:2] != (1, part.width)
            or value.shape[2:] != exemplar.shape[2:]
            or value.dtype != exemplar.dtype
            or value.device != exemplar.device
        ):
            raise ValueError("SP assembly has inconsistent slices or tensor domains")
    return torch.cat([value for _, value in pairs], dim=1)


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
        from arctic_platform.rl.processors.compute_logprobs import compute_logprobs_post

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
        local_logprobs = compute_logprobs_post(outputs, local_kwargs, {}, "unused")["logprobs"]
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
) -> dict:
    """Run the selected token-mean GRPO route without changing its loss recipe.

    Consume one full logical-DP microbatch already assigned to this actual SP
    group. Only native model inputs are token local; the existing run_pipeline
    sees gathered logprobs and the complete packed logical loss context. Requires
    a selected SP engine whose dense/expert gradient reduction sums sequence
    contributions and averages logical DP replicas. This source adapter is not
    installed-runtime, native-capacity, router-application or numerical proof.
    """
    from arctic_platform.rl.processors import run_pipeline
    from arctic_platform.rl.processors.base_loss import resolve_loss

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
    _, packed, mask, cu, part = prepare_token_partition(batch, sp_info=sp_info)
    _check_group(part, sp_group)
    logical_dp = part.world_size // len(part.ranks)
    if meta.get("dp_size", logical_dp) != logical_dp:
        raise ValueError("SP meta dp_size does not match actual logical-DP membership")
    loss_object = resolve_loss("grpo") if loss_object is None else loss_object
    adapter = _TokenLocalEngine(engine, part, sp_group)
    result = run_pipeline(
        adapter,
        (),
        packed,
        {**meta, "cu_seqlens": cu, "dp_size": logical_dp},
        processing,
        device,
        backward=backward,
        pack=False,
        return_tensors=True,
        loss_object=loss_object,
    )
    _check_group(part, sp_group)
    result["batch"] = {name: restore_token_rows(value, mask) for name, value in result.get("batch", {}).items()}
    return result
