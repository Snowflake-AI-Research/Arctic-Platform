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

"""Micro-batch loop shared by on-prem and Cortex.

Managed DeepSpeed engines run exactly ``gas`` calls and use their micro-step
counter as the accumulation boundary. Unmanaged engines may run a variable
``loop_count`` and accumulate until the caller's final ``step()``. This loop
never calls ``set_gradient_accumulation_boundary``: that override is sticky
for the life of the engine.
"""

from __future__ import annotations

import numbers
from typing import Any, Callable

import torch

from arctic_platform.common.utils.batch import merge_dict_shards
from arctic_platform.common.utils.batch import split_dict

_CYCLE_BROKEN = "_arctic_unpacked_cycle_broken"


def _managed_gradient_accumulation(engine: Any) -> bool:
    """Whether DeepSpeed owns accumulation boundaries.

    DeepSpeed 0.19.5 and newer expose the authoritative accessor. Lightweight
    test engines predate that contract: a GAS accessor identifies a managed
    fake, while a fake without one behaves as an unmanaged executor engine.
    """
    accessor = getattr(engine, "managed_gradient_accumulation", None)
    if accessor is not None:
        return bool(accessor())
    return callable(getattr(engine, "gradient_accumulation_steps", None))


def clear_gradient_accumulation_boundary_override(engine: Any) -> None:
    """Return DeepSpeed to its micro-step counter.

    ``set_gradient_accumulation_boundary`` stores the override forever.
    ``step()`` does not clear it. Call this after a path that set the
    override, before another unpacked step relies on the counter.
    """
    if getattr(engine, "_is_gradient_accumulation_boundary", None) is not None:
        engine._is_gradient_accumulation_boundary = None


def mark_unpacked_cycle_broken(engine: Any) -> None:
    """A between-chunk ``step()`` already ran and the request did not finish."""
    setattr(engine, _CYCLE_BROKEN, True)


def assert_unpacked_cycle_intact(engine: Any) -> None:
    if getattr(engine, _CYCLE_BROKEN, False):
        raise RuntimeError(
            "unpacked gradient-accumulation cycle was interrupted; "
            "this engine cannot accept another forward or step"
        )


def assert_ready_for_final_step(engine: Any) -> None:
    """The caller's ``step()`` is the last micro-step of the cycle."""
    assert_unpacked_cycle_intact(engine)
    if not _managed_gradient_accumulation(engine):
        # In unmanaged mode is_gradient_accumulation_boundary() becomes true
        # only while step() is executing. The executor's invariant is simply
        # that no partial request marked this engine broken.
        return
    if not engine.is_gradient_accumulation_boundary():
        micro_steps = getattr(engine, "micro_steps", None)
        raise RuntimeError(
            "step() is not at a gradient-accumulation boundary "
            f"(micro_steps={micro_steps})"
        )


def _batch_fits_gas(batch_data: Any, gas: int) -> bool:
    if isinstance(batch_data, list):
        return len(batch_data) == gas
    if not isinstance(batch_data, dict):
        return False
    for value in batch_data.values():
        if torch.is_tensor(value) and value.ndim >= 1 and int(value.shape[0]) < gas:
            return False
        if isinstance(value, list) and len(value) < gas:
            return False
    return True


def _every_rank_fits(engine: Any, ok: bool) -> bool:
    """True only when every rank can split. One short rank must not let the others step."""
    import torch.distributed as dist

    if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
        return ok
    device = getattr(engine, "device", None)
    if device is None:
        device = torch.device("cpu")
    flag = torch.tensor([1 if ok else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return int(flag.item()) == 1


def _require_aligned_cycle(engine: Any, loop_count: int, *, backward: bool) -> None:
    if loop_count < 1:
        raise ValueError(f"loop_count must be positive, got {loop_count}")
    assert_unpacked_cycle_intact(engine)
    if not backward:
        return
    if getattr(engine, "_is_gradient_accumulation_boundary", None) is not None:
        raise RuntimeError(
            "unpacked gradient accumulation requires the DeepSpeed boundary override to be unset"
        )
    if not _managed_gradient_accumulation(engine):
        return
    gas = int(engine.gradient_accumulation_steps())
    if loop_count != gas:
        raise RuntimeError(
            "managed gradient accumulation requires loop_count to equal "
            f"engine.gradient_accumulation_steps(); got loop_count={loop_count} gas={gas}"
        )
    micro_steps = int(getattr(engine, "micro_steps", 0))
    if micro_steps % gas != 0:
        raise RuntimeError(
            "unpacked forward is not at an accumulation-cycle boundary: "
            f"micro_steps={micro_steps} gas={gas}"
        )


def _combine_microbatch_outputs(outputs: list[dict], loss_reduction: Any) -> dict:
    if not outputs:
        return {}
    combined: dict[str, Any] = {}
    for key, value in outputs[0].items():
        if key == "metrics" and isinstance(value, dict):
            if loss_reduction is None:
                from arctic_platform.common.utils.batch import combine_metric_microbatches

                combined[key] = combine_metric_microbatches([item[key] for item in outputs])
            else:
                from arctic_platform.rl.processors import combine_packed_metrics

                combined[key] = combine_packed_metrics(
                    [item[key] for item in outputs],
                    loss_reduction.reporting_weights,
                )
        elif isinstance(value, dict):
            combined[key] = merge_dict_shards([item[key] for item in outputs])
        elif isinstance(value, numbers.Number):
            values = [item[key] for item in outputs]
            if key == "avg_loss" and loss_reduction is not None:
                from arctic_platform.rl.processors import combine_packed_losses

                combined[key] = combine_packed_losses(values, loss_reduction)
            else:
                combined[key] = sum(values) / len(values)
    return combined


def run_unpacked_gas(
    engine: Any,
    *,
    gas: int | None = None,
    loop_count: int | None = None,
    batch_data: Any,
    meta_data: dict,
    processing: dict,
    args: tuple = (),
    loss_object: Any = None,
    backward: bool = True,
    device: Any,
    return_tensors: bool = False,
    forward_engine: Any = None,
    step_fn: Callable[[], None] | None = None,
    prepare_call: Callable[[dict, dict], dict] | None = None,
    before_backward: Callable[[int, dict], None] | None = None,
    loss_reduction: Any = None,
    resolve_reduction: bool = True,
) -> dict:
    """Run unpacked ``run_pipeline`` calls on one rank.

    ``batch_data`` is one ``[rows, S]`` dict, or a list of ``loop_count``
    micro-batches. ``gas`` remains the compatibility spelling for callers
    using managed accumulation. Managed mode steps between calls and leaves
    the final step to the caller. Unmanaged mode never steps here; every
    ``step()`` is an optimizer update, so the caller alone owns it.
    """
    if loop_count is None:
        if gas is None:
            raise ValueError("one of loop_count or gas is required")
        loop_count = gas
    elif gas is not None and gas != loop_count:
        raise ValueError(f"gas={gas} and loop_count={loop_count} disagree")

    _require_aligned_cycle(engine, loop_count, backward=backward)
    fits = _batch_fits_gas(batch_data, loop_count)
    if backward and not _every_rank_fits(engine, fits):
        raise RuntimeError(
            f"unpacked loop requires at least {loop_count} rows on every rank"
        )
    if not fits:
        raise RuntimeError(
            f"unpacked loop requires at least {loop_count} rows on every rank"
        )

    if isinstance(batch_data, list):
        micro_batches = batch_data
    else:
        micro_batches = split_dict(batch_data, loop_count)

    from arctic_platform.rl.processors import apply_packed_loss_reduction
    from arctic_platform.rl.processors import resolve_loss
    from arctic_platform.rl.processors import resolve_packed_loss_reduction
    from arctic_platform.rl.processors import run_pipeline

    if loss_object is None and processing.get("loss_fn"):
        # One instance for the reduction and every call: class losses may
        # carry state from packed_reduction_callback into later callbacks.
        loss_object = resolve_loss(processing["loss_fn"])

    if resolve_reduction:
        # Callers that already resolved, including a forward-only pass, pass
        # the object in and set resolve_reduction False.
        loss_reduction = None
        if backward or processing.get("loss_fn") is not None:
            loss_reduction = resolve_packed_loss_reduction(
                processing,
                [{**meta_data, **micro_batch} for micro_batch in micro_batches],
                require_declared=False,
                loss_object=loss_object,
            )

    model_engine = engine if forward_engine is None else forward_engine

    def take_step() -> None:
        if step_fn is not None:
            step_fn()
            return
        engine.step()

    if backward and hasattr(engine, "train"):
        engine.train()

    outputs: list[dict] = []
    stepped = False
    try:
        for index, micro_batch in enumerate(micro_batches):
            if prepare_call is not None:
                micro_batch = prepare_call(micro_batch, meta_data)
            pipeline_backward: bool | str
            if backward and loss_reduction is not None:
                pipeline_backward = "loss_only"
            else:
                pipeline_backward = backward
            result = run_pipeline(
                engine=model_engine,
                args=args,
                batch=micro_batch,
                meta=dict(meta_data),
                processing=processing,
                device=device,
                backward=pipeline_backward,
                return_tensors=return_tensors,
                loss_object=loss_object,
            )
            if before_backward is not None and backward:
                before_backward(index, result)
            if backward and loss_reduction is not None:
                apply_packed_loss_reduction(
                    engine,
                    result.pop("loss_tensor"),
                    loss_reduction.loss_scales[index],
                    backward=True,
                )
            outputs.append(result)
            if (
                backward
                and _managed_gradient_accumulation(engine)
                and index < len(micro_batches) - 1
            ):
                take_step()
                stepped = True
    except Exception:
        if backward and (stepped or not _managed_gradient_accumulation(engine)):
            mark_unpacked_cycle_broken(engine)
        raise

    return _combine_microbatch_outputs(outputs, loss_reduction)


def _scheduled_source_payloads(calls: Any) -> list[Any]:
    source_payloads: dict[int, Any] = {}
    non_padding_sources: set[int] = set()
    for call in calls:
        source_payloads.setdefault(call.source_index, call.payload)
        if not call.is_padding:
            non_padding_sources.add(call.source_index)
    expected = list(range(len(source_payloads)))
    actual = sorted(source_payloads)
    if actual != expected:
        raise ValueError(
            "scheduled payload source indexes must be contiguous from zero; "
            f"got {actual}"
        )
    if sorted(non_padding_sources) != expected:
        raise ValueError(
            "every scheduled payload source requires one non-padding call; "
            f"got {sorted(non_padding_sources)}"
        )
    return [source_payloads[index] for index in expected]


def run_payload_schedule(
    engine: Any,
    calls: Any,
    *,
    backward: bool,
    device: Any,
    return_tensors: bool = False,
    step_fn: Callable[[], None] | None = None,
    before_backward: Callable[[int, dict], None] | None = None,
) -> tuple[list[dict], Any]:
    """Run scheduled AP payloads, including zero-scaled padding calls."""
    calls = list(calls)
    if not calls:
        raise ValueError("payload schedule must contain at least one call")
    _require_aligned_cycle(engine, len(calls), backward=backward)

    source_payloads = _scheduled_source_payloads(calls)
    first = source_payloads[0]
    loss_object = first.loss_object

    from arctic_platform.rl.processors import apply_packed_loss_reduction
    from arctic_platform.rl.processors import resolve_loss
    from arctic_platform.rl.processors import resolve_packed_loss_reduction
    from arctic_platform.rl.processors import run_pipeline

    if loss_object is None and first.processing.get("loss_fn"):
        loss_object = resolve_loss(first.processing["loss_fn"])
    loss_reduction = resolve_packed_loss_reduction(
        first.processing,
        [
            {**payload.meta_data, **payload.batch_data}
            for payload in source_payloads
        ],
        loss_object=loss_object,
    )

    if backward and hasattr(engine, "train"):
        engine.train()

    def take_step() -> None:
        if step_fn is not None:
            step_fn()
        else:
            engine.step()

    outputs: list[dict] = []
    stepped = False
    validated_sources: set[int] = set()
    try:
        for index, call in enumerate(calls):
            payload = call.payload
            pipeline_backward: bool | str = (
                "loss_only" if backward and loss_reduction is not None else backward
            )
            result = run_pipeline(
                engine=engine,
                args=payload.args,
                batch=payload.batch_data,
                meta=dict(payload.meta_data),
                processing=payload.processing,
                device=device,
                backward=pipeline_backward,
                return_tensors=return_tensors,
                validate_loss_callback=call.source_index not in validated_sources,
                loss_object=loss_object,
            )
            validated_sources.add(call.source_index)
            if before_backward is not None and backward:
                before_backward(index, result)
            if backward and loss_reduction is not None:
                apply_packed_loss_reduction(
                    engine,
                    result.pop("loss_tensor"),
                    0.0
                    if call.is_padding
                    else loss_reduction.loss_scales[call.source_index],
                    backward=True,
                )
            outputs.append(result)
            if (
                backward
                and _managed_gradient_accumulation(engine)
                and index < len(calls) - 1
            ):
                take_step()
                stepped = True
    except Exception:
        if backward and (stepped or not _managed_gradient_accumulation(engine)):
            mark_unpacked_cycle_broken(engine)
        raise
    return outputs, loss_reduction
