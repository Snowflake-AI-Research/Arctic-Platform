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

"""Composable processing pipeline for RL training forward passes.

Processors are registered by name into two phase-specific registries
and looked up at runtime from the ``processing`` dict embedded in each batch.

Phases
------
- **Post-forward** ``(model_outputs, meta, device) -> dict``
  Return a dict of *new* keys to add to model_outputs (e.g. logprobs from logits).
  Do NOT return the full model_outputs — only what the processor computed.
  This keeps the wire response compact (no raw logits sent over HTTP).
- **Loss function** ``(model_outputs, meta, device) -> (loss, metrics)``
  Compute a scalar loss and an arbitrary metrics dict.

Batch layout
------------
When the processing pipeline is active the batch dict should contain::

    {
        "args": (),
        "batch": { ... },      # passed to model.forward()
        "meta": { ... },     # extra data for post-processors / loss
        "processing": {
            "post": ["name1", "name2"],
            "loss_fn": "name",
            "config": { ... },
        },
    }

To add a new processor, decorate a function with ``@register_post_processor``
or ``@register_loss_fn``.  The name you pass becomes the string that clients
use in the ``processing`` dict.  Alternatively, pass a full dotted-path string
(e.g. ``"mypackage.module.my_loss_fn"``) -- the registry will import it on
first use.
"""

from __future__ import annotations

import importlib
from typing import Any
from typing import Callable
from typing import Dict

import torch

from arctic_platform.rl.utils.batch import detensorize
from arctic_platform.rl.utils.batch import log_dp_shard_tokens
from arctic_platform.rl.utils.debug import ProfilerContext
from arctic_platform.rl.utils.debug import pr0
from arctic_platform.rl.utils.debug import see_memory_usage

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


from .microbatch import DEFAULT_MAX_TOKENS_PER_MB

# PROFILER_TYPE = "c"
# PROFILER_TYPE = "torch"
PROFILER_TYPE = "none"


ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.rl.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

POST_PROCESSORS: Dict[str, Callable] = {}
LOSS_FNS: Dict[str, Callable] = {}


def register_post_processor(name: str):
    """Register a post-forward processor under *name*."""

    def decorator(fn: Callable) -> Callable:
        POST_PROCESSORS[name] = fn
        return fn

    return decorator


def register_loss_fn(name: str):
    """Register a loss function under *name*."""

    def decorator(fn: Callable) -> Callable:
        LOSS_FNS[name] = fn
        return fn

    return decorator


def _resolve_fn(registry: dict, name: str) -> Callable:
    """Look up *name* in registry; fall back to dotted-path import."""
    if name in registry:
        return registry[name]
    # Dotted-path import: "package.module.fn_name"
    module_path, fn_name = name.rsplit(".", 1)
    fn = getattr(importlib.import_module(module_path), fn_name)
    registry[name] = fn  # cache for next call
    return fn


def padded_tensor_2d_to_unpadded_tensor_1d(tensor_2d, attention_mask_2d_bool):
    pr0(f"{tensor_2d.shape=}")
    pr0(f"{attention_mask_2d_bool.shape=}")
    return tensor_2d[attention_mask_2d_bool].unsqueeze(0)


def padded_tensor_2d_dict_to_unpadded_tensor_1d_dict(tensor_dict, attention_mask_2d_bool):
    for key, value in tensor_dict.items():
        if torch.is_tensor(value) and value.shape == attention_mask_2d_bool.shape:
            new_value = padded_tensor_2d_to_unpadded_tensor_1d(value, attention_mask_2d_bool)
            pr0(f"2d->1d {key=} {value.shape=} -> {new_value.shape=} {value.sum()=} -> {new_value.sum()=}")
            # pr0(f"2d->1d: {value=}")
            # pr0(f"2d->1d: {new_value=}")

            tensor_dict[key] = new_value

        # else:
        #     pr0(f"2d->1d {key=} skipped")

    return tensor_dict


def unpadded_tensor_1d_to_padded_tensor_2d(tensor_1d, attention_mask_2d_bool, pad_value):

    if tensor_1d.shape != attention_mask_2d_bool.shape:
        ValueError(f"{tensor_1d.shape=} != {attention_mask_2d_bool.shape}")

    tensor_2d = torch.full(
        attention_mask_2d_bool.shape,
        fill_value=pad_value,
        dtype=tensor_1d.dtype,
        device=tensor_1d.device,
    )
    tensor_2d[attention_mask_2d_bool] = tensor_1d.view(-1)
    return tensor_2d


def unpadded_tensor_1d_dict_to_padded_tensor_2d_dict(tensor_dict, attention_mask_2d_bool, pad_value):
    for key, value in tensor_dict.items():
        if torch.is_tensor(value):
            pr0(f"1d->2d {key=} {value.shape=}")
            tensor_dict[key] = unpadded_tensor_1d_to_padded_tensor_2d(value, attention_mask_2d_bool, pad_value)
    return tensor_dict


def padded_tensor_2d_full_to_unpadded_tensor_1d_response(tensor_2d, attention_mask_2d_bool, max_prompt_len):
    pr0(f"{tensor_2d.shape=}")
    pr0(f"{attention_mask_2d_bool.shape=}")

    tensor_2d_response = tensor_2d[:, max_prompt_len:]
    attention_mask_2d_bool_response = attention_mask_2d_bool[:, max_prompt_len:]

    pr0(f"{tensor_2d_response.shape=}")
    pr0(f"{attention_mask_2d_bool_response.shape=}")

    return tensor_2d_response[attention_mask_2d_bool_response].unsqueeze(0)


def unpadded_tensor_1d_response_to_padded_tensor_2d_full(tensor_1d, attention_mask_2d_bool, max_prompt_len):

    # pad_value should be 0 for the return post-process tensors since the padding is just a shape placeholder
    pad_value = 0

    if tensor_1d.shape != attention_mask_2d_bool.shape:
        ValueError(f"{tensor_1d.shape=} != {attention_mask_2d_bool.shape}")

    pr0(f"{tensor_1d.shape=}")
    pr0(f"{tensor_1d.view(-1).shape=}")

    tensor_2d = torch.full(
        attention_mask_2d_bool.shape,
        fill_value=pad_value,
        dtype=tensor_1d.dtype,
        device=tensor_1d.device,
    )

    tensor_2d_response = tensor_2d[:, max_prompt_len:]
    attention_mask_2d_bool_response = attention_mask_2d_bool[:, max_prompt_len:]

    pr0(f"{tensor_2d_response.shape=}")
    pr0(f"{attention_mask_2d_bool_response.shape=}")

    tensor_2d_response[attention_mask_2d_bool_response] = tensor_1d.view(-1)

    return tensor_2d


def unpadded_tensor_1d_response_dict_to_padded_tensor_2d_full_dict(
    tensor_dict, attention_mask_2d_bool, max_prompt_len
):
    for key, value in tensor_dict.items():
        if torch.is_tensor(value):
            # pr0(f"1d->2d {key=} {value.shape=}")
            tensor_dict[key] = unpadded_tensor_1d_response_to_padded_tensor_2d_full(
                value, attention_mask_2d_bool, max_prompt_len
            )
    return tensor_dict


def compute_packing_info_for_batch(tensor_dict):
    attention_mask = tensor_dict["attention_mask"]
    sequence_offsets = attention_mask.long().sum(dim=1)
    tensor_dict["sequence_offsets"] = sequence_offsets.cumsum(dim=0)

    prompt_ids = tensor_dict["prompts"]
    tensor_dict["response_lens"] = attention_mask[:, prompt_ids.shape[1] :].sum(dim=1)

    return tensor_dict


def dump_dict_payload(payload: dict, tag: str):
    return
    for k, v in payload.items():
        if isinstance(v, torch.Tensor):
            pr0(f"{tag}: {k=} {v.shape=} {v=}")
        else:
            pr0(f"{tag}: {k=} {v=}")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

c = 0


def run_pipeline(
    engine,
    args: tuple,
    batch: dict,
    meta: dict,
    processing: dict,
    device: str,
    *,
    backward: bool = True,
    pack: bool = True,
    max_tokens_per_mb: int = DEFAULT_MAX_TOKENS_PER_MB,
    return_tensors: bool = False,
) -> dict:
    global c
    """Execute forward, post-processors, and optionally loss + backward.

    Parameters
    ----------
    engine
        DeepSpeed engine to call for the forward pass.
    args
        Positional arguments passed directly to ``engine()``.
    batch
        Dict with tensors belonging to this microbatch
    meta
        Extra batch data - flags and some tensors that shouldn't have been sharded
        (labels, advantages, ...) available to the model, post-processors and loss functions
    processing
        Dict with keys ``loss_fn`` (required for backward passes),
        and optional ``post`` and ``config``.
    device
        Target device string (e.g. ``"cuda:0"``).
    backward
        If *True*, ``loss_fn`` is required and ``engine.backward(loss)``
        is called.  Set to *False* for forward-only (no-grad) passes.
        Set to ``"loss_only"`` to compute the loss in train mode and return
        the raw loss tensor (requires_grad=True) without calling backward —
        the caller is responsible for scaling and calling engine.backward().
    pack
        If *True* (default), automatically handle sequence packing for flash
        attention: splits the input into microbatches by token budget, packs
        each microbatch from ``[B_mb, S]`` to ``[1, T]``, runs the pipeline,
        then unpacks and concatenates the results.  The caller passes padded
        ``[B, S]`` tensors and receives ``[B, S]`` outputs — packing is
        fully transparent.  Requires ``attention_mask`` in ``batch``.
        Auto-detects already-packed input: ``cu_seqlens`` is added by
        ``pack_for_dss`` and is the definitive sign that packing already
        happened — skip to avoid double-packing. ``pack=True`` is therefore
        safe to pass unconditionally.
    max_tokens_per_mb
        Token budget per microbatch when ``pack=True``.  Sequences are
        grouped by a first-fit-decreasing algorithm so no microbatch
        exceeds this limit.

    Returns
    -------
    dict
        ``{"avg_loss": float, "metrics": dict}`` when a loss function ran.
        ``{"avg_loss": ..., "metrics": ..., "batch": dict}`` also includes
        post-processor outputs (e.g. logprobs) when both loss and post-
        processors ran.
        ``{"batch": dict, "metrics": {}}`` when no loss function (forward-
        only).  ``batch`` contains only what post-processors added — never
        raw logits.
        When ``backward="loss_only"``: same as loss path but also includes
        ``"loss_tensor"`` (undetached, caller handles backward).
    """
    if pack:
        # Auto-detect already-packed input: pack_for_dss adds cu_seqlens as the
        # definitive signal that packing already happened — skip to avoid
        # double-packing. Any other case (including missing attention_mask) falls
        # through to packing, which will fail fast with a clear assertion error.
        all_input = {**batch, **meta}
        if "cu_seqlens" not in all_input:
            return _run_pipeline_with_packing(
                engine,
                args,
                batch,
                meta,
                processing,
                device,
                backward=backward,
                max_tokens_per_mb=max_tokens_per_mb,
            )

    tname_e2e = timers.start(f"run_pipeline e2e {engine.global_rank}")
    see_memory_usage("before fwd", force=True)
    post_names = processing.get("post", [])
    loss_fn_name = processing.get("loss_fn", "grpo")
    config = processing.get("config", {})

    # Skip entropy computation when it cannot affect the loss (entropy_coeff == 0).
    # Entropy is expensive: it requires a full-vocab softmax. In the non-zorro path
    # it is built by ``compute_entropy_and_logprobs_post``; in the zorro path it is
    # computed inside the model forward, which reads ``calculate_entropy`` from
    # ``**meta`` below. Gating ``meta["calculate_entropy"]`` here -- before the
    # forward -- covers both. Only override when ``actor_config`` explicitly carries
    # an ``entropy_coeff`` (i.e. the update_actor / loss path); the fwd-no-grad
    # ``compute_log_prob`` passes intentionally request entropy for logging and do
    # not send ``actor_config``, so they are left untouched. A fresh dict is used so
    # the shared per-call ``meta`` is not mutated.
    if meta.get("calculate_entropy"):
        actor_config = meta.get("actor_config")
        if isinstance(actor_config, dict) and "entropy_coeff" in actor_config:
            try:
                entropy_used = float(actor_config["entropy_coeff"]) != 0.0
            except (TypeError, ValueError):
                entropy_used = bool(actor_config["entropy_coeff"])
            if not entropy_used:
                meta = {**meta, "calculate_entropy": False}

    # --- forward ---
    # this is a future feature to avoid conflicts between `meta_data` and` model's `kwargs`, where the client could specify which meta_data keys to pass to the ending
    # if 'fwd_meta_keys' in meta_data:
    #     fwd_meta_data = {k:v for k,v in meta_data if k in fwd_meta_keys}
    # else:
    #     fwd_meta_data = meta_data
    # outputs = engine(*args, **kwargs, **fwd_meta_data)

    pack_with_unpad = True  # XXX: make configurable?

    # XXX: could the zorro parts be folded back into the model? this will also change when we start packing on the client side
    zorro_train_enable = meta.get("zorro_train_enable", False)
    # pr0(f"{zorro_train_enable=}")

    if pack_with_unpad or zorro_train_enable:
        batch = compute_packing_info_for_batch(batch)

    if zorro_train_enable:
        pack_with_unpad = False
        attention_mask_2d_bool = batch["attention_mask"].bool()
        max_prompt_len = meta["max_prompt_len"]

        non_model_input_keys = [k for k in batch.keys() if k not in ["input_ids", "position_ids", "attention_mask"]]
        for key in non_model_input_keys:
            if batch[key].shape == attention_mask_2d_bool.shape:
                batch[key] = padded_tensor_2d_full_to_unpadded_tensor_1d_response(
                    batch[key], attention_mask_2d_bool, max_prompt_len
                )

    # pr0(f"{pack_with_unpad=}")

    if pack_with_unpad:
        pad_token = meta["pad_token_id"]
        attention_mask_2d_bool = batch["attention_mask"].bool()
        batch = padded_tensor_2d_dict_to_unpadded_tensor_1d_dict(batch, attention_mask_2d_bool)

    log_dp_shard_tokens(
        engine.global_rank,
        "run_pipeline after_unpad",
        batch,
        meta,
    )

    pr0(f"effective {batch['input_ids'].shape=}")
    prof_fwd = ProfilerContext(type=PROFILER_TYPE, name="FWD")
    tname = timers.start(f"pipe fwd {engine.global_rank}")
    with prof_fwd():
        # passing **meta to the model since optimizations like zorro living in the model need to receive flags like calculate_entropy from the client
        if backward is False:
            engine.eval()
            with torch.no_grad():
                outputs = engine(*args, **batch, **meta)
        else:
            engine.train()
            # backward=True or backward="loss_only": train mode, grads enabled
            outputs = engine(*args, **batch, **meta)

    timers.stop_and_print_elapsed(tname)

    see_memory_usage("after fwd", force=True)
    prof_fwd.report()

    model_outputs: dict[str, Any] = {}
    if hasattr(outputs, "logits"):
        model_outputs["logits"] = outputs.logits
    if hasattr(outputs, "logprobs"):
        model_outputs["logprobs"] = outputs.logprobs
    if hasattr(outputs, "entropy"):
        model_outputs["entropy"] = outputs.entropy
    if hasattr(outputs, "loss") and outputs.loss is not None:
        model_outputs["loss"] = outputs.loss

    # --- post-forward ---
    prof_post_fwd = ProfilerContext(type=PROFILER_TYPE, name="POST-FWD")
    tname = timers.start(f"pipe post-fwd {engine.global_rank}")
    post_process_outputs = dict()
    with prof_post_fwd():
        for name in post_names:
            # tname_post = timers.start(f"pipe post-fwd [{name}]")
            fn = _resolve_fn(POST_PROCESSORS, name)
            post_process_outputs.update(**fn(model_outputs, batch, meta, device))
            # timers.stop_and_print_elapsed(tname_post)
    timers.stop_and_print_elapsed(tname)
    prof_post_fwd.report()

    model_outputs.update(**post_process_outputs)

    pipeline_outputs = dict(
        batch=post_process_outputs,
        metrics=dict(),
    )
    see_memory_usage("after post-fwd", force=True)

    # --- loss + backward ---
    if loss_fn_name is not None:
        tname = timers.start(f"pipe loss {engine.global_rank}")

        prof_loss = ProfilerContext(type=PROFILER_TYPE, name="LOSS")
        with prof_loss():
            fn = _resolve_fn(LOSS_FNS, loss_fn_name)
            meta.update(**post_process_outputs)
            loss, metrics = fn(model_outputs, batch, meta, config, device)
        timers.stop_and_print_elapsed(tname)
        prof_loss.report()

        # Exclude raw logits from response — large ([B,S,V]), no caller reads them.
        batch = {k: v for k, v in model_outputs.items() if k != "logits"}

        if backward is True:

            prof_bwd = ProfilerContext(type=PROFILER_TYPE, name="BWD")
            tname = timers.start(f"pipe bwd {engine.global_rank}")
            with prof_bwd():

                # GRAD-FIX: Arctic's user-side loss is already normalized as
                # (sum_pg_micro / T_global * dp_size) — i.e., when SUMMED across
                # the gas microbatches it equals the desired global per-token
                # mean × dp_size. DeepSpeed's default `scale_wrt_gas=True` then
                # divides each microbatch loss by gas inside backward(), giving
                # gradients exactly gas× too small. This was the dominant factor
                # in the ~44× grad_norm vs VeRL gap (predicted 64×, observed
                # 43.9×, residual ~1.46× = pg_loss aggregation convention).
                # Pass `scale_wrt_gas=False` so DS leaves our pre-normalized
                # loss alone.
                engine.backward(loss, scale_wrt_gas=False)

            timers.stop_and_print_elapsed(tname)
            prof_bwd.report()

        result = {
            "avg_loss": loss.detach().cpu().item(),
            "metrics": metrics,
        }
        if batch:
            result["batch"] = batch
        see_memory_usage("after bwd", force=True)

        if backward == "loss_only":
            result["loss_tensor"] = loss  # caller handles scaling + backward
        pipeline_outputs.update(**result)
        pipeline_outputs["metrics"].update(metrics)

    if zorro_train_enable:
        dump_dict_payload(pipeline_outputs["batch"], "zorro: post-fwd[before unpad_2_pad]")
        pipeline_outputs["batch"] = unpadded_tensor_1d_response_dict_to_padded_tensor_2d_full_dict(
            pipeline_outputs["batch"], attention_mask_2d_bool, max_prompt_len
        )
        dump_dict_payload(pipeline_outputs["batch"], "zorro: post-fwd[after unpad_2_pad]")

    if pack_with_unpad:
        dump_dict_payload(pipeline_outputs["batch"], "post-fwd[before unpad_2_pad]")
        pipeline_outputs["batch"] = unpadded_tensor_1d_dict_to_padded_tensor_2d_dict(
            pipeline_outputs["batch"], attention_mask_2d_bool, pad_token
        )
        dump_dict_payload(pipeline_outputs["batch"], "post-fwd[after unpad_2_pad]")

    # Consistency of metrics always being non-tensor
    pipeline_outputs["metrics"] = detensorize(pipeline_outputs["metrics"])

    if not return_tensors:
        dump_dict_payload(pipeline_outputs["batch"], "post-fwd[before detensorize]")
        pipeline_outputs["batch"] = detensorize(pipeline_outputs["batch"])
        dump_dict_payload(pipeline_outputs["batch"], "post-fwd[after detensorize]")

    timers.stop_and_print_elapsed(tname_e2e)

    return pipeline_outputs


def _run_pipeline_with_packing(
    engine,
    args: tuple,
    batch: dict,
    meta: dict,
    processing: dict,
    device: str,
    *,
    backward: bool,
    max_tokens_per_mb: int,
) -> dict:
    """Run the pipeline with automatic sequence packing/unpacking.

    Splits batch into microbatches, packs each to [1, T] for flash
    attention, runs the pipeline on each, then unpacks and concatenates.
    """
    from .microbatch import MicroBatchSpec
    from .microbatch import split_padded_tensor_dict_into_mb_list
    from .packing import pack_sequences
    from .packing import unpack_sequences

    # Combine for splitting — meta tensors are split the same way as batch
    all_input = {**meta, **batch}  # batch takes priority on key collision
    mb_spec = MicroBatchSpec(max_tokens_per_mb=max_tokens_per_mb)
    mb_list = split_padded_tensor_dict_into_mb_list(all_input, mb_spec)
    n_mbs = len(mb_list.mbs)

    loss_cfg = (processing or {}).get("config") or {}
    agg_level = loss_cfg.get("importance_sampling_level", "token")

    captured_losses: list[float] = []
    captured_weights: list[int] = []
    captured_metrics: dict[str, float] = {}
    collected_batch: dict[str, list[torch.Tensor]] = {}

    for i, mb in enumerate(mb_list.mbs):
        packed = pack_sequences(mb)
        pack_meta = packed.pop("_pack_meta")

        # 1D meta: packed tensors have shape [1, T] — squeeze for loss fns
        mb_1d = {
            k: v.squeeze(0) if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == 1 else v for k, v in packed.items()
        }

        # Model always receives packed [1, T] input_ids + position_ids
        mb_kwargs = {
            "input_ids": packed["input_ids"],
            "position_ids": packed["position_ids"],
            "use_cache": False,
        }

        if backward is True and hasattr(engine, "set_gradient_accumulation_boundary"):
            engine.set_gradient_accumulation_boundary(i == n_mbs - 1)

        result = run_pipeline(
            engine,
            args,
            mb_kwargs,
            meta,
            mb_1d,
            processing,
            device,
            backward=backward,
            return_tensors=True,
        )

        if "avg_loss" in result:
            captured_losses.append(result["avg_loss"])
            loss_mask = mb_1d.get("loss_mask")
            if agg_level == "sequence":
                weight = int(mb["input_ids"].shape[0]) if mb["input_ids"].ndim >= 2 else 1
            else:
                weight = int(loss_mask.sum()) if loss_mask is not None else 1
            captured_weights.append(weight)
            for k, v in result.get("metrics", {}).items():
                if isinstance(v, (int, float)):
                    captured_metrics[k] = captured_metrics.get(k, 0.0) + v * weight

        for k, v in result.get("batch", {}).items():
            if torch.is_tensor(v):
                v_1d = v.squeeze(0) if v.ndim == 2 and v.shape[0] == 1 else v
                unpacked = unpack_sequences(v_1d, pack_meta)
                collected_batch.setdefault(k, []).append(unpacked)

    # Restore original sequence order if microbatches were reordered
    idx = None
    if mb_list.backward_indices is not None:
        idx = torch.tensor(mb_list.backward_indices, dtype=torch.long)

    def _concat(tensors: list[torch.Tensor]) -> torch.Tensor:
        cat = torch.cat(tensors, dim=0)
        return cat[idx] if idx is not None else cat

    batch_out = {k: _concat(v) for k, v in collected_batch.items()}

    if captured_losses:
        total_weight = sum(captured_weights) or 1
        avg_loss = sum(loss * weight for loss, weight in zip(captured_losses, captured_weights)) / total_weight
        averaged_metrics = {k: v / total_weight for k, v in captured_metrics.items()}
        result = {"avg_loss": avg_loss, "metrics": averaged_metrics}
        if batch_out:
            result["batch"] = detensorize(batch_out)
        return result

    return {"batch": detensorize(batch_out), "metrics": {}}


# ---------------------------------------------------------------------------
# Built-in registered post-processors
# ---------------------------------------------------------------------------


@register_post_processor("identity")
def identity_post(model_outputs: dict, meta: dict, device: str) -> dict:
    """Pass-through — returns empty dict (nothing added)."""
    return {}


def logprobs_from_logits(logits, labels, inplace_backward=True):
    """
    Compute per-token log-probabilities for the given labels.

    Uses a Flash-Attention–based cross-entropy (if available) for efficient backward,
    otherwise falls back to a standard log-softmax+gather approach.

    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591

    Args:
        logits (Tensor): Model outputs of shape (..., vocab_size).
        labels (LongTensor): True class indices of shape matching logits[..., :-1].
        inplace_backward (bool): If True and Flash-Attn is available, perform backward in-place.

    Returns:
        Tensor: Log-probabilities of the target labels, shape logits.shape[:-1].
    """

    batch_dim = logits.shape[:-1]
    all_but_vocab_dim = logits.shape[-1]
    logits = logits.reshape(-1, all_but_vocab_dim)
    labels = labels.reshape(-1)
    fa_output = cross_entropy_loss(logits, labels, inplace_backward=inplace_backward)
    assert isinstance(
        fa_output, tuple
    ), "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
    output = -fa_output[0]
    return output.view(*batch_dim)


def chunked_logprobs_and_entropy_from_logits(logits, labels, calculate_entropy, peak_mem_gib=4.0):
    # Flatten to 2D for uniform chunked processing: [N, V] where N = B*S or T
    vocab_size = logits.shape[2]
    logits_2d = logits.reshape(-1, vocab_size)  # [N, V]
    labels_1d = labels.reshape(-1)  # [N]

    # Size each chunk so its [chunk_size, vocab] follow-up working set stays within the configured peak-memory
    # budget (arctic_rl.train.logits.optimization_peak_mem_size_in_gib). 4 bytes = fp32, the conservative accounting
    # for the logprob/entropy intermediates.
    budget_bytes = max(1, int(peak_mem_gib * 2**30))
    chunk_size = max(1, budget_bytes // max(1, vocab_size * 4))
    logprobs_chunks = []
    entropy_chunks = []
    for start in range(0, logits_2d.shape[0], chunk_size):
        end = min(start + chunk_size, logits_2d.shape[0])
        logits_chunk = logits_2d[start:end]
        labels_chunk = labels_1d[start:end]
        logprobs_chunk, entropy_chunk = fast_logprobs_and_entropy_from_logits(
            logits_chunk, labels_chunk, calculate_entropy
        )
        logprobs_chunks.append(logprobs_chunk)

        if calculate_entropy:
            entropy_chunks.append(entropy_chunk)

    # restore original shapes
    logprobs = torch.cat(logprobs_chunks).view_as(labels)

    if calculate_entropy:
        entropy = torch.cat(entropy_chunks).view_as(labels)
    else:
        entropy = None

    return logprobs, entropy


def fast_logprobs_and_entropy_from_logits(logits, labels, calculate_entropy):
    # tname_e2e = timers.start("logprob: e2e")

    if not calculate_entropy:
        entropy = None

    all_but_vocab_dim = logits.shape[:-1]
    vocab_dim = logits.shape[-1]
    # print(f"{logits.shape=}")
    # print(f"{labels.shape=}")
    flat_logits = logits.reshape(-1, vocab_dim)
    flat_labels = labels.reshape(-1)

    if FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE:
        # tname = timers.start("logprob: fa: 1. logprob")
        inplace_backward = logits.requires_grad
        output = cross_entropy_loss(flat_logits, flat_labels, inplace_backward=inplace_backward)
        logprobs = (-output[0]).view(*all_but_vocab_dim)
        # timers.stop_and_print_elapsed(tname)
        if calculate_entropy:
            # tname = timers.start("logprob: fa: 2. entropy")
            gathered_logits = torch.gather(flat_logits, -1, flat_labels.unsqueeze(-1)).squeeze(-1)
            logsumexp = gathered_logits - logprobs.reshape(-1)
            probs = torch.exp(flat_logits - logsumexp.unsqueeze(-1))
            entropy = (logsumexp - torch.sum(probs * flat_logits, dim=-1)).view(*all_but_vocab_dim)
            # timers.stop_and_print_elapsed(tname)
    else:
        # using 2 different implementations paths to optimize for whether calculate_entropy is needed or not
        if calculate_entropy:
            # tname = timers.start("logprob: (calculate_entropy=True) non-fa: 1. logprob")
            logsumexp = torch.logsumexp(flat_logits, dim=-1)
            logprobs = (torch.gather(flat_logits, -1, flat_labels.unsqueeze(-1)).squeeze(-1) - logsumexp).view(
                *all_but_vocab_dim
            )
            probs = torch.exp(flat_logits - logsumexp.unsqueeze(-1))
            # timers.stop_and_print_elapsed(tname)
            # tname = timers.start("logprob: (calculate_entropy=False) non-fa: 2. entropy")
            entropy = (logsumexp - torch.sum(probs * flat_logits, dim=-1)).view(*all_but_vocab_dim)
            # timers.stop_and_print_elapsed(tname)
        else:
            # Fastest logprobs-only: gather + logsumexp, but we can do better
            # with log_softmax fused kernel (single pass) + gather on the result
            # tname = timers.start("logprob: (calculate_entropy=False) non-fa: 0. logprob")
            logprobs = (
                torch.gather(torch.nn.functional.log_softmax(flat_logits, dim=-1), -1, flat_labels.unsqueeze(-1))
                .squeeze(-1)
                .view(*all_but_vocab_dim)
            )
            # timers.stop_and_print_elapsed(tname)

    # timers.stop_and_print_elapsed(tname_e2e)

    return logprobs, entropy


@register_post_processor("compute_entropy_and_logprobs")
def compute_entropy_and_logprobs_post(model_outputs: dict, batch: dict, meta: dict, device: str) -> dict:
    """Compute per-token log-probs from logits using torch.roll convention.

    Processes logits in chunks along the token dimension to avoid OOM on
    large vocabularies (e.g. 150k+ tokens).  Returns only logprobs and maybe entropy — raw
    logits are never included in the wire response.
    """
    # tname_e2e = timers.start("compute_entropy_and_logprobs_post")
    see_memory_usage("compute_entropy_and_logprobs_post start", force=True)

    calculate_entropy = meta.get("calculate_entropy")
    pr0(f"{calculate_entropy=}")
    processor_outputs = dict()
    if "logits" in model_outputs:
        # tname = timers.start("a3")
        # timers.stop_and_print_elapsed(tname)

        logits = model_outputs["logits"]
        input_ids = batch.get("input_ids")

        input_ids = input_ids.to(logits.device)
        # Align input_ids shape with logits if they differ (e.g. 1D packed meta)
        if input_ids.ndim < logits.ndim:
            input_ids = input_ids.view(logits.shape[:-1])
        labels = torch.roll(input_ids, shifts=-1, dims=-1)

        # arctic_rl.train.logits.compute_in_fp32: upcast the logits to fp32 before the logprob/entropy math runs on
        # them (improves numerical precision; no-op if the logits are already fp32).
        if meta.get("logits_compute_in_fp32", False):
            logits = logits.float()

        # arctic_rl.train.logits.optimization picks the logprob/entropy compute strategy ("memory" (tiling ) is not available unless zorro is used because the non-zorro path already has full logits manifested):
        #   none    -> single fused call over the full logits (fastest; manifests
        #              the full-size follow-up intermediates, i.e. logits memory
        #              more than once).
        #   compute -> manifest the full logits once, but run the follow-up in
        #              chunks so the full-size intermediates are never materialized.
        logits_optimization = meta.get("logits_optimization", "none")
        peak_mem_gib = meta.get("logits_optimization_peak_mem_size_in_gib", 4)
        if logits_optimization == "compute":
            logprobs, entropy = chunked_logprobs_and_entropy_from_logits(
                logits, labels, calculate_entropy, peak_mem_gib=peak_mem_gib
            )
        elif logits_optimization == "none":
            logprobs, entropy = fast_logprobs_and_entropy_from_logits(logits, labels, calculate_entropy)
        elif logits_optimization == "memory":
            raise ValueError("arctic_rl.train.logits.optimization=memory requires zorro enabled")
        else:
            raise ValueError(
                f"Unknown arctic_rl.train.logits.optimization={logits_optimization!r}; "
                "expected one of: none, memory, compute"
            )

        model_outputs["logprobs"] = logprobs
        model_outputs["entropy"] = entropy

        processor_outputs["logprobs"] = logprobs
        if entropy is not None:
            processor_outputs["entropy"] = entropy

        # tname = timers.start("post-fwd: token_logprobs")
        # see_memory_usage("compute_entropy_and_logprobs_post 3", force=True)
        # timers.stop_and_print_elapsed(tname)

        # free memory asap (not sure if another post-fwd consumer wants it?)
        model_outputs.pop("logits")

    elif "logprobs" in model_outputs:
        # zorro already computes these in CausalLM returning it as model_outputs["logprobs"] and model_outputs["entropy"]
        processor_outputs["logprobs"] = model_outputs["logprobs"]
        entropy = model_outputs.get("entropy")
        if entropy is not None:
            processor_outputs["entropy"] = entropy

    # timers.stop_and_print_elapsed(tname_e2e)

    return processor_outputs


# switch to compute_entropy_and_logprobs_post
# @register_post_processor("compute_logprobs")
# def compute_logprobs_post(model_outputs: dict, batch: dict, meta: dict, device: str) -> dict:
#     """Compute per-token log-probs from logits using torch.roll convention.

#     Processes logits in chunks along the token dimension to avoid OOM on
#     large vocabularies (e.g. 150k+ tokens).  Returns only logprobs — raw
#     logits are never included in the wire response.
#     """
#     logits = model_outputs["logits"]  # [B, S, V] or [1, T, V] packed
#     input_ids = batch.get("input_ids")
#     if input_ids is None:
#         return {}

#     if 1:
#         input_ids = input_ids.to(logits.device)
#         # Align input_ids shape with logits if they differ (e.g. 1D packed meta)
#         if input_ids.ndim < logits.ndim:
#             input_ids = input_ids.view(logits.shape[:-1])
#         labels = torch.roll(input_ids, shifts=-1, dims=-1)

#         # Flatten to 2D for uniform chunked processing: [N, V] where N = B*S or T
#         logits_2d = logits.reshape(-1, logits.shape[-1])   # [N, V]
#         labels_1d = labels.reshape(-1)                      # [N]

#         chunk_size = 1024
#         chunks = []
#         for start in range(0, logits_2d.shape[0], chunk_size):
#             end = min(start + chunk_size, logits_2d.shape[0])
#             lp = torch.log_softmax(logits_2d[start:end].float(), dim=-1)
#             chunks.append(lp.gather(-1, labels_1d[start:end].unsqueeze(-1)).squeeze(-1))

#         logprobs = torch.cat(chunks).view_as(labels)   # restore original shape
#         # pr0(f"compute_logprobs_post: {logprobs.shape=} {logprobs=}")
#     return {"logprobs": logprobs}


@register_post_processor("compute_entropy")
def compute_entropy_post(model_outputs: dict, batch: dict, meta: dict, device: str) -> dict:
    """Compute per-token entropy from logits."""
    processor_outputs = dict()
    if "entropy" in model_outputs:
        # zorro computes this already in CausalLM returning it as model_outputs["entropy"]
        processor_outputs["entropy"] = model_outputs["entropy"]
        # pr0(f"compute_entropy_post[zorro]: {processor_outputs['entropy'].shape=} {processor_outputs['entropy']=}")
    elif "logits" in model_outputs:
        logits = model_outputs["logits"]
        pd = torch.nn.functional.softmax(logits.float(), dim=-1)
        processor_outputs["entropy"] = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
        # pr0(f"compute_entropy_post[no-zorro]: {processor_outputs['entropy'].shape=} {processor_outputs['entropy']=}")
    return processor_outputs


@register_post_processor("apply_temperature")
def apply_temperature_post(model_outputs: dict, batch: dict, meta: dict, device: str) -> dict:
    """Apply temperature to logits."""
    see_memory_usage("apply_temperature_post start", force=True)
    if "logits" in model_outputs:
        temperature = meta["temperature"]
        logits = model_outputs["logits"]
        temperature = meta["temperature"]
        if temperature != 1.0:
            temperature = torch.tensor(temperature, device=logits.device)
            logits.div_(temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype))
        model_outputs["logits"] = logits
    elif "logprobs" in model_outputs:
        # zorro computes this already in patched CausalLM and we don't want to pass a huge logits tensor around
        pass

    see_memory_usage("apply_temperature_post end", force=True)
    return dict()
