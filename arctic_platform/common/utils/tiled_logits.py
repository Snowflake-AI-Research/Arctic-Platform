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

"""Memory-efficient logits / cross-entropy primitives shared by RL and SFT.

These building blocks turn hidden states (or a materialized ``[N, V]`` logits
block) into per-token log-probs (and optionally entropy) while bounding the
peak memory of the vocab-projection follow-up. They back the three
``logits_optimization`` modes used across the platform:

  * ``none``    -> full logits + a single-shot follow-up
    (:func:`tiled_logprobs_entropy_from_hidden`).
  * ``compute`` -> full logits once, but the softmax / entropy follow-up runs in
    token chunks so the full-size intermediates are never materialized at once
    (:func:`chunked_logprobs_entropy_from_hidden`).
  * ``memory``  -> the logits are never fully manifested; hidden states are
    tiled, projected per tile under ``no_grad``, and replayed in backward
    (:class:`TiledLogProbEntropy`). Requires a DeepSpeed engine for the tied
    lm-head / embedding weight grad bookkeeping. Used by ZoRRo and the no-ZoRRo
    GRPO pipeline (``run_pipeline`` + :func:`memory_logprobs_entropy_from_hidden`).

Originally lived in ``rl/zorro_train/qwen_model_patcher.py``; extracted here so
SFT (``sft_ce``) can reuse the exact same math. RL imports these back under its
historical names.
"""

from __future__ import annotations

from functools import partial

import torch

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = True
except ImportError:
    FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE = False


ENABLE_TIMERS = False
if ENABLE_TIMERS:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimple

    timers = SynchronizedWallClockTimerSimple(wall_clock_breakdown=True)
else:
    from arctic_platform.common.utils.debug import SynchronizedWallClockTimerSimpleDummy

    timers = SynchronizedWallClockTimerSimpleDummy(wall_clock_breakdown=True)


def logits_chunk_rows(vocab_size, peak_mem_gib, bytes_per_elem=4):
    """Max number of token rows whose ``[rows, vocab_size]`` logits block stays within ``peak_mem_gib`` GiB of
    peak memory overhead.

    Used to size the chunks/shards/tiles for the `memory` and `compute` logits-optimization modes from a single
    memory budget (``arctic_rl.train.logits.optimization_peak_mem_size_in_gib``). ``bytes_per_elem`` defaults to 4
    (fp32), the conservative accounting used by the logits follow-up math. Always returns at least 1.
    """
    budget_bytes = max(1, int(peak_mem_gib * 2**30))
    row_bytes = max(1, int(vocab_size) * bytes_per_elem)
    return max(1, budget_bytes // row_bytes)


def lm_head_logits(
    model, hidden_states, temperature=1.0, logits_compute_from_fp32_inputs=False, logits_compute_in_fp32=False
):
    """Project hidden states to vocab logits via the LM head, applying optional
    temperature scaling.

    Shared by the `none`/`compute` logits-optimization paths; the full logits are
    manifested here (the `memory` path avoids this by tiling inside the autograd
    function instead).

    When ``logits_compute_from_fp32_inputs`` is set, the LM-head input is upcast to fp32 so the projection
    (and hence the logits / logprob / entropy math) runs in fp32 (arctic_rl.train.logits.compute_from_fp32_inputs).

    When ``logits_compute_in_fp32`` is set, the produced logits are upcast to fp32 before they are consumed
    (temperature scaling + downstream logprob/entropy math) (arctic_rl.train.logits.compute_in_fp32). No-op if
    already fp32.
    """
    if logits_compute_from_fp32_inputs:
        hidden_states = hidden_states.float()
    logits = model.lm_head(hidden_states)
    if logits_compute_in_fp32:
        logits = logits.float()
    if temperature != 1.0:
        temperature = torch.tensor(temperature, device=logits.device)
        logits.div_(temperature.clamp(min=1e-8).unsqueeze(-1).to(logits.dtype))
    return logits


def logprobs_entropy_from_flat_logits(flat_logits, flat_labels, calculate_entropy):
    """Per-token logprobs (and optionally entropy) from a 2D ``[N, V]`` logits
    block, returned as 1D ``[N]`` tensors (entropy is ``None`` when not
    requested).

    Uses the fused flash-attn cross-entropy kernel when available, otherwise a
    logsumexp/gather fallback. This is the common core for both the single-shot
    (`tiled_...`) and chunked (`chunked_...`) entrypoints; callers own any
    reshape back to the original batch dims.
    """
    entropy = None
    # The flash-attn CE kernel is Triton and only runs on CUDA tensors; fall back
    # to the logsumexp/gather path on CPU so callers (e.g. sft_ce's ``none`` path)
    # stay valid off-GPU. GPU behavior is unchanged.
    if FLASH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE and flat_logits.is_cuda:
        inplace_backward = flat_logits.requires_grad
        output = cross_entropy_loss(flat_logits, flat_labels, inplace_backward=inplace_backward)
        logprobs = -output[0]
        if calculate_entropy:
            gathered_logits = torch.gather(flat_logits, -1, flat_labels.unsqueeze(-1)).squeeze(-1)
            logsumexp = gathered_logits - logprobs
            probs = torch.exp(flat_logits - logsumexp.unsqueeze(-1))
            entropy = logsumexp - torch.sum(probs * flat_logits, dim=-1)
    else:
        # using 2 different implementation paths to optimize for whether
        # calculate_entropy is needed or not
        if calculate_entropy:
            logsumexp = torch.logsumexp(flat_logits, dim=-1)
            logprobs = torch.gather(flat_logits, -1, flat_labels.unsqueeze(-1)).squeeze(-1) - logsumexp
            probs = torch.exp(flat_logits - logsumexp.unsqueeze(-1))
            entropy = logsumexp - torch.sum(probs * flat_logits, dim=-1)
        else:
            # Fastest logprobs-only: log_softmax fused kernel (single pass) +
            # gather on the result.
            logprobs = torch.gather(
                torch.nn.functional.log_softmax(flat_logits, dim=-1), -1, flat_labels.unsqueeze(-1)
            ).squeeze(-1)
    return logprobs, entropy


def tiled_logprobs_entropy_from_hidden(
    model,
    hidden_states,
    labels,
    temperature=1.0,
    calculate_entropy=True,
    logits_compute_from_fp32_inputs=False,
    logits_compute_in_fp32=False,
):
    # `none` mode: manifest the full logits, then compute logprobs/entropy in one shot. (Also reused per-shard
    # by TiledLogProbEntropy in `memory` mode, where it is called on tiled hidden_states so the full logits
    # aren't manifested.)
    tname_e2e = timers.start("logprob: tiled_logprobs_entropy_from_hidden e2e")

    logits = lm_head_logits(
        model,
        hidden_states,
        temperature,
        logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
        logits_compute_in_fp32=logits_compute_in_fp32,
    )

    batch_dim = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)

    logprobs, entropy = logprobs_entropy_from_flat_logits(flat_logits, flat_labels, calculate_entropy)
    logprobs = logprobs.view(*batch_dim)
    if entropy is not None:
        entropy = entropy.view(*batch_dim)

    timers.stop_and_print_elapsed(tname_e2e)

    return logprobs, entropy


def chunked_logprobs_entropy_from_hidden(
    model,
    hidden_states,
    labels,
    temperature=1.0,
    calculate_entropy=True,
    peak_mem_gib=4.0,
    logits_compute_from_fp32_inputs=False,
    logits_compute_in_fp32=False,
):
    """`compute` logits-optimization mode.

    Manifests the full logits once (a single ``model.lm_head`` over all tokens)
    and then runs the softmax/entropy follow-up in chunks along the token
    dimension, so the full-size follow-up intermediates (``probs`` /
    ``log_softmax``, each as large as the logits) are never materialized at once.
    Each chunk's follow-up working set is bounded by ``peak_mem_gib`` GiB
    (arctic_rl.train.logits.optimization_peak_mem_size_in_gib).
    Memory cost: the full logits tensor, once. Compute cost: a Python loop over
    token chunks.

    Contrast with the other modes:
      * ``none``   -> :func:`tiled_logprobs_entropy_from_hidden`:
                      full logits + full-size follow-up intermediates.
      * ``memory`` -> :class:`TiledLogProbEntropy`: logits never fully manifested,
                      at the cost of an extra forward replay in backward.
    """
    tname_e2e = timers.start("logprob: chunked_logprobs_entropy_from_hidden e2e")

    logits = lm_head_logits(
        model,
        hidden_states,
        temperature,
        logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
        logits_compute_in_fp32=logits_compute_in_fp32,
    )

    batch_dim = logits.shape[:-1]
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)

    chunk_size = logits_chunk_rows(flat_logits.shape[-1], peak_mem_gib)

    logprobs_chunks = []
    entropy_chunks = [] if calculate_entropy else None
    for start in range(0, flat_logits.shape[0], chunk_size):
        end = min(start + chunk_size, flat_logits.shape[0])
        logprobs_chunk, entropy_chunk = logprobs_entropy_from_flat_logits(
            flat_logits[start:end], flat_labels[start:end], calculate_entropy
        )
        logprobs_chunks.append(logprobs_chunk)
        if calculate_entropy:
            entropy_chunks.append(entropy_chunk)

    logprobs = torch.cat(logprobs_chunks, dim=0).view(*batch_dim)
    entropy = torch.cat(entropy_chunks, dim=0).view(*batch_dim) if calculate_entropy else None

    timers.stop_and_print_elapsed(tname_e2e)

    return logprobs, entropy


class TiledLogProbEntropy(torch.autograd.Function):
    """
    TiledLogProbEntropy implementation using gradient hooks (the grad hooks were copied from Axolotl). This has been adapted from TiledMLP in Deepspeed.
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        model,
        hidden_states,
        labels,
        temperature,
        calculate_entropy,
        shards,
        compute_params,
    ) -> torch.Tensor:

        # don't store anything for bwd if this is a torch.no_grad forward
        if hidden_states.requires_grad:
            ctx.fn = fn
            ctx.model = model
            ctx.shards = shards
            ctx.compute_params = [p for p in compute_params if p.requires_grad]
            ctx.temperature = temperature
            ctx.calculate_entropy = calculate_entropy
            ctx.save_for_backward(hidden_states, labels)

        hidden_states_shards = list(torch.chunk(hidden_states, chunks=shards, dim=0))
        labels_shards = list(torch.chunk(labels, chunks=shards, dim=0))

        with torch.no_grad():
            logprobs_shards, entropy_shards = list(
                zip(
                    *[
                        fn(model, hidden_states_shards[idx], labels_shards[idx], temperature, calculate_entropy)
                        for idx in range(shards)
                    ]
                )
            )

        if calculate_entropy:
            entropy = torch.cat(entropy_shards, dim=0)
            # pr(f"{entropy.shape=}")
        else:
            entropy = None

        logprobs = torch.cat(logprobs_shards, dim=0)

        return logprobs, entropy

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (hidden_states, labels) = ctx.saved_tensors
        model = ctx.model
        shards = ctx.shards
        compute_params = ctx.compute_params

        temperature = ctx.temperature
        calculate_entropy = ctx.calculate_entropy

        hs = hidden_states

        hs_requires_grad = hs.requires_grad
        hs = hs.detach()
        hs.requires_grad_(hs_requires_grad)

        logprobs_grads, entropy_grads = grads

        hs_grad = torch.zeros_like(hs)

        # return (None, None, hs_grad, None, None, None, None, None)

        hs_shards = list(torch.chunk(hs, chunks=shards, dim=0))
        labels_shards = list(torch.chunk(labels, chunks=shards, dim=0))

        # not using GradientAccumulator since it's not needed under deepspeed (needed for ddp/fsdp, so leaving the code here, but commented out)
        # Create a gradient accumulator for parameters
        # grad_accumulator = GradientAccumulator(compute_params, shards, dtype=hs.dtype)

        # Tell deepspeed not to add a new grad to its ipg bucket during this backward
        # oddly because of self.lm_head.weight being tied with self.model.embed_tokens.weight we have to tell DS that the grad isn't ready and it'll be reduced when model.embed_tokens.weight grad is reduced
        # otherwise it asserts the parameter model.embed_tokens.weight has already been reduced.
        for param in compute_params:
            param.ds_grad_is_ready = False

        labels_step = labels_shards[0].shape[0]
        shard_step = hs_shards[0].numel()
        for i, hs_shard in enumerate(hs_shards):
            hs_shard.requires_grad_(hs_requires_grad)

            shard_offset = i * shard_step
            hs_shard.grad = hs_grad.view(-1).narrow(0, shard_offset, hs_shard.numel()).view_as(hs_shard)

            # Install hooks for this shard
            # is_last_shard = i + 1 == shards
            # grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                logprobs_shard, entropy_shard = fn(model, hs_shard, labels_shards[i], temperature, calculate_entropy)

            incoming_grad_shards = []
            tensors = []
            if entropy_shard is not None:
                tensors += [entropy_shard]
                incoming_grad_shards += [
                    (
                        entropy_grads.view(-1)
                        .narrow(0, i * labels_step, labels_shards[i].shape[0])
                        .view(labels_shards[i].shape[0])
                    )
                ]

            tensors += [logprobs_shard]
            incoming_grad_shards += [
                (
                    logprobs_grads.view(-1)
                    .narrow(0, i * labels_step, labels_shards[i].shape[0])
                    .view(labels_shards[i].shape[0])
                )
            ]

            torch.autograd.backward(tensors, incoming_grad_shards)

        # Clean up hooks
        # grad_accumulator.cleanup()
        # del grad_accumulator

        return (
            None,
            None,
            hs_grad,
            None,
            None,
            None,
            None,
            None,
        )


_LOGITS_OPT_META_KEYS = (
    "logits_optimization",
    "logits_optimization_peak_mem_size_in_gib",
    "logits_compute_from_fp32_inputs",
    "logits_compute_in_fp32",
)


def fill_logits_opt_from_worker_config(meta: dict, ds_worker_config: dict | None) -> dict:
    """Copy worker-init logits knobs into per-call meta when the client left them at ``none``."""
    worker = ds_worker_config or {}
    worker_opt = worker.get("logits_optimization") or "none"
    meta_opt = meta.get("logits_optimization") or "none"
    if meta_opt != "none" or worker_opt == "none":
        return meta
    filled = dict(meta)
    for key in _LOGITS_OPT_META_KEYS:
        if key in worker:
            filled[key] = worker[key]
    return filled


def sync_logits_num_shards(num_shards: int, device) -> int:
    """MAX-reduce tile count across DP so ranks with different packed lengths stay in lockstep."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return num_shards
    if torch.distributed.get_world_size() <= 1:
        return num_shards
    t = torch.tensor(num_shards, dtype=torch.long, device=device)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return int(t.item())


def memory_logprobs_entropy_from_hidden(
    model,
    hidden_states,
    labels,
    *,
    temperature=1.0,
    calculate_entropy=True,
    peak_mem_gib=4.0,
    logits_compute_from_fp32_inputs=False,
    logits_compute_in_fp32=False,
    device=None,
):
    """Tile ``hidden_states`` through the LM head; never materializes full ``[N, V]`` logits.

    ``labels`` must already be next-token targets (``torch.roll(input_ids, -1)``).
    """
    batch_dim = labels.shape
    hidden_size = hidden_states.shape[-1]
    flat_hidden = hidden_states.reshape(-1, hidden_size)
    flat_labels = labels.reshape(-1)
    if flat_hidden.shape[0] != flat_labels.shape[0]:
        raise ValueError(f"memory logits: hidden rows {flat_hidden.shape[0]} != labels {flat_labels.shape[0]}")
    tied = getattr(getattr(model, "config", None), "tie_word_embeddings", None)
    if tied is False:
        raise ValueError(
            "logits_optimization=memory requires tie_word_embeddings=True so DeepSpeed reduces "
            "lm_head.weight with embed_tokens.weight (TiledLogProbEntropy sets ds_grad_is_ready=False)."
        )
    vocab_size = model.config.vocab_size
    chunk_rows = logits_chunk_rows(vocab_size, peak_mem_gib)
    num_shards = max(1, -(-flat_hidden.shape[0] // chunk_rows))
    if device is None:
        device = flat_hidden.device
    num_shards = sync_logits_num_shards(num_shards, device)
    tiled_fn = partial(
        tiled_logprobs_entropy_from_hidden,
        logits_compute_from_fp32_inputs=logits_compute_from_fp32_inputs,
        logits_compute_in_fp32=logits_compute_in_fp32,
    )
    logprobs, entropy = TiledLogProbEntropy.apply(
        tiled_fn,
        model,
        flat_hidden,
        flat_labels,
        temperature,
        calculate_entropy,
        num_shards,
        [model.lm_head.weight],
    )
    logprobs = logprobs.view(*batch_dim)
    if entropy is not None:
        entropy = entropy.view(*batch_dim)
    return logprobs, entropy
