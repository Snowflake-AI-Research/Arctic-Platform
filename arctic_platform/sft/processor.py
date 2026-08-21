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

"""SFT processors — separate from the GRPO pipeline.

Phase-1 loss mirrors ArcticTraining ``SFTTrainer.loss`` for
``sequence_parallel_size == 1``: hand ``labels`` to the HF causal-LM and
read ``outputs.loss`` (HF does the shift + CE with ``ignore_index=-100``).

The GRPO ``run_pipeline`` path is intentionally not reused: it requires a
``prompts`` key, unpads every ``[B, S]`` tensor, and forwards all of
``**meta`` into the model. SFT needs none of that.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import _resolve_fn
from arctic_platform.common.registry import register_loss_fn
from arctic_platform.common.utils.batch import detensorize
from arctic_platform.common.utils.debug import pr0
from arctic_platform.common.utils.debug import see_memory_usage
from arctic_platform.common.utils.tiled_logits import TiledLogProbEntropy
from arctic_platform.common.utils.tiled_logits import chunked_logprobs_entropy_from_hidden
from arctic_platform.common.utils.tiled_logits import logits_chunk_rows
from arctic_platform.common.utils.tiled_logits import logprobs_entropy_from_flat_logits
from arctic_platform.common.utils.tiled_logits import tiled_logprobs_entropy_from_hidden

# Valid ``logits_optimization`` modes for sft_ce. ``none`` keeps the classic
# full-logits path (``sft_ce_loss``); ``compute`` / ``memory`` compute the CE
# from hidden states without materializing the full ``[B, S, V]`` logits.
SFT_CE_LOGITS_OPTIMIZATIONS = {"none", "compute", "memory"}


def _require_labels(batch: dict, who: str) -> torch.Tensor:
    """Return ``batch['labels']`` or raise a clear error (not a bare KeyError)."""
    labels = batch.get("labels")
    if labels is None:
        raise ValueError(f"{who} requires batch['labels'] ([B, S] with -100 on ignored positions)")
    return labels


def count_valid_target_tokens(batch: Any) -> int | None:
    """Valid next-token targets after HF's ``labels[:, 1:]`` / ``-100`` shift.

    ``batch`` is one microbatch dict or a gas list of them. ``None`` means no
    labels (skip global-token injection); ``0`` means all positions masked.
    """
    if isinstance(batch, list):
        mbs = [mb for mb in batch if isinstance(mb, dict) and mb.get("labels") is not None]
        if not mbs:
            return None
        return sum(int((mb["labels"][:, 1:] != -100).sum().item()) for mb in mbs)
    if not isinstance(batch, dict) or batch.get("labels") is None:
        return None
    return int((batch["labels"][:, 1:] != -100).sum().item())


def _paired_loss_metrics(loss_sum: float, tokens: float) -> dict:
    """``loss.sum`` / ``loss.tokens`` pair → global token-mean via ``combine_metric_*``."""
    return {"loss.sum": float(loss_sum), "loss.tokens": float(tokens)}


@register_loss_fn("sft")
def sft_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """HF causal-LM CE loss (ArcticTraining SFT parity).

    Expected ``model_outputs``: ``loss`` — scalar from ``AutoModelForCausalLM``.
    Expected ``batch``: ``labels`` ``[B, S]`` with ``-100`` on ignored positions.

    Emits paired ``loss.sum`` / ``loss.tokens`` so
    ``combine_metric_shards`` / ``combine_metric_microbatches`` produce a
    global token-mean across DP ranks and gas microbatches.
    """
    loss = model_outputs["loss"]
    if loss is None:
        raise ValueError("SFT loss is None — model returned no loss (check that labels are present and not all -100)")

    labels = _require_labels(batch, "sft loss")
    # HF CE targets are labels[:, 1:], ignore_index=-100.
    n_valid = int((labels[:, 1:] != -100).sum().item())
    if n_valid == 0:
        # HF CE is NaN when every target is ignore_index.
        loss = torch.zeros_like(loss, requires_grad=loss.requires_grad)
    # Reconstruct Σ CE from HF's token-mean so cross-rank aggregation stays exact.
    loss_sum = float(loss.detach().float().item()) * n_valid
    return loss, _paired_loss_metrics(loss_sum, n_valid)


# Loss fns that need raw logits (rather than HF's scalar ``outputs.loss``).
LOGIT_LOSS_FNS = {"sft_ce"}

# Dispatched by ``run_sft_pipeline`` in the DeepSpeed worker.
SFT_LOSS_FNS = {"sft", "sft_ce"}

# Opt-in: worker injects ``global_num_tokens`` + ``dp_size`` before the loss runs.
SFT_GLOBAL_TOKEN_LOSS_FNS = {"sft_ce"}


@register_loss_fn("sft_ce")
def sft_ce_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Explicit CE from logits (fp32 upcast). Scaled via ``_scale_ce_sum`` / ``meta``."""
    logits = model_outputs.get("logits")
    if logits is None:
        raise ValueError("sft_ce requires logits — run_sft_pipeline must capture them for this loss_fn")

    labels = _require_labels(batch, "sft_ce loss").to(logits.device)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab = shift_logits.size(-1)

    # Compute CE through the shared logprobs core rather than F.cross_entropy so
    # the full-logits ``none`` path is numerically identical to the ``compute`` /
    # ``memory`` hidden-state paths (which also go through this core). The two
    # kernels agree on the forward loss but their *backward* diverges at large
    # vocab (~1.5% on the lm-head grad), which otherwise desyncs the three modes'
    # training curves after the first step. Mirror sft_ce_sum_from_hidden's -100
    # masking: clamp ignored targets to a safe gather index, then zero them out.
    valid = shift_labels != -100
    safe_labels = shift_labels.clamp_min(0)
    flat_logits = shift_logits.reshape(-1, vocab).float()
    logprobs, _ = logprobs_entropy_from_flat_logits(flat_logits, safe_labels.reshape(-1), False)
    ce_sum = -(logprobs * valid.reshape(-1).to(logprobs.dtype)).sum()
    n_valid = int(valid.sum().item())

    loss = _scale_ce_sum(ce_sum, n_valid, meta)
    return loss, _paired_loss_metrics(float(ce_sum.detach().float().item()), n_valid)


def _scale_ce_sum(ce_sum: torch.Tensor, n_valid: int, meta: dict) -> torch.Tensor:
    """Token-mean CE. With ``meta["global_num_tokens"]``: ``ce_sum / T_global * dp_size``
    so DP all-reduce (which averages) reconstructs the global mean; else per-shard."""
    global_tokens = meta.get("global_num_tokens")
    dp_size = int(meta.get("dp_size", 1) or 1)
    if global_tokens:
        return ce_sum / float(global_tokens) * dp_size
    return ce_sum / float(max(n_valid, 1))


def sft_ce_sum_from_hidden(
    model,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    *,
    mode: str,
    peak_mem_gib: float = 4.0,
) -> tuple[torch.Tensor, int]:
    """Summed causal-LM CE over valid targets, computed from the last hidden
    states without ever materializing the full ``[B, S, V]`` logits.

    Applies the next-token shift here: ``hidden[:, :-1]`` predicts
    ``labels[:, 1:]``. Ignored positions (``labels == -100``) are given a safe
    clamped index for the gather in the shared kernel, then zeroed out of the
    sum so their gradient contribution is exactly zero.

    Modes (reusing the shared primitives in ``common.utils.tiled_logits``):
      * ``compute`` -> :func:`chunked_logprobs_entropy_from_hidden`: manifests
        the full logits once, chunks the softmax follow-up under ``peak_mem_gib``.
      * ``memory``  -> :class:`TiledLogProbEntropy`: tiles the hidden states,
        projects per tile under ``no_grad``, replays in backward — the full
        logits are never manifested. Requires a DeepSpeed engine (tied lm-head /
        embedding grad bookkeeping).

    CE math runs in fp32 (``logits_compute_in_fp32``) for parity with the
    full-logits ``sft_ce_loss`` path. Returns ``(ce_sum, num_valid_tokens)``.
    """
    if mode not in ("compute", "memory"):
        raise ValueError(f"sft_ce_sum_from_hidden: mode must be 'compute' or 'memory', got {mode!r}")

    shift_hidden = hidden[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    valid = shift_labels != -100
    safe_labels = shift_labels.clamp_min(0)

    if mode == "compute":
        logprobs, _ = chunked_logprobs_entropy_from_hidden(
            model,
            shift_hidden,
            safe_labels,
            temperature=1.0,
            calculate_entropy=False,
            peak_mem_gib=peak_mem_gib,
            logits_compute_in_fp32=True,
        )
    else:  # memory
        flat_hidden = shift_hidden.reshape(-1, shift_hidden.shape[-1])
        flat_labels = safe_labels.reshape(-1)
        chunk_rows = logits_chunk_rows(model.config.vocab_size, peak_mem_gib)
        num_shards = max(1, -(flat_hidden.shape[0] // -chunk_rows))  # ceil division
        # Bind fp32 CE so it flows through the (replayed) tiled forward, matching
        # the compute/none paths. lm_head.weight is the tied compute param whose
        # grad DeepSpeed reduces with the embedding weight.
        tiled_fn = partial(tiled_logprobs_entropy_from_hidden, logits_compute_in_fp32=True)
        logprobs, _ = TiledLogProbEntropy.apply(
            tiled_fn,
            model,
            flat_hidden,
            flat_labels,
            1.0,  # temperature
            False,  # calculate_entropy
            num_shards,
            [model.lm_head.weight],
        )
        logprobs = logprobs.view(*safe_labels.shape)

    ce_sum = -(logprobs * valid.to(logprobs.dtype)).sum()
    n_valid = int(valid.sum().item())
    return ce_sum, n_valid


def _batch_is_packed(batch: dict, meta: dict) -> bool:
    """True when the wire batch carries Axolotl-style sample packing.

    HF FA2 derives varlen ``cu_seqlens`` from ``position_ids`` that reset to 0
    at each packed-document boundary (``_is_packed_sequence``). Prefer an
    explicit ``meta['sample_packing']`` flag from the client; fall back to
    detecting a reset inside ``position_ids``.
    """
    if meta.get("sample_packing"):
        return True
    pos = batch.get("position_ids")
    if pos is None or not torch.is_tensor(pos) or pos.numel() < 2:
        return False
    # A reset to 0 after the first token marks a pack boundary.
    flat = pos.reshape(-1)
    return bool(((flat[1:] == 0) & (flat[:-1] != 0)).any().item())


def _build_sft_model_kwargs(batch: dict, meta: dict, labels: torch.Tensor, need_logits: bool) -> dict[str, Any]:
    """Build HF causal-LM kwargs, honoring packed vs dense batches.

    Packed (FA2 varlen):
      * Pass ``position_ids`` (required for HF to derive ``cu_seqlens``).
      * Omit a synthetic/all-ones ``attention_mask`` — HF casts segment-id
        masks to bool anyway and uses ``position_ids`` for boundaries.
      * Prefer ``batch_size == 1`` per DeepSpeed microbatch (worker GAS split).

    Dense (unpacked):
      * Pass ``attention_mask`` when present (skip pad in attention scores;
        still rectangular compute unless a future unpad path is added).
    """
    model_kwargs: dict[str, Any] = {
        "input_ids": batch["input_ids"],
        "use_cache": False,
    }
    packed = _batch_is_packed(batch, meta)
    if "position_ids" in batch and batch["position_ids"] is not None:
        model_kwargs["position_ids"] = batch["position_ids"]
    if not packed and "attention_mask" in batch and batch["attention_mask"] is not None:
        model_kwargs["attention_mask"] = batch["attention_mask"]
    elif packed and "attention_mask" in batch and batch["attention_mask"] is not None:
        # Optional: forward a real mask when the client kept one (e.g. segment
        # ids or bool pad). HF FA2 will bool-cast it; boundaries still come
        # from position_ids. Skipping an all-ones mask avoids dense leakage
        # if a buggy client synthesized one — detect via all-ones check.
        attn = batch["attention_mask"]
        if not bool(torch.all(attn != 0).item()):
            model_kwargs["attention_mask"] = attn
    if not need_logits:
        model_kwargs["labels"] = labels
    return model_kwargs


def run_sft_pipeline(
    engine,
    batch: dict,
    meta: dict,
    processing: dict,
    device: str,
    *,
    backward: bool = True,
) -> dict:
    """Forward (+ optional loss/backward) for SFT.

    Dense batches pass ``input_ids`` + ``attention_mask`` + ``labels``.
    Packed batches (Axolotl ``sample_packing``) additionally carry
    ``position_ids`` with per-document resets; those are forwarded so HF FA2
    can run varlen attention instead of a dense padded rectangle. No GRPO
    packing/unpad pipeline — SFT stays on the plain HF causal-LM surface.
    """
    from arctic_platform import sft_profile

    see_memory_usage("run_sft_pipeline start", force=True)
    loss_fn_name = processing.get("loss_fn", "sft")
    config = processing.get("config", {}) or {}

    # sft_ce can trade the full [B, S, V] logits for a hidden-state CE that runs
    # in token chunks (``compute``) or tiles that never manifest the full logits
    # (``memory``). ``none`` keeps the classic full-logits path (``sft_ce_loss``).
    logits_opt = str(config.get("logits_optimization", "none") or "none")
    if logits_opt not in SFT_CE_LOGITS_OPTIMIZATIONS:
        raise ValueError(
            f"logits_optimization must be one of {sorted(SFT_CE_LOGITS_OPTIMIZATIONS)}, got {logits_opt!r}"
        )
    peak_mem_gib = float(config.get("logits_optimization_peak_mem_size_in_gib", 4) or 4)
    hidden_ce = loss_fn_name == "sft_ce" and logits_opt in ("compute", "memory")

    # ``need_logits`` = keep the full logits tensor for the loss (only the classic
    # sft_ce path). ``omit_labels`` also drops HF's own CE when we compute it.
    need_logits = loss_fn_name in LOGIT_LOSS_FNS and not hidden_ce
    omit_labels = need_logits or hidden_ce

    labels = _require_labels(batch, "run_sft_pipeline (SFT wire batch)")
    model_kwargs = _build_sft_model_kwargs(batch, meta, labels, omit_labels)
    if hidden_ce:
        # Request last-layer hidden states (post-final-norm) and keep only the
        # last token's logits so HF doesn't project the whole sequence to vocab.
        model_kwargs["output_hidden_states"] = True
        model_kwargs["logits_to_keep"] = 1
    packed = _batch_is_packed(batch, meta)
    pr0(
        f"run_sft_pipeline: {model_kwargs['input_ids'].shape=} {backward=} "
        f"{loss_fn_name=} {packed=} {logits_opt=} keys={sorted(model_kwargs)}"
    )

    def _cuda_sync() -> None:
        if sft_profile.enabled() and torch.cuda.is_available():
            torch.cuda.synchronize()

    _cuda_sync()
    with sft_profile.timed("fwd"):
        if backward is False:
            engine.eval()
            with torch.no_grad():
                outputs = engine(**model_kwargs)
        else:
            engine.train()
            outputs = engine(**model_kwargs)
        _cuda_sync()

    if hidden_ce:
        # Tiled / chunked CE straight from hidden states — the full logits are
        # never materialized (memory) or never fully softmaxed at once (compute).
        hf_model = getattr(engine, "module", engine)
        hidden = outputs.hidden_states[-1]
        with sft_profile.timed("loss"):
            ce_sum, n_valid = sft_ce_sum_from_hidden(
                hf_model, hidden, labels.to(hidden.device), mode=logits_opt, peak_mem_gib=peak_mem_gib
            )
            loss = _scale_ce_sum(ce_sum, n_valid, meta)
        metrics = _paired_loss_metrics(float(ce_sum.detach().float().item()), n_valid)
    else:
        model_outputs: dict[str, Any] = {}
        if hasattr(outputs, "loss") and outputs.loss is not None:
            model_outputs["loss"] = outputs.loss
        if need_logits:
            # Consumed locally by the loss_fn only; never placed on the wire response.
            model_outputs["logits"] = outputs.logits

        fn = _resolve_fn(LOSS_FNS, loss_fn_name)
        loss, metrics = fn(model_outputs, batch, meta, config, device)

    if backward is True:
        # Match the GRPO GRAD-FIX: loss is already a mean, do not let DeepSpeed
        # divide by gas again.
        _cuda_sync()
        with sft_profile.timed("bwd"):
            engine.backward(loss, scale_wrt_gas=False)
            _cuda_sync()

    result = {
        "avg_loss": loss.detach().cpu().item(),
        "metrics": detensorize(metrics),
        "batch": {},
    }
    see_memory_usage("run_sft_pipeline end", force=True)
    return result
