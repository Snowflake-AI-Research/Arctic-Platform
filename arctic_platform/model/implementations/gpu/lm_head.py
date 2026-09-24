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
"""Chunked LM-head logprobs and FP32 LM-head projection.

These two transforms reduce memory and improve precision around the final
``hidden @ lm_head_weight.T`` projection used to compute per-token logprobs:

* The full logits tensor for a backbone output is ``[B, S, V]``, which for
  large ``V`` (e.g. 150k) dominates activation memory. Materializing it in
  ``token_chunk_size x vocab_chunk_size`` tiles keeps peak memory bounded
  while still producing exact per-token logprobs via an online (streaming)
  log-sum-exp over the vocab tiles.
* Casting the final projection to fp32 avoids low-precision noise in
  ``log(softmax(logits))`` (and its gradient), which matters for stable RL
  grad-log-prob updates.

Public API:

* :func:`enable_fp32_lm_head` — monkey-patch ``model.lm_head.forward`` to
  upcast hidden states, weights, and bias to fp32 before the projection, and,
  on a Liger-patched model, set its fused loss to accumulate in fp32.
* :func:`enable_chunked_lm_head_logprobs` — install a ``dss_compute_logprobs``
  branch on ``model.forward`` that runs the backbone once and computes
  per-token logprobs via tiled matmuls instead of materializing full logits.
* :func:`chunked_lm_head_logprobs` — the underlying tiled-matmul logprob
  function, exposed for direct invocation and testing.

The chunked autograd Function is built lazily on first use so this module's
top-level import stays torch-free.
"""

from __future__ import annotations

import functools
import sys
import types

from arctic_platform.model.implementations.gpu.action_masks import action_masks_to_lm_head
from arctic_platform.model.implementations.gpu.action_masks import apply_lm_head_action_masks_
from arctic_platform.model.implementations.gpu.action_masks import filter_lm_head_action_masks
from arctic_platform.model.implementations.gpu.action_masks import slice_action_masks_for_logits_to_keep
from arctic_platform.model.implementations.gpu.action_masks import slice_lm_head_action_masks
from arctic_platform.model.implementations.gpu.action_masks import validate_action_mask_targets
from arctic_platform.model.implementations.gpu.packing import IGNORE_INDEX

# Vocabulary index the ignore-index sentinel is scored at. Any index in ``[0, vocab_size)`` would do: the
# substitute only has to be gatherable from some vocab tile, since callers exclude ignored positions from the
# loss and :func:`filter_lm_head_action_masks` drops the constraints recorded at them.
IGNORE_INDEX_SUBSTITUTE = 0


def safe_chunked_labels(labels):
    """Map ``IGNORE_INDEX`` onto a scoreable vocabulary index, and report which positions carried it.

    Returns the substituted labels and a boolean mask of the positions that carried the sentinel.

    The tiled forward gathers a position's target logit only from the vocab tile that contains its label, so a
    label outside ``[0, vocab_size)`` is gathered from no tile and the position keeps the zero its target logit
    was initialized to. It is then reported as ``0 - logsumexp(logits)``, which is finite, is not the
    log-probability of any token, and is positive whenever every logit at that position is negative.

    Any other out-of-range label is a caller error rather than a position to substitute, and
    :func:`validate_lm_head_targets` is what rejects it. That check runs on the host before the labels reach
    here, so nothing in this function reads them back from the device.
    """
    import torch

    ignore_mask = labels == IGNORE_INDEX
    substitute = labels.new_full((), IGNORE_INDEX_SUBSTITUTE)
    return torch.where(ignore_mask, substitute, labels).contiguous(), ignore_mask.contiguous()


_VALIDATED_TARGETS_ATTR = "_dss_validated_lm_head_vocab_size"


def mark_lm_head_targets_validated(labels, *, vocab_size: int):
    """Mark a tensor whose ids were checked on CPU before transfer."""
    setattr(labels, _VALIDATED_TARGETS_ATTR, int(vocab_size))
    return labels


def inherit_lm_head_target_validation(source, transformed):
    """Carry CPU validation metadata through a view or shape transform of the same target ids."""
    vocab_size = getattr(source, _VALIDATED_TARGETS_ATTR, None)
    if vocab_size is not None:
        mark_lm_head_targets_validated(transformed, vocab_size=vocab_size)
    return transformed


def validated_lm_head_targets_to(labels, *, device, vocab_size: int):
    """Validate recoverably on CPU, transfer, and mark the device tensor for the chunked head."""
    if labels.device.type != "cpu":
        raise ValueError("LM-head targets must be validated before their first device transfer")
    validate_lm_head_targets(labels, vocab_size=vocab_size)
    return mark_lm_head_targets_validated(labels.to(device), vocab_size=vocab_size)


def validate_lm_head_targets(labels_flat, *, vocab_size: int) -> None:
    """Reject targets that no vocabulary tile can claim, for any chunked LM head.

    A chunked head reads each token's target logit out of the tile that covers it, selected with
    ``(labels >= vocab_start) & (labels < vocab_end)``. A label outside ``[0, vocab_size)`` matches no tile,
    so that token's target logit keeps the zero it was initialized with: a finite logprob, a finite gradient,
    and nothing to distinguish it from a token the head really scored. Unfused cross-entropy raises on the
    same input, and a head that silently trains on it is the more dangerous of the two.

    ``IGNORE_INDEX`` is the one value allowed through: SFT marks the positions it does not train with it and
    masks them out of the loss, so those tokens are expected to carry a meaningless logprob.
    """
    import torch

    from arctic_platform.model.implementations.gpu.packing import IGNORE_INDEX

    if labels_flat.dtype == torch.bool or labels_flat.is_floating_point() or labels_flat.is_complex():
        raise ValueError(f"chunked LM-head targets must have an integer dtype, got {labels_flat.dtype}")
    if getattr(labels_flat, _VALIDATED_TARGETS_ATTR, None) == int(vocab_size):
        return
    # Direct callers can still hand the public helper a device tensor. Copy it back and raise recoverably;
    # production validates before transfer and reaches the marker fast path above, so valid requests do not
    # pay this synchronization and malformed requests cannot poison the long-lived CUDA context.
    labels_for_validation = labels_flat.detach().cpu() if labels_flat.device.type != "cpu" else labels_flat
    invalid_targets = (labels_for_validation != IGNORE_INDEX) & (
        (labels_for_validation < 0) | (labels_for_validation >= vocab_size)
    )
    if bool(invalid_targets.any().item()):
        first_invalid = int(labels_for_validation[invalid_targets][0].item())
        raise ValueError(
            f"chunked LM-head target {first_invalid} is outside the vocabulary [0, {vocab_size}); "
            f"only IGNORE_INDEX ({IGNORE_INDEX}) may lie outside it"
        )


# TODO: replace ``SequenceChunkedLogProbFn`` with an OSS kernel such as
# LigerFusedLinearCrossEntropy once it is a viable drop-in for DSS per-token
# logprob output.
@functools.lru_cache(maxsize=None)
def _get_sequence_chunked_logprob_fn():
    """Build (and cache) the chunked autograd Function. Defers torch import
    so this module stays import-light for callers that only need the
    enable_* helpers at config-validation time."""
    import torch

    def online_logsumexp_update(m, s, chunk_logits):
        """Streaming logsumexp: combine the running ``(max, sum_exp)`` state
        ``(m, s)`` with a new ``[N, k]`` chunk's logits, returning the
        updated state. Standard max-shift trick keeps every exp() in [0, 1].
        """
        chunk_m = torch.amax(chunk_logits, dim=-1)
        m_new = torch.maximum(m, chunk_m)
        exp_old = torch.where(torch.isfinite(m), torch.exp(m - m_new), torch.zeros_like(m))
        finite_chunk = torch.isfinite(chunk_logits)
        shifted = torch.where(finite_chunk, chunk_logits - m_new.unsqueeze(-1), torch.zeros_like(chunk_logits))
        chunk_exp = torch.where(finite_chunk, torch.exp(shifted), torch.zeros_like(chunk_logits))
        return m_new, s * exp_old + chunk_exp.sum(dim=-1)

    class SequenceChunkedLogProbFn(torch.autograd.Function):
        """Compute ``log_softmax(hidden @ weight.T / temperature)[label]`` per
        row of ``hidden``, tiling over (token, vocab) so peak memory is
        ``O(token_chunk_size * vocab_chunk_size)`` instead of ``O(N * V)``.

        The forward caches the per-row log-partition ``logz`` for backward;
        backward then re-materializes logits tile by tile (memory-vs-time
        tradeoff) and assembles
        ``d/d(logits) = grad_logprob * (onehot(target) - softmax(logits))``.
        """

        @staticmethod
        def forward(
            ctx,
            hidden,
            weight,
            labels,
            inv_temperature,
            bias,
            token_chunk_size: int,
            vocab_chunk_size: int,
            fp32_lm_head: bool,
            action_masks=None,
        ):
            if hidden.dim() != 2 or weight.dim() != 2:
                raise ValueError(
                    f"expected hidden [N,H] and weight [V,H], got {tuple(hidden.shape)} and {tuple(weight.shape)}"
                )
            if labels.dim() != 1 or inv_temperature.dim() != 1:
                raise ValueError(
                    "expected labels [N] and inv_temperature [N], got "
                    f"{tuple(labels.shape)} and {tuple(inv_temperature.shape)}"
                )
            n_tokens, hidden_size = hidden.shape
            vocab_size = weight.shape[0]
            if not (n_tokens == labels.shape[0] == inv_temperature.shape[0]):
                raise ValueError("hidden/labels/inv_temperature N mismatch")
            if hidden_size != weight.shape[1]:
                raise ValueError("hidden/weight H mismatch")
            if token_chunk_size <= 0 or vocab_chunk_size <= 0:
                raise ValueError("chunk sizes must be positive")

            device = hidden.device
            labels = labels.to(torch.long)
            inv_temperature = inv_temperature.to(torch.float32)

            logprobs = torch.empty(n_tokens, device=device, dtype=torch.float32)
            logz = torch.empty(n_tokens, device=device, dtype=torch.float32)

            for start in range(0, n_tokens, token_chunk_size):
                end = min(start + token_chunk_size, n_tokens)
                hidden_chunk = hidden[start:end]
                labels_chunk = labels[start:end]
                inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
                tk = end - start

                # Running (max, sum_exp) state for the online logsumexp.
                m = torch.full((tk,), float("-inf"), device=device, dtype=torch.float32)
                s = torch.zeros(tk, device=device, dtype=torch.float32)
                # Logit at each row's target label, gathered across vocab tiles.
                target_logits = torch.zeros(tk, device=device, dtype=torch.float32)
                token_masks = slice_lm_head_action_masks(action_masks, token_start=start, token_end=end)

                for vocab_start in range(0, vocab_size, vocab_chunk_size):
                    vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                    weight_chunk = weight[vocab_start:vocab_end]
                    if fp32_lm_head:
                        logits_chunk = hidden_chunk.float() @ weight_chunk.float().t()
                    else:
                        logits_chunk = hidden_chunk @ weight_chunk.t()
                    if bias is not None:
                        logits_chunk = logits_chunk + bias[vocab_start:vocab_end].to(logits_chunk.dtype)
                    scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                    apply_lm_head_action_masks_(
                        scaled_logits,
                        token_masks,
                        token_start=0,
                        vocab_start=vocab_start,
                        vocab_end=vocab_end,
                    )

                    m, s = online_logsumexp_update(m, s, scaled_logits)

                    # Pick out logits at target labels that fall in this vocab tile.
                    mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                    if torch.any(mask):
                        idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                        target_logits[mask] = scaled_logits[mask, idx]

                # log_softmax(x)[t] = x[t] - logsumexp(x). Cache logz so
                # backward can rebuild softmax without re-streaming logits.
                logz_chunk = m + torch.log(s)
                validate_action_mask_targets(labels_chunk, token_masks, token_start=start, target_logits=target_logits)
                logz[start:end] = logz_chunk
                logprobs[start:end] = target_logits - logz_chunk

            # save_for_backward only takes tensors, so signal bias presence via
            # an attribute and append bias to the saved tuple when present.
            saved: tuple[torch.Tensor, ...] = (hidden, weight, labels, inv_temperature, logz)
            if bias is not None:
                saved = saved + (bias,)
            ctx.save_for_backward(*saved)
            ctx.has_bias = bias is not None
            ctx.action_masks = action_masks
            ctx.token_chunk_size = token_chunk_size
            ctx.vocab_chunk_size = vocab_chunk_size
            ctx.fp32_lm_head = fp32_lm_head
            return logprobs

        @staticmethod
        def backward(ctx, grad_logprobs):
            saved = ctx.saved_tensors
            hidden, weight, labels, inv_temperature, logz = saved[:5]
            bias = saved[5] if ctx.has_bias else None
            token_chunk_size: int = ctx.token_chunk_size
            vocab_chunk_size: int = ctx.vocab_chunk_size
            fp32_lm_head: bool = ctx.fp32_lm_head
            action_masks = ctx.action_masks

            n_tokens = hidden.shape[0]
            vocab_size = weight.shape[0]

            # Accumulate in fp32 when requested, otherwise in each input's
            # native dtype; we cast back to input dtypes on return.
            accum = torch.float32 if fp32_lm_head else None
            grad_hidden = torch.zeros_like(hidden, dtype=accum or hidden.dtype)
            grad_weight = torch.zeros_like(weight, dtype=accum or weight.dtype)
            grad_bias = torch.zeros_like(bias, dtype=accum or bias.dtype) if bias is not None else None

            for start in range(0, n_tokens, token_chunk_size):
                end = min(start + token_chunk_size, n_tokens)
                hidden_chunk = hidden[start:end]
                labels_chunk = labels[start:end]
                grad_chunk = grad_logprobs[start:end].to(torch.float32)
                inv_t_chunk = inv_temperature[start:end].to(torch.float32).unsqueeze(-1)
                logz_chunk = logz[start:end]
                token_masks = slice_lm_head_action_masks(action_masks, token_start=start, token_end=end)

                for vocab_start in range(0, vocab_size, vocab_chunk_size):
                    vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
                    weight_chunk = weight[vocab_start:vocab_end]
                    h_proj = hidden_chunk.float() if fp32_lm_head else hidden_chunk
                    w_proj = weight_chunk.float() if fp32_lm_head else weight_chunk
                    logits_chunk = h_proj @ w_proj.t()
                    if bias is not None:
                        bias_chunk = bias[vocab_start:vocab_end]
                        logits_chunk = logits_chunk + (bias_chunk.float() if fp32_lm_head else bias_chunk)
                    scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                    apply_lm_head_action_masks_(
                        scaled_logits,
                        token_masks,
                        token_start=0,
                        vocab_start=vocab_start,
                        vocab_end=vocab_end,
                    )
                    # softmax(scaled) recovered from cached logz; this is the
                    # memory-vs-time tradeoff (recompute logits, skip storing).
                    probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                    # d log_softmax(z)[t] / d z = onehot(t) - softmax(z).
                    grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                    mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                    if torch.any(mask):
                        idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                        grad_logits[mask, idx] += grad_chunk[mask]
                    # Chain rule for temperature scaling: scaled = logits * inv_t.
                    grad_logits = grad_logits * inv_t_chunk

                    grad_hidden[start:end].add_(grad_logits.to(grad_hidden.dtype) @ w_proj)
                    grad_weight[vocab_start:vocab_end].add_(grad_logits.to(grad_weight.dtype).t() @ h_proj)
                    if grad_bias is not None:
                        grad_bias[vocab_start:vocab_end].add_(grad_logits.to(grad_bias.dtype).sum(dim=0))

            return (
                grad_hidden.to(hidden.dtype),
                grad_weight.to(weight.dtype),
                None,
                None,
                grad_bias.to(bias.dtype) if grad_bias is not None else None,
                None,
                None,
                None,
                None,
            )

    return SequenceChunkedLogProbFn


def _coerce_BS_shape(name: str, t, batch_size: int, seq_len: int):
    """Reshape ``t`` to ``[B, S]`` if its element count matches, else error."""
    if t.shape == (batch_size, seq_len):
        return t
    if t.numel() != batch_size * seq_len:
        raise ValueError(
            f"{name} shape {tuple(t.shape)} is incompatible with hidden states shape [{batch_size}, {seq_len}, *]"
        )
    return t.reshape(batch_size, seq_len)


def chunked_lm_head_logprobs(
    hidden_states,
    weight,
    labels,
    *,
    bias=None,
    temperature=None,
    token_chunk_size: int,
    vocab_chunk_size: int,
    fp32_lm_head: bool,
    action_masks=None,
):
    """Per-token logprobs at ``labels`` via tiled (token x vocab) matmuls.

    Flattens ``[B, S, *]`` to ``[B*S, *]`` so the autograd Function can iterate
    a 1-D token axis; reshapes the output back to ``[B, S]``.
    """
    import torch

    if hidden_states.dim() != 3:
        raise ValueError(f"expected hidden_states [B,S,H], got {tuple(hidden_states.shape)}")
    if labels is None:
        raise ValueError("chunked LM-head logprobs require labels")

    batch_size, seq_len, hidden_size = hidden_states.shape
    n_tokens = batch_size * seq_len
    validate_lm_head_targets(labels, vocab_size=int(weight.shape[0]))
    labels = _coerce_BS_shape("labels", labels.to(hidden_states.device), batch_size, seq_len)

    hidden_flat = hidden_states.reshape(n_tokens, hidden_size).contiguous()
    labels_flat, ignore_mask = safe_chunked_labels(labels.reshape(n_tokens).to(torch.long))
    lm_head_action_masks = filter_lm_head_action_masks(
        action_masks_to_lm_head(action_masks, device=hidden_states.device), ~ignore_mask
    )

    if temperature is None:
        inv_temperature = torch.ones(n_tokens, device=hidden_states.device, dtype=torch.float32)
    else:
        temperature = _coerce_BS_shape("temperature", temperature.to(hidden_states.device), batch_size, seq_len)
        temperature_flat = temperature.reshape(n_tokens).to(torch.float32)
        safe_temperature = temperature_flat.masked_fill(temperature_flat == 0, 1.0)
        inv_temperature = safe_temperature.reciprocal().contiguous()

    fn = _get_sequence_chunked_logprob_fn()
    logprobs = fn.apply(
        hidden_flat,
        weight,
        labels_flat,
        inv_temperature,
        bias,
        int(token_chunk_size),
        int(vocab_chunk_size),
        bool(fp32_lm_head),
        lm_head_action_masks,
    )
    return logprobs.reshape(batch_size, seq_len)


def enable_chunked_lm_head_logprobs(
    model,
    *,
    token_chunk_size: int,
    vocab_chunk_size: int,
    fp32_lm_head: bool,
) -> None:
    """Install a chunked-logprob branch on ``model.forward``.

    The patched forward keeps default-arg behavior unchanged (delegates to
    the original ``forward``), and only switches into chunked-logprob mode
    when the caller passes ``dss_compute_logprobs=True``. That lets us share
    the same ``HF AutoModelForCausalLM`` instance for both training loss
    forward/backward and RL grad-log-prob passes.
    """
    if getattr(model, "lm_head", None) is None:
        raise ValueError("fused_lm_head_token_chunk_size requires model.lm_head")
    if getattr(model, "model", None) is None:
        raise ValueError("fused_lm_head_token_chunk_size requires model.model")
    if token_chunk_size <= 0:
        raise ValueError("fused_lm_head_token_chunk_size must be positive")
    if vocab_chunk_size <= 0:
        raise ValueError("fused_lm_head_vocab_chunk_size must be positive")

    # Re-enabling on an already-patched model just updates the chunk sizes;
    # avoids stacking wrappers if a job is reconfigured.
    if getattr(model, "_dss_chunked_lm_head_logprobs", False):
        model._dss_chunked_lm_head_token_chunk_size = int(token_chunk_size)
        model._dss_chunked_lm_head_vocab_chunk_size = int(vocab_chunk_size)
        model._dss_chunked_lm_head_fp32 = bool(fp32_lm_head)
        return

    old_forward = model.forward

    def dss_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep: int = 0,
        temperature=None,
        action_masks=None,
        dss_compute_logprobs: bool = False,
        **kwargs,
    ):
        if not dss_compute_logprobs:
            # ``logits_to_keep`` is either an int count of trailing positions or a tensor of column indices,
            # so it cannot be put in a truth test: a multi-element tensor raises, and a one-element one answers
            # about the index it holds rather than about whether a selection was asked for. Only the int form
            # carries "0 means keep everything", which is the sole reason the argument is ever dropped here.
            if isinstance(logits_to_keep, int):
                if logits_to_keep:
                    kwargs["logits_to_keep"] = logits_to_keep
            elif logits_to_keep is not None:
                kwargs["logits_to_keep"] = logits_to_keep
            return old_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs,
            )

        # Labels are the caller's to supply, and cannot be reconstructed here. This forward sees one ``[B, S]``
        # tensor and cannot tell a whole row from a shard of one or from several packed rows, so the token that
        # follows the window's last position may not be present at all. Deriving targets from ``input_ids`` scores
        # that position against the window's own first token and returns a plausible number: wrong at one position
        # per window, with nothing to signal it. The whole row does exist before it is split, which is where
        # ``ensure_next_token_labels`` in ``dss/ray_dss/jobs/gpu/sp/data_plane.py`` builds them.
        if labels is None:
            raise ValueError(
                "dss_compute_logprobs requires labels: a next-token target cannot be reconstructed from "
                "input_ids in this forward, because the window it receives may end mid-row and the target "
                "for its last position is then a token it does not hold. Build labels on the whole row "
                "before it is split (ensure_next_token_labels in dss/ray_dss/jobs/gpu/sp/data_plane.py)."
            )

        # Run only the backbone — we explicitly skip the (full-vocab) lm_head
        # call that the default HF forward would do, since the whole point is
        # to compute logprobs without ever materializing [B, S, V] logits.
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        if labels.dim() != 2:
            labels = inherit_lm_head_target_validation(labels, labels.reshape(hidden_states.shape[:2]))
        if temperature is not None and temperature.dim() != 2:
            temperature = temperature.reshape(hidden_states.shape[:2])

        # ``logits_to_keep > 0`` is the HF convention for "only score the last
        # k tokens" (e.g. RL rollouts of length k); a 1D integer tensor names the kept columns directly.
        # Action-mask positions index the full sequence, so they have to move into the kept columns' space
        # before the hidden states and labels are narrowed to it.
        action_masks = slice_action_masks_for_logits_to_keep(
            action_masks,
            batch_size=int(hidden_states.shape[0]),
            original_seq_len=int(hidden_states.shape[1]),
            logits_to_keep=logits_to_keep,
        )
        if isinstance(logits_to_keep, int):
            slice_indices = slice(-logits_to_keep, None) if logits_to_keep > 0 else slice(None)
        else:
            slice_indices = logits_to_keep
        hidden_states = hidden_states[:, slice_indices, :]
        labels = inherit_lm_head_target_validation(labels, labels[:, slice_indices])
        if temperature is not None:
            temperature = temperature[:, slice_indices]

        return {
            "logprobs": chunked_lm_head_logprobs(
                hidden_states,
                self.lm_head.weight,
                labels,
                bias=getattr(self.lm_head, "bias", None),
                temperature=temperature,
                token_chunk_size=self._dss_chunked_lm_head_token_chunk_size,
                vocab_chunk_size=self._dss_chunked_lm_head_vocab_chunk_size,
                fp32_lm_head=self._dss_chunked_lm_head_fp32,
                action_masks=action_masks,
            )
        }

    model.forward = types.MethodType(dss_forward, model)
    model._dss_chunked_lm_head_old_forward = old_forward
    model._dss_chunked_lm_head_token_chunk_size = int(token_chunk_size)
    model._dss_chunked_lm_head_vocab_chunk_size = int(vocab_chunk_size)
    model._dss_chunked_lm_head_fp32 = bool(fp32_lm_head)
    model._dss_chunked_lm_head_logprobs = True


def enable_fp32_lm_head(model) -> None:
    """Replace ``model.lm_head.forward`` with an fp32 projection.

    Hidden states, weight, and bias are upcast to fp32 before the linear so
    the resulting logits — and the softmax/logprob that follow — get fp32
    precision regardless of the model's native dtype (bf16/fp16).
    """
    import torch.nn.functional as F

    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        raise ValueError("fp32_lm_head requires model.lm_head")
    old_forward = lm_head.forward

    def fp32_forward(self, hidden_states):
        bias = getattr(self, "bias", None)
        return F.linear(
            hidden_states.float(),
            self.weight.float(),
            bias.float() if bias is not None else None,
        )

    lm_head.forward = types.MethodType(fp32_forward, lm_head)
    lm_head._dss_fp32_lm_head_old_forward = old_forward
    lm_head._dss_fp32_lm_head = True

    liger_module = _liger_patched_module(model)
    if liger_module is not None:
        _enable_fp32_liger_fused_loss(liger_module)


def _liger_patched_module(model):
    """Return the ``liger_kernel`` module that owns ``model``'s forward, or ``None`` if Liger has not patched it.

    Liger replaces the causal-LM ``forward`` with one defined in ``liger_kernel``, so the defining module is what
    tells us whether the fused path is in play. Reading it off the instance covers both patch styles: loading
    through ``AutoLigerKernelForCausalLM`` rebinds the class attribute, while ``apply_liger_kernel_to_*(model=...)``
    rebinds only this instance.
    """
    forward = getattr(model, "forward", None)
    module_name = getattr(forward, "__module__", "") or ""
    if not module_name.startswith("liger_kernel"):
        return None
    return sys.modules.get(module_name)


def _enable_fp32_liger_fused_loss(module) -> None:
    """Make the Liger fused loss in ``module`` accumulate the lm_head gradient in fp32.

    While training with labels, a Liger-patched forward hands ``lm_head.weight`` to the fused cross-entropy
    kernel instead of calling ``lm_head.forward``, so upcasting the projection cannot reach the loss. That kernel
    derives its chunk count from the token count and by default accumulates the weight gradient in the weight's
    dtype, which makes the bf16 sum depend on how the tokens were split across calls; ``accum_dtype`` keeps the
    accumulator and each chunk's product in fp32 instead. See docs/lm_head_gradient_precision.md.
    """
    import torch

    loss_function = getattr(module, "LigerForCausalLMLoss", None)
    if loss_function is None:
        raise ValueError(
            f"fp32_lm_head cannot reach the fused loss: Liger module {module.__name__!r} exposes no "
            "LigerForCausalLMLoss to configure"
        )
    module.LigerForCausalLMLoss = functools.partial(loss_function, accum_dtype=torch.float32)
