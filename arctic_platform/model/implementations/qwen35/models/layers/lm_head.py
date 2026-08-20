from __future__ import annotations

import types
from typing import TypedDict

import torch
import torch.nn as nn
from torch import Tensor

from ...gpu.action_masks import (
    action_masks_to_lm_head,
    apply_lm_head_action_masks_,
    validate_action_mask_targets,
)

from ...logging_utils import get_logger

FUSED_CE_IGNORE_INDEX = -100


class PrimeLmOutput(TypedDict, total=False):
    """Output from LM head - a TypedDict so pytree can find tensors for FSDP2 hooks."""

    logits: Tensor | None
    logprobs: Tensor | None
    entropy: Tensor | None
    loss: Tensor | None


def _zero_lm_head_loss(hidden_states: Tensor, weight: Tensor) -> Tensor:
    hidden_zero = hidden_states.reshape(-1)[0].float() * 0.0
    weight_zero = weight.reshape(-1)[0].float() * 0.0
    return hidden_zero + weight_zero


def cast_float_and_contiguous(output: PrimeLmOutput) -> PrimeLmOutput:
    """Convert tensors in PrimeLmOutput to float and make contiguous."""

    def _float_and_contiguous(tensor: Tensor | None) -> Tensor | None:
        return tensor.float().contiguous() if tensor is not None else None

    return PrimeLmOutput(
        logits=_float_and_contiguous(output.get("logits")),
        logprobs=_float_and_contiguous(output.get("logprobs")),
        entropy=_float_and_contiguous(output.get("entropy")),
        loss=output.get("loss"),
    )


class FusedOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, chunk_size: int, fp32_lm_head: bool = False):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size
        self.fp32_lm_head = fp32_lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        action_masks: object | None = None,
        dss_force_zero_loss: bool = False,
    ) -> PrimeLmOutput:
        if dss_force_zero_loss:
            return PrimeLmOutput(loss=_zero_lm_head_loss(hidden_states, self.weight))
        assert labels is not None, "FusedOutputLinear requires labels for chunked logprob computation"
        assert temperature is not None, "FusedOutputLinear requires per-token temperatures"

        b, s, h = hidden_states.shape
        hidden_states = hidden_states.reshape(b * s, h).contiguous()
        labels = labels.reshape(b * s).contiguous()
        temperature = temperature.reshape(b * s).contiguous()
        safe_temperature = temperature.masked_fill(temperature == 0, 1.0)
        inv_t = safe_temperature.reciprocal()  # [N]
        lm_head_action_masks = action_masks_to_lm_head(action_masks, device=hidden_states.device)

        logprobs, entropy = _SequenceChunkedLogProbEntropyFn.apply(
            hidden_states,
            self.weight,
            labels,
            inv_t,
            self.chunk_size,
            self.fp32_lm_head,
            lm_head_action_masks,
        )

        logprobs = logprobs.reshape(b, s)
        entropy = entropy.reshape(b, s)
        return PrimeLmOutput(logprobs=logprobs, entropy=entropy)


class VanillaOutputLinear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, fp32_lm_head: bool = False):
        super().__init__(in_features, out_features, bias=False)
        self.fp32_lm_head = fp32_lm_head

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        dss_force_zero_loss: bool = False,
    ) -> PrimeLmOutput:
        if dss_force_zero_loss:
            return PrimeLmOutput(loss=_zero_lm_head_loss(hidden_states, self.weight))
        # VanillaOutputLinear just returns logits - temperature scaling is done externally in train.py
        if self.fp32_lm_head:
            return PrimeLmOutput(logits=nn.functional.linear(hidden_states.float(), self.weight.float()))
        return PrimeLmOutput(logits=super().forward(hidden_states))


class FusedCrossEntropyOutputLinear(torch.nn.Linear):
    """Fused lm_head + cross-entropy loss using Liger kernel.

    Avoids materializing the full [N, V] logits tensor by fusing the linear
    projection with the cross-entropy loss computation.
    """

    IGNORE_INDEX = FUSED_CE_IGNORE_INDEX

    def __init__(self, in_features: int, out_features: int, softcap: float | None = None):
        super().__init__(in_features, out_features, bias=False)
        from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

        self.fused_ce = LigerFusedLinearCrossEntropyLoss(
            ignore_index=self.IGNORE_INDEX, reduction="mean", softcap=softcap
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        dss_force_zero_loss: bool = False,
    ) -> PrimeLmOutput:
        if dss_force_zero_loss:
            return PrimeLmOutput(loss=_zero_lm_head_loss(hidden_states, self.weight))
        if labels is None:
            return PrimeLmOutput(logits=super().forward(hidden_states))

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        labels_flat = labels.reshape(b * s).contiguous()
        loss = self.fused_ce(self.weight, hidden_flat, labels_flat)
        return PrimeLmOutput(loss=loss)


class QuackFusedCrossEntropyOutputLinear(torch.nn.Linear):
    """Fused lm_head + cross-entropy loss using quack-kernels.

    Chunks the linear projection and cross-entropy computation to avoid
    materializing the full [N, V] logits tensor, using quack's optimized
    CuTe DSL kernels for CE and GEMM.
    """

    IGNORE_INDEX = FUSED_CE_IGNORE_INDEX

    def __init__(self, in_features: int, out_features: int, chunk_size: int = 4096):
        super().__init__(in_features, out_features, bias=False)
        self.chunk_size = chunk_size

    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor | None = None,
        temperature: Tensor | None = None,
        dss_force_zero_loss: bool = False,
    ) -> PrimeLmOutput:
        if dss_force_zero_loss:
            return PrimeLmOutput(loss=_zero_lm_head_loss(hidden_states, self.weight))
        if labels is None:
            return PrimeLmOutput(logits=super().forward(hidden_states))

        from quack.linear_cross_entropy import chunked_linear_cross_entropy

        b, s, h = hidden_states.shape
        hidden_flat = hidden_states.reshape(b * s, h).contiguous()
        labels_flat = labels.reshape(b * s).contiguous()
        loss = chunked_linear_cross_entropy(
            hidden_flat,
            self.weight,
            labels_flat,
            chunk_size=self.chunk_size,
            ignore_index=self.IGNORE_INDEX,
            reduction="mean",
        )
        return PrimeLmOutput(loss=loss)


def _online_logsumexp_and_weighted_update(
    m: torch.Tensor, s: torch.Tensor, t: torch.Tensor, chunk_logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    chunk_m = torch.amax(chunk_logits, dim=-1)
    m_new = torch.maximum(m, chunk_m)
    exp_old = torch.where(torch.isfinite(m), torch.exp(m - m_new), torch.zeros_like(m))
    finite_chunk = torch.isfinite(chunk_logits)
    shifted = torch.where(finite_chunk, chunk_logits - m_new.unsqueeze(-1), torch.zeros_like(chunk_logits))
    chunk_exp = torch.where(finite_chunk, torch.exp(shifted), torch.zeros_like(chunk_logits))
    weighted_logits = torch.where(finite_chunk, chunk_logits, torch.zeros_like(chunk_logits))
    s_new = s * exp_old + chunk_exp.sum(dim=-1)
    t_new = t * exp_old + (chunk_exp * weighted_logits).sum(dim=-1)
    return m_new, s_new, t_new


class _SequenceChunkedLogProbEntropyFn(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        labels: torch.Tensor,  # [N]
        inv_temperature: torch.Tensor,  # [N]
        chunk_size: int,
        fp32_lm_head: bool,
        action_masks=None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns per-token logprobs and entropy by chunking over flattened sequence tokens.
        """
        assert hidden.dim() == 2, f"expected hidden [N,H], got {tuple(hidden.shape)}"
        assert weight.dim() == 2, f"expected weight [V,H], got {tuple(weight.shape)}"
        assert labels.dim() == 1, f"expected labels [N], got {tuple(labels.shape)}"
        assert inv_temperature.dim() == 1, f"expected inv_temperature [N], got {tuple(inv_temperature.shape)}"
        assert hidden.shape[0] == labels.shape[0], "hidden/labels N mismatch"
        assert hidden.shape[1] == weight.shape[1], "hidden/weight H mismatch"
        assert hidden.shape[0] == inv_temperature.shape[0], "hidden/inv_temperature N mismatch"
        assert chunk_size > 0

        device = hidden.device
        n = hidden.shape[0]
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)
        logprobs = torch.empty((n,), device=device, dtype=torch.float32)
        entropy = torch.empty((n,), device=device, dtype=torch.float32)
        logz = torch.empty((n,), device=device, dtype=torch.float32)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            token_count = end - start

            m = torch.full((token_count,), float("-inf"), device=device, dtype=torch.float32)
            s = torch.zeros((token_count,), device=device, dtype=torch.float32)
            t = torch.zeros((token_count,), device=device, dtype=torch.float32)
            target_logits = torch.zeros((token_count,), device=device, dtype=torch.float32)

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                if fp32_lm_head:
                    logits_chunk = hidden_chunk.float() @ weight_chunk.float().t()
                else:
                    logits_chunk = hidden_chunk @ weight_chunk.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                apply_lm_head_action_masks_(
                    scaled_logits,
                    action_masks,
                    token_start=start,
                    vocab_start=vocab_start,
                    vocab_end=vocab_end,
                )

                m, s, t = _online_logsumexp_and_weighted_update(m, s, t, scaled_logits)

                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    target_logits[mask] = scaled_logits[mask, idx]

            logz_chunk = m + torch.log(s)
            validate_action_mask_targets(labels_chunk, action_masks, token_start=start, target_logits=target_logits)
            logz[start:end] = logz_chunk
            logprobs[start:end] = target_logits - logz_chunk
            entropy[start:end] = logz_chunk - (t / s)

        ctx.save_for_backward(hidden, weight, labels, inv_temperature, logz)
        ctx.action_masks = action_masks
        ctx.chunk_size = chunk_size
        ctx.fp32_lm_head = fp32_lm_head

        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs: torch.Tensor, grad_entropy: torch.Tensor | None):
        assert grad_entropy is None or torch.all(grad_entropy == 0.0), (
            "Backward through entropy is not implemented in FusedOutputLinear"
        )

        hidden, weight, labels, inv_temperature, logz = ctx.saved_tensors
        action_masks = ctx.action_masks
        chunk_size: int = ctx.chunk_size
        fp32_lm_head: bool = ctx.fp32_lm_head

        n, _ = hidden.shape
        vocab = weight.shape[0]
        vocab_chunk_size = min(vocab, 8192)

        grad_hidden = torch.zeros_like(hidden, dtype=torch.float32 if fp32_lm_head else hidden.dtype)
        grad_weight = torch.zeros_like(weight, dtype=torch.float32 if fp32_lm_head else weight.dtype)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            hidden_chunk = hidden[start:end]
            labels_chunk = labels[start:end]
            grad_chunk = grad_logprobs[start:end].to(torch.float32)
            inv_t_chunk = inv_temperature[start:end].unsqueeze(-1)
            logz_chunk = logz[start:end]

            for vocab_start in range(0, vocab, vocab_chunk_size):
                vocab_end = min(vocab_start + vocab_chunk_size, vocab)
                weight_chunk = weight[vocab_start:vocab_end]
                hidden_for_logits = hidden_chunk.float() if fp32_lm_head else hidden_chunk
                weight_for_logits = weight_chunk.float() if fp32_lm_head else weight_chunk
                logits_chunk = hidden_for_logits @ weight_for_logits.t()
                scaled_logits = logits_chunk.to(torch.float32) * inv_t_chunk
                apply_lm_head_action_masks_(
                    scaled_logits,
                    action_masks,
                    token_start=start,
                    vocab_start=vocab_start,
                    vocab_end=vocab_end,
                )
                probs = torch.exp(scaled_logits - logz_chunk.unsqueeze(-1))

                grad_logits = (-grad_chunk).unsqueeze(-1) * probs
                mask = (labels_chunk >= vocab_start) & (labels_chunk < vocab_end)
                if torch.any(mask):
                    idx = (labels_chunk[mask] - vocab_start).to(torch.long)
                    grad_logits[mask, idx] += grad_chunk[mask]
                grad_logits = grad_logits * inv_t_chunk

                grad_logits_for_hidden = grad_logits if fp32_lm_head else grad_logits.to(hidden.dtype)
                grad_logits_for_weight = grad_logits if fp32_lm_head else grad_logits.to(weight.dtype)
                grad_hidden[start:end].add_(grad_logits_for_hidden @ weight_for_logits)
                grad_weight[vocab_start:vocab_end].add_(grad_logits_for_weight.t() @ hidden_for_logits)

        return grad_hidden.to(hidden.dtype), grad_weight.to(weight.dtype), None, None, None, None, None


def inject_prime_lm_head(
    model: nn.Module,
    chunk_size: int | None = None,
    fused_cross_entropy: bool | str = False,
    fp32_lm_head: bool = False,
) -> None:
    """
    Inject a PrimeRL LM head into a model.

    This replaces the model's lm_head and overrides the forward method to use labels
    and temperature for chunked loss computation.

    Args:
        model: The model to wrap.
        chunk_size: When set to an int, uses FusedOutputLinear with sequence-token chunked
            logprob/entropy computation (for RL).
        fused_cross_entropy: Controls fused lm_head + CE loss. Accepts:
            - False: no fusion
            - True or "liger": Liger kernel fusion
            - "quack": quack-kernels fusion (chunked linear + CE with CuTe DSL kernels)
    """
    # Guards so we have nicer error messages when a non-standard model is used
    assert hasattr(model, "model"), f"model doesnt have backbone in model.model:\n{model}"
    assert isinstance(model.model, nn.Module), f"model.model is not a nn.Module: {type(model.model)}\n{model}"
    assert hasattr(model, "lm_head"), f"model doesnt have lm_head in model.lm_head:\n{model}"
    assert isinstance(model.lm_head, nn.Linear), f"model.lm_head is not a nn.Linear: {type(model.lm_head)}\n{model}"
    assert not hasattr(model.lm_head, "bias") or model.lm_head.bias is None, (
        f"model.lm_head.bias is not supported: {model.lm_head}\n{model}"
    )

    logger = get_logger()

    # Gemma-style softcapping is not supported by this carved-out package (Qwen3.5
    # has no final_logit_softcapping). Fail loudly rather than silently misbehaving.
    final_logit_softcapping = getattr(model.config, "final_logit_softcapping", None)
    if final_logit_softcapping:
        if fused_cross_entropy == "quack":
            raise ValueError("quack_fused does not support Gemma logit softcapping.")
        if not fused_cross_entropy:
            raise NotImplementedError(
                "Gemma-style final_logit_softcapping is not supported in the carved-out qwen35 "
                "package; use fused_cross_entropy='liger'."
            )

    # Replace the lm_head with the appropriate wrapper
    old_lm_head = model.lm_head
    if fused_cross_entropy == "quack":
        if fp32_lm_head:
            raise ValueError("fp32_lm_head is not supported with fused_cross_entropy='quack'")
        logger.info("Injecting fused cross-entropy LM head (quack-kernels)")
        model.lm_head = QuackFusedCrossEntropyOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
        )
    elif fused_cross_entropy:
        if fp32_lm_head:
            raise ValueError("fp32_lm_head is not supported with fused_cross_entropy=True/'liger'")
        logger.info("Injecting fused cross-entropy LM head (Liger kernel)")
        model.lm_head = FusedCrossEntropyOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            softcap=final_logit_softcapping,
        )
    elif isinstance(chunk_size, int):
        logger.info(
            f"Injecting chunked LM head with chunk size {chunk_size}" + (" and float32 matmul" if fp32_lm_head else "")
        )
        model.lm_head = FusedOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            chunk_size=chunk_size,
            fp32_lm_head=fp32_lm_head,
        )
    else:
        logger.info("Injecting vanilla LM head" + (" with float32 matmul" if fp32_lm_head else ""))
        model.lm_head = VanillaOutputLinear(
            in_features=old_lm_head.in_features,
            out_features=old_lm_head.out_features,
            fp32_lm_head=fp32_lm_head,
        )
    model.lm_head.weight = old_lm_head.weight
    del old_lm_head

    _patch_model_forward(model)


def _patch_model_forward(model: nn.Module) -> None:
    # Patch the forward method to use the new lm_head with labels and temperature
    def new_forward(
        self: nn.Module,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        logits_to_keep: int = 0,
        temperature: torch.Tensor | None = None,
        action_masks: object | None = None,
        dss_compute_logprobs: bool = False,
        dss_force_zero_loss: bool = False,
        use_cache: bool | None = None,
        **kwargs: object,
    ) -> PrimeLmOutput:
        if use_cache not in (None, False):
            raise ValueError("use_cache=True is not supported by the custom Qwen3.5 LM head")
        # For VLM with images, don't create position_ids - let model compute MRoPE internally
        is_multimodal = kwargs.get("pixel_values") is not None
        if position_ids is None and not is_multimodal:
            reference_tensor = input_ids if input_ids is not None else inputs_embeds
            position_ids = torch.arange(1, reference_tensor.shape[1] + 1, device=reference_tensor.device).unsqueeze(0)
        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # Slice hidden states for logits_to_keep
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) and logits_to_keep > 0 else slice(None)
        )

        # action_masks is RL-only (chunked-logprob head); SFT/CE heads reject it, so forward only when set.
        lm_head_kwargs = {}
        if action_masks is not None:
            lm_head_kwargs["action_masks"] = action_masks
        return self.lm_head(
            hidden_states[:, slice_indices, :],
            labels[:, slice_indices] if labels is not None else None,
            temperature=temperature[:, slice_indices] if temperature is not None else None,
            dss_force_zero_loss=dss_force_zero_loss,
            **lm_head_kwargs,
        )

    # Bind the new forward to the model
    model.forward = types.MethodType(new_forward, model)
