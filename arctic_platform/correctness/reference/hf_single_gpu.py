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

"""Single-GPU HuggingFace forward-backward, with no parallelism and no offload.

Two properties make this usable as a golden truth for a config that needs eight GPUs.

The output loss follows the config under test. When Arctic Platform enables its chunked LM head, the reference calls
the same tiled log-probability autograd path with the same token and vocabulary tile sizes. Vocabulary
tiling changes the backward even when the scalar loss agrees, so replacing it with ordinary cross entropy
would measure the loss implementation rather than distributed training. Configs without that path use
chunked cross entropy to bound the logits.

Accumulation makes the peak independent of the global batch. The request gradient is
``sum_i (w_i / W) * grad(CE_i)``: each sequence contributes independently and backward accumulates
additively, so grouping sequences into microbatches any way at all gives the same result. This engine
therefore chooses its own microbatch size from its own memory budget, and does not reproduce the split Arctic Platform
uses. The only hard constraint is that a sequence is never split, since attention would change.

Accumulation happens in float32. With a thousand-sequence global batch the gradient receives on the order of
a thousand additions, and bfloat16 carries two to three decimal digits, so a bfloat16 accumulator would
inject error far above the third decimal place the comparison is trying to resolve.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional

import torch

IGNORE_INDEX = -100


@dataclass
class ReferenceResult:
    loss: float
    grad_norms: Dict[str, float]
    active_tokens: int
    microbatches: int
    peak_gib: float
    skipped_no_grad: List[str] = field(default_factory=list)
    optimizer_state_manifest: Optional[str] = None
    optimizer_gradient_norm: Optional[float] = None


@dataclass
class ForwardBackwardResult:
    """One request's loss and the float32 gradient accumulator it drained into."""

    loss: float
    gradients: Dict[str, "torch.Tensor"]
    active_tokens: int
    microbatches: int
    skipped_no_grad: List[str] = field(default_factory=list)


def _build_liger_fused_cross_entropy():
    """Build the same fp32-accumulating fused CE used by the Arctic Platform PrimeRL MoE head."""
    import torch
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        reduction="mean",
        accum_dtype=torch.float32,
    )


@dataclass
class ReferenceEngine:
    """A built model with the tensors the loss path reaches by identity resolved once.

    The output embedding is found through ``get_output_embeddings`` and its parameter name by identity
    against that tensor, because the attribute path differs between architectures. Resolving them per
    request would repeat a whole ``named_parameters`` scan on every step of a multi-step run.
    """

    model: object
    decoder: object
    head: object
    hidden_size: int
    head_name: Optional[str]
    device: str


def row_groups(lengths: List[int], seq_len: int, token_budget: int) -> List[List[int]]:
    """Group whole rows into microbatches under a token budget, never splitting a row."""
    groups: List[List[int]] = []
    current: List[int] = []
    for index in range(len(lengths)):
        if current and (len(current) + 1) * seq_len > token_budget:
            groups.append(current)
            current = []
        current.append(index)
    if current:
        groups.append(current)
    return groups


def build_engine(
    model_path: str,
    *,
    dtype: Optional[str] = "bfloat16",
    attn_implementation: str = "sdpa",
    peft_config: Optional[dict] = None,
    peft_adapter_path: str | None = None,
    seed: int = 0,
    matmul_precision: str = "highest",
    device: str = "cuda",
) -> ReferenceEngine:
    """Load the checkpoint, apply the config's PEFT, and leave the model ready to forward-backward.

    Separate from the request path because a multi-step run builds one engine and drives it many times.
    Loading per step would restart every trajectory from the checkpoint.
    """
    import torch
    import transformers
    from transformers import AutoConfig

    # ``high`` runs every float32 matmul on the tensor cores at a 10-bit mantissa, which is a relative
    # error near 1e-3 -- the size of the agreement being measured. The engine sets it, so the reference
    # has to be told which one it is being compared against.
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cls = getattr(transformers, cfg.architectures[0])
    model = cls.from_pretrained(model_path, dtype=torch_dtype, attn_implementation=attn_implementation).to(device)
    if peft_adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, peft_adapter_path, is_trainable=True)
    elif peft_config:
        from arctic_platform.model.patches.peft import apply_peft

        model = apply_peft(model, peft_config, optimization_dtype=torch_dtype)
    model.gradient_checkpointing_enable()
    model.train()

    head = model.get_output_embeddings()
    # Chunked cross-entropy calls F.linear against head.weight rather than the head module, so a bias would
    # be dropped silently instead of being added to every logit.
    if getattr(head, "bias", None) is not None:
        raise ValueError("reference lm_head has a bias, which the chunked projection below does not apply")
    return ReferenceEngine(
        model=model,
        decoder=model.get_decoder(),
        head=head,
        hidden_size=getattr(cfg, "text_config", cfg).hidden_size,
        # The output-embedding gradient is the one tensor every cross-entropy chunk writes to, so it is
        # drained per chunk below instead of per microbatch. Located by identity because the attribute path
        # differs between architectures.
        head_name=next((n for n, q in model.named_parameters() if q is head.weight), None),
        device=device,
    )


def mixer_boundaries(row_length: int, rows: int, device: str, *, flattened: bool = False) -> Dict[str, "torch.Tensor"]:
    """The sequence boundaries the gated delta net reads and does not build.

    A caller that omits ``seq_idx`` and ``cu_seq_lens_q`` gets the batched convolution and delta-rule
    kernels instead of the varlen ones. A row holds one sequence, so the boundaries are the row edges.
    """
    import torch

    if rows < 1:
        raise ValueError(f"mixer packing needs at least one row, got {rows}")
    boundaries = torch.arange(0, (rows + 1) * row_length, row_length, dtype=torch.int32, device=device)
    seq_idx = torch.arange(rows, dtype=torch.int32, device=device).repeat_interleave(row_length)
    seq_idx_shape = (1, rows * row_length) if flattened else (rows, row_length)
    return {
        "seq_idx": seq_idx.reshape(seq_idx_shape),
        "cu_seq_lens_q": boundaries,
    }


def mixer_call_inputs(input_ids, labels, device: str):
    """Prepare an opt-in mixer-packed reference call.

    Qwen3.6's varlen gated-delta path requires flattened inputs when a single model call carries more
    than one row segment. The default one-row reference path keeps the historical 2-D shape.
    """
    rows_here, row_len = input_ids.shape
    flatten_varlen = rows_here > 1
    mixer_kwargs = mixer_boundaries(row_len, rows_here, device, flattened=flatten_varlen)
    if flatten_varlen:
        return (
            input_ids.reshape(1, rows_here * row_len),
            labels.reshape(1, rows_here * row_len),
            mixer_kwargs,
        )
    return input_ids, labels, mixer_kwargs


def fixed_row_groups(lengths: List[int], group_rows: int) -> List[List[int]]:
    """Group whole rows by count, preserving order."""
    if group_rows < 1:
        raise ValueError(f"group_rows must be positive, got {group_rows}")
    return [list(range(start, min(start + group_rows, len(lengths)))) for start in range(0, len(lengths), group_rows)]


def reference_row_groups(
    lengths,
    row_length: int,
    token_budget: int,
    *,
    mixer_packing: bool,
    mixer_packing_group_rows: int = 1,
):
    """Choose reference model-call groups without changing the default golden path.

    Hybrid-mixer references historically used one row per model call, while Arctic Platform can hand the
    mixer several row segments in one packed call. ``mixer_packing_group_rows`` is an opt-in diagnostic knob
    for matching that call shape; production correctness keeps the default of one row.
    """
    if mixer_packing:
        return fixed_row_groups(lengths, mixer_packing_group_rows)
    return row_groups(lengths, row_length, token_budget)


def forward_backward(
    engine: ReferenceEngine,
    batch,
    *,
    token_budget: int = 65536,
    ce_chunk: int = 2048,
    fp32_lm_head: bool = False,
    fused_cross_entropy: bool | str = False,
    mixer_packing: bool = False,
    mixer_packing_group_rows: int = 1,
    lm_head_token_chunk_size: int | None = None,
    lm_head_vocab_chunk_size: int = 8192,
    accumulate_dtype: "torch.dtype | None" = None,
) -> ForwardBackwardResult:
    """Forward-backward one whole request and return its loss beside a float32 gradient accumulator.

    The model's own gradient buffers are drained and cleared before returning, so the caller receives the
    request's gradient and the engine is left ready for the next request.
    """
    import torch
    import torch.nn.functional as F

    from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs

    model, decoder, head = engine.model, engine.decoder, engine.head
    hidden_size, head_name, device = engine.hidden_size, engine.head_name, engine.device

    fused_ce = None
    if fused_cross_entropy:
        if fused_cross_entropy not in {True, "liger"}:
            raise ValueError(f"reference does not support fused_cross_entropy={fused_cross_entropy!r}")
        if lm_head_token_chunk_size is not None:
            raise ValueError("fused cross-entropy and chunked LM-head logprobs are mutually exclusive")
        fused_ce = _build_liger_fused_cross_entropy()

    input_ids = batch.input_ids.to(device)
    # The batch carries HuggingFace-convention labels. The Arctic Platform driver pre-shifts them for PrimeRL before
    # packing; apply the identical shift here so both engines score the same targets. Without this the two optimize
    # against positions one apart, which a randomly-initialized model hides almost perfectly: every target
    # is equally unpredictable, so the gradient changes while its norm barely moves.
    labels = batch.shifted_labels().to(device)
    seq_len = int(input_ids.shape[1])

    # The request-wide denominator. Every microbatch divides by this same total, which is what makes the
    # grouping below irrelevant to the result.
    active_tokens = int((labels != IGNORE_INDEX).sum())
    if active_tokens == 0:
        raise ValueError("batch has no active tokens")

    # float32 unless a caller is deliberately measuring what a narrower accumulator costs.
    accum_dtype = accumulate_dtype or torch.float32
    accumulator: Dict[str, "torch.Tensor"] = {}
    total_loss = 0.0

    lengths = [seq_len] * int(input_ids.shape[0])
    groups = reference_row_groups(
        lengths,
        seq_len,
        token_budget,
        mixer_packing=mixer_packing,
        mixer_packing_group_rows=mixer_packing_group_rows,
    )
    for rows in groups:
        index = torch.tensor(rows, device=device)
        ids_mb = input_ids.index_select(0, index)
        labels_mb = labels.index_select(0, index)

        mixer_kwargs = {}
        if mixer_packing:
            ids_mb, labels_mb, mixer_kwargs = mixer_call_inputs(ids_mb, labels_mb, device)
        hidden = decoder(input_ids=ids_mb, use_cache=False, **mixer_kwargs).last_hidden_state
        detached = hidden.detach().requires_grad_(True)
        flat_labels = labels_mb.reshape(-1)

        if lm_head_token_chunk_size is not None:
            logprobs = chunked_lm_head_logprobs(
                detached,
                head.weight,
                labels_mb,
                bias=getattr(head, "bias", None),
                token_chunk_size=lm_head_token_chunk_size,
                vocab_chunk_size=lm_head_vocab_chunk_size,
                fp32_lm_head=fp32_lm_head,
            )
            valid = labels_mb != IGNORE_INDEX
            loss = -torch.where(valid, logprobs.float(), 0.0).sum() / active_tokens
            loss.backward()
            total_loss += float(loss.item())
        elif fused_ce is not None:
            valid_tokens = int((flat_labels != IGNORE_INDEX).sum())
            if valid_tokens == 0:
                continue
            # Arctic Platform's Liger head returns a mean for each model call, then the training loop weights that mean by
            # the call's share of request-wide active tokens. Use the same fused kernel and weighting here.
            mean_loss = fused_ce(head.weight, detached.reshape(-1, hidden_size), flat_labels)
            loss = mean_loss * (valid_tokens / active_tokens)
            loss.backward()
            total_loss += float(loss.item())
        else:
            for start in range(0, flat_labels.numel(), ce_chunk):
                chunk_labels = flat_labels[start : start + ce_chunk]
                if int((chunk_labels != IGNORE_INDEX).sum()) == 0:
                    continue
                chunk_hidden = detached.view(-1, hidden_size)[start : start + ce_chunk]
                # ``fp32_lm_head`` widens the projection's operands, not just its output. The difference reaches
                # every gradient below the head, so the reference applies the same operand conversion.
                if fp32_lm_head:
                    logits = F.linear(chunk_hidden.float(), head.weight.float())
                else:
                    logits = F.linear(chunk_hidden, head.weight).float()
                loss = (
                    F.cross_entropy(logits, chunk_labels, ignore_index=IGNORE_INDEX, reduction="sum") / active_tokens
                )
                loss.backward()
                total_loss += float(loss.item())

                # Every cross-entropy chunk adds to this one gradient, so leaving the running sum in the
                # parameter's bfloat16 buffer would round each addition. The production chunked LM head has
                # one autograd invocation and is drained once below instead.
                if head_name is not None and head.weight.grad is not None:
                    if head_name not in accumulator:
                        accumulator[head_name] = head.weight.grad.detach().to(device="cpu", dtype=accum_dtype)
                    else:
                        accumulator[head_name].add_(head.weight.grad.detach().to(device="cpu", dtype=accum_dtype))
                    head.weight.grad = None

        hidden.backward(detached.grad)

        # Drain bfloat16 gradients into the float32 accumulator after every microbatch, so the running sum
        # never lives in the parameter dtype.
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if name not in accumulator:
                accumulator[name] = param.grad.detach().to(device="cpu", dtype=accum_dtype)
            else:
                accumulator[name].add_(param.grad.detach().to(device="cpu", dtype=accum_dtype))
        model.zero_grad(set_to_none=True)

    return ForwardBackwardResult(
        loss=total_loss,
        gradients=accumulator,
        active_tokens=active_tokens,
        microbatches=len(groups),
        skipped_no_grad=[name for name, _ in model.named_parameters() if name not in accumulator],
    )


def gradient_norms(gradients: Dict[str, "torch.Tensor"], device: str) -> Dict[str, float]:
    """One L2 norm per accumulated gradient.

    Keeping the request-wide accumulator on CPU prevents it from competing with the longest sequence
    activation graph. One completed tensor at a time moves back to CUDA so norm reduction semantics still
    match the GPU-side Arctic Platform metric.
    """
    import torch

    return {
        name: float(torch.linalg.vector_norm(tensor.to(device=device)).cpu()) for name, tensor in gradients.items()
    }


def run(
    model_path: str,
    batch,
    *,
    token_budget: int = 65536,
    ce_chunk: int = 2048,
    dtype: Optional[str] = "bfloat16",
    attn_implementation: str = "sdpa",
    fp32_lm_head: bool = False,
    fused_cross_entropy: bool | str = False,
    device: str = "cuda",
    accumulate_dtype: "torch.dtype | None" = None,
    mixer_packing: bool = False,
    mixer_packing_group_rows: int = 1,
    matmul_precision: str = "highest",
    peft_config: Optional[dict] = None,
    peft_adapter_path: str | None = None,
    seed: int = 0,
    lm_head_token_chunk_size: int | None = None,
    lm_head_vocab_chunk_size: int = 8192,
    optimizer_config: dict | None = None,
    learning_rate: float | None = None,
    gradient_clipping: float | None = None,
    optimizer_dtype: str = "float32",
    optimizer_output_dir: str | None = None,
) -> ReferenceResult:
    """Forward-backward the whole request and return one L2 gradient norm per parameter."""
    import torch

    engine = build_engine(
        model_path,
        dtype=dtype,
        attn_implementation=attn_implementation,
        peft_config=peft_config,
        peft_adapter_path=peft_adapter_path,
        seed=seed,
        matmul_precision=matmul_precision,
        device=device,
    )
    model = engine.model
    torch.cuda.reset_peak_memory_stats(device)
    forward = forward_backward(
        engine,
        batch,
        token_budget=token_budget,
        ce_chunk=ce_chunk,
        fp32_lm_head=fp32_lm_head,
        fused_cross_entropy=fused_cross_entropy,
        mixer_packing=mixer_packing,
        mixer_packing_group_rows=mixer_packing_group_rows,
        lm_head_token_chunk_size=lm_head_token_chunk_size,
        lm_head_vocab_chunk_size=lm_head_vocab_chunk_size,
        accumulate_dtype=accumulate_dtype,
    )
    accumulator = forward.gradients
    skipped = forward.skipped_no_grad
    grad_norms = gradient_norms(accumulator, device)
    optimizer_state_manifest = None
    optimizer_gradient_norm = None
    if optimizer_config is not None:
        if learning_rate is None or optimizer_output_dir is None:
            raise ValueError("optimizer_config requires learning_rate and optimizer_output_dir")
        from arctic_platform.correctness.reference.optimizer_step import run_adamw_step

        optimizer_artifact = run_adamw_step(
            model,
            accumulator,
            optimizer_config,
            learning_rate=learning_rate,
            gradient_clipping=gradient_clipping,
            optimizer_dtype=optimizer_dtype,
            output_dir=Path(optimizer_output_dir),
        )
        optimizer_state_manifest = str(optimizer_artifact.manifest_path)
        optimizer_gradient_norm = optimizer_artifact.gradient_norm_before_clip

    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    del engine, model, accumulator
    torch.cuda.empty_cache()
    return ReferenceResult(
        loss=forward.loss,
        grad_norms=grad_norms,
        active_tokens=forward.active_tokens,
        microbatches=forward.microbatches,
        peak_gib=peak,
        skipped_no_grad=skipped,
        optimizer_state_manifest=optimizer_state_manifest,
        optimizer_gradient_norm=optimizer_gradient_norm,
    )
