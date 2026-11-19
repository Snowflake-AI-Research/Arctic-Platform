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

"""Single-GPU HuggingFace forward-backward, with no parallelism.

Qwen4's frozen PLE lookup table stays on CPU; its upstream lookup returns the selected rows to CUDA.

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

import hashlib
import inspect
import json
import marshal
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import CodeType
from typing import Dict
from typing import List
from typing import Optional

import torch

from arctic_platform.correctness.harness.storage import require_data_fast_path
from arctic_platform.correctness.reference.glm5_next_reference import GLM5_NEXT_MODEL_TYPES
from arctic_platform.correctness.reference.glm5_next_reference import load_glm5_next_reference
from arctic_platform.correctness.reference.qwen3_5_moe_reference import QWEN3_5_MOE_MODEL_TYPES
from arctic_platform.correctness.reference.qwen3_5_moe_reference import load_qwen3_5_moe_reference

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
    reference_execution: Optional[dict] = None
    participation: Optional[dict] = None


@dataclass
class ForwardBackwardResult:
    """One request's loss and the float32 gradient accumulator it drained into."""

    loss: float
    gradients: Dict[str, "torch.Tensor"]
    active_tokens: int
    microbatches: int
    skipped_no_grad: List[str] = field(default_factory=list)
    participation: Optional[dict] = None


def _build_liger_fused_cross_entropy(*, fp32_lm_head: bool):
    """Use the DSS reference's explicit LM-head precision choice."""
    import torch
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

    return LigerFusedLinearCrossEntropyLoss(
        ignore_index=IGNORE_INDEX,
        reduction="mean",
        accum_dtype=torch.float32 if fp32_lm_head else None,
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
    # Gemma's final_logit_softcapping, which the Transformers forward applies to the logits and the target's chunked
    # head too (arctic_platform/model/implementations/gpu/lm_head.py _final_logit_softcapping); the loss paths below
    # project the head themselves, so they apply it.
    softcap: Optional[float] = None


def execution_witness(engine: ReferenceEngine, *, requested_attention: str) -> dict:
    """Describe the actual loaded model and decoder, without reading weight tensors."""
    from arctic_platform.correctness.harness.storage import file_identity

    def portable_code(code):
        # Python embeds checkout paths in nested code objects; exclude those paths from the code digest.
        return code.replace(
            co_filename="",
            co_consts=tuple(
                portable_code(value) if isinstance(value, CodeType) else value for value in code.co_consts
            ),
        )

    def implementation(module):
        cls = type(module)
        forward = getattr(module.forward, "__func__", module.forward)
        code = getattr(forward, "__code__", None)
        if not isinstance(code, CodeType):
            raise ValueError("HF reference execution witness requires an inspectable Python forward")
        source = Path(code.co_filename)
        class_source = inspect.getsourcefile(cls)
        if class_source is None:
            raise ValueError("HF reference execution witness requires an inspectable Python model class")
        return {
            "class": {
                "module": cls.__module__,
                "qualname": cls.__qualname__,
                "source": file_identity(class_source, refresh=True),
            },
            "forward": {
                "module": forward.__module__,
                "qualname": forward.__qualname__,
                "source": file_identity(source, refresh=True),
                "python_code_sha256": hashlib.sha256(marshal.dumps(portable_code(code))).hexdigest(),
            },
        }

    def configuration(module):
        cfg = module.config
        # HF's to_dict may omit the effective attention backend; retain it explicitly.
        value = {
            "config": cfg.to_dict(),
            "attention_implementation": getattr(cfg, "_attn_implementation", None),
        }
        # A Mamba-2 config bounds its time step by time_step_limit = (0.0, inf) (Nemotron-H); a non-finite value is
        # recorded by its JSON extension name ("Infinity") so the witness stays strict JSON.
        return json.loads(json.dumps(value, sort_keys=True), parse_constant=str)

    configurations = {"model": configuration(engine.model), "decoder": configuration(engine.decoder)}
    return {
        "schema": "hf-reference-execution-v1",
        "requested_attention": requested_attention,
        "implementations": {"model": implementation(engine.model), "decoder": implementation(engine.decoder)},
        "effective_configurations": configurations,
        "effective_configurations_sha256": (
            hashlib.sha256(
                json.dumps(configurations, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
            ).hexdigest()
        ),
    }


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


# FlashAttention 3's largest head size ("FlashAttention forward only supports head dimension at most 256").
FLASH_ATTENTION_3_MAX_HEAD_DIM = 256


def route_wide_heads_to_sdpa() -> None:
    """Run the attention calls FlashAttention 3 cannot, head sizes above 256, on torch's memory-efficient SDPA.

    Gemma 4's sliding layers are 256 wide and its full-attention layers 512. The target splits them the same way:
    DSS's heterogeneous Ulysses keeps the 256-wide layers on FlashAttention and sends the 512-wide ones to Arctic
    Platform's TileLang head-512 kernel (dss/ray_dss/jobs/gpu/sp/heterogeneous_ulysses.py _wide_head_attention_forward,
    arctic_platform/model/attention_hd512.py). The reference takes SDPA's memory-efficient kernel for them, an
    independent implementation with the same contract (bf16 operands, fp32 accumulation and softmax), and never its
    math backend. Calls of 256 or less reach FlashAttention 3 unchanged.
    """
    from torch.nn.attention import SDPBackend
    from torch.nn.attention import sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    flash = ALL_ATTENTION_FUNCTIONS["flash_attention_3"]
    if getattr(flash, "routes_wide_heads_to_sdpa", False):
        return

    def attention(module, query, key, value, attention_mask, **kwargs):
        if query.shape[-1] <= FLASH_ATTENTION_3_MAX_HEAD_DIM:
            return flash(module, query, key, value, attention_mask, **kwargs)
        unsupported = [
            name
            for name in ("sliding_window", "softcap", "cu_seq_lens_q", "cu_seq_lens_k")
            if kwargs.get(name) is not None
        ]
        if attention_mask is not None or unsupported or not getattr(module, "is_causal", True):
            raise ValueError(
                f"the reference's head-{query.shape[-1]} SDPA covers unpadded causal attention only; this call has "
                f"{'an attention mask' if attention_mask is not None else ', '.join(unsupported) or 'no causal mask'}"
            )
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
            return sdpa_attention_forward(module, query, key, value, None, **kwargs)

    attention.routes_wide_heads_to_sdpa = True
    ALL_ATTENTION_FUNCTIONS["flash_attention_3"] = attention


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
    huggingface_loader: bool = False,
) -> ReferenceEngine:
    """Load the checkpoint, apply the config's PEFT, and leave the model ready to forward-backward.

    Separate from the request path because a multi-step run builds one engine and drives it many times.
    Loading per step would restart every trajectory from the checkpoint.

    ``huggingface_loader`` builds the class the target's huggingface loader builds, ``AutoModelForCausalLM``'s for
    the checkpoint (``LoadedConfig.huggingface_loader``), so both train the same parameters; otherwise the class
    ``architectures[0]`` names.
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
    if attn_implementation == "flash_attention_3":
        route_wide_heads_to_sdpa()
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cls = getattr(transformers, cfg.architectures[0])
    if getattr(cfg, "model_type", None) in {"qwen4_exp", "qwen4_exp_text"}:
        # Qwen3.8-Flash-Next at the sampler's precision (FP32 hyper-connection mix and combine, GatedDeltaNet beta and
        # output norm, QSA RoPE and output gate; reference/qwen4_exp_precision.py).
        from arctic_platform.correctness.reference.qwen4_reference import load_qwen4_reference

        model = load_qwen4_reference(
            cls, model_path, dtype=torch_dtype, attn_implementation=attn_implementation, device=device
        )
    elif getattr(cfg, "model_type", None) in GLM5_NEXT_MODEL_TYPES:
        # GLM-5.3-Flash at the sampler's precision (FP32 mHC remix and base/scale, KDA beta, one-rounding norms).
        model = load_glm5_next_reference(
            transformers.AutoModelForCausalLM if huggingface_loader else cls,
            model_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device=device,
        )
    elif getattr(cfg, "model_type", None) in QWEN3_5_MOE_MODEL_TYPES:
        # Qwen3.5-MoE at the sampler's precision (FP32 residual norms, gated norm, beta, RoPE and routing weights).
        model = load_qwen3_5_moe_reference(
            transformers.AutoModelForCausalLM if huggingface_loader else cls,
            model_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
            device=device,
        )
    else:
        if huggingface_loader:
            cls = transformers.AutoModelForCausalLM
        model = cls.from_pretrained(model_path, dtype=torch_dtype, attn_implementation=attn_implementation).to(device)
    if peft_adapter_path:
        from peft import PeftModel

        from arctic_platform.correctness.harness.lora import shared_adapter_identity as adapter_identity
        from arctic_platform.correctness.harness.lora import shared_initialize_adapter as initialize_adapter

        supplied_identity = adapter_identity(peft_adapter_path, peft_config)
        model = PeftModel.from_pretrained(model, peft_adapter_path, is_trainable=True)
        initialize_adapter(model, peft_adapter_path, supplied_identity)
    elif peft_config:
        from arctic_platform.model.patches.peft import apply_peft

        model = apply_peft(model, peft_config, optimization_dtype=torch_dtype)
    model.gradient_checkpointing_enable()
    model.train()

    head = model.get_output_embeddings()
    text_cfg = cfg.get_text_config() if hasattr(cfg, "get_text_config") else cfg
    softcap = getattr(text_cfg, "final_logit_softcapping", None)
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
        softcap=float(softcap) if softcap else None,
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


def accumulator_device(model, gradient_device, accum_dtype) -> "torch.device":
    """Hold the float32 accumulator on the gradients' GPU when it fits there, else on the CPU.

    Chosen once, after the first microbatch's backward, when that microbatch's activations are released: the GPU
    when it can hold the whole trainable set's accumulator plus twice the activation peak the microbatch measured
    (later microbatches have the same rows, but MoE routing moves their peak). A host accumulator costs a pageable
    copy, a host-side widening and a host addition of every gradient after every microbatch: 98% of a 9.7B-parameter
    full-FT reference's time, its GPU near idle. A fixture that only fits beside a host accumulator keeps it.
    """
    import torch

    if gradient_device.type != "cuda":
        return torch.device("cpu")
    itemsize = torch.empty((), dtype=accum_dtype).element_size()
    need = itemsize * sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    free, _ = torch.cuda.mem_get_info(gradient_device)
    allocated = torch.cuda.memory_allocated(gradient_device)
    obtainable = free + torch.cuda.memory_reserved(gradient_device) - allocated
    transient = torch.cuda.max_memory_allocated(gradient_device) - allocated
    return gradient_device if need + 2 * transient <= obtainable else torch.device("cpu")


def drain_gradient(accumulator: Dict[str, "torch.Tensor"], name: str, gradient, accum_dtype, device) -> None:
    """Add one gradient into its accumulator, which lives on ``device``.

    On the gradient's own device the mixed-dtype ``add_`` widens the bfloat16 operand inside the kernel: the same
    exact widening and the same correctly rounded float32 addition as the host path, without a temporary.
    """
    gradient = gradient.detach()
    if name not in accumulator:
        accumulator[name] = gradient.to(device=device, dtype=accum_dtype)
    elif accumulator[name].device == gradient.device:
        accumulator[name].add_(gradient)
    else:
        accumulator[name].add_(gradient.to(device=accumulator[name].device, dtype=accum_dtype))


def _forward_backward(
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
        if engine.softcap is not None:
            raise ValueError("the reference's Liger fused cross-entropy does not apply final_logit_softcapping")
        if lm_head_token_chunk_size is not None:
            raise ValueError("fused cross-entropy and chunked LM-head logprobs are mutually exclusive")
        fused_ce = _build_liger_fused_cross_entropy(fp32_lm_head=fp32_lm_head)

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
    # The CPU until the first microbatch's backward chooses (accumulator_device).
    device_for_accumulator = None
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
                softcap=engine.softcap,
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
                if engine.softcap is not None:
                    logits = torch.tanh(logits / engine.softcap) * engine.softcap
                loss = (
                    F.cross_entropy(logits, chunk_labels, ignore_index=IGNORE_INDEX, reduction="sum") / active_tokens
                )
                loss.backward()
                total_loss += float(loss.item())

                # Every cross-entropy chunk adds to this one gradient, so leaving the running sum in the
                # parameter's bfloat16 buffer would round each addition. The production chunked LM head has
                # one autograd invocation and is drained once below instead.
                if head_name is not None and head.weight.grad is not None:
                    drain_gradient(
                        accumulator, head_name, head.weight.grad, accum_dtype, device_for_accumulator or "cpu"
                    )
                    head.weight.grad = None

        hidden.backward(detached.grad)
        if device_for_accumulator is None:
            device_for_accumulator = accumulator_device(model, head.weight.device, accum_dtype)
            for name in accumulator:
                accumulator[name] = accumulator[name].to(device_for_accumulator)

        # Drain bfloat16 gradients into the float32 accumulator after every microbatch, so the running sum
        # never lives in the parameter dtype.
        for name, param in model.named_parameters():
            if param.grad is not None:
                drain_gradient(accumulator, name, param.grad, accum_dtype, device_for_accumulator)
        model.zero_grad(set_to_none=True)

    return ForwardBackwardResult(
        loss=total_loss,
        gradients=accumulator,
        active_tokens=active_tokens,
        microbatches=len(groups),
        skipped_no_grad=[name for name, _ in model.named_parameters() if name not in accumulator],
    )


def forward_backward(engine: ReferenceEngine, batch, **options) -> ForwardBackwardResult:
    from arctic_platform.correctness.harness.optimizer import ParticipationCapture

    with ParticipationCapture(engine.model, decoder=engine.decoder, head=engine.head) as capture:
        result = _forward_backward(engine, batch, **options)
        result.participation = capture.result()
    return result


def gradient_norms(gradients: Dict[str, "torch.Tensor"], device: str) -> Dict[str, float]:
    """One L2 norm per accumulated gradient.

    A host accumulator (one the GPU had no room for, accumulator_device) moves back to CUDA one completed tensor
    at a time, so norm reduction semantics still match the GPU-side Arctic Platform metric.
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
    huggingface_loader: bool = False,
) -> ReferenceResult:
    """Forward-backward the whole request and return one L2 gradient norm per parameter."""
    if optimizer_output_dir is not None:
        optimizer_output_dir = str(require_data_fast_path(optimizer_output_dir, label="reference optimizer output"))
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
        huggingface_loader=huggingface_loader,
    )
    model = engine.model
    reference_execution = execution_witness(engine, requested_attention=attn_implementation)
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
    from arctic_platform.correctness.harness.optimizer import active_gradient_norms

    active_gradient_norms(forward.participation, grad_norms)
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
            consume_gradients=True,
            participation=forward.participation,
        )
        optimizer_state_manifest = str(optimizer_artifact.manifest_path)
        optimizer_gradient_norm = optimizer_artifact.gradient_norm_before_clip

    if execution_witness(engine, requested_attention=attn_implementation) != reference_execution:
        raise ValueError("HF reference implementation or effective configuration changed during execution")
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
        reference_execution=reference_execution,
        participation=forward.participation,
    )
