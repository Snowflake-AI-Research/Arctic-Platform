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

"""Opt-in: let GLM-5.3-Flash (``glm5next``) run under ``VLLM_BATCH_INVARIANT=1``.

vLLM refuses batch-invariant mode for every sparse MLA backend. On Hopper ``FLASH_ATTN_MLA_SPARSE`` runs each query
token as its own one-token sequence over its top-k cached tokens, the same call in prefill and decode; the only
batch-dependent choice is FA3's split-KV heuristic, which ``num_splits=1`` turns off, as vLLM's dense FlashAttention
backend does in this mode. The KDA layers use the GatedDeltaNet backend (allowed by
:mod:`arctic_platform.inference.vllm.gdn_batch_invariance`) and their own autotuned Triton kernels, pinned here to
one config. The sparse indexer's backends (its fp8 key cache and kpool tail) are allowed as well: their kernels have
no autotuning, and whether their top-k selection is batch-invariant is measured, not assumed. The hyper-connection
pre-norm GEMM (DeepGEMM ``tf32_hc_prenorm_gemm``) splits its reduction over more blocks the fewer tokens a step has,
and small steps take a fused post + pre-norm kernel instead; every step runs the one-split unfused path here. The MoE
router's fp32 logits go through vLLM's batch-invariant matmul instead of a kernel chosen by token count. With
``ARCTIC_GLM5_BATCH_INVARIANT=1`` all of this is applied; the GDN patch is applied too.
"""

from __future__ import annotations

import os

import torch

ENV_VAR = "ARCTIC_GLM5_BATCH_INVARIANT"
_KDA_MODULES = (
    "vllm.models.glm5next.nvidia.ops.third_party.kda.kernels",
    "vllm.models.glm5next.nvidia.ops.third_party.kda.fused_recurrent",
)

_PINNED = False
_APPLIED = False


def glm5_batch_invariance_requested() -> bool:
    return os.environ.get(ENV_VAR, "0") == "1"


def pin_kda_kernels() -> int:
    """Pin GLM-5.3-Flash's KDA Triton autotuners to their first config; returns how many were pinned."""
    global _PINNED
    if _PINNED:
        return 0
    import importlib

    from triton.runtime.autotuner import Autotuner

    from arctic_platform.inference.vllm.gdn_batch_invariance import _pin_autotuner

    pinned = 0
    for name in _KDA_MODULES:
        module = importlib.import_module(name)
        for obj in list(vars(module).values()):
            pinned += _pin_autotuner(obj, Autotuner)
    _PINNED = True
    return pinned


def _single_split(flash_attn_varlen_func):
    def call(*args, **kwargs):
        kwargs["num_splits"] = 1
        return flash_attn_varlen_func(*args, **kwargs)

    call.__wrapped__ = flash_attn_varlen_func
    return call


def _batch_invariant_fp8_block_linear() -> None:
    """Run every block-FP8 linear layer on vLLM's Triton block-scaled GEMM.

    On CUDA vLLM picks ``FlashInferFp8DeepGEMMDynamicBlockScaledKernel``, which runs FlashInfer below 32 tokens and
    DeepGEMM from 32 on, so a token's projection depends on the step's size. The Triton kernel uses one fixed tile
    config under ``VLLM_BATCH_INVARIANT`` (vLLM already forces it on ROCm for that reason).
    """
    from vllm.model_executor.kernels import linear

    original = linear.choose_scaled_mm_linear_kernel

    def choose(config, possible_kernels, compute_capability=None, force_kernel=None, *, quantization):
        if quantization == "fp8_block_w8a8" and force_kernel is None:
            force_kernel = linear.TritonFp8BlockScaledMMKernel
        return original(config, possible_kernels, compute_capability, force_kernel, quantization=quantization)

    linear.choose_scaled_mm_linear_kernel = choose


def _ordered_sum(x: torch.Tensor) -> torch.Tensor:
    """Sum over the last dimension adding one column at a time, so the order does not depend on the row count."""
    total = x[..., 0]
    for i in range(1, x.shape[-1]):
        total = total + x[..., i]
    return total.unsqueeze(-1)


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM's torch grouped top-k with sorted selection and every sum in a fixed order."""
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        top2 = scores.view(num_token, num_expert_group, -1).topk(2, dim=-1, sorted=True)[0]
        group_scores = top2[..., 0] + top2[..., 1]
    else:
        group_scores = scores.view(num_token, num_expert_group, -1).max(dim=-1).values
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=True)[1]
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=True)
    if renormalize:
        topk_weights = topk_weights / _ordered_sum(topk_weights)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def deterministic_topk(
    logits: torch.Tensor, starts: torch.Tensor, ends: torch.Tensor, k: int, out: torch.Tensor
) -> torch.Tensor:
    """Write each row's top ``k`` columns of ``logits`` within ``[start, end)`` into ``out``, as ``column - start``.

    Every score is made unique by its column, so equal scores go to the later column and the selected set never
    depends on thread scheduling; the selection is written in descending order with ``-1`` for missing entries.
    """
    rows, cols = logits.shape
    col = torch.arange(cols, device=logits.device)
    valid = (col[None, :] >= starts[:, None].long()) & (col[None, :] < ends[:, None].long())
    bits = logits.float().contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits >= 0, bits, bits ^ 0x7FFFFFFF)
    key = torch.where(valid, (ordered << 32) | col[None, :], torch.iinfo(torch.int64).min)
    take = min(k, cols)
    top = torch.topk(key, take, dim=-1, sorted=False).indices
    picked = torch.where(valid.gather(1, top), top - starts[:, None].long(), -1)
    out[:, :take] = picked.sort(dim=-1, descending=True).values.to(out.dtype)
    out[:, take:] = -1
    return out


def _deterministic_indexer_topk() -> None:
    """Replace GLM-5.3-Flash's sparse-indexer top-k (prefill and decode) by :func:`deterministic_topk`.

    The prefill kernel breaks score ties by thread scheduling and the decode dispatcher picks a kernel by batch
    size, so the pools a token attends to could change from run to run and with its batch-mates.
    """
    import logging

    from vllm.models.glm5next.nvidia import sparse_indexer

    log = logging.getLogger(__name__)
    original_prefill = torch.ops._C.top_k_per_row_prefill
    checks = {"left": 3}

    def top_k_per_row_prefill(logits, ks, ke, out, num_rows, stride0, stride1, k):
        reference = None
        if checks["left"] > 0:  # the kernel's own selection, to confirm the index convention matches
            checks["left"] -= 1
            reference = torch.empty_like(out)
            original_prefill(logits, ks, ke, reference, num_rows, stride0, stride1, k)
        deterministic_topk(logits[:num_rows], ks[:num_rows], ke[:num_rows], k, out)
        if reference is not None:
            same = (reference.sort(dim=-1).values == out.sort(dim=-1).values).all(dim=-1).float().mean().item()
            log.warning("ARCTIC_GLM5_BATCH_INVARIANT indexer top-k: %.4f of rows select the kernel's set", same)

    torch.ops._C.top_k_per_row_prefill = top_k_per_row_prefill

    def decode_topk(backend):
        def run(logits, seq_lens, next_n, out, k, max_seq_len):
            if next_n != 1:
                raise NotImplementedError("deterministic indexer top-k covers decode without speculative tokens")
            lengths = seq_lens.reshape(-1)[: logits.shape[0]].to(torch.int64)
            if os.environ.get("ARCTIC_LAYER_DIGEST_DIR"):
                from arctic_platform.inference.vllm.layer_digest import record_numbered

                col = torch.arange(logits.shape[1], device=logits.device)
                record_numbered("indexer_logits", torch.where(col[None, :] < lengths[:, None], logits, 0))
            deterministic_topk(logits, torch.zeros_like(lengths), lengths, k, out)

        return run

    sparse_indexer.get_indexer_topk = decode_topk


def _batch_invariant_indexer_head_weights() -> None:
    """Compute the sparse indexer's fp32 per-head weights with the batch-invariant matmul.

    ``Glm5NextIndexer`` computes them as ``torch.mm(hidden_states.float(), wp_fp32)``. On Hopper
    ``VLLM_BATCH_INVARIANT`` keeps cuBLAS and only disables split-K, but cuBLAS still picks another fp32 kernel by row
    count (it changes at 16 decode rows), so a request's weights, and then its pool scores, depend on its
    batch-mates. Both operands hold bf16 values (the hidden state, and the bf16 projection weight upcast once), whose
    products are exact in fp32, so the bf16-operand, fp32-accumulating batch-invariant matmul loses no precision.
    That ``torch.mm`` is the module's only one.
    """
    import types

    from vllm.models.glm5next.common import attention

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import matmul_fp32_out

    def mm(a, b, *args, **kwargs):
        if args or kwargs or a.dtype != torch.float32 or b.dtype != torch.float32 or a.dim() != 2:
            return torch.mm(a, b, *args, **kwargs)
        return matmul_fp32_out(a.to(torch.bfloat16), b.to(torch.bfloat16))

    class _Torch(types.ModuleType):
        def __getattr__(self, name):
            return getattr(torch, name)

    proxy = _Torch("torch")
    proxy.mm = mm
    attention.torch = proxy


def _split_mixed_kda_steps() -> None:
    """Run a mixed step's decode requests through KDA's decode path, as a decode-only step runs them.

    With any prefill in the step, ``Glm5NextLinearAttention`` runs every request (decodes included) through the
    chunked prefill kernels; in a decode-only step it runs the recurrent kernel. A decoded token's value would
    then depend on whether another request was prefilling. vLLM orders decodes first, and the metadata builder
    already keeps the prefill requests' own boundaries, states and initial-state flags.
    """
    import dataclasses

    from vllm.forward_context import get_forward_context
    from vllm.models.glm5next.common.kda import Glm5NextLinearAttention
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    original = Glm5NextLinearAttention._forward

    def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
        raw = get_forward_context().attn_metadata
        md = raw.get(self.prefix) if isinstance(raw, dict) else None
        if md is None or md.spec_sequence_masks is not None or not (md.num_prefills > 0 and md.num_decodes > 0):
            return original(self, qkv_proj_states, g1, beta, core_attn_out)
        d, n = md.num_decode_tokens, md.num_actual_tokens
        decode = dataclasses.replace(
            md,
            num_prefills=0,
            num_prefill_tokens=0,
            num_actual_tokens=d,
            has_initial_state=None,
            non_spec_query_start_loc=md.non_spec_query_start_loc[: md.num_decodes + 1],
            non_spec_state_indices_tensor=md.non_spec_state_indices_tensor[: md.num_decodes],
        )
        prefill_cu = md.prefill_query_start_loc
        nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(
            prefill_cu.cpu(), device=prefill_cu.device
        )
        prefill = dataclasses.replace(
            md,
            num_decodes=0,
            num_decode_tokens=0,
            num_actual_tokens=n - d,
            has_initial_state=md.prefill_has_initial_state,
            non_spec_query_start_loc=prefill_cu,
            non_spec_state_indices_tensor=md.prefill_state_indices,
            nums_dict=nums_dict,
            batch_ptr=batch_ptr,
            token_chunk_offset_ptr=token_chunk_offset_ptr,
        )
        try:
            raw[self.prefix] = decode
            original(self, qkv_proj_states[:d], g1[:, :d], beta[:, :d], core_attn_out[:, :d])
            raw[self.prefix] = prefill
            original(self, qkv_proj_states[d:n], g1[:, d:n], beta[:, d:n], core_attn_out[:, d:n])
        finally:
            raw[self.prefix] = md

    Glm5NextLinearAttention._forward = _forward


def _batch_invariant_router_gate() -> None:
    """Compute the MoE router's fp32 logits with the batch-invariant matmul.

    ``GateLinear`` picks a kernel by token count (a CuteDSL kernel up to 16 tokens, a specialized one up to 32, then
    cuBLAS ``mm(out_dtype=fp32)``, which batch-invariant mode does not replace), so a token's routing scores depend
    on how many tokens share the step.
    """
    from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear

    from arctic_platform.inference.vllm.batch_invariant_fp32_logits import linear_fp32_out

    original = GateLinear.forward

    def forward(self, x):
        if self.out_dtype != torch.float32 or getattr(self, "bias", None) is not None:
            return original(self, x)
        if self.weight.dtype == torch.float32:
            output = torch.mm(x.float(), self.weight.t())
        else:
            output = linear_fp32_out(x.to(self.weight.dtype), self.weight)
        return self._return(output)

    GateLinear.forward = forward


PROXY_ENV_VAR = "ARCTIC_GLM5_PROXY_LAYERS"


def _load_proxy_layers_only(num_layers: int) -> None:
    """Debug: let a model served with ``num_hidden_layers=N`` skip the checkpoint's weights for layers N and up."""
    import re

    from vllm.models.glm5next.common.model import Glm5NextModel

    original = Glm5NextModel.load_weights
    layer = re.compile(r"(?:^|\.)layers\.(\d+)\.")

    def load_weights(self, weights):
        kept = ((name, w) for name, w in weights if not ((m := layer.search(name)) and int(m.group(1)) >= num_layers))
        return original(self, kept)

    Glm5NextModel.load_weights = load_weights


def ensure_glm5_batch_invariance() -> None:
    global _APPLIED
    if _APPLIED or not glm5_batch_invariance_requested():
        return
    if os.environ.get(PROXY_ENV_VAR):
        _load_proxy_layers_only(int(os.environ[PROXY_ENV_VAR]))
    os.environ.setdefault("ARCTIC_GDN_BATCH_INVARIANT", "1")
    # The fused grouped top-k CUDA kernel's routing weights depend on the step; vLLM's torch path sorts for batch
    # invariance.
    os.environ.setdefault("VLLM_USE_FUSED_MOE_GROUPED_TOPK", "0")
    from arctic_platform.inference.vllm.gdn_batch_invariance import ensure_gdn_batch_invariance

    ensure_gdn_batch_invariance()

    from vllm.v1.attention.backends.mla import flashattn_mla_sparse
    from vllm.v1.attention.backends.mla import indexer

    for backend in (
        flashattn_mla_sparse.FlashAttnMLASparseBackend,
        indexer.Glm5NextIndexerBackend,
        indexer.KpoolTailBackend,
    ):
        backend.supports_batch_invariance = classmethod(lambda cls: True)
    flashattn_mla_sparse.flash_attn_varlen_func = _single_split(flashattn_mla_sparse.flash_attn_varlen_func)
    from vllm.model_executor.kernels.mhc import tilelang_kernels
    from vllm.model_executor.kernels.mhc import warmup

    # The hyper-connection pre-norm GEMM splits its reduction by the step's token count; callers import these from
    # both modules at call time.
    for module in (tilelang_kernels, warmup):
        module.compute_num_split = lambda *args, **kwargs: 1
        module.compute_mhc_pre_num_splits = lambda *args, **kwargs: 1
    # Small steps would take a fused post + pre-norm kernel with its own split; every step takes the split-free path.
    tilelang_kernels.mhc_fused_post_pre_split_config = lambda *args, **kwargs: None
    _batch_invariant_router_gate()
    _batch_invariant_fp8_block_linear()
    _split_mixed_kda_steps()
    _deterministic_indexer_topk()
    _batch_invariant_indexer_head_weights()
    from vllm.model_executor.layers.fused_moe.router import grouped_topk_router

    grouped_topk_router.grouped_topk = grouped_topk
    pin_kda_kernels()
    _APPLIED = True
