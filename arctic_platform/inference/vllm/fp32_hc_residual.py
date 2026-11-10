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

"""FP32 hyper-connection residual streams for Qwen3.8-Flash-Next (``ARCTIC_FP32_HC_RESIDUAL=1``).

vLLM carries the ``hc_count`` residual streams of every decoder layer in the
model dtype, so each injection (``hc_combine``, ``hc_combine_norm``) adds in
fp32 and then rounds the streams to bf16. With this patch the streams stay fp32
from the first decoder layer to the final mixer, as the trainer's
``loader_options.fp32_residual`` keeps them:

- the first decoder layer upcasts the streams; the combine kernels already add
  in fp32 and store the dtype of the streams they are given;
- the norms in front of the block GEMMs store the weight dtype, so every block
  still receives a bf16 input;
- the PLE layer reads the streams in bf16, as its fused gate requires, and adds
  its output to the fp32 streams;
- the final mixer hands the multi-stream state to the MTP drafter in bf16, as
  before.

Under pipeline parallelism vLLM copies the streams into model-dtype transport
buffers between stages, which rounds them there.
"""

from __future__ import annotations

import torch
from vllm.models.common.ops.sequence_parallel import sp_all_gather, sp_shard
from vllm.models.qwen4_exp.nvidia import hyperconnection
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpDecoderLayer
from vllm.models.qwen4_exp.nvidia.ops import hc as hc_ops
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpPLELayer, ple_gate
from vllm.platforms import current_platform

from arctic_platform.inference.patching import ArcticPatch


def _grouped_gemma_rmsnorm(x, weight, eps, num_groups):
    """vLLM's launcher, storing the normalized streams in the weight dtype."""
    y = torch.empty(x.shape, dtype=weight.dtype, device=x.device)
    hc_ops._grouped_gemma_rmsnorm_kernel[(x.shape[0] * num_groups,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        x.shape[1],
        num_groups,
        W_SHARED=weight.numel() == x.shape[1] // num_groups,
        EPS=eps,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return y


def _hc_combine_norm(residual, block_output, injection_logits, norm_weight,
                     eps, hc_count):
    """vLLM's launcher, storing the normalized streams in the weight dtype and
    the combined streams in the dtype of ``residual``."""
    hc_dim = residual.shape[1] // hc_count
    out = torch.empty_like(residual)
    y = torch.empty(residual.shape, dtype=norm_weight.dtype,
                    device=residual.device)
    hc_ops._hc_combine_norm_kernel[(residual.shape[0] * hc_count,)](
        block_output,
        residual,
        injection_logits,
        norm_weight,
        out,
        y,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0) if injection_logits is not None else 0,
        out.stride(0),
        y.stride(0),
        hc_dim,
        hc_count,
        W_SHARED=norm_weight.numel() == hc_dim,
        EPS=eps,
        BLOCK_SIZE=512,
        launch_pdl=current_platform.is_arch_support_pdl(),
    )
    return out, y


class HyperConnectionModulePatch(ArcticPatch[hyperconnection]):
    """The names ``GatedResidual.mix`` and ``combine_and_mix`` call."""

    grouped_gemma_rmsnorm = _grouped_gemma_rmsnorm
    hc_combine_norm = _hc_combine_norm


class GatedResidualPatch(ArcticPatch[GatedResidual]):
    _orig_combine_and_mix = GatedResidual.combine_and_mix

    def combine_and_mix(self, hidden_states, prev_block_output,
                        prev_injection):
        hidden_states, block_input, injection = self._orig_combine_and_mix(
            hidden_states, prev_block_output, prev_injection)
        if not self.use_combine:
            # The final mixer's streams only feed the MTP drafter.
            hidden_states = hidden_states.to(self.hc_norm.weight.dtype)
        return hidden_states, block_input, injection


class Qwen4ExpDecoderLayerPatch(ArcticPatch[Qwen4ExpDecoderLayer]):
    _orig_forward = Qwen4ExpDecoderLayer.forward

    def forward(self, hidden_states, *args, **kwargs):
        return self._orig_forward(hidden_states.float(), *args, **kwargs)


class Qwen4ExpPLELayerPatch(ArcticPatch[Qwen4ExpPLELayer]):

    def forward(self, hidden_states, input_ids, query_start_loc,
                ngram_context):
        """The upstream forward on bf16 streams, with the outer residual
        added to the fp32 streams instead of inside the short convolution."""
        streams = hidden_states
        hidden_states = streams.to(self.model_config.dtype)
        input_ids = input_ids.reshape(-1)
        embeddings = self.ple_embedding(hidden_states, input_ids,
                                        query_start_loc, ngram_context)
        embeddings = self._dequantize_embeddings(embeddings,
                                                 hidden_states.dtype)
        kv, _ = self.kv_proj(embeddings)
        key, value = kv.split(self.kv_proj.output_sizes, dim=-1)
        gated_output, conv_input = ple_gate(
            key,
            value,
            hidden_states,
            self.norm_key.weight,
            self.norm_query.weight,
            self.norm_conv.weight,
            self.norm_key.eps,
        )
        if self.use_sequence_parallel:
            conv_input = sp_all_gather(conv_input)[:input_ids.shape[0]]
        # Without an outer residual the kernel stores the convolution over its
        # residual argument instead of adding to it.
        conv_output = torch.zeros_like(conv_input)
        self._short_conv(conv_input, conv_output, outer_residual=None)
        gated_output.add_(sp_shard(conv_output)
                          if self.use_sequence_parallel else conv_output)
        return streams + gated_output


def apply_fp32_hc_residual_patches() -> None:
    """Carry Qwen3.8-Flash-Next's hyper-connection streams in fp32, once per process."""
    if "forward" in Qwen4ExpPLELayer.__dict__.get("_arctic_patches", {}):
        return
    HyperConnectionModulePatch.apply_patch()
    GatedResidualPatch.apply_patch()
    Qwen4ExpDecoderLayerPatch.apply_patch()
    Qwen4ExpPLELayerPatch.apply_patch()
