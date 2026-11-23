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
"""With ``fp32_residual``, Qwen3.8-Flash-Next carries its hyper-connection streams in fp32 between blocks.

A stack of real decoder layers (hyper connections, mix and combine) with BF16 weights and linear stand-ins for the
attention and MoE blocks runs three ways: BF16 streams, FP32 streams, and the same weights in fp64. Each block still
receives a BF16 input either way; only the rounding of the streams after every injection goes away.
"""

import copy
import os

import pytest
import torch

from arctic_platform.model import loader as loader_mod
from tests.model.test_flash_trainer_rounding import _Branch
from tests.model.test_qwen4_exp import _tiny_config

LAYERS, HC, HIDDEN, TOKENS = 12, 4, 16, 256


class _PLE(torch.nn.Module):
    """Stands in for the PLE layer of layer 1: records the dtype of the streams it reads."""

    def __init__(self):
        super().__init__()
        self.key_proj = torch.nn.Linear(HIDDEN, HC * HIDDEN, bias=False)
        self.read = []

    def forward(self, hidden_states, *_args, **_kwargs):
        self.read.append(hidden_states.dtype)
        return torch.tanh(hidden_states.float()).to(self.key_proj.weight.dtype)


def _language_model(fp32_residual: bool):
    from arctic_platform.model.implementations.moe.vlm import get_language_model
    from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import Qwen4ExpForConditionalGenerationPrimeRL

    config = _tiny_config()
    config.text_config.hc_count = HC
    config.text_config.num_hidden_layers = LAYERS
    config.text_config.layer_types = ["linear_attention"] * LAYERS
    config.fp32_residual = fp32_residual
    language_model = get_language_model(Qwen4ExpForConditionalGenerationPrimeRL(config))
    for layer in language_model.layers:
        layer.ple = _PLE() if layer is language_model.layers[1] else None
        layer.linear_attn = _Branch(HIDDEN, HIDDEN, bias=False)
        layer.mlp = _Branch(HIDDEN, HIDDEN, bias=False)
    return language_model


def _forward(language_model, streams):
    hidden_states = []
    for layer in language_model.layers:
        streams = layer(streams, position_embeddings=(torch.ones(1), torch.zeros(1)))
        hidden_states.append(streams)
    return hidden_states, language_model.hyper_connection_mixer(streams)


def _rel(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.double() - reference).norm() / reference.norm()).item()


def test_qwen4_exp_loader_takes_fp32_residual_only_with_fp32_parity():
    from arctic_platform.model.implementations.qwen35.deepspeed_integration import _build_model_config
    from arctic_platform.model.loaders import flash_moe

    options = loader_mod._LOADERS["qwen4_exp"].options
    assert options().fp32_residual is False
    spec_options = options.model_validate({"fp32_residual": True}).model_dump(exclude_unset=True)
    config = _build_model_config("qwen4_exp", 1, 1, "bfloat16", "sdpa", flash_moe._FlashLoadOptions(**spec_options))
    assert config.fp32_residual is True
    with pytest.raises(ValueError, match="fp32_residual requires fp32_parity"):
        options.model_validate({"fp32_residual": True, "fp32_parity": False})
    assert "fp32_residual" not in loader_mod._LOADERS["glm5_next"].options.model_fields


def test_qwen38_fp32_residual_streams_are_closer_to_fp64():
    torch.manual_seed(0)
    bf16_streams = _language_model(fp32_residual=False)
    with torch.no_grad():
        for parameter in bf16_streams.parameters():
            parameter.normal_(std=0.5)
    bf16_streams.to(torch.bfloat16)
    fp32_streams = _language_model(fp32_residual=True).to(torch.bfloat16)
    fp32_streams.load_state_dict(bf16_streams.state_dict())
    exact = copy.deepcopy(bf16_streams).double()
    x = (2 * torch.randn(1, TOKENS, HC * HIDDEN)).bfloat16()

    with torch.no_grad():
        reference_states, reference = _forward(exact, x.double())
        bf16_states, bf16_mixed = _forward(bf16_streams, x)
        fp32_states, fp32_mixed = _forward(fp32_streams, x)

    assert all(state.dtype == torch.bfloat16 for state in bf16_states)
    assert fp32_streams.layers[1].ple.read == [torch.bfloat16]  # as vLLM's fused PLE gate reads them
    assert all(state.dtype == torch.float32 for state in fp32_states)
    assert bf16_mixed.dtype == fp32_mixed.dtype == torch.bfloat16
    for layer, (bf16_state, fp32_state, exact_state) in enumerate(zip(bf16_states, fp32_states, reference_states)):
        assert _rel(fp32_state, exact_state) < 0.8 * _rel(bf16_state, exact_state), layer
    assert _rel(fp32_mixed, reference) < _rel(bf16_mixed, reference)


@pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="the trainer's and vLLM's hyper-connection kernels need CUDA or the Triton interpreter",
)
def test_qwen38_fp32_residual_combine_and_norm_match_the_sampler():
    """The trainer's combine and norm kernels and the sampler's patched combine-and-norm, on the same fp32 streams:
    equal streams, and the same bf16 block input but for values whose fp32 order of operations rounds apart."""
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.hc")
    from arctic_platform.inference.vllm import fp32_hc_residual
    from arctic_platform.model.implementations.qwen38 import hyper_connection

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(1)
    streams = 2 * torch.randn(TOKENS, HC * HIDDEN, device=device)
    block = torch.randn(TOKENS, HIDDEN, device=device).bfloat16()
    logits = (4 * torch.randn(TOKENS, HC, device=device)).bfloat16()
    weight = (0.3 * torch.randn(HC * HIDDEN, device=device)).bfloat16()

    out, normed = fp32_hc_residual._hc_combine_norm(streams, block, logits, weight, 1e-6, HC)
    trainer_out = hyper_connection.combine(streams, block, logits, HC)
    trainer_normed = torch.empty_like(trainer_out, dtype=torch.bfloat16)
    rstd = torch.empty(TOKENS * HC, device=device)
    hyper_connection._norm_fwd_kernel[(TOKENS * HC,)](
        trainer_out, weight, trainer_normed, rstd, HIDDEN, 1e-6, HC=HC, BLOCK=HIDDEN
    )
    assert out.dtype == trainer_out.dtype == torch.float32 and normed.dtype == torch.bfloat16
    torch.testing.assert_close(out, trainer_out, rtol=2e-7, atol=1e-6)
    assert int((normed != trainer_normed).sum()) <= normed.numel() // 100
