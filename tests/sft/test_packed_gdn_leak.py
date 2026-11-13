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
"""GPU leak test: logits of sample B packed after sample A against B alone, on tiny random models.

Acceptance is top-1 agreement on every B position and a mean absolute logit difference of at most 0.05. The Qwen3.5
case has GatedDeltaNet layers and needs the packed boundary hook; the Qwen3 case has only full attention and is the
negative control, where the hook installs nothing.
"""

from __future__ import annotations

import importlib.util

import pytest
import torch

from arctic_platform.common.utils.packed_gdn import install_packed_gdn_boundaries
from arctic_platform.testing_utils import set_seed

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
    pytest.mark.skipif(importlib.util.find_spec("flash_attn_interface") is None, reason="needs FA3"),
]

_SEQ = 12
_VOCAB = 64
_MAX_MEAN_ABS_LOGIT_DIFF = 0.05


def _randomize(model):
    # Zero-initialized (1 + weight) norm weights would hide a dropped weight; give every parameter a value.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return model


def _qwen3_5():
    modeling = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=_VOCAB,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_3"
    return modeling.Qwen3_5ForCausalLM(config)


def _qwen3():
    from transformers import Qwen3Config
    from transformers import Qwen3ForCausalLM

    config = Qwen3Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=_VOCAB,
    )
    config._attn_implementation = "flash_attention_3"
    return Qwen3ForCausalLM(config)


def _model(build):
    set_seed(0)
    return _randomize(build()).to(device="cuda", dtype=torch.bfloat16).eval()


def _inputs():
    generator = torch.Generator().manual_seed(1)
    sample_a = torch.randint(0, _VOCAB, (1, _SEQ), generator=generator)
    sample_b = torch.randint(0, _VOCAB, (1, _SEQ), generator=generator)
    return sample_a.cuda(), sample_b.cuda()


@torch.no_grad()
def _b_logits(model):
    sample_a, sample_b = _inputs()
    positions = torch.arange(_SEQ, device="cuda").unsqueeze(0)
    alone = model(input_ids=sample_b, position_ids=positions).logits.float()
    packed_ids = torch.cat([sample_a, sample_b], dim=1)
    packed_positions = torch.cat([positions, positions], dim=1)
    packed = model(input_ids=packed_ids, position_ids=packed_positions).logits.float()[:, _SEQ:]
    return packed, alone


def _report(packed, alone):
    top1 = (packed.argmax(-1) == alone.argmax(-1)).float().mean().item()
    diff = (packed - alone).abs()
    print(f"top1_agreement={top1:.4f} mean_abs_logit_diff={diff.mean().item():.4e} max={diff.max().item():.4e}")
    return top1, diff.mean().item()


def test_gdn_leaks_without_the_hook():
    top1, mean_diff = _report(*_b_logits(_model(_qwen3_5)))
    assert top1 < 1.0 or mean_diff > _MAX_MEAN_ABS_LOGIT_DIFF


def test_gdn_does_not_leak_with_the_hook():
    model = _model(_qwen3_5)
    install_packed_gdn_boundaries(model)
    top1, mean_diff = _report(*_b_logits(model))
    assert top1 == 1.0
    assert mean_diff <= _MAX_MEAN_ABS_LOGIT_DIFF


def test_full_attention_does_not_leak_and_gets_no_hook():
    model = _model(_qwen3)
    hooks_before = len(model._forward_pre_hooks)
    install_packed_gdn_boundaries(model)
    assert len(model._forward_pre_hooks) == hooks_before
    top1, mean_diff = _report(*_b_logits(model))
    assert top1 == 1.0
    assert mean_diff <= _MAX_MEAN_ABS_LOGIT_DIFF
