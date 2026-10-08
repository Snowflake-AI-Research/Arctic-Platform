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

"""Parameter-name alignment follows the PrimeRL-to-Hugging Face weight conversion."""

import math

from arctic_platform.correctness.harness.names import align


def test_qwen35_moe_norms_align_across_fused_and_prime_layouts() -> None:
    dss = {
        "model.layers.0._checkpoint_wrapped_module.mlp.router.gate.weight": 1.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w1": 3.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w2": 5.0,
        "model.layers.0._checkpoint_wrapped_module.mlp.experts.w3": 4.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert.w1.weight": 6.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert.w2.weight": 7.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert.w3.weight": 8.0,
        "model.layers.0._checkpoint_wrapped_module.shared_expert_gate.weight": 9.0,
    }
    reference = {
        "model.language_model.layers.0.mlp.gate.weight": 1.0,
        "model.language_model.layers.0.mlp.experts.gate_up_proj": 5.0,
        "model.language_model.layers.0.mlp.experts.down_proj": 5.0,
        "model.language_model.layers.0.mlp.shared_expert.gate_proj.weight": 6.0,
        "model.language_model.layers.0.mlp.shared_expert.down_proj.weight": 7.0,
        "model.language_model.layers.0.mlp.shared_expert.up_proj.weight": 8.0,
        "model.language_model.layers.0.mlp.shared_expert_gate.weight": 9.0,
    }

    pairs, only_dss, only_reference = align(dss, reference)

    assert not only_dss
    assert not only_reference
    assert dict((name, (actual, expected)) for name, actual, expected in pairs)[
        "layers.0.mlp.experts.gate_up_proj"
    ] == (math.hypot(3.0, 4.0), 5.0)


def test_equal_checkpoint_aliases_collapse_without_losing_coverage() -> None:
    dss = {
        "model.layers.0.input_layernorm.weight": 2.0,
        "model.language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight": 2.0,
    }
    reference = {"model.language_model.layers.0.input_layernorm.weight": 2.0}

    pairs, only_dss, only_reference = align(dss, reference)

    assert pairs == [("layers.0.input_layernorm.weight", 2.0, 2.0)]
    assert not only_dss
    assert not only_reference


def test_unequal_checkpoint_aliases_fail_coverage() -> None:
    dss = {
        "model.layers.0.input_layernorm.weight": 2.0,
        "model.language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight": 3.0,
    }
    reference = {"model.language_model.layers.0.input_layernorm.weight": 2.0}

    _, only_dss, _ = align(dss, reference)

    assert only_dss == ["model.language_model.layers.0._checkpoint_wrapped_module.input_layernorm.weight"]


def test_unused_zero_vision_gradients_are_excluded() -> None:
    dss = {
        "model.visual.blocks.0.attn.qkv.weight": 0.0,
        "model.layers.0.input_layernorm.weight": 2.0,
    }
    reference = {"model.language_model.layers.0.input_layernorm.weight": 2.0}

    pairs, only_dss, only_reference = align(dss, reference)

    assert pairs == [("layers.0.input_layernorm.weight", 2.0, 2.0)]
    assert not only_dss
    assert not only_reference


def test_nonzero_vision_gradient_fails_text_only_coverage() -> None:
    dss = {"model.visual.blocks.0.attn.qkv.weight": 1.0}

    _, only_dss, _ = align(dss, {})

    assert only_dss == ["model.visual.blocks.0.attn.qkv.weight"]
