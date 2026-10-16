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

"""A spec may only name checks that a module registers."""

import pytest

from arctic_platform.correctness.harness.runner import assert_applicable_tests_registered
from arctic_platform.correctness.harness.spec import ArmSpec
from arctic_platform.correctness.harness.spec import ModelSpec
from arctic_platform.correctness.harness.spec import TestSpec as ConfigSpec


def _spec(applicable):
    return ConfigSpec(
        config_id="unit",
        config_path="unit.config",
        model=ModelSpec(
            source_checkpoint="model",
            num_hidden_layers=1,
            vision_depth=0,
            seed=0,
            param_count=1,
            cache_path="model",
            content_hash="hash",
            sized_against_gib=1.0,
        ),
        arms=[
            ArmSpec(
                name="fixed",
                global_batch_size=1,
                max_seq_len=8,
                total_tokens=8,
                active_tokens=7,
                dss_microbatches=1,
                pad_fraction=0.125,
            )
        ],
        attn_implementations=["flash_attention_3"],
        applicable_tests=applicable,
    )


def test_a_registered_id_is_accepted():
    assert_applicable_tests_registered(_spec(["single-step-grads"]), "unit")


def test_an_unregistered_id_is_refused_rather_than_filtered_away():
    with pytest.raises(ValueError, match="nothing registers"):
        assert_applicable_tests_registered(_spec(["t01_grad_norms"]), "unit")


def test_the_refusal_names_every_unregistered_id():
    """An operator fixing the spec needs all of them, not the first one found."""
    with pytest.raises(ValueError) as failure:
        assert_applicable_tests_registered(_spec(["t01_grad_norms", "t02_delta_update"]), "unit")
    assert "t01_grad_norms" in str(failure.value)
    assert "t02_delta_update" in str(failure.value)
