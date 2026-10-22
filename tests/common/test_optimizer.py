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

import pytest

from arctic_platform.common.optimizer import SUPPORTED_OPTIMIZER_KEYS
from arctic_platform.common.optimizer import normalize_optimizer_config


class TestReportedRegression:
    def test_beta1_beta2_become_canonical_betas(self):
        """beta1/beta2 were dropped and betas defaulted to [0.9, 0.999]."""
        optimizer, gradient_clipping = normalize_optimizer_config(
            {
                "beta1": 0.9,
                "beta2": 0.95,
                "gradient_clipping": 1.0,
                "lr": 1e-5,
            }
        )

        assert optimizer == {"lr": 1e-5, "betas": [0.9, 0.95]}
        assert gradient_clipping == 1.0

    def test_scheduler_fields_from_the_reported_payload_are_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            normalize_optimizer_config(
                {
                    "beta1": 0.9,
                    "beta2": 0.95,
                    "lr_scheduler_type": "cosine",
                    "warmup_steps_proportion": 0.05,
                }
            )

        message = str(excinfo.value)
        assert "lr_scheduler_type" in message
        assert "warmup_steps_proportion" in message
        assert "step()" in message
        assert 'ds_config["scheduler"]' in message


class TestAliases:
    def test_type_is_rewritten_to_name(self):
        optimizer, _ = normalize_optimizer_config({"type": "adamw", "lr": 2e-5})

        assert optimizer == {"name": "adamw", "lr": 2e-5}

    def test_deepspeed_params_block_is_flattened(self):
        optimizer, _ = normalize_optimizer_config(
            {
                "type": "AdamW",
                "params": {"lr": 3e-5, "betas": [0.9, 0.95], "weight_decay": 0.01},
            }
        )

        assert optimizer == {
            "name": "AdamW",
            "lr": 3e-5,
            "betas": [0.9, 0.95],
            "weight_decay": 0.01,
        }

    def test_params_conflicting_with_top_level_is_rejected(self):
        with pytest.raises(ValueError, match="Remove one"):
            normalize_optimizer_config({"lr": 1e-5, "params": {"lr": 2e-5}})

    def test_params_repeating_the_same_value_is_accepted(self):
        optimizer, _ = normalize_optimizer_config({"lr": 1e-5, "params": {"lr": 1e-5}})

        assert optimizer == {"lr": 1e-5}

    def test_non_dict_params_is_rejected(self):
        with pytest.raises(ValueError, match=r"optimizer\.params must be an object"):
            normalize_optimizer_config({"params": [1, 2]})

    def test_conflicting_name_and_type_is_rejected(self):
        with pytest.raises(ValueError, match="Use name"):
            normalize_optimizer_config({"name": "fused_adam", "type": "torch_adamw"})

    def test_matching_name_and_type_is_accepted(self):
        optimizer, _ = normalize_optimizer_config({"name": "fused_adam", "type": "fused_adam"})

        assert optimizer == {"name": "fused_adam"}


class TestBetas:
    def test_canonical_betas_pass_through(self):
        optimizer, _ = normalize_optimizer_config({"betas": [0.9, 0.95]})

        assert optimizer == {"betas": [0.9, 0.95]}

    def test_tuple_betas_are_normalized_to_a_list(self):
        optimizer, _ = normalize_optimizer_config({"betas": (0.9, 0.95)})

        assert optimizer == {"betas": [0.9, 0.95]}

    def test_lone_beta1_fills_beta2_from_the_default(self):
        optimizer, _ = normalize_optimizer_config({"beta1": 0.85})

        assert optimizer == {"betas": [0.85, 0.999]}

    def test_lone_beta2_fills_beta1_from_the_default(self):
        optimizer, _ = normalize_optimizer_config({"beta2": 0.95})

        assert optimizer == {"betas": [0.9, 0.95]}

    def test_betas_conflicting_with_an_alias_is_rejected(self):
        with pytest.raises(ValueError, match="Use betas"):
            normalize_optimizer_config({"betas": [0.9, 0.999], "beta2": 0.95})

    def test_betas_matching_its_aliases_is_accepted(self):
        optimizer, _ = normalize_optimizer_config({"betas": [0.9, 0.95], "beta1": 0.9, "beta2": 0.95})

        assert optimizer == {"betas": [0.9, 0.95]}

    @pytest.mark.parametrize("betas", [[0.9], [0.9, 0.95, 0.99], []])
    def test_wrong_arity_is_rejected(self, betas):
        with pytest.raises(ValueError, match="exactly two values"):
            normalize_optimizer_config({"betas": betas})

    @pytest.mark.parametrize("betas", ["0.9,0.95", 0.9, {"beta1": 0.9}])
    def test_non_sequence_betas_is_rejected(self, betas):
        with pytest.raises(ValueError, match="must be a list of two numbers"):
            normalize_optimizer_config({"betas": betas})

    @pytest.mark.parametrize("value", [1.0, 1.5, -0.1])
    def test_out_of_range_betas_is_rejected(self, value):
        with pytest.raises(ValueError, match=r"must be in \[0, 1\)"):
            normalize_optimizer_config({"betas": [0.9, value]})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_betas_is_rejected(self, value):
        with pytest.raises(ValueError, match="must be finite"):
            normalize_optimizer_config({"betas": [0.9, value]})

    def test_non_numeric_betas_is_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            normalize_optimizer_config({"betas": [0.9, "0.95"]})

    def test_boolean_betas_is_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            normalize_optimizer_config({"beta1": True})


class TestScalars:
    def test_all_supported_fields_pass_through(self):
        optimizer, gradient_clipping = normalize_optimizer_config(
            {
                "name": "AdamW",
                "lr": 1e-5,
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "fused": False,
            }
        )

        assert optimizer == {
            "name": "AdamW",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "fused": False,
        }
        assert gradient_clipping is None

    def test_defaults_are_not_injected(self):
        """The server's OptimizerConfig owns defaults, so the block stays sparse."""
        optimizer, _ = normalize_optimizer_config({"lr": 1e-5})

        assert optimizer == {"lr": 1e-5}

    @pytest.mark.parametrize("value", [0, -1e-5])
    def test_non_positive_eps_is_rejected(self, value):
        with pytest.raises(ValueError, match="must be greater than zero"):
            normalize_optimizer_config({"eps": value})

    def test_negative_lr_is_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            normalize_optimizer_config({"lr": -1e-5})

    def test_zero_lr_is_accepted(self):
        """A run that steps the optimizer without moving the weights is a control, not a misconfiguration."""
        optimizer, _ = normalize_optimizer_config({"lr": 0.0})

        assert optimizer == {"lr": 0.0}

    def test_negative_weight_decay_is_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            normalize_optimizer_config({"weight_decay": -0.01})

    def test_zero_weight_decay_is_accepted(self):
        optimizer, _ = normalize_optimizer_config({"weight_decay": 0.0})

        assert optimizer == {"weight_decay": 0.0}

    def test_non_finite_lr_is_rejected(self):
        with pytest.raises(ValueError, match="must be finite"):
            normalize_optimizer_config({"lr": float("inf")})

    def test_string_lr_is_rejected(self):
        with pytest.raises(ValueError, match="must be a number"):
            normalize_optimizer_config({"lr": "1e-5"})

    def test_non_boolean_fused_is_rejected(self):
        with pytest.raises(ValueError, match="must be a boolean"):
            normalize_optimizer_config({"fused": "no"})

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="must be a non-empty string"):
            normalize_optimizer_config({"name": "  "})


class TestGradientClipping:
    def test_it_is_hoisted_out_of_the_optimizer_block(self):
        optimizer, gradient_clipping = normalize_optimizer_config({"lr": 1e-5, "gradient_clipping": 0.5})

        assert "gradient_clipping" not in optimizer
        assert gradient_clipping == 0.5

    def test_an_explicit_null_reads_as_absent(self):
        optimizer, gradient_clipping = normalize_optimizer_config({"lr": 1e-5, "gradient_clipping": None})

        assert optimizer == {"lr": 1e-5}
        assert gradient_clipping is None

    def test_negative_clipping_is_rejected(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            normalize_optimizer_config({"gradient_clipping": -1.0})


class TestUnknownFields:
    def test_a_typo_is_rejected_and_lists_supported_fields(self):
        with pytest.raises(ValueError) as excinfo:
            normalize_optimizer_config({"lr": 1e-5, "weigth_decay": 0.01})

        message = str(excinfo.value)
        assert "['weigth_decay']" in message
        for field in SUPPORTED_OPTIMIZER_KEYS:
            assert field in message

    def test_every_unknown_field_is_reported_at_once(self):
        with pytest.raises(ValueError, match=r"\['alpha', 'momentum'\]"):
            normalize_optimizer_config({"momentum": 0.9, "alpha": 1.0})

    def test_location_prefixes_the_message(self):
        with pytest.raises(ValueError, match="training.optimizer contains unsupported"):
            normalize_optimizer_config({"nope": 1}, location="training.optimizer")

    @pytest.mark.parametrize("optimizer", [None, [], "adamw", 5])
    def test_a_non_dict_block_is_rejected(self, optimizer):
        with pytest.raises(ValueError, match="must be an object"):
            normalize_optimizer_config(optimizer)

    def test_an_empty_block_normalizes_to_empty(self):
        optimizer, gradient_clipping = normalize_optimizer_config({})

        assert optimizer == {}
        assert gradient_clipping is None

    def test_the_input_is_not_mutated(self):
        submitted = {"beta1": 0.9, "beta2": 0.95, "gradient_clipping": 1.0}

        normalize_optimizer_config(submitted)

        assert submitted == {"beta1": 0.9, "beta2": 0.95, "gradient_clipping": 1.0}
