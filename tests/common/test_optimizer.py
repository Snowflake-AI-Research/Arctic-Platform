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

from arctic_platform.common.optimizer import OptimizerSettings


def _explicit(data):
    return OptimizerSettings.model_validate(data).model_dump(exclude_unset=True)


class TestReportedRegression:
    def test_beta1_beta2_become_canonical_betas(self):
        """beta1/beta2 were dropped and betas defaulted to [0.9, 0.999]."""
        settings = OptimizerSettings.model_validate(
            {"beta1": 0.9, "beta2": 0.95, "gradient_clipping": 1.0, "lr": 1e-5}
        )

        assert settings.model_dump(exclude_unset=True) == {"lr": 1e-5, "betas": [0.9, 0.95]}
        assert settings.gradient_clipping == 1.0

    def test_scheduler_fields_from_the_reported_payload_are_rejected(self):
        with pytest.raises(ValueError) as excinfo:
            _explicit({"beta1": 0.9, "beta2": 0.95, "lr_scheduler_type": "cosine", "warmup_steps_proportion": 0.05})

        message = str(excinfo.value)
        assert "lr_scheduler_type" in message
        assert "warmup_steps_proportion" in message
        assert "step()" in message
        assert 'ds_config["scheduler"]' in message


class TestAliases:
    def test_type_is_rewritten_to_name(self):
        assert _explicit({"type": "adamw", "lr": 2e-5}) == {"name": "adamw", "lr": 2e-5}

    def test_deepspeed_params_block_is_flattened(self):
        optimizer = _explicit({"type": "AdamW", "params": {"lr": 3e-5, "betas": [0.9, 0.95], "weight_decay": 0.01}})

        assert optimizer == {"name": "AdamW", "lr": 3e-5, "betas": [0.9, 0.95], "weight_decay": 0.01}

    def test_params_conflicting_with_top_level_is_rejected(self):
        with pytest.raises(ValueError, match="Remove one"):
            _explicit({"lr": 1e-5, "params": {"lr": 2e-5}})

    def test_params_repeating_the_same_value_is_accepted(self):
        assert _explicit({"lr": 1e-5, "params": {"lr": 1e-5}}) == {"lr": 1e-5}

    def test_non_dict_params_is_rejected(self):
        with pytest.raises(ValueError, match=r"optimizer\.params must be an object"):
            _explicit({"params": [1, 2]})

    def test_conflicting_name_and_type_is_rejected(self):
        with pytest.raises(ValueError, match="Use name"):
            _explicit({"name": "fused_adam", "type": "torch_adamw"})

    def test_matching_name_and_type_is_accepted(self):
        assert _explicit({"name": "fused_adam", "type": "fused_adam"}) == {"name": "fused_adam"}


class TestBetas:
    def test_canonical_betas_pass_through(self):
        assert _explicit({"betas": [0.9, 0.95]}) == {"betas": [0.9, 0.95]}

    def test_tuple_betas_are_normalized_to_a_list(self):
        assert _explicit({"betas": (0.9, 0.95)}) == {"betas": [0.9, 0.95]}

    def test_null_betas_reads_as_absent(self):
        assert _explicit({"betas": None}) == {}

    def test_lone_beta1_fills_beta2_from_the_default(self):
        assert _explicit({"beta1": 0.85}) == {"betas": [0.85, 0.999]}

    def test_lone_beta2_fills_beta1_from_the_default(self):
        assert _explicit({"beta2": 0.95}) == {"betas": [0.9, 0.95]}

    def test_betas_conflicting_with_an_alias_is_rejected(self):
        with pytest.raises(ValueError, match="Use betas"):
            _explicit({"betas": [0.9, 0.999], "beta2": 0.95})

    def test_betas_matching_its_aliases_is_accepted(self):
        assert _explicit({"betas": [0.9, 0.95], "beta1": 0.9, "beta2": 0.95}) == {"betas": [0.9, 0.95]}

    @pytest.mark.parametrize("betas", [[0.9], [0.9, 0.95, 0.99], []])
    def test_wrong_arity_is_rejected(self, betas):
        with pytest.raises(ValueError, match="2 items"):
            _explicit({"betas": betas})

    @pytest.mark.parametrize("betas", ["0.9,0.95", 0.9, {"beta1": 0.9}])
    def test_non_sequence_betas_is_rejected(self, betas):
        with pytest.raises(ValueError, match="valid list"):
            _explicit({"betas": betas})

    @pytest.mark.parametrize("value", [1.0, 1.5, -0.1])
    def test_out_of_range_betas_is_rejected(self, value):
        with pytest.raises(ValueError, match="less than 1|greater than or equal to 0"):
            _explicit({"betas": [0.9, value]})

    @pytest.mark.parametrize("value", [float("nan"), float("inf")])
    def test_non_finite_betas_is_rejected(self, value):
        with pytest.raises(ValueError, match="finite number"):
            _explicit({"betas": [0.9, value]})

    def test_non_numeric_betas_is_rejected(self):
        with pytest.raises(ValueError, match="valid number"):
            _explicit({"betas": [0.9, "0.95"]})

    def test_boolean_betas_is_rejected(self):
        with pytest.raises(ValueError, match="valid number"):
            _explicit({"beta1": True})


class TestScalars:
    def test_all_supported_fields_pass_through(self):
        submitted = {
            "name": "AdamW",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "fused": False,
        }
        settings = OptimizerSettings.model_validate(submitted)

        assert settings.model_dump(exclude_unset=True) == submitted
        assert settings.gradient_clipping is None

    def test_defaults_are_not_marked_explicit(self):
        assert _explicit({"lr": 1e-5}) == {"lr": 1e-5}

    def test_defaults(self):
        assert OptimizerSettings().model_dump() == {
            "name": "AdamW",
            "lr": 1e-5,
            "weight_decay": 0.0,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "fused": None,
        }

    def test_int_values_are_accepted(self):
        assert _explicit({"lr": 0, "weight_decay": 1}) == {"lr": 0.0, "weight_decay": 1.0}

    @pytest.mark.parametrize("value", [0, -1e-5])
    def test_non_positive_eps_is_rejected(self, value):
        with pytest.raises(ValueError, match="greater than 0"):
            _explicit({"eps": value})

    def test_negative_lr_is_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            _explicit({"lr": -1e-5})

    def test_zero_lr_is_accepted(self):
        """A run that steps the optimizer without moving the weights is a control, not a misconfiguration."""
        assert _explicit({"lr": 0.0}) == {"lr": 0.0}

    def test_negative_weight_decay_is_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            _explicit({"weight_decay": -0.01})

    def test_non_finite_lr_is_rejected(self):
        with pytest.raises(ValueError, match="finite number"):
            _explicit({"lr": float("inf")})

    def test_string_lr_is_rejected(self):
        with pytest.raises(ValueError, match="valid number"):
            _explicit({"lr": "1e-5"})

    def test_non_boolean_fused_is_rejected(self):
        with pytest.raises(ValueError, match="valid boolean"):
            _explicit({"fused": "no"})

    def test_empty_name_is_rejected(self):
        with pytest.raises(ValueError, match="should match pattern"):
            _explicit({"name": "  "})


class TestGradientClipping:
    def test_it_is_never_dumped_with_the_block(self):
        settings = OptimizerSettings.model_validate({"lr": 1e-5, "gradient_clipping": 0.5})

        assert "gradient_clipping" not in settings.model_dump()
        assert settings.gradient_clipping == 0.5

    def test_an_explicit_null_reads_as_absent(self):
        assert OptimizerSettings.model_validate({"gradient_clipping": None}).gradient_clipping is None

    def test_negative_clipping_is_rejected(self):
        with pytest.raises(ValueError, match="greater than or equal to 0"):
            _explicit({"gradient_clipping": -1.0})


class TestUnknownFields:
    def test_a_typo_is_rejected_and_lists_supported_fields(self):
        with pytest.raises(ValueError) as excinfo:
            _explicit({"lr": 1e-5, "weigth_decay": 0.01})

        message = str(excinfo.value)
        assert "['weigth_decay']" in message
        for field in OptimizerSettings.model_fields:
            assert field in message

    def test_every_unknown_field_is_reported_at_once(self):
        with pytest.raises(ValueError, match=r"\['alpha', 'momentum'\]"):
            _explicit({"momentum": 0.9, "alpha": 1.0})

    @pytest.mark.parametrize("optimizer", [None, [], "adamw", 5])
    def test_a_non_dict_block_is_rejected(self, optimizer):
        with pytest.raises(ValueError, match="valid dictionary"):
            _explicit(optimizer)

    def test_an_empty_block_is_all_defaults(self):
        assert _explicit({}) == {}

    def test_the_input_is_not_mutated(self):
        submitted = {"beta1": 0.9, "beta2": 0.95, "gradient_clipping": 1.0, "params": {"lr": 1e-5}}

        _explicit(submitted)

        assert submitted == {"beta1": 0.9, "beta2": 0.95, "gradient_clipping": 1.0, "params": {"lr": 1e-5}}
