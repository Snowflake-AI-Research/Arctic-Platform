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
"""CPU tests of the ``dflash2`` processing loss: forward routing, packed-window weights, and metrics."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.common.utils.batch import metric_is_summed
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors.dflash2 import TERMS_OUTPUT_KEY
from arctic_platform.rl.processors.dflash2 import DFlash2Loss

# Platform loss_mask: entry i weights the logits at i that predict token i + 1. A response of [A, B] after two prompt
# tokens is supervised at [0, 1, 1, 0].
TWO_TOKEN_RESPONSE = [0, 1, 1, 0]  # trainable; 2 token-aligned supervised tokens
THREE_TOKEN_RESPONSE = [1, 1, 1, 0]  # trainable; 3
ONE_TOKEN_RESPONSE = [0, 1, 0]  # one response token cannot hold an anchor and a target, so the model drops the row


def packed(*row_masks):
    """A packed ``[1, T]`` microbatch with ``position_ids`` restarting at each row."""
    return dict(
        loss_mask=torch.tensor([[w for row in row_masks for w in row]], dtype=torch.float32),
        position_ids=torch.tensor([[i for row in row_masks for i in range(len(row))]]),
    )


def padded(*row_masks):
    """A right-padded ``[B, S]`` microbatch with its ``attention_mask``."""
    width = max(len(row) for row in row_masks)
    return dict(
        loss_mask=torch.tensor([row + [0] * (width - len(row)) for row in row_masks], dtype=torch.float32),
        attention_mask=torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in row_masks]),
    )


def reduce(*microbatches):
    return DFlash2Loss().packed_reduction_callback(list(microbatches), {}, "dflash2")


def test_dflash2_is_a_registered_loss():
    assert isinstance(resolve_loss("dflash2"), DFlash2Loss)


def test_forward_callback_routes_the_prediction_aligned_mask_and_step_options():
    loss_mask = torch.tensor(TWO_TOKEN_RESPONSE + THREE_TOKEN_RESPONSE, dtype=torch.float32)
    model_kwargs: dict = {}
    output_keys = ["logits", "logprobs"]

    DFlash2Loss().model_forward_callback(
        model_kwargs,
        {"loss_mask": loss_mask},
        {"selector_loss_alpha": 2, "detailed_metrics": False, "per_position_metrics": True},
        output_keys,
    )

    # The mask is handed over unshifted, as a [1, T] window; the model converts each row to token alignment.
    assert model_kwargs["loss_mask"].shape == (1, 8)
    assert torch.equal(model_kwargs["loss_mask"][0], loss_mask)
    assert model_kwargs["selector_loss_alpha"] == 2.0
    assert model_kwargs["collect_detailed_metrics"] is False
    assert model_kwargs["per_position_metrics"] is True
    assert output_keys == ["logits", "logprobs", TERMS_OUTPUT_KEY]


def test_forward_callback_leaves_unset_step_options_to_the_model_defaults():
    model_kwargs: dict = {}

    DFlash2Loss().model_forward_callback(model_kwargs, {"loss_mask": torch.ones(1, 4)}, {}, [])

    assert "selector_loss_alpha" not in model_kwargs
    assert model_kwargs["collect_detailed_metrics"] is True
    assert model_kwargs["per_position_metrics"] is False


def test_forward_callback_requires_a_loss_mask():
    with pytest.raises(ValueError, match="requires context\\['loss_mask'\\]"):
        DFlash2Loss().model_forward_callback({}, {}, {}, [])


def test_validation_rejects_a_mask_of_the_wrong_size_and_a_negative_selector_weight():
    loss = DFlash2Loss()

    with pytest.raises(ValueError, match="loss_mask has 3 entries but input_ids has 4"):
        loss.validation_callback({"loss_mask": torch.ones(3), "input_ids": torch.ones(1, 4)}, {})
    with pytest.raises(ValueError, match="selector_loss_alpha"):
        loss.validation_callback({}, {"selector_loss_alpha": -1})


@pytest.mark.parametrize("alpha", [float("nan"), float("inf"), True, "1.0"])
def test_validation_rejects_a_selector_weight_that_is_not_a_finite_number(alpha):
    with pytest.raises(ValueError, match="finite non-negative number"):
        DFlash2Loss().validation_callback({}, {"selector_loss_alpha": alpha})


@pytest.mark.parametrize("alpha", [None, 0, 0.0, 2, 0.5])
def test_validation_accepts_unset_zero_and_positive_selector_weights(alpha):
    DFlash2Loss().validation_callback({}, {} if alpha is None else {"selector_loss_alpha": alpha})


def test_packed_window_weight_counts_only_rows_the_model_trains_on():
    # Row 2 is dropped by the model, so its supervised token must not weigh the window (the raw mask sums to 3).
    reduction = reduce(packed(TWO_TOKEN_RESPONSE, ONE_TOKEN_RESPONSE), packed(THREE_TOKEN_RESPONSE))

    assert reduction.reporting_weights == pytest.approx((2.0, 3.0))
    assert reduction.loss_scales == pytest.approx((2 / 5, 3 / 5))
    assert not reduction.loss_is_additive


def test_padded_microbatch_weight_uses_the_attention_mask_rows():
    reduction = reduce(
        padded(TWO_TOKEN_RESPONSE, ONE_TOKEN_RESPONSE, THREE_TOKEN_RESPONSE), packed(THREE_TOKEN_RESPONSE)
    )

    assert reduction.reporting_weights == pytest.approx((5.0, 3.0))


def test_padded_and_packed_layouts_weigh_the_same_rows_identically():
    rows = (TWO_TOKEN_RESPONSE, ONE_TOKEN_RESPONSE, THREE_TOKEN_RESPONSE)

    assert reduce(padded(*rows)).reporting_weights == reduce(packed(*rows)).reporting_weights == pytest.approx((5.0,))


def test_a_window_with_no_trainable_row_carries_no_weight():
    reduction = reduce(packed(ONE_TOKEN_RESPONSE, [0, 0, 0]), packed(TWO_TOKEN_RESPONSE))

    assert reduction.reporting_weights == (0.0, 2.0)
    assert reduction.loss_scales == (0.0, 1.0)


def test_a_step_with_no_trainable_row_anywhere_still_reduces():
    reduction = reduce(packed(ONE_TOKEN_RESPONSE), packed([0, 0, 0]))

    assert sum(reduction.loss_scales) == pytest.approx(1.0)


def test_fractional_weights_are_summed():
    assert reduce(packed([0, 0.75, 0.9, 0])).reporting_weights == pytest.approx((1.65,))


def test_packed_reduction_requires_a_loss_mask():
    with pytest.raises(ValueError, match="requires context\\['loss_mask'\\]"):
        reduce({"position_ids": torch.arange(4)[None]})


def test_a_padded_mask_without_row_lengths_is_rejected():
    with pytest.raises(ValueError, match="cannot split a padded loss_mask"):
        reduce({"loss_mask": torch.ones(2, 5)})


def test_loss_reports_each_ratio_term_as_an_additive_pair():
    loss = torch.tensor(1.5)
    outputs = {
        "loss": loss,
        TERMS_OUTPUT_KEY: {"dflash2/selector/greedy_accepted_length": (torch.tensor(6.0), torch.tensor(4.0))},
    }

    result, metrics = DFlash2Loss().loss(outputs, {}, {}, {}, "cpu")

    assert result is loss
    assert metrics == {
        "loss_term_dflash2/selector/greedy_accepted_length__num": 6.0,
        "loss_term_dflash2/selector/greedy_accepted_length__den": 4.0,
    }
    assert all(metric_is_summed(name) for name in metrics)


def test_loss_requires_the_online_model_outputs():
    with pytest.raises(ValueError, match="needs the DFlash2 online model"):
        DFlash2Loss().loss({"logits": torch.zeros(1)}, {}, {}, {}, "cpu")


def test_output_callback_drops_the_scalar_and_term_outputs_from_the_response():
    outputs = {"logits": 1, "loss": 2, TERMS_OUTPUT_KEY: {}, "logprobs": 3}

    DFlash2Loss().output_callback(outputs)

    assert outputs == {"logprobs": 3}
