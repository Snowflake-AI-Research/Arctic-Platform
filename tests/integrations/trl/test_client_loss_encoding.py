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

"""CPU tests for how the two-pass path encodes ``w = dL/dlogprobs`` for the server.

The client-side path is only correct if the server, differentiating whatever we
send, lands on exactly ``w``. The `weighted_logprob_sum` encoding says so
literally, so there is little to check. The `grpo` encoding says it indirectly --
through advantages, a pinned behavioural policy and a normalization that has to
collapse -- so these tests differentiate the real ``grpo`` loss and compare.

That is the property worth pinning: not the payload's shape, but that its
gradient is the weights we asked for.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.integrations.trl.loss import _surrogate_payload
from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.rl.processors.weighted_logprob import weighted_logprob_sum

B, S = 2, 6


def _batch() -> dict:
    return {
        "input_ids": torch.randint(0, 50, (B, S)),
        "attention_mask": torch.ones(B, S, dtype=torch.long),
        "position_ids": torch.arange(S).expand(B, S).contiguous(),
    }


def _server_gradient(payload: dict, config: dict, log_probs: torch.Tensor) -> torch.Tensor:
    """d(grpo)/d(logprobs) as the server would compute it, at the returned log-probs.

    The server recomputes log-probs; in the noiseless case they equal what the
    forward pass returned, which is the point at which the ratio is 1.
    """
    leaf = log_probs.clone().requires_grad_(True)
    loss, _ = grpo_loss({"logprobs": leaf}, payload, config, "cpu")
    (grad,) = torch.autograd.grad(loss, leaf)
    return grad


class TestGrpoEncoding:
    def test_server_gradient_is_the_requested_weights(self):
        """The whole contract: differentiate what we send, get back w."""
        torch.manual_seed(0)
        batch = _batch()
        log_probs = torch.randn(B, S, dtype=torch.float64)
        weights = torch.randn(B, S, dtype=torch.float64)

        payload, loss_fn, config = _surrogate_payload("grpo", batch, weights, log_probs, "unused")

        assert loss_fn == "grpo"
        assert torch.allclose(_server_gradient(payload, config, log_probs), weights, atol=1e-12)

    def test_it_holds_when_some_weights_are_zero(self):
        """Masked-out positions must contribute no gradient, not a small one."""
        torch.manual_seed(1)
        batch = _batch()
        log_probs = torch.randn(B, S, dtype=torch.float64)
        weights = torch.randn(B, S, dtype=torch.float64)
        weights[0, :3] = 0.0
        weights[1, -1] = 0.0

        payload, _, config = _surrogate_payload("grpo", batch, weights, log_probs, "unused")
        grad = _server_gradient(payload, config, log_probs)

        assert torch.allclose(grad, weights, atol=1e-12)
        assert float(grad[0, :3].abs().max()) == 0.0

    def test_normalization_collapses_to_a_plain_sum(self):
        """token-mean is masked_sum / batch_num_tokens * dp_size.

        Both must be 1 or the gradient picks up a scale factor, which is the one
        way this encoding can be wrong while still looking plausible.
        """
        _, _, config = _surrogate_payload("grpo", _batch(), torch.zeros(B, S), torch.zeros(B, S), "unused")

        assert config == {"batch_num_tokens": 1, "dp_size": 1}

    def test_the_behavioural_policy_is_pinned_to_the_returned_logprobs(self):
        """Ratio 1 is what removes clipping and leaves -advantages."""
        log_probs = torch.randn(B, S, dtype=torch.float64)
        payload, _, _ = _surrogate_payload("grpo", _batch(), torch.randn(B, S, dtype=torch.float64), log_probs, "x")

        assert torch.equal(payload["old_log_probs_shifted"], log_probs)

    def test_a_large_weight_still_maps_exactly(self):
        """Advantages are -w unscaled, so nothing clips however big w gets."""
        batch = _batch()
        log_probs = torch.zeros(B, S, dtype=torch.float64)
        weights = torch.full((B, S), 37.5, dtype=torch.float64)

        payload, _, config = _surrogate_payload("grpo", batch, weights, log_probs, "unused")

        assert torch.allclose(_server_gradient(payload, config, log_probs), weights, atol=1e-12)


class TestEquivalenceWithTheDefaultEncoding:
    def test_both_encodings_produce_the_same_gradient(self):
        """The two paths must be interchangeable in the only way that matters."""
        torch.manual_seed(2)
        batch = _batch()
        log_probs = torch.randn(B, S, dtype=torch.float64)
        weights = torch.randn(B, S, dtype=torch.float64)
        weights[:, -1] = 0.0  # as _unpack_to_padded leaves padded positions

        grpo_payload, _, config = _surrogate_payload("grpo", batch, weights, log_probs, "x")
        wls_payload, _, _ = _surrogate_payload("weighted_logprob_sum", batch, weights, log_probs, "x")

        leaf = log_probs.clone().requires_grad_(True)
        wls_loss, _ = weighted_logprob_sum({"logprobs": leaf}, wls_payload, {}, {}, "cpu")
        (wls_grad,) = torch.autograd.grad(wls_loss, leaf)

        assert torch.allclose(_server_gradient(grpo_payload, config, log_probs), wls_grad, atol=1e-12)

    def test_the_reported_loss_value_differs_and_that_is_fine(self):
        """Only the gradient is shared; the scalars are not the same number.

        At ratio 1 the per-token GRPO term is ``-advantages``, so grpo reports
        ``sum(w)`` where the default reports ``sum(w * logprobs)``. The client
        reports its own loss to TRL and drops the server's, so this shows up only
        in server-side metrics -- but it would be a real surprise when comparing
        runs across encodings, so it is pinned rather than left implicit.
        """
        torch.manual_seed(3)
        batch = _batch()
        log_probs = torch.randn(B, S, dtype=torch.float64)
        weights = torch.randn(B, S, dtype=torch.float64)

        grpo_payload, _, config = _surrogate_payload("grpo", batch, weights, log_probs, "x")
        wls_payload, _, _ = _surrogate_payload("weighted_logprob_sum", batch, weights, log_probs, "x")

        grpo_value, _ = grpo_loss({"logprobs": log_probs}, grpo_payload, config, "cpu")
        wls_value, _ = weighted_logprob_sum({"logprobs": log_probs}, wls_payload, {}, {}, "cpu")

        assert float(grpo_value) == pytest.approx(float(weights.sum()))
        assert float(wls_value) == pytest.approx(float((weights * log_probs).sum()))
        assert float(grpo_value) != pytest.approx(float(wls_value))


class TestDefaultEncoding:
    def test_default_is_unchanged(self):
        """The existing path must keep naming its own loss and its own key."""
        weights = torch.randn(B, S)
        payload, loss_fn, config = _surrogate_payload(
            "weighted_logprob_sum", _batch(), weights, torch.randn(B, S), "pkg.weighted_logprob_sum"
        )

        assert loss_fn == "pkg.weighted_logprob_sum"
        assert config == {}
        assert torch.equal(payload["logprob_weights_shifted"], weights)
        assert "advantages" not in payload

    def test_an_unknown_encoding_is_rejected_at_construction(self):
        # The client module needs trl itself; the encoding above does not, which is
        # why it lives in loss.py and is testable without it.
        client_mod = pytest.importorskip("arctic_platform.integrations.trl.client")

        with pytest.raises(ValueError, match="client_loss_encoding"):
            client_mod.ArcticTrainingClient(client=object(), client_loss_encoding="nope")
