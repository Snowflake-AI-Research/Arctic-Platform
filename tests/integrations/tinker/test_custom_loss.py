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
"""Tinker's ``forward_backward_custom``, lowered onto Cortex.

The SDK computes a user loss locally and ships ``w = dC/dlogprobs`` as
cross-entropy ``weights``, relying on the backend's cross-entropy being
``L = sum(-logprobs * weights)`` so the chain rule reconstitutes ``C``. Nothing
downstream reads the loss *value*, so a sign or frame error here is invisible
end to end -- it just trains the wrong way. These tests pin the gradient.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from arctic_platform.integrations.tinker.cortex import _grpo_surrogate
from arctic_platform.integrations.tinker.router import Datum
from arctic_platform.integrations.tinker.router import EncodedTextChunk
from arctic_platform.integrations.tinker.router import ModelInput
from arctic_platform.integrations.tinker.router import TensorData
from arctic_platform.integrations.tinker.router import datum_list_to_arctic_batch

MPL, MRL = 4, 4


def _ce_datum(tokens, weights):
    """A pass-2 datum: exactly what ``forward_backward_custom`` sends."""
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs={
            "target_tokens": TensorData(
                dtype="int64", data=tokens[1:] + [tokens[-1] + 1], shape=[len(tokens)]
            ),
            "weights": TensorData(dtype="float32", data=weights, shape=[len(weights)]),
        },
    )


def _pack(datums, **kw):
    return datum_list_to_arctic_batch(
        datums, "cross_entropy", None,
        max_prompt_length=MPL, max_response_length=MRL, pad_token_id=0, **kw,
    )


def _grpo_loss(logprobs, advantages, loss_mask, batch_num_tokens, eps_clip=0.2, dp_size=1):
    """Cortex's registered ``grpo``, as the surrogate relies on it.

    ``old_log_probs`` defaults to ``logprobs.detach()`` when the payload omits
    ``old_log_probs_shifted``, which is what pins the ratio to exactly 1.
    """
    old = logprobs.detach()
    ratio = torch.exp(logprobs - old)
    per_token = -torch.minimum(
        ratio * advantages,
        torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_clip) * advantages,
    )
    masked = torch.where(loss_mask.bool(), per_token, torch.zeros_like(per_token)).sum()
    return masked / batch_num_tokens * dp_size


class TestRouterPacking:
    def test_cross_entropy_maps_to_weighted_logprob_sum(self):
        out, _ = _pack([_ce_datum([1, 2, 3], [0.0, 1.0, 1.0])])
        assert out["processing"]["loss_fn"] == "weighted_logprob_sum"

    def test_weights_sign_flips_into_the_batch(self):
        """Tinker's CE is ``sum(-logprobs * w)``; the registered loss is
        ``sum(logprobs * w)``. The negation has to happen exactly once."""
        out, slices = _pack([_ce_datum([1, 2, 3, 4, 5], [0.0, 0.0, 0.5, -1.5, 2.0])])
        start, end, _ = slices[0]
        packed = out["batch"]["logprob_weights_shifted"][0]
        got = [float(packed[start + k]) for k in range(end - start)]
        assert got == pytest.approx([0.0, 0.0, -0.5, 1.5, -2.0])

    def test_ratio_losses_keep_their_own_loss_name(self):
        out, _ = datum_list_to_arctic_batch(
            [_ce_datum([1, 2, 3], [0.0, 1.0, 1.0])], "ppo", None,
            max_prompt_length=MPL, max_response_length=MRL, pad_token_id=0,
        )
        assert out["processing"]["loss_fn"] == "verl_grpo"


class TestForwardAppendsScoringToken:
    """Pass 1 is a ``forward`` whose log-probs the client differentiates, so its
    final position must score the real final target, not a pad."""

    def test_forward_only_appends_the_final_target(self):
        datum = _ce_datum([1, 2, 3], [0.0, 1.0, 1.0])
        final_target = int(datum.loss_fn_inputs["target_tokens"].data[-1])
        out, slices = _pack([datum], forward_only=True)
        start, end, expected_len = slices[0]
        input_ids = out["batch"]["input_ids"][0]
        # The whole input is prompt in forward_only, so the appended target
        # lands one past the returned window and is attended but never scored.
        assert int(input_ids[end]) == final_target
        assert int(out["batch"]["attention_mask"][0][end]) == 1
        assert end - start == expected_len == 3


class TestGrpoSurrogate:
    def test_weights_become_negated_advantages(self):
        body = {"logprob_weights_shifted": torch.tensor([[0.5, -1.5]])}
        out, meta = _grpo_surrogate(body, {"batch_num_tokens": 9})
        assert out["advantages"].tolist() == [[-0.5, 1.5]]
        assert "logprob_weights_shifted" not in out

    def test_token_mean_divisor_is_cancelled(self):
        """Tinker's CE is an unnormalized sum and the client has already scaled
        the weights, so grpo's ``masked_sum / batch_num_tokens`` must not
        rescale the gradient."""
        _, meta = _grpo_surrogate(
            {"logprob_weights_shifted": torch.zeros(1, 2)}, {"batch_num_tokens": 9}
        )
        assert meta["batch_num_tokens"] == 1

    def test_missing_weights_raises(self):
        with pytest.raises(ValueError, match="logprob_weights_shifted"):
            _grpo_surrogate({"advantages": torch.zeros(1, 2)}, {})


class TestSurrogateIsScopedToCrossEntropy:
    """A datum may carry ``weights`` under a ratio loss (the SDK reuses the
    datum shape); the surrogate must not fire on it and rewrite advantages."""

    class _Stub:
        def __init__(self):
            self.sent = []

        async def fwd_bwd(self, payload, processing=None, router_replay=None):
            self.sent.append(payload)
            return {"batch": {"logprobs": payload["kwargs"]["input_ids"].to(torch.float32)}}

    @pytest.mark.parametrize(
        "loss_fn,backend_loss", [("ppo", "verl_grpo"), ("importance_sampling", "verl_grpo")]
    )
    def test_ratio_loss_advantages_survive(self, loss_fn, backend_loss):
        import asyncio

        from arctic_platform.integrations.tinker.cortex import CortexTinkerBackend

        datum = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            loss_fn_inputs={
                "advantages": TensorData(dtype="float32", data=[0.0, 0.25, 0.25]),
                "weights": TensorData(dtype="float32", data=[0.0, 9.0, 9.0]),
                "logprobs": TensorData(dtype="float32", data=[-1.0, -1.0, -1.0]),
            },
        )
        batch, _ = datum_list_to_arctic_batch(
            [datum], loss_fn, None,
            max_prompt_length=MPL, max_response_length=MRL, pad_token_id=0,
        )
        assert batch["processing"]["loss_fn"] == backend_loss

        stub = self._Stub()
        asyncio.run(CortexTinkerBackend(stub).fwd_bwd(batch))
        sent = stub.sent[0]
        # Advantages are the datum's own, not -weights (which would be -9.0).
        assert float(sent["context"]["advantages"].max()) == pytest.approx(0.25)
        assert "logprob_weights_shifted" not in sent["kwargs"]
        assert "logprob_weights_shifted" not in sent["context"]
        assert sent["processing"]["config"]["batch_num_tokens"] != 1


class TestGradientEquivalence:
    """The load-bearing test: the gradient Cortex computes must equal the one
    Tinker's cross-entropy contract promises, ``dL/dlogprobs = -weights``."""

    @pytest.mark.parametrize(
        "tokens,weights",
        [
            ([1, 2, 3, 4, 5], [0.0, 0.0, 0.5, -1.5, 2.0]),   # mixed signs
            ([1, 2, 3], [0.0, 1.0, 1.0]),                      # plain SFT weights
            ([1, 2, 3, 4], [0.0, -0.25, -0.5, -0.75]),         # all negative
        ],
    )
    def test_surrogate_reproduces_tinker_cross_entropy(self, tokens, weights):
        out, slices = _pack([_ce_datum(tokens, weights)])
        body, meta = _grpo_surrogate(
            {k: torch.as_tensor(v) for k, v in out["batch"].items()}, out["meta"]
        )

        logprobs = torch.zeros(body["advantages"].shape, dtype=torch.float32)
        logprobs.requires_grad_(True)
        loss = _grpo_loss(
            logprobs,
            body["advantages"].to(torch.float32),
            body["response_mask"],
            meta["batch_num_tokens"],
        )
        loss.backward()

        start, end, _ = slices[0]
        got = [float(logprobs.grad[0, start + k]) for k in range(end - start)]
        # Tinker's contract: L = sum(-logprobs * weights), so dL/dlogprob_k is
        # -weights[k]. Positions outside the response window are masked and
        # carry zero weight, so they agree trivially.
        assert got == pytest.approx([-w for w in weights], abs=1e-6)

    def test_ratio_is_exactly_one_so_clipping_cannot_engage(self):
        """The surrogate leans on ratio == 1. grpo reaches that by defaulting
        π_old to this same forward's log-probs, so it is exact rather than
        approximate -- a tiny eps_clip must still not bite."""
        out, slices = _pack([_ce_datum([1, 2, 3, 4], [0.0, 3.0, -3.0, 3.0])])
        body, meta = _grpo_surrogate(
            {k: torch.as_tensor(v) for k, v in out["batch"].items()}, out["meta"]
        )
        logprobs = torch.randn(body["advantages"].shape, dtype=torch.float32)
        logprobs.requires_grad_(True)
        _grpo_loss(
            logprobs, body["advantages"].to(torch.float32),
            body["response_mask"], meta["batch_num_tokens"], eps_clip=1e-6,
        ).backward()

        start, end, _ = slices[0]
        got = [float(logprobs.grad[0, start + k]) for k in range(end - start)]
        assert got == pytest.approx([0.0, -3.0, 3.0, -3.0], abs=1e-6)

    def test_a_dropped_negation_would_be_caught(self):
        """Guard the guard: the same pipeline with the sign left unflipped must
        disagree, or the test above would pass on a broken implementation."""
        weights = [0.0, 0.5, -1.5]
        out, slices = _pack([_ce_datum([1, 2, 3], weights)])
        body = {k: torch.as_tensor(v) for k, v in out["batch"].items()}
        wrong = body["logprob_weights_shifted"].to(torch.float32)  # no negation

        logprobs = torch.zeros(wrong.shape, dtype=torch.float32, requires_grad=True)
        _grpo_loss(logprobs, wrong, body["response_mask"], 1).backward()
        start, end, _ = slices[0]
        got = np.array([float(logprobs.grad[0, start + k]) for k in range(end - start)])
        assert not np.allclose(got, np.array([-w for w in weights]))
