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
"""The Tinker router's verbs, lowered onto Cortex.

CPU only: the client is a stub that records what the binder sent, so these pin
the wire shape and the frame arithmetic without a Cortex job.
"""

from __future__ import annotations

import asyncio

import pytest
import torch

from arctic_platform.integrations.tinker.cortex import CortexTinkerBackend
from arctic_platform.integrations.tinker.cortex import _align
from arctic_platform.integrations.tinker.cortex import _align_plan
from arctic_platform.integrations.tinker.cortex import _unalign_rows
from arctic_platform.integrations.tinker.cortex import build_handlers
from arctic_platform.testing_utils import torch_assert_equal


_ECHO_INPUT_IDS = object()  # distinct from None, which means "omit log-probs"


class _StubClient:
    """Records the payload and replays log-probs the caller chooses.

    Default echoes ``input_ids`` back as log-probs, which makes a frame shift
    visible: the value at a position names the token it belongs to.
    """

    def __init__(self, logprobs=_ECHO_INPUT_IDS, batch_key="batch", metrics=None):
        self.sent: list[dict] = []
        self.stepped: list[float | None] = []
        self._logprobs = logprobs
        self._batch_key = batch_key
        self._metrics = metrics or {"loss": 1.0}

    async def _respond(self, payload):
        self.sent.append(payload)
        lp = self._logprobs
        if lp is _ECHO_INPUT_IDS:
            lp = payload["kwargs"]["input_ids"].to(torch.float32)
        body = {} if lp is None else {"logprobs": lp}
        return {self._batch_key: body, "metrics": self._metrics}

    async def fwd_bwd(self, payload, processing=None, router_replay=None):
        return await self._respond(payload)

    async def fwd_no_grad(self, payload, processing=None, reference_model=False):
        return await self._respond(payload)

    async def step(self, learning_rate=None):
        self.stepped.append(learning_rate)
        return {"ok": True}


def _router_batch():
    """A batch in the router's layout: ``[pad… prompt][response pad…]``."""
    attention_mask = torch.tensor(
        [[0, 0, 1, 1, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    input_ids = torch.arange(1, 25, dtype=torch.long).reshape(3, 8)
    response_mask = torch.tensor(
        [[0, 0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]],
        dtype=torch.long,
    )
    advantages = response_mask.to(torch.float32) * 0.5
    return {
        "batch": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "advantages": advantages,
            "old_log_probs": torch.zeros(3, 8),
        },
        "meta": {"global_batch_size": 3},
    }


class TestRowAlignment:
    def test_real_tokens_move_to_leading_columns(self):
        batch = _router_batch()["batch"]
        order, valid = _align_plan(batch["attention_mask"])
        aligned = _align(batch, order, valid)
        # Cortex's packer requires exactly this: leading real tokens, tail pads.
        lengths = batch["attention_mask"].sum(1)
        for row, n in enumerate(lengths.tolist()):
            assert aligned["attention_mask"][row, :n].all()
            assert not aligned["attention_mask"][row, n:].any()

    def test_scoring_tensors_ride_the_same_permutation(self):
        """`advantages` must stay on the token it scored, not just get sorted."""
        batch = _router_batch()["batch"]
        order, valid = _align_plan(batch["attention_mask"])
        aligned = _align(batch, order, valid)
        for row in range(batch["input_ids"].shape[0]):
            before = {
                int(t): float(a)
                for t, a, m in zip(batch["input_ids"][row], batch["advantages"][row], batch["attention_mask"][row])
                if m
            }
            after = {
                int(t): float(a)
                for t, a, m in zip(
                    aligned["input_ids"][row], aligned["advantages"][row], aligned["attention_mask"][row]
                )
                if m
            }
            assert before == after

    def test_unalign_restores_the_original_frame(self):
        batch = _router_batch()["batch"]
        order, valid = _align_plan(batch["attention_mask"])
        aligned = _align(batch, order, valid)
        restored = _unalign_rows(aligned["input_ids"].to(torch.float32), order)
        mask = batch["attention_mask"]
        torch_assert_equal(restored.to(torch.long) * mask, batch["input_ids"] * mask)

    def test_skipping_the_inverse_would_shift_rows(self):
        """Discriminative: the un-align is load-bearing, not decorative.

        Without it the log-probs stay in the aligned frame while the router
        slices the original one, which is a silent per-row shift rather than an
        error. If this ever stops differing, the inverse has become untested.
        """
        batch = _router_batch()["batch"]
        order, valid = _align_plan(batch["attention_mask"])
        aligned = _align(batch, order, valid)["input_ids"].to(torch.float32)
        mask = batch["attention_mask"]
        padded_rows = (mask.sum(1) != mask.shape[1]).nonzero().flatten()
        assert padded_rows.numel() > 0, "fixture must contain a padded row"
        assert not torch.equal(aligned * mask, batch["input_ids"] * mask)


class TestForwardBackwardWire:
    def test_payload_uses_a_loss_cortex_registers(self):
        client = _StubClient()
        backend = CortexTinkerBackend(client)
        asyncio.run(backend.fwd_bwd(_router_batch()))
        (payload,) = client.sent
        # ArcticTraining-dss registers causal_cross_entropy / grpo / grpo_echo_v1.
        # The router asks for verl_grpo, which would not resolve there.
        assert payload["processing"]["loss_fn"] == "grpo"
        # Cortex zones register `compute_logprobs`; `compute_entropy_and_logprobs`
        # does not exist there and the zone refuses before any model call.
        assert payload["processing"]["post"] == ["compute_logprobs"]
        assert set(payload) == {"args", "kwargs", "context", "processing"}

    def test_sends_left_aligned_rows(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(_router_batch()))
        mask = client.sent[0]["kwargs"]["attention_mask"]
        lengths = mask.sum(1)
        for row, n in enumerate(lengths.tolist()):
            assert mask[row, :n].all() and not mask[row, n:].any()

    def test_logprobs_come_back_in_the_routers_frame(self):
        """End to end through the binder: what the router reads must line up."""
        batch = _router_batch()
        client = _StubClient()  # echoes input_ids as logprobs
        out = asyncio.run(CortexTinkerBackend(client).fwd_bwd(batch))
        mask = batch["batch"]["attention_mask"]
        torch_assert_equal(
            out["batch"]["logprobs"].to(torch.long) * mask,
            batch["batch"]["input_ids"] * mask,
        )

    def test_old_log_probs_are_not_sent(self):
        """The server re-derives pi_old; shipping ours would be dead weight."""
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(_router_batch()))
        assert "old_log_probs" not in client.sent[0]["kwargs"]
        assert "old_log_probs" not in client.sent[0]["context"]


class TestMissingLogprobsFailLoud:
    @pytest.mark.parametrize("response", ["no_batch", "empty_batch"])
    def test_absent_logprobs_raise_instead_of_defaulting(self, response):
        """The router's fallback is an empty dict, which surfaces in the cookbook
        as a bare KeyError frames away. These log-probs also feed the
        sampler-vs-trainer KL check, so zeros would disable that alarm."""
        client = _StubClient(logprobs=None, batch_key="batch" if response == "empty_batch" else "other")
        with pytest.raises(RuntimeError, match="no per-token log-probs"):
            asyncio.run(CortexTinkerBackend(client).fwd_bwd(_router_batch()))


class TestStepAndHandlers:
    def test_step_forwards_the_learning_rate(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).step({"lr": 3e-6}))
        assert client.stepped == [3e-6]

    def test_step_without_overrides_sends_none(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).step(None))
        assert client.stepped == [None]

    def test_build_handlers_matches_init_tinker_state(self):
        import inspect

        from arctic_platform.integrations.tinker.router import init_tinker_state

        handlers = build_handlers(_StubClient())
        params = inspect.signature(init_tinker_state).parameters
        assert set(handlers) <= set(params), "handler kwargs must be accepted by the router"
        required = {n for n, p in params.items() if p.default is inspect.Parameter.empty and n.endswith("_handler")}
        assert required <= set(handlers)


class TestForwardVerb:
    def test_forward_sends_no_loss_and_no_context(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_no_grad(_router_batch()))
        (payload,) = client.sent
        assert "context" not in payload
        assert "loss_fn" not in payload["processing"]
        assert payload["processing"]["post"] == ["compute_logprobs"]

    def test_forward_logprobs_return_in_the_routers_frame(self):
        batch = _router_batch()
        out = asyncio.run(CortexTinkerBackend(_StubClient()).fwd_no_grad(batch))
        mask = batch["batch"]["attention_mask"]
        torch_assert_equal(
            out["batch"]["logprobs"].to(torch.long) * mask,
            batch["batch"]["input_ids"] * mask,
        )
