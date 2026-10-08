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
import math
from types import SimpleNamespace

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

    def __init__(self, logprobs=_ECHO_INPUT_IDS, batch_key="batch", metrics=None, training_gpus=1, peft=None):
        self.config = SimpleNamespace(training_gpus=training_gpus, training=SimpleNamespace(peft=peft))
        self.sent: list[dict] = []
        self.stepped: list[float | None] = []
        self.synced: list[str | None] = []
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

    async def sync_weights(self, weight_format=None):
        self.synced.append(weight_format)
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
            "old_log_probs_shifted": -0.01 * input_ids.to(torch.float32) * response_mask,
        },
        "meta": {"global_batch_size": 3},
    }


def _ratio_loss_batch(ratio_clip=(0.8, 1.2)):
    batch = _router_batch()
    batch["processing"] = {"loss_fn": "verl_grpo", "ratio_clip": ratio_clip}
    return batch


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

    def test_old_log_probs_are_not_sent_without_a_ratio_loss(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(_router_batch()))
        assert "old_log_probs" not in client.sent[0]["kwargs"]
        assert "old_log_probs_shifted" not in client.sent[0]["context"]


class TestRatioLosses:
    """Tinker's ``importance_sampling`` and ``ppo`` divide by the sampler's log-probs.

    If Cortex falls back to the trainer's own log-probs the ratio is always 1,
    and a sampler/trainer mismatch goes uncorrected until training collapses.
    """

    def test_sampler_log_probs_reach_cortex_in_the_sent_frame(self):
        batch = _ratio_loss_batch()
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(batch))
        (payload,) = client.sent
        order, valid = _align_plan(batch["batch"]["attention_mask"])
        expected = _align(batch["batch"], order, valid)["old_log_probs_shifted"]
        torch_assert_equal(payload["context"]["old_log_probs_shifted"], expected)
        assert payload["context"]["old_log_probs_shifted"].dtype == torch.float32

    def test_ppo_bounds_become_grpo_clip_range(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(_ratio_loss_batch((0.9, 1.3))))
        config = client.sent[0]["processing"]["config"]
        assert config["eps_clip"] == pytest.approx(0.1)
        assert config["eps_clip_higher"] == pytest.approx(0.3)

    def test_unbounded_ratio_is_never_clamped(self):
        """``importance_sampling``: ``grpo`` clamps to ``[1 - eps, 1 + eps_higher]``,
        so the bounds must contain every positive finite ratio."""
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(_ratio_loss_batch((0.0, math.inf))))
        config = client.sent[0]["processing"]["config"]
        assert 1.0 - config["eps_clip"] == 0.0
        assert 1.0 + config["eps_clip_higher"] == torch.finfo(torch.float32).max
        assert math.isfinite(config["eps_clip_higher"]), "the config travels as JSON"
        ratio = torch.tensor([1e-30, 0.5, 1.0, 7.0, 1e30])
        clamped = torch.clamp(ratio, 1.0 - config["eps_clip"], 1.0 + config["eps_clip_higher"])
        torch_assert_equal(clamped, ratio)

    def test_ratio_clip_is_not_forwarded(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client).fwd_bwd(_ratio_loss_batch()))
        assert "ratio_clip" not in client.sent[0]["processing"]

    def test_missing_sampler_log_probs_raise(self):
        batch = _ratio_loss_batch()
        del batch["batch"]["old_log_probs_shifted"]
        with pytest.raises(ValueError, match="old_log_probs_shifted"):
            asyncio.run(CortexTinkerBackend(_StubClient()).fwd_bwd(batch))


class TestRowPadding:
    """Cortex refuses a batch with fewer rows than training ranks."""

    def test_short_batch_is_padded_without_loss_signal(self):
        batch = _ratio_loss_batch()
        client = _StubClient()
        out = asyncio.run(CortexTinkerBackend(client, min_rows=5).fwd_bwd(batch))
        (payload,) = client.sent
        assert payload["kwargs"]["input_ids"].shape[0] == 5
        torch_assert_equal(
            payload["kwargs"]["attention_mask"][3:], payload["kwargs"]["attention_mask"][:1].expand(2, -1)
        )
        assert not payload["context"]["loss_mask"][3:].any()
        assert not payload["context"]["advantages"][3:].any()
        assert not payload["context"]["old_log_probs_shifted"][3:].any()
        assert out["batch"]["logprobs"].shape[0] == 3
        mask = batch["batch"]["attention_mask"]
        torch_assert_equal(out["batch"]["logprobs"].to(torch.long) * mask, batch["batch"]["input_ids"] * mask)

    def test_full_batch_is_untouched(self):
        client = _StubClient()
        asyncio.run(CortexTinkerBackend(client, min_rows=3).fwd_bwd(_router_batch()))
        assert client.sent[0]["kwargs"]["input_ids"].shape[0] == 3

    def test_build_handlers_pads_to_the_jobs_training_gpus(self):
        client = _StubClient(training_gpus=4)
        asyncio.run(build_handlers(client)["fwd_bwd_handler"](_router_batch()))
        assert client.sent[0]["kwargs"]["input_ids"].shape[0] == 4


class TestIsolatedSequences:
    """Rows lengthened past half a micro-batch, so Cortex never packs two together."""

    CAPACITY = 8  # the router frame's width, as serve provisions it

    def _run(self, min_rows=1):
        batch = _ratio_loss_batch()
        client = _StubClient()
        backend = CortexTinkerBackend(client, min_rows=min_rows, isolate_capacity=self.CAPACITY)
        out = asyncio.run(backend.fwd_bwd(batch))
        (payload,) = client.sent
        return batch, payload, out

    def test_no_two_rows_fit_one_micro_batch(self):
        from arctic_platform.rl.processors.microbatch import _ffd_allocate

        _, payload, _ = self._run()
        lengths = payload["kwargs"]["attention_mask"].sum(dim=1).tolist()
        assert min(lengths) == self.CAPACITY // 2 + 1
        groups = _ffd_allocate(lengths, self.CAPACITY, min_groups=1)
        assert sorted(len(g) for g in groups) == [1, 1, 1]

    def test_filler_follows_the_real_tokens_and_carries_no_loss(self):
        batch, payload, _ = self._run()
        real = batch["batch"]["attention_mask"].sum(dim=1).tolist()
        sent_mask = payload["kwargs"]["attention_mask"]
        for row, n in enumerate(real):
            assert sent_mask[row, :n].all()
            assert not payload["context"]["loss_mask"][row, n:].any()
            assert not payload["context"]["advantages"][row, n:].any()
            assert not payload["context"]["old_log_probs_shifted"][row, n:].any()

    def test_logprobs_return_in_the_routers_frame(self):
        batch, _, out = self._run(min_rows=4)
        mask = batch["batch"]["attention_mask"]
        assert out["batch"]["logprobs"].shape == mask.shape
        torch_assert_equal(out["batch"]["logprobs"].to(torch.long) * mask, batch["batch"]["input_ids"] * mask)

    def test_rows_wider_than_the_frame_refused(self):
        backend = CortexTinkerBackend(_StubClient(), isolate_capacity=64)
        with pytest.raises(ValueError, match="cannot extend rows of width 8 to 33"):
            asyncio.run(backend.fwd_bwd(_router_batch()))


class TestCortexResponseShapes:
    """Cortex returns log-probs in a different place per verb.

    Observed against a live QA6 job: ``forward`` replies
    ``{job_id, logprobs: tensor[B, T]}`` while ``forward-backward`` replies
    ``{avg_loss, job_id, metrics, post_process_outputs: {logprobs: [[...]]}}``.
    Both are padded to full width. The published API spec says
    ``post_process_outputs`` is always empty, so it is these shapes -- not the
    document -- that the code has to match.
    """

    def _client_returning(self, response: dict):
        class _Fixed:
            def __init__(self):
                self.sent = []

            async def fwd_bwd(self, payload, processing=None, router_replay=None):
                self.sent.append(payload)
                return response

            async def fwd_no_grad(self, payload, processing=None, reference_model=False):
                self.sent.append(payload)
                return response

        return _Fixed()

    def test_nested_lists_under_post_process_outputs(self):
        batch = _router_batch()
        ids = batch["batch"]["input_ids"]
        order, valid = _align_plan(batch["batch"]["attention_mask"])
        aligned = _align(batch["batch"], order, valid)["input_ids"]
        client = self._client_returning(
            {
                "avg_loss": -0.5,
                "metrics": {"approx_kl": 0.0},
                "post_process_outputs": {"logprobs": aligned.to(torch.float32).tolist()},
            }
        )
        out = asyncio.run(CortexTinkerBackend(client).fwd_bwd(batch))
        mask = batch["batch"]["attention_mask"]
        torch_assert_equal(out["batch"]["logprobs"].to(torch.long) * mask, ids * mask)

    def test_top_level_tensor(self):
        batch = _router_batch()
        ids = batch["batch"]["input_ids"]
        order, valid = _align_plan(batch["batch"]["attention_mask"])
        aligned = _align(batch["batch"], order, valid)["input_ids"]
        client = self._client_returning({"job_id": "j", "logprobs": aligned.to(torch.float32)})
        out = asyncio.run(CortexTinkerBackend(client).fwd_bwd(batch))
        mask = batch["batch"]["attention_mask"]
        torch_assert_equal(out["batch"]["logprobs"].to(torch.long) * mask, ids * mask)


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

    @pytest.mark.parametrize(("peft", "weight_format"), [(None, None), ({"peft_type": "Lora", "r": 8}, "lora")])
    def test_lora_runs_sync_only_the_adapter(self, peft, weight_format):
        client = _StubClient(peft=peft)
        asyncio.run(CortexTinkerBackend(client).sync_weights())
        assert client.synced == [weight_format]


class TestPromptLogprobs:
    """`compute_logprobs` (the distillation teacher's verb) reads these.

    The shape is what a live Cortex sampler returned for ``prompt_logprobs``:
    one entry per prompt position, ``None`` first, then a dict keyed by token
    id that may hold top-k extras besides the prompt token.
    """

    class _Sampler:
        def __init__(self, prompt_logprobs):
            self.prompt_logprobs = prompt_logprobs
            self.params = []

        async def generate(self, prompts, sampling_params=None):
            self.params.append(sampling_params)
            result = {"token_ids": [9], "logprobs": [{"9": {"logprob": -0.2, "rank": 1}}], "finish_reason": "length"}
            if self.prompt_logprobs is not None:
                result["prompt_logprobs"] = self.prompt_logprobs
            return [result for _ in prompts]

    def _generate(self, sampler, prompt, **params):
        return asyncio.run(CortexTinkerBackend(sampler).generate(prompt, {"n": 1, "max_tokens": 1, **params}))

    def test_each_prompt_token_is_read_by_id(self):
        sampler = self._Sampler(
            [
                None,
                {"5": {"logprob": -9.0, "rank": 40}, "11": {"logprob": -0.1, "rank": 1}},
                {"6": {"logprob": -0.5, "rank": 1}},
            ]
        )
        out = self._generate(sampler, [4, 5, 6], prompt_logprobs=0)
        assert out["prompt_logprobs"] == [None, -9.0, -0.5]
        assert sampler.params[0]["prompt_logprobs"] == 0

    def test_not_asked_not_returned(self):
        out = self._generate(self._Sampler([None, {"5": -1.0}]), [4, 5])
        assert "prompt_logprobs" not in out

    @pytest.mark.parametrize(
        "prompt_logprobs, match",
        [
            (None, "prompt log-prob positions"),
            ([None, {"5": -1.0}], "prompt log-prob positions"),
            ([None, {"7": -1.0}, {"6": -1.0}], "prompt token 5"),
            ([None, None, {"6": -1.0}], "position 1"),
        ],
    )
    def test_a_gap_raises(self, prompt_logprobs, match):
        with pytest.raises(RuntimeError, match=match):
            self._generate(self._Sampler(prompt_logprobs), [4, 5, 6], prompt_logprobs=0)
