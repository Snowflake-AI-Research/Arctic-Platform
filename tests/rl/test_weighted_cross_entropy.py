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

"""``weighted_cross_entropy``: value, gradient, distributed scale, packed microbatches and every rejection.

CPU only. The loss runs no collective, so a data- and sequence-parallel split is reproduced by calling it on each
rank's packed token window with the ``dp_size`` the server stamps, then combining the gradients as DeepSpeed does:
summed over the sequence-parallel ranks of a replica and averaged over the data-parallel replicas.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import torch

from arctic_platform.common.registry import LOSS_FNS
from arctic_platform.common.registry import PUBLIC_LOSS_FNS
from arctic_platform.common.utils.batch import finalize_fwd_bwd_metrics
from arctic_platform.common.utils.batch import ray_split_batch
from arctic_platform.common.utils.batch import unpack_batch
from arctic_platform.rl.processors import prepare_request_loss
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors.base_loss import REQUIRES_TOKEN_LOGPROBS
from arctic_platform.rl.processors.causal_cross_entropy import causal_cross_entropy_loss
from arctic_platform.rl.processors.packed_reduction import resolve_packed_loss_reduction
from arctic_platform.rl.processors.pipeline import metric_is_summed
from arctic_platform.rl.processors.pipeline import run_pipeline
from arctic_platform.rl.processors.weighted_cross_entropy import OBJECTIVE_SUM_METRIC
from arctic_platform.rl.processors.weighted_cross_entropy import WEIGHTED_TOKEN_COUNT_METRIC
from arctic_platform.rl.processors.weighted_cross_entropy import WEIGHTS_KEY
from arctic_platform.rl.processors.weighted_cross_entropy import weighted_cross_entropy_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

_LOSS = "weighted_cross_entropy"


def _request(seed: int = 0) -> dict:
    """Four left-aligned rows of different lengths, prediction-aligned signed weights zero outside loss_mask."""
    set_seed(seed)
    lengths = torch.tensor([8, 7, 6, 3])
    width = 8
    attention_mask = (torch.arange(width).unsqueeze(0) < lengths.unsqueeze(1)).long()
    # A row's last real position predicts nothing, and the first is a prompt token.
    loss_mask = ((torch.arange(width).unsqueeze(0) < (lengths - 1).unsqueeze(1)) & (torch.arange(width) > 0)).float()
    weights = (torch.randn(4, width) * loss_mask).to(torch.float32)
    logprobs = -torch.rand(4, width) * 3.0
    return dict(
        input_ids=torch.arange(1, 4 * width + 1).view(4, width) * attention_mask,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        weights=weights,
        logprobs=logprobs,
        objective=float(-(weights * logprobs).sum()),
    )


class TestRegistration(TestCasePlus):
    def test_public_and_needs_token_logprobs(self):
        self.assertIn(_LOSS, PUBLIC_LOSS_FNS)
        self.assertIs(LOSS_FNS[_LOSS], weighted_cross_entropy_loss)
        loss_object = resolve_loss(_LOSS)
        self.assertTrue(loss_object.has_capability(REQUIRES_TOKEN_LOGPROBS))
        self.assertFalse(loss_object.preserves_explicit_loss_scale_config())
        self.assertTrue(metric_is_summed(OBJECTIVE_SUM_METRIC))
        self.assertTrue(metric_is_summed(WEIGHTED_TOKEN_COUNT_METRIC))


class TestValueAndGradient(TestCasePlus):
    def test_unnormalized_signed_sum(self):
        logprobs = torch.tensor([[-1.0, -2.0, -0.5], [-0.25, -3.0, -1.5]], requires_grad=True)
        loss_mask = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
        weights = torch.tensor([[0.0, 0.5, -2.0], [1.5, -0.25, 0.0]])
        loss, metrics = weighted_cross_entropy_loss(
            {"logprobs": logprobs}, {"loss_mask": loss_mask, WEIGHTS_KEY: weights}, {}, {}, "cpu"
        )
        loss.backward()
        # -(0.5 * -2 + -2 * -0.5 + 1.5 * -0.25 + -0.25 * -3), exact in binary floating point.
        self.assertEqual(loss.item(), -0.375)
        torch_assert_equal(logprobs.grad, -weights)
        self.assertEqual(metrics, {OBJECTIVE_SUM_METRIC: -0.375, WEIGHTED_TOKEN_COUNT_METRIC: 4.0})

    def test_masked_positions_are_ignored_even_when_not_finite(self):
        logprobs = torch.tensor([-1.0, float("-inf"), float("nan")], requires_grad=True)
        loss_mask = torch.tensor([1.0, 0.0, 0.0])
        weights = torch.tensor([2.0, 0.0, 0.0])
        loss, _ = weighted_cross_entropy_loss(
            {"logprobs": logprobs}, {"loss_mask": loss_mask, WEIGHTS_KEY: weights}, {}, {}, "cpu"
        )
        loss.backward()
        self.assertEqual(loss.item(), 2.0)
        torch_assert_equal(logprobs.grad, torch.tensor([-2.0, 0.0, 0.0]))

    def test_zero_weights_give_a_connected_zero(self):
        logprobs = torch.tensor([-1.0, -2.0], requires_grad=True)
        loss, metrics = weighted_cross_entropy_loss(
            {"logprobs": logprobs}, {"loss_mask": torch.ones(2), WEIGHTS_KEY: torch.zeros(2)}, {}, {}, "cpu"
        )
        loss.backward()
        self.assertEqual(loss.item(), 0.0)
        torch_assert_equal(logprobs.grad, torch.zeros(2))
        self.assertEqual(metrics[WEIGHTED_TOKEN_COUNT_METRIC], 0.0)

    def test_loss_mask_weights_are_the_summed_causal_cross_entropy(self):
        request = _request()
        mask = request["loss_mask"]
        # 16 positions, a power of two, so dividing and multiplying by it is exact.
        batch_num_tokens = float(mask.sum())
        self.assertEqual(batch_num_tokens, 16.0)
        for dp_size in (1, 4):
            meta = {"batch_num_tokens": batch_num_tokens, "dp_size": dp_size}
            cce_logprobs = request["logprobs"].clone().requires_grad_(True)
            cce, _ = causal_cross_entropy_loss({"logprobs": cce_logprobs}, {"loss_mask": mask}, meta, {}, "cpu")
            cce.backward()
            wce_logprobs = request["logprobs"].clone().requires_grad_(True)
            wce, _ = weighted_cross_entropy_loss(
                {"logprobs": wce_logprobs}, {"loss_mask": mask, WEIGHTS_KEY: mask.clone()}, meta, {}, "cpu"
            )
            wce.backward()
            # A different summation order than the mean; 16 fp32 terms below 3 in magnitude.
            torch_assert_close(wce, cce * batch_num_tokens, rtol=0, atol=1e-5, msg=f"dp_size={dp_size}")
            torch_assert_equal(wce_logprobs.grad, cce_logprobs.grad * batch_num_tokens)


def _pack_rows(rows: list[int], request: dict, sp_size: int) -> dict:
    """Pack rows' valid tokens into one call padded to a multiple of ``sp_size``; padding has -inf logprobs."""
    lengths = request["attention_mask"][rows].sum(dim=1)
    valid = request["attention_mask"][rows].bool()
    packed = {key: request[key][rows][valid] for key in ("loss_mask", "weights", "logprobs")}
    pad = (-int(lengths.sum())) % sp_size
    packed["loss_mask"] = torch.cat([packed["loss_mask"], torch.zeros(pad)])
    packed["weights"] = torch.cat([packed["weights"], torch.zeros(pad)])
    packed["logprobs"] = torch.cat([packed["logprobs"], torch.full((pad,), float("-inf"))])
    packed["cu_seqlens"] = torch.nn.functional.pad(torch.cumsum(lengths, dim=0, dtype=torch.int32), (1, 0))
    packed["index"] = [(row, column) for row, n in zip(rows, lengths.tolist()) for column in range(n)] + [None] * pad
    return packed


class TestDistributedScale(TestCasePlus):
    def _run(self, dp_size: int, sp_size: int):
        request = _request(seed=dp_size * 10 + sp_size)
        leaf = request["logprobs"].clone().requires_grad_(True)
        rank_losses, objective_sums = [], []
        for replica in range(dp_size):
            call = _pack_rows(list(range(replica, 4, dp_size)), request, sp_size)
            width = call["loss_mask"].numel() // sp_size
            for sp_rank in range(sp_size):
                window = slice(sp_rank * width, (sp_rank + 1) * width)
                pieces = [leaf[i] if i is not None else torch.tensor(float("-inf")) for i in call["index"][window]]
                logprobs = (torch.stack(pieces) if pieces else torch.zeros(0)).unsqueeze(0)
                loss, metrics = weighted_cross_entropy_loss(
                    {"logprobs": logprobs},
                    {"loss_mask": call["loss_mask"][window].unsqueeze(0), WEIGHTS_KEY: call["weights"][window][None]},
                    {"cu_seqlens": call["cu_seqlens"], "dp_size": dp_size, "batch_num_tokens": 16.0},
                    {"dp_size": dp_size, "batch_num_tokens": 16.0},
                    "cpu",
                )
                loss.backward()
                rank_losses.append(loss.item())
                objective_sums.append(metrics[OBJECTIVE_SUM_METRIC])
        return request, leaf.grad / dp_size, rank_losses, objective_sums

    def test_gradient_and_reported_loss_are_split_invariant(self):
        for dp_size, sp_size in ((1, 1), (2, 1), (4, 1), (2, 2), (1, 4), (2, 3)):
            with self.subTest(dp_size=dp_size, sp_size=sp_size):
                request, grad, rank_losses, objective_sums = self._run(dp_size, sp_size)
                # Every position's gradient is exactly one weight times dp_size / dp_size.
                torch_assert_equal(grad, -request["weights"])
                # DSS reports the SP-summed rank losses averaged over replicas. The rank sums group the fp32
                # products differently from the reference; 16 terms of magnitude below 10 stay within 1e-5.
                self.assertAlmostEqual(sum(rank_losses) / dp_size, request["objective"], delta=1e-5)
                self.assertAlmostEqual(sum(objective_sums), request["objective"], delta=1e-5)

    def test_the_native_coordinator_reports_L_surrogate_in_the_summed_metric(self):
        for dp_size, sp_size in ((1, 1), (2, 1), (4, 1), (2, 2), (1, 4), (2, 3)):
            with self.subTest(dp_size=dp_size, sp_size=sp_size):
                request, _, rank_losses, objective_sums = self._run(dp_size, sp_size)
                results = [
                    {"avg_loss": loss, "metrics": {OBJECTIVE_SUM_METRIC: objective}}
                    for loss, objective in zip(rank_losses, objective_sums)
                ]
                metrics, summed = finalize_fwd_bwd_metrics(results)
                # The summed rank losses carry the dp_size training factor.
                self.assertAlmostEqual(summed, dp_size * request["objective"], delta=1e-4)
                self.assertAlmostEqual(metrics[OBJECTIVE_SUM_METRIC], request["objective"], delta=1e-5)


def _awaitable(value):
    async def _done():
        return value

    return _done()


class TestNativeServer(TestCasePlus):
    def test_forward_backward_reports_L_surrogate_in_the_summed_metric(self):
        import arctic_platform.common.ray_server as ray_server

        request = _request(seed=5)
        token_logprobs = -torch.rand(4 * 8 + 1) * 3.0
        objective = float(-(request["weights"] * token_logprobs[request["input_ids"]]).sum())
        frame = {
            "kwargs": {"input_ids": request["input_ids"], "attention_mask": request["attention_mask"]},
            "context": {"loss_mask": request["loss_mask"], WEIGHTS_KEY: request["weights"]},
            "processing": {"loss_fn": _LOSS, "post": ["compute_logprobs"], "config": {}},
        }

        class Remote:
            def remote(self, shard):
                _, batch, meta, processing = unpack_batch(shard)
                engine = _TableEngine(token_logprobs.clone().requires_grad_(True))
                out = run_pipeline(engine, (), batch, meta, processing, "cpu", backward=True, pack=True)
                return _awaitable({"avg_loss": out["avg_loss"], "metrics": out["metrics"]})

        original_get = ray_server.ray.get
        ray_server.ray.get = lambda refs: refs
        try:
            for dp_size in (1, 2, 4):
                with self.subTest(dp_size=dp_size):
                    server = object.__new__(ray_server.ArcticRLRayServer)
                    server._training_op_lock = asyncio.Lock()
                    server._training_failed = False
                    server._training_failure = None
                    server.jobs = {1: {"job_type": "training", "sp_size": 1}}
                    server.training_workers = [
                        type("Worker", (), {"forward_backward": Remote()})() for _ in range(dp_size)
                    ]
                    response = asyncio.run(
                        server.forward_backward(1, {**frame, "processing": dict(frame["processing"])})
                    )
                    # The summed rank losses carry the dp_size training factor.
                    self.assertAlmostEqual(response["avg_loss"], dp_size * objective, delta=1e-4)
                    self.assertAlmostEqual(response["metrics"][OBJECTIVE_SUM_METRIC], objective, delta=1e-5)
        finally:
            ray_server.ray.get = original_get


class _TableEngine:
    """Returns each token's logprob from a trainable table, so the backward reaches one leaf."""

    def __init__(self, table: torch.Tensor):
        self.table = table
        self.global_rank = 0
        self.backward_scales = []

    def train(self):
        pass

    def eval(self):
        pass

    def __call__(self, *args, **kwargs):
        ids = kwargs["input_ids"]
        ids = ids.unsqueeze(0) if ids.ndim == 1 else ids
        return SimpleNamespace(logits=torch.zeros(*ids.shape, 2), logprobs=self.table[ids])

    def backward(self, loss, scale_wrt_gas=True):
        self.backward_scales.append(scale_wrt_gas)
        loss.backward()


class _RefusingEngine(_TableEngine):
    def __call__(self, *args, **kwargs):
        raise AssertionError("a rejected request must fail before the forward")


def _pipeline_batch(request: dict) -> dict:
    return {
        "input_ids": request["input_ids"],
        "attention_mask": request["attention_mask"],
        "loss_mask": request["loss_mask"],
        WEIGHTS_KEY: request["weights"],
    }


class TestPackedMicrobatches(TestCasePlus):
    def test_microbatch_contributions_add(self):
        request = _request(seed=3)
        token_logprobs = -torch.rand(4 * 8 + 1) * 3.0
        processing = {"loss_fn": _LOSS, "post": [], "config": {}}
        meta = {"pad_token_id": 0, "batch_num_tokens": 16.0, "dp_size": 2}
        expected_objective = float(-(request["weights"] * token_logprobs[request["input_ids"]]).sum())

        runs = []
        for max_tokens_per_mb in (64, 8):
            engine = _TableEngine(token_logprobs.clone().requires_grad_(True))
            out = run_pipeline(
                engine,
                (),
                _pipeline_batch(request),
                meta,
                processing,
                "cpu",
                backward=True,
                pack=True,
                max_tokens_per_mb=max_tokens_per_mb,
            )
            self.assertTrue(all(scale is False for scale in engine.backward_scales))
            runs.append((len(engine.backward_scales), out, engine.table.grad))

        (one_call, single, single_grad), (calls, split, split_grad) = runs
        self.assertEqual(one_call, 1)
        self.assertGreater(calls, 1)
        expected_grad = torch.zeros_like(token_logprobs).index_add_(
            0, request["input_ids"].reshape(-1), -request["weights"].reshape(-1) * 2
        )
        for out, grad in ((single, single_grad), (split, split_grad)):
            # Every real token id is distinct, so each table entry receives exactly one weight times dp_size.
            torch_assert_equal(grad, expected_grad)
            self.assertAlmostEqual(out["avg_loss"], 2 * expected_objective, delta=1e-5)
            self.assertAlmostEqual(out["metrics"][OBJECTIVE_SUM_METRIC], expected_objective, delta=1e-5)
            self.assertEqual(out["metrics"][WEIGHTED_TOKEN_COUNT_METRIC], float((request["weights"] != 0).sum()))

    def test_packed_reduction_is_additive(self):
        request = _request()
        microbatches = [
            {"input_ids": request["input_ids"][rows], "loss_mask": request["loss_mask"][rows], WEIGHTS_KEY: w}
            for rows, w in (([0, 1], request["weights"][[0, 1]]), ([2, 3], request["weights"][[2, 3]]))
        ]
        reduction = resolve_packed_loss_reduction({"loss_fn": _LOSS, "config": {}}, microbatches)
        self.assertTrue(reduction.loss_is_additive)
        self.assertEqual(reduction.loss_scales, (1.0, 1.0))


def _valid_context() -> dict:
    return {
        "input_ids": torch.tensor([[5, 6, 7]]),
        "loss_mask": torch.tensor([[0.0, 1.0, 1.0]]),
        WEIGHTS_KEY: torch.tensor([[0.0, -1.0, 2.0]]),
    }


# name -> (context update, config, error pattern); a None value removes the key.
_REJECTIONS = {
    "missing weights": ({WEIGHTS_KEY: None}, {}, f"requires context\\['{WEIGHTS_KEY}'\\]"),
    "float64 weights": ({WEIGHTS_KEY: torch.tensor([[0.0, -1.0, 2.0]], dtype=torch.float64)}, {}, "float32"),
    "bfloat16 weights": ({WEIGHTS_KEY: torch.tensor([[0.0, -1.0, 2.0]], dtype=torch.bfloat16)}, {}, "float32"),
    "integer weights": ({WEIGHTS_KEY: torch.tensor([[0, -1, 2]])}, {}, "float32"),
    "list weights": ({WEIGHTS_KEY: [[0.0, -1.0, 2.0]]}, {}, "float32"),
    "shape mismatch": ({WEIGHTS_KEY: torch.tensor([[0.0, -1.0]])}, {}, "must match loss_mask"),
    "nan weight": ({WEIGHTS_KEY: torch.tensor([[0.0, float("nan"), 2.0]])}, {}, "must be finite"),
    "infinite weight": ({WEIGHTS_KEY: torch.tensor([[0.0, float("inf"), 2.0]])}, {}, "must be finite"),
    "weight outside loss_mask": (
        {WEIGHTS_KEY: torch.tensor([[0.5, -1.0, 2.0]])},
        {},
        "must be zero where loss_mask is 0; found 1 nonzero weight",
    ),
    "temperature": ({"temperature": 0.7}, {}, "requires untempered model log probabilities"),
    "tensor temperature": ({"temperature": torch.tensor(2.0)}, {}, "requires untempered model log probabilities"),
    "config key": ({}, {"loss_agg_mode": "token-mean"}, "Unknown config keys for loss_fn 'weighted_cross_entropy'"),
    "missing loss_mask": ({"loss_mask": None}, {}, "requires context\\['loss_mask'\\]"),
    "ignored label": ({"labels": torch.tensor([[6, -100, 8]])}, {}, "must be zero where labels use ignore_index"),
}


def _rejected_context(update: dict) -> dict:
    context = _valid_context()
    for key, value in update.items():
        if value is None:
            context.pop(key)
        else:
            context[key] = value
    return context


class TestRejections(TestCasePlus):
    def test_the_loss_rejects(self):
        for name, (update, config, pattern) in _REJECTIONS.items():
            with self.subTest(name), self.assertRaisesRegex(ValueError, pattern):
                context = _rejected_context(update)
                logprobs = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
                weighted_cross_entropy_loss({"logprobs": logprobs}, context, {}, config, "cpu")

    def test_the_whole_request_is_rejected_before_sharding(self):
        for name, (update, config, pattern) in _REJECTIONS.items():
            context = _rejected_context(update)
            input_ids = context.pop("input_ids")
            requests = {
                # The Cortex frame the zone receives, and the flattened request DSS hands the loss.
                "cortex": {"kwargs": {"input_ids": input_ids}, "context": context},
                "dss": {"input_ids": input_ids, "context": context},
                "ap": {"batch": {"input_ids": input_ids, **context}, "meta": {}},
            }
            for form, request in requests.items():
                request["processing"] = {"loss_fn": _LOSS, "post": ["compute_logprobs"], "config": dict(config)}
                with self.subTest(name=name, form=form), self.assertRaisesRegex(ValueError, pattern):
                    prepare_request_loss(request)

    def test_a_packed_call_is_rejected_before_the_forward(self):
        for name, (update, config, pattern) in _REJECTIONS.items():
            context = _rejected_context(update)
            batch = {"attention_mask": torch.ones(1, 3, dtype=torch.long)}
            meta = {"pad_token_id": 0}
            for key, value in context.items():
                # Forward scalars ride on meta; everything else is per token.
                (meta if key == "temperature" else batch)[key] = value
            processing = {"loss_fn": _LOSS, "post": [], "config": dict(config)}
            with self.subTest(name), self.assertRaisesRegex(ValueError, pattern):
                run_pipeline(
                    _RefusingEngine(torch.zeros(9)), (), batch, meta, processing, "cpu", backward=True, pack=True
                )

    def test_a_valid_request_passes_every_preflight(self):
        request = {"kwargs": {"input_ids": torch.tensor([[5, 6, 7]])}, "context": _valid_context()}
        request["context"].pop("input_ids")
        request["processing"] = {"loss_fn": _LOSS, "config": {"dp_size": 2, "batch_num_tokens": 2.0}}
        self.assertTrue(resolve_loss(_LOSS).is_legacy_adapter_for(weighted_cross_entropy_loss))
        prepare_request_loss(request)

    def test_non_finite_logprob_on_a_loss_mask_position(self):
        logprobs = torch.tensor([[-1.0, float("-inf"), -3.0]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "1 non-finite logprob value"):
            weighted_cross_entropy_loss({"logprobs": logprobs}, _valid_context(), {}, {}, "cpu")
        logprobs = torch.tensor([[-1.0, -2.0, float("nan")]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "non-finite logprob"):
            weighted_cross_entropy_loss({"logprobs": logprobs}, _valid_context(), {}, {}, "cpu")

    def test_missing_logprobs_names_compute_logprobs(self):
        with self.assertRaisesRegex(ValueError, "compute_logprobs"):
            weighted_cross_entropy_loss({"logits": torch.zeros(1, 3, 4)}, _valid_context(), {}, {}, "cpu")

    def test_conflicting_or_invalid_scale(self):
        logprobs = torch.tensor([[-1.0, -2.0, -3.0]], requires_grad=True)
        with self.assertRaisesRegex(ValueError, "conflicting dp_size"):
            weighted_cross_entropy_loss(
                {"logprobs": logprobs}, _valid_context(), {"dp_size": 2}, {"dp_size": 4}, "cpu"
            )
        with self.assertRaisesRegex(ValueError, "dp_size must be an integer"):
            weighted_cross_entropy_loss({"logprobs": logprobs}, _valid_context(), {"dp_size": 0}, {}, "cpu")
        with self.assertRaisesRegex(ValueError, "batch_num_tokens requires an explicit dp_size"):
            weighted_cross_entropy_loss({"logprobs": logprobs}, _valid_context(), {"batch_num_tokens": 2.0}, {}, "cpu")


class TestShardingWithRows(TestCasePlus):
    def test_weights_shard_with_their_rows_on_the_native_server(self):
        request = _request()
        frame = {
            "kwargs": {"input_ids": request["input_ids"], "attention_mask": request["attention_mask"]},
            "context": {"loss_mask": request["loss_mask"], WEIGHTS_KEY: request["weights"]},
            "processing": {"loss_fn": _LOSS, "post": ["compute_logprobs"], "config": {}},
        }
        shards, _ = ray_split_batch(frame, 2)
        for shard, rows in zip(shards, ([0, 1], [2, 3])):
            self.assertNotIn(WEIGHTS_KEY, shard["meta"])
            torch_assert_equal(shard["batch"][WEIGHTS_KEY], request["weights"][rows])
            torch_assert_equal(shard["batch"]["loss_mask"], request["loss_mask"][rows])
            self.assertEqual(shard["meta"]["dp_size"], 2)
