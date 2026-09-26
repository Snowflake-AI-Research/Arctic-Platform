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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Grouped class losses against full-vocabulary float64 references."""

from __future__ import annotations

import math

import pytest
import torch

from arctic_platform.rl.processors import LOSS_FNS
from arctic_platform.rl.processors import resolve_loss
from arctic_platform.rl.processors import run_pipeline

_B, _S, _V, _M = 2, 6, 40, 4
_KD_COEF = 0.7
_POLICY = dict(use_cispo_loss=True, is_weight_clip_max=5.0, dp_size=1)
_TEACHER = ("teacher_token_ids", "teacher_log_probs", "teacher_tail_log_prob")
_CONTEXT = ("labels", "old_log_probs_shifted", "advantages", "loss_mask", "kd_mask", *_TEACHER)
# The implementation intentionally performs model projection/log-softmax in
# float32 before the grouped objective compares against this float64 reference.
_FLOAT32_REFERENCE_ATOL = 2e-5
_FLOAT32_GRADIENT_ATOL = 3e-6


def _frame():
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randperm(_V, generator=generator)[: _B * _S].reshape(_B, _S)
    table = torch.randn(_V, _V, generator=generator) * 2
    teacher_logits = torch.randn(_B, _S, _V, generator=generator, dtype=torch.float64) * 2
    teacher_logits[0, 1] = -table[input_ids[0, 1]].double() * 3
    ids = teacher_logits.topk(_M, dim=-1).indices.int()
    table[input_ids[0, 2], ids[0, 2, 0]] = 40.0
    ids[1, 0, 2:] = -1
    teacher_probs = torch.softmax(teacher_logits, -1)
    teacher_log_probs = teacher_probs.gather(-1, ids.clamp(min=0)).log().masked_fill(ids < 0, math.nan).float()
    in_group = (torch.arange(_V) == ids[..., None]).any(-2)
    teacher_tail_log_prob = teacher_probs.masked_fill(in_group, 0.0).sum(-1).log().float()
    # Inactive rows deliberately carry unusable teacher data.
    ids[0, 4] = 7
    teacher_log_probs[0, 4] = math.nan
    teacher_tail_log_prob[:, 5] = math.nan
    return dict(
        table=table,
        teacher_probs=teacher_probs,
        input_ids=input_ids,
        attention_mask=torch.ones(_B, _S, dtype=torch.long),
        labels=torch.roll(input_ids, -1, dims=-1),
        old_log_probs_shifted=torch.full((_B, _S), -1.0),
        advantages=torch.linspace(-1, 1, _B * _S).reshape(_B, _S),
        loss_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1, 0, 1, 1, 0, 0]], dtype=torch.bool),
        kd_mask=torch.tensor([[1.0, 0.5, 1.0, 0.25, 0.0, 0.0], [1.0, 1.0, 0.75, 1.0, 1.0, 0.0]]),
        teacher_token_ids=ids,
        teacher_log_probs=teacher_log_probs,
        teacher_tail_log_prob=teacher_tail_log_prob,
    )


def _divergence(teacher, student, divergence, beta):
    def kl(left, right):
        values = left * (left / right).log()
        return values.masked_fill(left == 0, 0.0).sum()

    if divergence == "jsd":
        mixture = beta * teacher + (1 - beta) * student
        return beta * kl(teacher, mixture) + (1 - beta) * kl(student, mixture)
    return beta * kl(teacher, student) + (1 - beta) * kl(student, teacher)


def _reference(student_logits, frame, weights, divergence, beta):
    student_probs = torch.softmax(student_logits.double(), -1)
    kd_sum = student_tail = teacher_tail = 0.0
    for index in map(tuple, (weights > 0).nonzero().tolist()):
        slots = frame["teacher_token_ids"][index].long()
        slots = slots[slots >= 0]
        outside = torch.ones(student_probs.shape[-1], dtype=torch.bool)
        outside[slots] = False
        teacher, student = (
            torch.cat([probs[index][slots], probs[index][outside].sum()[None]])
            for probs in (frame["teacher_probs"], student_probs)
        )
        weight = float(weights[index])
        kd_sum = kd_sum + weight * _divergence(teacher, student, divergence, beta)
        teacher_tail += weight * float(teacher[-1])
        student_tail += weight * float(student[-1].detach())
    return kd_sum, teacher_tail, student_tail


class _Bigram:
    global_rank = 0

    def __init__(self, table, grouped=False):
        self.table = torch.nn.Parameter(table.clone())
        self.grouped = grouped
        self.forward_calls = 0

    def __call__(self, input_ids=None, labels=None, group_token_ids=None, **kwargs):
        self.forward_calls += 1
        logits = self.table[input_ids]
        if not self.grouped:
            assert group_token_ids is None
            return {"logits": logits}
        assert kwargs["dss_compute_logprobs"] is True
        log_probs = logits.log_softmax(-1)
        in_group = (torch.arange(logits.shape[-1]) == group_token_ids[..., None]).any(-2)
        head = log_probs.gather(-1, group_token_ids.clamp(min=0)).masked_fill(group_token_ids < 0, -math.inf)
        tail = log_probs.masked_fill(in_group, -math.inf).logsumexp(-1, keepdim=True)
        return {
            "logprobs": log_probs.gather(-1, labels[..., None]).squeeze(-1),
            "group_log_probs": torch.cat([head, tail], -1),
        }

    def train(self):
        pass

    def eval(self):
        pass

    def backward(self, loss, scale_wrt_gas=False):
        assert scale_wrt_gas is False
        loss.backward()


def _run(frame, config, *, loss_fn="grpo", max_tokens_per_mb=12, context_extra=None, engine=None):
    engine = engine or _Bigram(frame["table"])
    context = {key: frame[key] for key in _CONTEXT} | (context_extra or {})
    result = run_pipeline(
        engine,
        (),
        {key: frame[key] for key in ("input_ids", "attention_mask", "labels")}
        | ({"dss_compute_logprobs": True} if engine.grouped else {}),
        {key: value for key, value in context.items() if value is not None},
        {
            "loss_fn": loss_fn,
            "post": [] if engine.grouped else ["compute_logprobs"],
            "config": config,
        },
        "cpu",
        backward=True,
        pack=True,
        max_tokens_per_mb=max_tokens_per_mb or 10_000,
    )
    return engine, result


def _grpo_config(frame, **extra):
    return {
        **_POLICY,
        "batch_num_tokens": float(frame["loss_mask"].sum()),
        **extra,
    }


def _family(divergence, beta, prefix):
    return {} if (divergence, beta) == ("kl", 1.0) else {f"{prefix}divergence": divergence, f"{prefix}beta": beta}


def _kd_config(frame, divergence="kl", beta=1.0):
    normalizer = float(frame["kd_mask"].sum()) + 1.5
    return dict(
        kd_coef=_KD_COEF,
        kd_batch_num_tokens=normalizer,
        **_family(divergence, beta, "kd_"),
    )


@pytest.mark.parametrize(
    ("divergence", "beta", "grouped_head", "max_tokens_per_mb"),
    [
        ("kl", 1.0, True, None),
        ("kl", 0.0, False, 12),
        ("jsd", 0.5, True, 6),
        ("kl", 0.6, False, 6),
    ],
)
def test_grouped_distillation_matches_float64_value_metrics_and_gradient(
    divergence, beta, grouped_head, max_tokens_per_mb
):
    frame = _frame()
    weights = frame["kd_mask"]
    table = frame["table"].double().requires_grad_()
    logits = table[frame["input_ids"]]
    kd_sum, teacher_tail, _ = _reference(logits, frame, weights, divergence, beta)
    nll = -(weights * logits.log_softmax(-1).gather(-1, frame["labels"][..., None]).squeeze(-1)).sum()
    config = dict(
        kd_batch_num_tokens=float(weights.sum()) + 1.5,
        dp_size=2,
        **_family(divergence, beta, "kd_"),
    )
    expected = (0.75 * nll + 0.25 * kd_sum) * 2 / config["kd_batch_num_tokens"]
    (expected_grad,) = torch.autograd.grad(expected, table)

    engine, result = _run(
        frame,
        config,
        loss_fn="grouped_distillation",
        max_tokens_per_mb=max_tokens_per_mb,
        context_extra={
            "kd_mask": weights,
            "teacher_token_ids": frame["teacher_token_ids"].long(),
        },
        engine=_Bigram(frame["table"], grouped=grouped_head),
    )

    assert result["avg_loss"] == pytest.approx(expected.item(), rel=0, abs=_FLOAT32_REFERENCE_ATOL)
    torch.testing.assert_close(
        engine.table.grad.double(),
        expected_grad,
        rtol=0,
        atol=_FLOAT32_GRADIENT_ATOL,
    )
    keys = ("sft_nll_sum", "kd_sum", "teacher_tail_mass_sum", "kd_weight_sum")
    truth = (nll.item(), kd_sum.item(), teacher_tail, float(weights.sum()))
    assert [result["metrics"][key] for key in keys] == pytest.approx(
        truth,
        rel=0,
        abs=_FLOAT32_REFERENCE_ATOL,
    )
    assert set(result["batch"]) == {"logprobs"}


def test_native_packing_validates_each_final_window_once_before_model(monkeypatch):
    import arctic_platform.rl.processors.grouped_distillation as grouped_module

    frame = _frame()
    events = []
    original_validation = grouped_module.GroupedDistillationLoss.validation_callback

    def record_validation(self, context, config):
        assert context["input_ids"].shape == (1, 6)
        assert context["cu_seqlens"].tolist() == [0, 6]
        events.append("validation")
        return original_validation(self, context, config)

    class OrderedBigram(_Bigram):
        def __call__(self, *args, **kwargs):
            events.append("model")
            return super().__call__(*args, **kwargs)

    monkeypatch.setattr(
        grouped_module.GroupedDistillationLoss,
        "validation_callback",
        record_validation,
    )
    engine, _ = _run(
        frame,
        {
            "kd_batch_num_tokens": float(frame["kd_mask"].sum()),
            "dp_size": 1,
        },
        loss_fn="grouped_distillation",
        max_tokens_per_mb=6,
        engine=OrderedBigram(frame["table"], grouped=True),
    )

    assert engine.forward_calls == 2
    assert events == ["validation", "model", "validation", "model"]


@pytest.mark.parametrize(
    ("divergence", "beta", "max_tokens_per_mb"),
    [
        ("kl", 1.0, None),
        ("kl", 0.0, 12),
        ("kl", 0.3, 6),
        ("jsd", 0.5, 6),
    ],
)
def test_grpo_kd_term_adds_float64_value_metrics_and_gradient(divergence, beta, max_tokens_per_mb):
    frame = _frame()
    table = frame["table"].double().requires_grad_()
    kd_sum, teacher_tail, student_tail = _reference(
        table[frame["input_ids"]], frame, frame["kd_mask"], divergence, beta
    )
    kd = _kd_config(frame, divergence, beta)
    expected = _KD_COEF * kd_sum / kd["kd_batch_num_tokens"]
    (expected_grad,) = torch.autograd.grad(expected, table)
    config = _grpo_config(frame)

    (base_engine, base), (engine, result) = (
        _run(frame, terms, max_tokens_per_mb=max_tokens_per_mb) for terms in (config, config | kd)
    )

    assert result["avg_loss"] - base["avg_loss"] == pytest.approx(
        expected.item(),
        rel=0,
        abs=_FLOAT32_REFERENCE_ATOL,
    )
    torch.testing.assert_close(
        (engine.table.grad - base_engine.table.grad).double(),
        expected_grad,
        rtol=0,
        atol=_FLOAT32_GRADIENT_ATOL,
    )
    keys = (
        "kd_coef",
        "loss_term_kd",
        "kd_weight_sum",
        "kd_sum",
        "teacher_tail_mass_sum",
        "student_tail_mass_sum",
    )
    truth = (
        _KD_COEF,
        expected.item(),
        float(frame["kd_mask"].sum()),
        kd_sum.item(),
        teacher_tail,
        student_tail,
    )
    assert [result["metrics"][key] for key in keys] == pytest.approx(
        truth,
        rel=0,
        abs=_FLOAT32_REFERENCE_ATOL,
    )


def test_grpo_kd_off_matches_legacy_function_exactly():
    frame = _frame()
    config = _grpo_config(frame, kd_coef=0.0)
    context = {key: frame[key] for key in _CONTEXT if key not in _TEACHER}
    first = torch.randn(_B, _S, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)

    class_loss, class_metrics = resolve_loss("grpo").loss({"logprobs": first}, {}, context, config, "cpu")
    legacy_loss, legacy_metrics = LOSS_FNS["grpo"]({"logprobs": second}, {}, context, config, "cpu")
    class_loss.backward()
    legacy_loss.backward()

    assert torch.equal(class_loss, legacy_loss)
    assert class_metrics == legacy_metrics
    assert torch.equal(first.grad, second.grad)


def test_grpo_callbacks_overwrite_global_count_map_head_names_and_sum_metrics():
    frame = _frame()
    request = {
        "processing": {"loss_fn": "grpo", "config": {"kd_coef": 0.5, "dp_size": 1}},
        "context": {"kd_mask": frame["kd_mask"]},
    }
    loss_object = resolve_loss("grpo")
    loss_object.batching_callback(request)
    assert request["processing"]["config"]["kd_batch_num_tokens"] == pytest.approx(float(frame["kd_mask"].sum()))

    kwargs = {"dss_compute_logprobs": True}
    output_keys = ["logprobs"]
    context = {key: frame[key] for key in _CONTEXT} | {"input_ids": frame["input_ids"]}
    config = request["processing"]["config"]
    loss_object.validation_callback(context, config)
    loss_object.model_forward_callback(kwargs, context, config, output_keys)
    expected_ids = torch.where(
        frame["kd_mask"][..., None] > 0,
        frame["teacher_token_ids"],
        -1,
    )
    assert torch.equal(kwargs["group_token_ids"], expected_ids)
    assert output_keys == ["logprobs", "group_log_probs"]

    metrics = {"kd_sum": 1.0, "loss_term_kd": 2.0}
    loss_object.metrics_callback(
        [
            {"kd_sum": 1.0, "loss_term_kd": 2.0},
            {"kd_sum": 3.0, "loss_term_kd": 4.0},
        ],
        metrics,
    )
    assert metrics == {"kd_sum": 4.0, "loss_term_kd": 6.0}
    outputs = {
        "logits": torch.ones(1),
        "group_log_probs": torch.ones(1),
        "logprobs": torch.ones(1),
    }
    loss_object.output_callback(outputs)
    assert set(outputs) == {"logprobs"}


def test_standalone_callbacks_use_public_kd_names():
    frame = _frame()
    request = {
        "processing": {
            "loss_fn": "grouped_distillation",
            "config": {
                "kd_coef": 0.5,
                "kd_divergence": "jsd",
                "kd_beta": 0.4,
                "dp_size": 1,
            },
        },
        "context": {"kd_mask": frame["kd_mask"]},
    }
    loss_object = resolve_loss("grouped_distillation")
    loss_object.batching_callback(request)
    config = request["processing"]["config"]

    assert config["kd_batch_num_tokens"] == pytest.approx(float(frame["kd_mask"].sum()))
    context = {key: frame[key] for key in _CONTEXT} | {"input_ids": frame["input_ids"]}
    loss_object.validation_callback(context, config)

    with pytest.raises(ValueError, match="lambda_kd"):
        loss_object.validation_callback(context, {**config, "lambda_kd": 0.5})
    with pytest.raises(ValueError, match="kd_mask"):
        loss_object.validation_callback(
            {key: value for key, value in context.items() if key != "kd_mask"},
            config,
        )


def test_grpo_batching_and_objective_share_canonical_underflowed_weights():
    frame = _frame()
    kd_mask = torch.zeros_like(frame["kd_mask"], dtype=torch.float64)
    kd_mask[0, 0] = 1e-50
    config = _grpo_config(frame, kd_coef=0.5)
    request = {
        "processing": {"loss_fn": "grpo", "config": config},
        "context": {"kd_mask": kd_mask},
    }
    loss_object = resolve_loss("grpo")
    loss_object.batching_callback(request)

    assert config["kd_batch_num_tokens"] == 0.0
    context = {key: kd_mask if key == "kd_mask" else frame[key] for key in _CONTEXT} | {
        "input_ids": frame["input_ids"]
    }
    loss_object.validation_callback(context, config)

    class_logprobs = torch.randn(_B, _S, requires_grad=True)
    legacy_logprobs = class_logprobs.detach().clone().requires_grad_(True)
    logits = torch.randn(_B, _S, _V)
    class_loss, metrics = loss_object.loss(
        {"logprobs": class_logprobs, "logits": logits},
        {},
        context,
        config,
        "cpu",
    )
    legacy_loss, _ = LOSS_FNS["grpo"](
        {"logprobs": legacy_logprobs},
        {},
        context,
        config,
        "cpu",
    )
    class_loss.backward()
    legacy_loss.backward()

    assert torch.equal(class_loss, legacy_loss)
    assert torch.equal(class_logprobs.grad, legacy_logprobs.grad)
    assert metrics["kd_weight_sum"] == 0.0
    assert metrics["kd_sum"] == 0.0
    assert metrics["loss_term_kd"] == 0.0


def test_grpo_batching_rejects_weights_that_overflow_float32():
    request = {
        "processing": {"loss_fn": "grpo", "config": {"kd_coef": 0.5}},
        "context": {
            "kd_mask": torch.tensor(
                [[torch.finfo(torch.float64).max]],
                dtype=torch.float64,
            )
        },
    }

    with pytest.raises(ValueError, match="representable as finite float32"):
        resolve_loss("grpo").batching_callback(request)


@pytest.mark.parametrize(
    ("temperature", "match"),
    [
        (0.7, r"received \[0.7\]"),
        (torch.tensor([1.0, 0.5]), r"received \[1.0, 0.5\]"),
    ],
)
def test_temperature_refusal_reports_received_values(temperature, match):
    frame = _frame()
    context = {key: frame[key] for key in _CONTEXT} | {"input_ids": frame["input_ids"], "temperature": temperature}
    with pytest.raises(ValueError, match=match):
        resolve_loss("grpo").validation_callback(context, _grpo_config(frame) | _kd_config(frame))


@pytest.mark.parametrize("kd_coef", [0.0, 0.5])
def test_grpo_rejects_unknown_kd_config_even_when_kd_is_off(kd_coef):
    with pytest.raises(ValueError, match="kd_divergance"):
        resolve_loss("grpo").validation_callback(
            {},
            {**_POLICY, "kd_coef": kd_coef, "kd_divergance": "jsd"},
        )


@pytest.mark.parametrize("loss_fn", ["grpo", "grouped_distillation"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("kd_coef", True),
        ("kd_coef", "0.5"),
        ("kd_coef", None),
        ("kd_coef", 0.5 + 0j),
        ("kd_coef", math.inf),
        ("kd_beta", True),
        ("kd_beta", "0.5"),
        ("kd_beta", None),
        ("kd_beta", 0.5 + 0j),
        ("kd_beta", math.nan),
    ],
)
def test_kd_numeric_config_requires_finite_real_non_boolean_values(loss_fn, field, value):
    config = {
        **(_POLICY if loss_fn == "grpo" else {}),
        "kd_coef": 0.5,
        "kd_beta": 0.5,
        "kd_batch_num_tokens": 1.0,
        "dp_size": 1,
        field: value,
    }

    with pytest.raises(ValueError, match=field):
        resolve_loss(loss_fn).validation_callback({}, config)


@pytest.mark.parametrize("value", [True, "0.5", None, 0.5 + 0j, math.nan])
def test_grpo_kd_off_still_rejects_malformed_explicit_beta(value):
    with pytest.raises(ValueError, match="kd_beta"):
        resolve_loss("grpo").validation_callback(
            {},
            {**_POLICY, "kd_coef": 0.0, "kd_beta": value},
        )


def test_teacher_tensor_leading_shapes_must_match_exactly():
    context = {
        "input_ids": torch.zeros(2, 3, dtype=torch.long),
        "kd_mask": torch.ones(2, 3),
        "teacher_token_ids": torch.zeros(2, 3, 1, dtype=torch.int32),
        "teacher_log_probs": torch.full((3, 2, 1), math.log(0.4)),
        "teacher_tail_log_prob": torch.full((3, 2), math.log(0.6)),
    }
    with pytest.raises(ValueError, match="teacher_log_probs shape"):
        resolve_loss("grouped_distillation").validation_callback(
            context,
            {"kd_batch_num_tokens": 6.0, "dp_size": 1},
        )


@pytest.mark.parametrize(
    ("loss_fn", "weight_name", "config"),
    [
        (
            "grouped_distillation",
            "kd_mask",
            {"kd_batch_num_tokens": 2.0, "dp_size": 1},
        ),
        (
            "grpo",
            "kd_mask",
            {
                **_POLICY,
                "kd_coef": 0.5,
                "kd_batch_num_tokens": 2.0,
            },
        ),
    ],
)
def test_teacher_candidate_dimension_must_be_at_least_one(loss_fn, weight_name, config):
    context = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        weight_name: torch.ones(1, 2),
        "teacher_token_ids": torch.empty(1, 2, 0, dtype=torch.int32),
        "teacher_log_probs": torch.empty(1, 2, 0),
        "teacher_tail_log_prob": torch.zeros(1, 2),
    }

    with pytest.raises(ValueError, match="candidate dimension M must be at least 1"):
        resolve_loss(loss_fn).validation_callback(context, config)


@pytest.mark.parametrize("name", ["teacher_log_probs", "teacher_tail_log_prob"])
@pytest.mark.parametrize("dtype", [torch.complex64, torch.int64])
def test_teacher_log_probabilities_require_real_floating_point_dtype(name, dtype):
    context = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "kd_mask": torch.ones(1, 2),
        "teacher_token_ids": torch.zeros(1, 2, 1, dtype=torch.int32),
        "teacher_log_probs": torch.full((1, 2, 1), math.log(0.4)),
        "teacher_tail_log_prob": torch.full((1, 2), math.log(0.6)),
    }
    invalid = context[name].to(dtype)
    if invalid.is_complex():
        invalid = invalid + 7j
    context[name] = invalid

    with pytest.raises(ValueError, match=rf"{name} must use a real floating-point dtype"):
        resolve_loss("grouped_distillation").validation_callback(
            context,
            {"kd_batch_num_tokens": 2.0, "dp_size": 1},
        )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.uint16, torch.uint32, torch.uint64])
@pytest.mark.parametrize("grouped_head", [False, True])
def test_unsigned_teacher_ids_are_signed_before_masking_and_loss(dtype, grouped_head):
    inactive_id = torch.iinfo(dtype).max
    context = {
        "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "labels": torch.tensor([[1, 2]], dtype=torch.long),
        "kd_mask": torch.tensor([[1.0, 0.0]]),
        "teacher_token_ids": torch.tensor(
            [[[1, 2], [inactive_id, inactive_id]]],
            dtype=dtype,
        ),
        "teacher_log_probs": torch.tensor(
            [[[math.log(0.25), math.log(0.25)], [math.nan, math.nan]]],
        ),
        "teacher_tail_log_prob": torch.tensor([[math.log(0.5), math.nan]]),
    }
    config = {
        "kd_coef": 0.5,
        "kd_batch_num_tokens": 1.0,
        "dp_size": 1,
    }
    loss_object = resolve_loss("grouped_distillation")
    loss_object.validation_callback(context, config)

    model_kwargs = {"dss_compute_logprobs": True}
    output_keys = ["logprobs"]
    loss_object.model_forward_callback(model_kwargs, context, config, output_keys)
    group_token_ids = model_kwargs["group_token_ids"]
    assert group_token_ids.dtype == torch.int64
    assert group_token_ids.tolist() == [[[1, 2], [-1, -1]]]

    logits = torch.randn(1, 2, 5, requires_grad=True)
    log_probs = logits.log_softmax(-1)
    model_outputs = {
        "logprobs": log_probs.gather(-1, context["labels"][..., None]).squeeze(-1),
    }
    if grouped_head:
        in_group = (torch.arange(logits.shape[-1]) == group_token_ids[..., None]).any(-2)
        head = log_probs.gather(-1, group_token_ids.clamp(min=0)).masked_fill(group_token_ids < 0, -math.inf)
        tail = log_probs.masked_fill(in_group, -math.inf).logsumexp(-1, keepdim=True)
        model_outputs["group_log_probs"] = torch.cat([head, tail], -1)
    else:
        model_outputs["logits"] = logits

    loss, _ = loss_object.loss(model_outputs, {}, context, config, "cpu")
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()


def test_active_uint64_teacher_id_must_fit_signed_int64():
    context = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "kd_mask": torch.ones(1, 1),
        "teacher_token_ids": torch.full(
            (1, 1, 1),
            torch.iinfo(torch.uint64).max,
            dtype=torch.uint64,
        ),
        "teacher_log_probs": torch.full((1, 1, 1), math.log(0.5)),
        "teacher_tail_log_prob": torch.full((1, 1), math.log(0.5)),
    }

    with pytest.raises(ValueError, match="must fit in signed int64"):
        resolve_loss("grouped_distillation").validation_callback(
            context,
            {"kd_batch_num_tokens": 1.0, "dp_size": 1},
        )


@pytest.mark.parametrize(
    ("divergence", "beta"),
    [("kl", 1.0), ("jsd", 0.5)],
)
def test_full_logits_reject_active_teacher_ids_outside_student_vocabulary(divergence, beta):
    logits = torch.randn(1, 1, 3, requires_grad=True)
    context = {
        "input_ids": torch.zeros(1, 1, dtype=torch.long),
        "labels": torch.ones(1, 1, dtype=torch.long),
        "kd_mask": torch.ones(1, 1),
        "teacher_token_ids": torch.tensor([[[3]]], dtype=torch.int32),
        "teacher_log_probs": torch.full((1, 1, 1), math.log(0.4)),
        "teacher_tail_log_prob": torch.full((1, 1), math.log(0.6)),
    }
    config = {
        "kd_batch_num_tokens": 1.0,
        "dp_size": 1,
        "kd_divergence": divergence,
        "kd_beta": beta,
    }
    loss_object = resolve_loss("grouped_distillation")
    loss_object.validation_callback(context, config)
    with pytest.raises(ValueError, match=r"outside the student vocabulary \[0, 3\)"):
        loss_object.loss(
            {
                "logits": logits,
                "logprobs": logits.log_softmax(-1)[..., 1],
            },
            {},
            context,
            config,
            "cpu",
        )


def test_unpacked_batch_size_one_standalone_loss_preserves_tensor_ranks():
    frame = _frame()
    context = {
        "input_ids": frame["input_ids"][:1],
        "labels": frame["labels"][:1],
        "kd_mask": frame["kd_mask"][:1],
        **{name: frame[name][:1] for name in _TEACHER},
    }
    logits = frame["table"][context["input_ids"]].requires_grad_()
    model_outputs = {
        "logits": logits,
        "logprobs": logits.log_softmax(-1).gather(-1, context["labels"][..., None]).squeeze(-1),
    }
    config = {
        "kd_batch_num_tokens": float(context["kd_mask"].sum()),
        "dp_size": 1,
    }
    loss_object = resolve_loss("grouped_distillation")
    loss_object.validation_callback(context, config)
    loss, metrics = loss_object.loss(model_outputs, {}, context, config, "cpu")
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert metrics["kd_weight_sum"] == pytest.approx(
        float(context["kd_mask"].sum()),
        rel=0,
        abs=1e-7,
    )
    assert logits.grad is not None
