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

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

from arctic_platform.opd.examples.run_on_policy_distill import _SYNC_ZERO_RUN
from arctic_platform.opd.examples.run_on_policy_distill import _ds_config
from arctic_platform.opd.examples.run_on_policy_distill import _student_gpu_memory_utilization
from arctic_platform.opd.examples.run_on_policy_distill import build_parser
from arctic_platform.opd.examples.run_on_policy_distill import build_step_record
from arctic_platform.opd.examples.run_on_policy_distill import check_metrics
from arctic_platform.opd.examples.run_on_policy_distill import check_sync_result
from arctic_platform.opd.examples.run_on_policy_distill import iter_batches
from arctic_platform.opd.examples.run_on_policy_distill import kl_per_token_len_adj
from arctic_platform.opd.examples.run_on_policy_distill import load_prompt_records
from arctic_platform.opd.examples.run_on_policy_distill import lr_at
from arctic_platform.opd.examples.run_on_policy_distill import resolve_train_attn
from arctic_platform.opd.examples.run_on_policy_distill import prompt_content_from_record
from arctic_platform.opd.examples.run_on_policy_distill import student_sampling_params
from arctic_platform.opd.examples.run_on_policy_distill import tokenize_and_filter
from arctic_platform.opd.examples.run_on_policy_distill import tokenize_prompt
from arctic_platform.opd.examples.run_on_policy_distill import vllm_fa2_engine_kwargs
from arctic_platform.opd.examples.run_on_policy_distill import vllm_fp32_engine_kwargs
from arctic_platform.opd.examples.run_on_policy_distill import vllm_supports_fp32_lm_head


def test_linear_warmup_then_constant():
    assert lr_at(0, 2e-5, 8) == 2e-5 * 1 / 8
    assert lr_at(7, 2e-5, 8) == 2e-5
    assert lr_at(20, 2e-5, 8) == 2e-5


def test_cosine_after_warmup():
    peak = 2e-5
    assert lr_at(0, peak, 8, schedule="cosine", total_steps=152) == peak * 1 / 8
    assert lr_at(7, peak, 8, schedule="cosine", total_steps=152) == peak
    mid = lr_at(80, peak, 8, schedule="cosine", total_steps=152)
    expected_frac = (80 - 8) / (152 - 8)
    expected = peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * expected_frac)))
    assert math.isclose(mid, expected)
    last = lr_at(151, peak, 8, schedule="cosine", total_steps=152)
    last_frac = (151 - 8) / (152 - 8)
    last_expected = peak * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * last_frac)))
    assert math.isclose(last, last_expected)
    assert last > 0.1 * peak * 0.99


def test_load_prompt_records_and_max_prompt_len(tmp_path: Path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"prompt": "short"})
        + "\n"
        + json.dumps({"text": "also short"})
        + "\n"
        + json.dumps({"messages": [{"role": "user", "content": "chat"}]})
        + "\n",
        encoding="utf-8",
    )
    records = load_prompt_records(str(path))
    assert [prompt_content_from_record(row) for row in records] == [
        "short",
        "also short",
        [{"role": "user", "content": "chat"}],
    ]

    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            return list(range(len(content)))

    kept = tokenize_and_filter(FakeTokenizer(), records, max_prompt_len=5)
    assert kept == [[0, 1, 2, 3, 4], [0, 1, 2, 3]]


def test_tokenize_prompt_pins_non_thinking():
    class CaptureTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return [1, 2, 3]

    tok = CaptureTokenizer()
    assert tokenize_prompt(tok, "hello") == [1, 2, 3]
    assert tok.kwargs["enable_thinking"] is False
    assert tok.kwargs["add_generation_prompt"] is True
    assert tok.kwargs["tokenize"] is True


def test_tokenize_prompt_can_enable_thinking():
    class CaptureTokenizer:
        def __init__(self):
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):
            self.kwargs = kwargs
            return [7, 8, 9]

    tok = CaptureTokenizer()
    assert tokenize_prompt(tok, "hello", enable_thinking=True) == [7, 8, 9]
    assert tok.kwargs["enable_thinking"] is True
    parser = build_parser()
    assert parser.parse_args([]).enable_thinking is False
    assert parser.parse_args(["--enable-thinking"]).enable_thinking is True


def test_tokenize_prompt_drops_enable_thinking_if_unsupported():
    class StrictTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected enable_thinking")
            return [4, 5]

    assert tokenize_prompt(StrictTokenizer(), "hello") == [4, 5]


def test_resolve_train_attn_keeps_fa2():
    assert resolve_train_attn("flash_attention_2") == "flash_attention_2"


def test_parser_defaults_match_xyu_train_attn_and_graphs():
    ns = build_parser().parse_args([])
    assert ns.attn == "flash_attention_3"
    assert ns.enforce_eager is False
    assert ns.vllm_fp32_lm_head is True
    assert ns.train_fp32_lm_head is True
    off = build_parser().parse_args(["--no-vllm-fp32-lm-head", "--no-train-fp32-lm-head"])
    assert off.vllm_fp32_lm_head is False
    assert off.train_fp32_lm_head is False


def test_student_sampling_params_match_xyu():
    params = student_sampling_params(16384)
    assert params["n"] == 1
    assert params["temperature"] == 1.0
    assert params["top_p"] == 1.0
    assert params["max_tokens"] == 16384
    assert params["logprobs"] == 0
    assert params.get("top_k", -1) == -1
    assert params.get("min_p", 0.0) == 0.0
    assert isinstance(vllm_supports_fp32_lm_head(), bool)
    fp32_engine = vllm_fp32_engine_kwargs()
    assert fp32_engine in (
        {},
        {"fp32_lm_head": True},
        {"hf_overrides": {"head_dtype": "float32"}},
    )
    fa2_engine = vllm_fa2_engine_kwargs()
    assert fa2_engine in ({}, {"attention_config": {"flash_attn_version": 2}})


def test_tokenize_prompt_unwraps_qwen35_mapping():
    class MappingTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [10, 11, 12], "attention_mask": [1, 1, 1]}

    assert tokenize_prompt(MappingTokenizer(), "hello") == [10, 11, 12]


def test_tokenize_prompt_unwraps_nested_batch_input_ids():
    class NestedTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return {"input_ids": [[10, 11]], "attention_mask": [[1, 1]]}

    assert tokenize_prompt(NestedTokenizer(), "hello") == [10, 11]


def test_iter_batches_shuffle_is_seed_deterministic():
    ids = [[i] for i in range(10)]
    a = iter_batches(ids, batch_size=3, steps=4, shuffle=True, seed=0)
    b = iter_batches(ids, batch_size=3, steps=4, shuffle=True, seed=0)
    c = iter_batches(ids, batch_size=3, steps=4, shuffle=True, seed=1)
    assert a == b
    assert a != c
    assert a != iter_batches(ids, batch_size=3, steps=4, shuffle=False, seed=0)
    unshuffled = iter_batches(ids, batch_size=3, steps=2, shuffle=False, seed=0)
    assert unshuffled[0] == [[0], [1], [2]]
    assert unshuffled[1] == [[3], [4], [5]]


def test_ds_config_cosine_and_gas():
    args = SimpleNamespace(
        batch_size=8,
        training_gpus=2,
        lr=2e-5,
        lr_schedule="cosine",
        steps=152,
        warmup_steps=8,
    )
    ds = _ds_config(args)
    assert ds["train_batch_size"] == 8
    assert ds["train_micro_batch_size_per_gpu"] == 1
    assert ds["gradient_accumulation_steps"] == 4
    assert ds["zero_optimization"]["stage"] == 1
    assert "scheduler" not in ds
    packed = _ds_config(
        SimpleNamespace(
            batch_size=8,
            training_gpus=2,
            lr=2e-5,
            lr_schedule="cosine",
            steps=152,
            warmup_steps=8,
            max_tokens_per_mb=46592,
        )
    )
    assert packed["gradient_accumulation_steps"] == 1
    assert packed["train_micro_batch_size_per_gpu"] == 4
    assert packed["train_batch_size"] == 8


def test_student_util_defaults_and_xyu_aliases():
    colocated = SimpleNamespace(student_gpu_memory_utilization=None, colocate=True, gpu_memory_utilization=0.85)
    disjoint = SimpleNamespace(student_gpu_memory_utilization=None, colocate=False, gpu_memory_utilization=0.85)
    assert _student_gpu_memory_utilization(colocated) == 0.35
    assert _student_gpu_memory_utilization(disjoint) == 0.85
    args = build_parser().parse_args(
        [
            "--learning-rate",
            "2e-5",
            "--seq-len",
            "32768",
            "--max-new-tokens",
            "1024",
            "--train-gpus",
            "2",
            "--student-infer-gpus",
            "2",
            "--teacher-infer-gpus",
            "4",
            "--student-tp",
            "1",
            "--teacher-tp",
            "2",
            "--no-colocate",
            "--wandb-run-name",
            "opd3n_20260812_194823_lr2e-5-1node",
        ]
    )
    assert args.lr == 2e-5
    assert args.max_seq_len == 32768
    assert args.max_tokens == 1024
    assert args.training_gpus == 2
    assert args.sampling_gpus == 2
    assert args.teacher_sampling_gpus == 4
    assert args.student_tp == 1
    assert args.teacher_tp == 2
    assert args.colocate is False
    assert args.wandb_run_name == "opd3n_20260812_194823_lr2e-5-1node"
    assert args.wandb_len_ref == 4096.0
    assert args.wandb_len_exponent == 0.486
    three_node = build_parser().parse_args(
        [
            "--training-gpus",
            "8",
            "--sampling-gpus",
            "8",
            "--teacher-sampling-gpus",
            "8",
            "--teacher-tp",
            "2",
            "--no-colocate",
            "--student-ray-hostfile",
            "/tmp/opd_student_hostfile",
            "--teacher-ray-hostfile",
            "/tmp/opd_teacher_hostfile",
            "--teacher-server-cuda-visible-devices",
            "",
        ]
    )
    assert three_node.training_gpus == 8
    assert three_node.sampling_gpus == 8
    assert three_node.teacher_sampling_gpus == 8
    assert three_node.student_ray_hostfile == "/tmp/opd_student_hostfile"
    assert three_node.teacher_ray_hostfile == "/tmp/opd_teacher_hostfile"
    assert three_node.teacher_server_cuda_visible_devices == ""


def test_kl_len_adj_matches_xyu_446z0mnb():
    # Last logged values from yak/opd-qwen35/446z0mnb.
    per_token = 0.007215610657604029
    tokens_per_rollout = 1663.8125
    adj = kl_per_token_len_adj(per_token, tokens_per_rollout, 4096.0, 0.486)
    assert math.isclose(adj, 0.004657178530134034, rel_tol=1e-9)


def test_build_step_record_xyu_groups():
    result = {
        "forward_backward": {
            "metrics": {
                "loss": 0.1,
                "distill_kl": 0.08,
                "distill_k1": 0.09,
                "distill_kl_count": 32,
                "sampler_train_abs_delta_max": 0.2,
                "sampler_train_abs_delta_mean": 0.01,
            }
        },
        "step": {"metrics": {"grad_norm": [1.5, 1.5]}},
        "n_rollouts": 16,
        "tokens_scored": 26624,
        "times": {
            "gen_s": 10.0,
            "score_s": 4.0,
            "fwdbwd_s": 2.0,
            "step_sync_s": 1.0,
            "total_s": 17.0,
        },
    }
    record = build_step_record(
        step=3,
        learning_rate=2e-5,
        result=result,
        tokens_cumulative=26624,
        elapsed_s=99.0,
        wandb_len_ref=4096.0,
        wandb_len_exponent=0.486,
    )
    assert record["loss/avg"] == 0.1
    assert record["kl/per_token"] == 0.08
    assert record["kl/k1_per_token"] == 0.09
    assert record["optim/lr"] == 2e-5
    assert record["optim/grad_norm"] == 1.5
    assert record["optim/update_successful"] == 1.0
    assert record["tokens/scored"] == 26624
    assert record["tokens/per_rollout"] == 1664.0
    assert record["tokens/cumulative"] == 26624
    assert record["time/gen_s"] == 10.0
    assert record["time/score_s"] == 4.0
    assert record["time/fwdbwd_s"] == 2.0
    assert record["time/step_sync_s"] == 1.0
    assert record["time/total_s"] == 17.0
    assert record["sync/abs_delta_max"] == 0.2
    assert record["kl/per_token_len_adj"] == kl_per_token_len_adj(0.08, 1664.0, 4096.0, 0.486)
    assert record["loss"] == 0.1
    assert record["grad_norm"] == 1.5


def test_check_sync_result_flags_zero_params_loaded():
    _SYNC_ZERO_RUN[0] = 0
    warnings, summary = check_sync_result({"recv": {"params_loaded": 0, "workers": [{"params_loaded": 0}]}})
    assert any("params_loaded=0" in w for w in warnings)
    assert "params_loaded" in summary


def test_check_sync_result_consecutive_zero_l2_is_a_stall():
    _SYNC_ZERO_RUN[0] = 0
    payload = {"model_l2_sq_before": 1.0, "model_l2_sq_after": 1.0, "params_loaded": 10}
    for _ in range(4):
        warnings, _ = check_sync_result(payload)
        assert not any("consecutive" in w for w in warnings)
    warnings, summary = check_sync_result(payload)
    assert any("consecutive" in w for w in warnings)
    assert "zero-delta run" in summary
    _SYNC_ZERO_RUN[0] = 0


def test_check_sync_result_onprem_status_ok_is_unverifiable():
    _SYNC_ZERO_RUN[0] = 0
    warnings, summary = check_sync_result({"status": "ok"})
    assert warnings == []
    assert "unverifiable" in summary


def test_check_metrics_requires_parity_and_flags_large_gap():
    missing = check_metrics({"distill_kl_coef": 1.0, "distill_estimator_is_k3": 1.0}, step=0)
    assert any("parity gate did NOT run" in w for w in missing)

    ok = check_metrics(
        {
            "distill_kl_coef": 1.0,
            "distill_estimator_is_k3": 1.0,
            "distill_kl_count": 8.0,
            "sampler_train_kl_sum": 0.0,
            "sampler_train_abs_delta_max": 0.0,
        },
        step=0,
    )
    assert ok == []

    bad = check_metrics(
        {
            "distill_kl_coef": 1.0,
            "distill_estimator_is_k3": 1.0,
            "distill_kl_count": 4.0,
            "sampler_train_kl_sum": 1.0,
            "sampler_train_abs_delta_max": 1.0,
        },
        step=0,
    )
    assert any("RMS delta" in w for w in bad)

    abs_only = check_metrics(
        {
            "distill_kl_coef": 1.0,
            "distill_estimator_is_k3": 1.0,
            "distill_kl_count": 8.0,
            "sampler_train_kl_sum": 0.004,
            "sampler_train_abs_delta_max": 0.34,
        },
        step=0,
    )
    assert any("abs_delta_max" in w for w in abs_only)
    assert not any("RMS delta" in w for w in abs_only)
