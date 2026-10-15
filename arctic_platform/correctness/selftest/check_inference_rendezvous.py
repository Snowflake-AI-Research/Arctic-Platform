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

"""Inference reload must reuse the correctness runtime's standard rendezvous."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.checks import inference_checkpoint_loss as check


@pytest.fixture
def inference_run(tmp_path, monkeypatch):
    tokenizer = SimpleNamespace(save_pretrained=lambda path: None)
    distribution = {"examples": 1, "minimum_tokens": 4, "median_tokens": 4, "maximum_tokens": 4}
    cfg = SimpleNamespace(
        model_name="Qwen/Qwen3-8B",
        training={"model_provider": "huggingface", "dtype": "bfloat16"},
        sub_job={},
        n_gpus=16,
        dp_size=16,
        max_tokens_per_mb=8192,
    )
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=SimpleNamespace(model=SimpleNamespace(cache_path=str(tmp_path / "model"))),
        workdir=tmp_path,
        attn_implementation="flash_attention_3",
    )
    monkeypatch.setattr(check, "configured_learning_rate", lambda cfg: 0.01)
    monkeypatch.setattr(check, "configured_rows_per_step", lambda cfg: 1)
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: tokenizer)
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda value: 0)
    monkeypatch.setattr(check.gsm8k, "read_rows", lambda *args, **kwargs: [])
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda *args: [])
    monkeypatch.setattr(check.gsm8k, "length_distribution", lambda rows: distribution)
    example = SimpleNamespace(prompt="question", completion=" answer")
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: ([example], 4))
    monkeypatch.setattr(check.gsm8k, "assert_disjoint", lambda *args: None)
    monkeypatch.setattr(check.gsm8k, "build_batch", lambda *args: object())
    monkeypatch.setattr(check, "training_batches", lambda *args: [object()] * check.TRAIN_STEPS)
    monkeypatch.setattr(check, "pack_logit_aligned", lambda batch: {})
    monkeypatch.setattr(check, "pack", lambda *args, **kwargs: {})
    monkeypatch.setattr(check, "build_payload", lambda *args, **kwargs: {"job_type": "training"})
    monkeypatch.setattr(check, "sampling_payload", lambda *args, **kwargs: {"job_type": "sampling"})
    monkeypatch.setattr(check, "fwd_bwd_step", lambda *args, **kwargs: SimpleNamespace(avg_loss=2.0))
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: None)
    monkeypatch.setattr(check, "reference_answer_logprobs", lambda *args: [[-2.0]])
    monkeypatch.setattr(check, "target_answer_logprobs", lambda *args: [[-2.0]])
    monkeypatch.setattr(check, "log_probs", lambda *args, **kwargs: None)
    monkeypatch.setattr(check, "save_weights_only_checkpoint", lambda *args: str(tmp_path / "hf"))
    monkeypatch.setattr(check, "verdict", lambda *args: "verified")
    ports, jobs, allocations = [], [], []

    @contextmanager
    def gateway(path, slots, *, backend):
        assert isinstance(backend, OnPremConfig)
        yield object()

    @contextmanager
    def running_job(url, payload):
        ports.append(os.environ.get("MASTER_PORT"))
        jobs.append(payload["job_type"])
        if len(jobs) == 3 and failure[0]:
            raise RuntimeError("reload failed")
        yield object()

    def allocate_port(*args, **kwargs):
        allocations.append((args, kwargs))
        return 30001

    failure = [False]
    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", running_job)
    monkeypatch.setattr(check, "get_unique_port_number", lambda: 30000, raising=False)
    monkeypatch.setattr(check, "reserve_free_port", allocate_port, raising=False)
    return SimpleNamespace(context=ctx, ports=ports, jobs=jobs, allocations=allocations, failure=failure)


@pytest.mark.parametrize("configured_port", [None, "29500"])
def test_reload_leaves_standard_runtime_rendezvous_and_caller_environment_unchanged(
    inference_run, monkeypatch, configured_port
):
    if configured_port is None:
        monkeypatch.delenv("MASTER_PORT", raising=False)
    else:
        monkeypatch.setenv("MASTER_PORT", configured_port)
    assert check._run(inference_run.context) == "verified"
    assert inference_run.jobs == ["training", "sampling", "training"]
    assert not inference_run.allocations
    assert inference_run.ports == [configured_port] * 3
    assert os.environ.get("MASTER_PORT") == configured_port


def test_reload_error_does_not_allocate_an_alternate_port(inference_run, monkeypatch):
    monkeypatch.setenv("MASTER_PORT", "29500")
    inference_run.failure[0] = True
    with pytest.raises(RuntimeError, match="reload failed"):
        check._run(inference_run.context)
    assert not inference_run.allocations
    assert inference_run.ports == ["29500"] * 3
    assert os.environ["MASTER_PORT"] == "29500"
