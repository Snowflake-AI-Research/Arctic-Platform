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

"""Normal correctness execution preserves DSS requests and independent reference semantics."""

import inspect
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.reference import hf_single_gpu
from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory


@pytest.mark.parametrize("provider", ["huggingface", "prime_rl"])
@pytest.mark.parametrize("case,rows,calls", [("gas1", 4, 1), ("gas4", 16, 4)])
def test_checkpoint_cases_use_plain_supervised_requests(provider, case, rows, calls):
    ctx = SimpleNamespace(
        config_id="qwen3-8b-h200-train-sft-full-4gpus-64k",
        cfg=SimpleNamespace(training={"model_provider": provider}),
        vocab_size=128,
    )
    arm = SimpleNamespace(name=case, global_batch_size=rows, max_seq_len=8, dss_microbatches=calls)
    bodies = checkpoint_resume._iteration_bodies(ctx, arm)
    batches = checkpoint_resume.iteration_batches(ctx, arm)
    for body, batch in zip(bodies, batches, strict=True):
        assert isinstance(body["batch"], dict)
        assert body["processing"] == {"loss_fn": "sft"}
        lowered = dss_driver.cortex_training_body(body)
        assert "processing" not in lowered
        assert lowered["kwargs"]["input_ids"].shape == (rows, 8)
        labels = batch.shifted_labels() if provider == "prime_rl" else batch.labels
        torch.testing.assert_close(lowered["kwargs"]["labels"], labels, rtol=0, atol=0)


def test_payload_does_not_invent_deepep_chunking():
    payload = dss_driver.build_payload({"ep_size": 4, "mb_spec": {"max_tokens_per_mb": 10240}}, "model", 42)
    assert "deepep_token_chunk_size" not in payload["training_config"]
    configured = dss_driver.build_payload({"ep_size": 4, "deepep_token_chunk_size": 4096}, "model", 42)
    assert configured["training_config"]["deepep_token_chunk_size"] == 4096


def test_normal_runner_has_no_execution_overlay_or_config_specific_loss_selector():
    assert "optimizer_capture_worker" not in vars(runner)
    assert not hasattr(dss_driver, "ap_batch_processing_options")
    assert not hasattr(dss_driver, "pack_microbatches")
    assert "gradient_accumulation_steps" not in inspect.signature(dss_driver.build_payload).parameters
    assert "processing_config" not in inspect.signature(dss_driver.pack).parameters


def test_core_fixture_binding_has_no_upload_account_or_subprocess_dependency():
    assert not hasattr(cortex_fixture, "require_upload_credentials")
    assert not hasattr(cortex_fixture, "DEV_STAGE")
    assert not hasattr(cortex_fixture, "subprocess")


@pytest.mark.parametrize("torch_adam", [False, True])
def test_golden_trajectory_cannot_select_product_optimizer(torch_adam):
    assert "optimizer_backend" not in inspect.signature(AdamWTrajectory).parameters

    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.bfloat16, device="cpu")
    config = {
        "name": "AdamW",
        "torch_adam": torch_adam,
        "betas": [0.8, 0.95],
        "eps": 1e-6,
        "weight_decay": 0.125,
    }
    trajectory = AdamWTrajectory(
        model,
        config,
        learning_rate=3e-3,
        gradient_clipping=None,
        optimizer_dtype="float32",
    )
    trajectory.step({"weight": torch.tensor([[0.25, -0.5]], dtype=torch.float32, device="cpu")})

    optimizer = trajectory._optimizer
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.defaults["fused"] is False
    assert optimizer.defaults["foreach"] is False
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert group["lr"] == 3e-3
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == 1e-6
    assert group["weight_decay"] == 0.125
    assert all(parameter.device.type == "cpu" and parameter.dtype == torch.float32 for parameter in group["params"])


@pytest.mark.parametrize("fp32_lm_head,accum_dtype", [(False, None), (True, torch.float32)])
def test_golden_fused_cross_entropy_uses_configured_accumulation(monkeypatch, fp32_lm_head, accum_dtype):
    loss = object()
    constructor = Mock(return_value=loss)
    monkeypatch.setitem(sys.modules, "liger_kernel", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "liger_kernel.transformers", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "liger_kernel.transformers.fused_linear_cross_entropy",
        SimpleNamespace(LigerFusedLinearCrossEntropyLoss=constructor),
    )

    assert hf_single_gpu._build_liger_fused_cross_entropy(fp32_lm_head=fp32_lm_head) is loss
    constructor.assert_called_once_with(ignore_index=-100, reduction="mean", accum_dtype=accum_dtype)


def test_sampling_does_not_select_gdn_backend_to_match_training():
    payload = dss_driver.sampling_payload("model", 42, dtype="bfloat16", n_gpus=4, max_seq_len=128)
    assert "gdn_prefill_backend" not in payload["inference_config"]["vllm_config"]
