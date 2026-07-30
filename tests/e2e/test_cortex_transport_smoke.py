# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end plumbing smoke: `CortexTransport` against `fake_cortex_gs`.

Validates that:

- `CortexTransport.initialize()` calls CreateJob + polls GetJob and produces
  `JobHandles` covering training/sampling/log_prob sub-jobs.
- Every `call(Request)` op the transport dispatches (fwd_bwd, fwd_no_grad,
  step, save, generate, sync_weights, reset_prefix_cache, log_probs, all the
  colocation lifecycle no-ops) round-trips through the fake GS without
  raising.
- Response shapes returned by `_shape_train_response` match what verl and
  SkyRL then read from them: `.get("grad_norm")`, `.get("avg_loss")`,
  `response["batch"]["log_probs"]` after the `model_outputs -> batch` alias.

Nothing here validates *training correctness* — losses are random. It's a
plumbing gate: prove the client's REST + wire path works end-to-end without
mocks at any layer.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import torch

# tests/ isn't a package in this repo, so relative-style imports don't work
# out of the box; put the current directory on the path so `import
# fake_cortex_gs` resolves the sibling module.
sys.path.insert(0, str(Path(__file__).parent))

from arctic_platform.client.config import ArcticRLClientConfig as UnifiedClientConfig
from arctic_platform.client.transport import Request
from arctic_platform.client.transports.cortex import CortexTransport


def _free_port() -> int:
    """Grab an unused TCP port so parallel runs don't collide."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def fake_gs_url() -> str:
    from fake_cortex_gs import serve_in_background

    port = _free_port()
    serve_in_background(host="127.0.0.1", port=port)
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def transport(fake_gs_url: str) -> CortexTransport:
    cfg = UnifiedClientConfig(
        backend="cortex",
        model_name="Qwen/Qwen3-0.6B",
        training_gpus=1,
        sampling_gpus=1,
        log_prob_gpus=0,
        cortex_base_url=fake_gs_url,
        cortex_database="fake_db",
        cortex_schema="fake_sch",
        cortex_endpoint="cortex-training",
        max_seq_len=512,
    )
    tp = CortexTransport(cfg)
    tp.initialize()
    return tp


class TestInitialize:
    def test_creates_job_and_captures_sub_jobs(self, transport: CortexTransport):
        assert transport.job_id is not None
        # Training + sampling sub-jobs should be recognized. Log-prob was
        # 0 GPUs so it's the fallback synthesized handle.
        assert "training" in transport.sub_jobs
        assert "sampling" in transport.sub_jobs


class TestCoreOps:
    def test_fwd_bwd_returns_loss_and_grad_norm(self, transport: CortexTransport):
        # Build a verl-GRPO-shaped body: {batch: {...}, meta: {...}, processing: {...}}
        batch = {
            "input_ids": torch.zeros(2, 8, dtype=torch.long),
            "attention_mask": torch.ones(2, 8, dtype=torch.long),
        }
        body = {"batch": batch, "meta": {"rollout_n": 1}, "processing": {"loss_fn": "verl_grpo"}}
        resp = transport.call(Request(op="fwd-bwd", body=body))
        # verl reads `.get("avg_loss")`; SkyRL reads `.get("grad_norm")`.
        assert "avg_loss" in resp
        assert resp["metrics"].get("grad_norm") is not None

    def test_fwd_no_grad_returns_batch_log_probs(self, transport: CortexTransport):
        batch = {"input_ids": torch.zeros(2, 16, dtype=torch.long)}
        body = {"batch": batch, "meta": {}, "reference_model": True}
        resp = transport.call(Request(op="fwd-no-grad", body=body))
        # _shape_train_response aliases model_outputs -> batch; verl then
        # renames `logprobs` -> `log_probs`. Assert the alias took.
        assert "batch" in resp
        assert "logprobs" in resp["batch"]
        lp = resp["batch"]["logprobs"]
        assert torch.is_tensor(lp)
        assert lp.shape == (2, 16)

    def test_step_returns_metrics(self, transport: CortexTransport):
        resp = transport.call(Request(op="step", body={"learning_rate": 1e-5}))
        assert "metrics" in resp
        assert resp["metrics"].get("grad_norm") is not None

    def test_save_checkpoint(self, transport: CortexTransport):
        resp = transport.call(Request(op="save-checkpoint", body={}))
        assert isinstance(resp, dict)
        assert "path" in resp

    def test_generate_returns_results_list(self, transport: CortexTransport):
        body = {
            "prompts": ["hello world", "second prompt"],
            "sampling_params": {"temperature": 0.7, "max_tokens": 8},
            "routing_key": None,
        }
        resp = transport.call(Request(op="generate", body=body))
        assert isinstance(resp, dict)
        assert "results" in resp
        assert len(resp["results"]) >= 1

    def test_log_probs(self, transport: CortexTransport):
        body = {
            "prompts": ["hello world"],
            "completions": [[1, 2, 3, 4, 5]],
            "top_k": None,
        }
        resp = transport.call(Request(op="log-probs", body=body))
        assert isinstance(resp, dict)
        assert "model_outputs" in resp
        assert "logprobs" in resp["model_outputs"]

    def test_sync_weights(self, transport: CortexTransport):
        resp = transport.call(Request(op="sync-weights", body={"cuda_ipc": False, "low_memory": False}))
        assert isinstance(resp, dict)

    def test_reset_prefix_cache(self, transport: CortexTransport):
        resp = transport.call(Request(op="reset-prefix-cache", body={"drain": False, "timeout_s": 5.0}))
        assert isinstance(resp, dict)


class TestColocationNoops:
    """Every colocation lifecycle op is a no-op on Cortex (returns {}).
    SkyRL calls these unconditionally when `colocate=True`; verl calls
    `wake_inference` / `sleep_inference`. Regression here means SkyRL's
    colocated recipe would raise on the very first weight sync."""

    @pytest.mark.parametrize(
        "op",
        [
            "wake-training",
            "sleep-training",
            "wake-inference",
            "sleep-inference",
            "wake-log-prob",
            "sleep-log-prob",
            "empty-training-cache",
            "weight-norm",
            "save-weights",
        ],
    )
    def test_op_is_noop(self, transport: CortexTransport, op: str):
        resp = transport.call(Request(op=op, body={}))
        assert resp == {}
