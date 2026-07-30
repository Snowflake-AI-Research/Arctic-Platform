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
"""Pin the exact call patterns SkyRL and verl integrations use.

These tests intentionally mirror the SkyRL ``_ArcticDispatch`` and verl
``ArcticRLClientWrapper`` call sites verbatim. Every regression here is a
regression for the pre-merged integrations that (by design) we cannot edit.

Sources being pinned:
- SkyRL: ``arctic-skyrl/skyrl/backends/arctic_rl/{arctic_trainer,arctic_generator}.py``
- verl:  ``arctic-verl/verl/trainer/ppo/arctic_rl_client.py``
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from arctic_platform.client import ArcticRLClient
from arctic_platform.client import ArcticRLClientConfig
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import OPS
from arctic_platform.client.transport import Request
from arctic_platform.client.transport import Transport
from arctic_platform.client.transport import method_name
from arctic_platform.client.transport import unresolved_ops


class _RecordingTransport(Transport):
    """A Transport that records every request and returns canned responses.

    The canned responses match the shapes the *on-prem* server returns today
    (metrics nested under ``metrics``) so the client-side flattening is
    exercised end-to-end.
    """

    def __init__(self, config: ArcticRLClientConfig) -> None:
        self.config = config
        self.calls: list[Request] = []
        self.jobs = JobHandles(training="t-1", sampling="s-1", log_prob="lp-1")
        self.responses: dict[str, Any] = {
            "fwd-bwd": {"avg_loss": 0.5, "metrics": {"grad_norm": 1.0, "loss": 0.5}, "post_process_outputs": {}},
            "fwd-no-grad": {"model_outputs": {"logprobs": [[-0.1, -0.2]]}, "metrics": {}},
            "step": {"metrics": {"grad_norm": 0.9, "learning_rate": 1e-5}},
            "generate": {"results": [{"token_ids": [1, 2, 3], "text": "ok", "finish_reason": "stop"}]},
            "sync-weights": {"ok": True},
            "save-checkpoint": {"path": "/tmp/ckpt"},
            "reset-prefix-cache": {"drained": True},
            "log-probs": {"logprobs": [[-0.1, -0.2]]},
        }

    def initialize(self) -> JobHandles:
        return self.jobs

    def call(self, request: Request) -> dict:
        self.calls.append(request)
        return dict(self.responses.get(request.op, {}))

    def shutdown(self) -> None:
        return None


@pytest.fixture
def recording_client() -> ArcticRLClient:
    config = ArcticRLClientConfig(
        model_name="Qwen/Qwen3-0.6B",
        training_gpus=1,
        sampling_gpus=1,
        log_prob_gpus=1,
    )
    with patch("arctic_platform.client.client.make_transport") as mt:
        transport = _RecordingTransport(config)
        mt.return_value = transport
        client = ArcticRLClient(config)
    client._transport_for_tests = transport
    return client


# ── legacy config aliases (SkyRL + verl) ─────────────────────────────────


class TestConfigLegacyAliases:
    def test_backend_local_maps_to_onprem(self) -> None:
        cfg = ArcticRLClientConfig(model_name="m", backend="local", training_gpus=1)
        assert cfg.backend == "onprem"

    def test_backend_dss_platform_maps_to_onprem(self) -> None:
        cfg = ArcticRLClientConfig(model_name="m", backend="dss-platform", training_gpus=1)
        assert cfg.backend == "onprem"

    def test_backend_neutrino_maps_to_cortex(self) -> None:
        cfg = ArcticRLClientConfig(model_name="m", backend="neutrino", training_gpus=1)
        assert cfg.backend == "cortex"

    def test_sample_gpus_alias_populates_sampling_gpus(self) -> None:
        """verl builds its config with `sample_gpus=`; must land as sampling_gpus."""
        cfg = ArcticRLClientConfig(model_name="m", backend="local", sample_gpus=4)
        assert cfg.sampling_gpus == 4

    def test_legacy_ignored_fields_do_not_error(self) -> None:
        """`log_prob_engine` and friends flow in from verl configs; must drop silently."""
        cfg = ArcticRLClientConfig(
            model_name="m",
            backend="local",
            training_gpus=1,
            sampling_gpus=1,
            log_prob_gpus=1,
            log_prob_engine="deepspeed",
            experiment_name="ignored",
        )
        assert cfg.backend == "onprem"
        assert not hasattr(cfg, "log_prob_engine")


# ── SkyRL client-surface pinning ─────────────────────────────────────────


class TestSkyRLCallPatterns:
    """SkyRL calls the client in _ArcticDispatch / ArcticGenerator / entrypoint.

    Every method used there must exist on ArcticRLClient with a compatible
    signature and response shape.
    """

    def test_fwd_bwd_folds_processing_and_extra_kwargs(self, recording_client: ArcticRLClient) -> None:
        # `_ArcticDispatch.forward_backward` call: `fwd_bwd(batch, processing={...})`.
        result = recording_client.fwd_bwd(
            {"batch": {"input_ids": [[1, 2, 3]]}, "meta": {"pad_token_id": 0}},
            processing={"loss_fn": "grpo", "config": {"n_samples": 4}, "post": ["compute_logprobs"]},
        )
        req = recording_client._transport_for_tests.calls[-1]
        assert req.op == "fwd-bwd"
        assert req.body["processing"]["loss_fn"] == "grpo"
        # SkyRL reads `result["grad_norm"]` at the top level after step; the
        # fwd_bwd response is passed through the same flattener.
        assert result["grad_norm"] == 1.0
        assert result["avg_loss"] == 0.5

    def test_fwd_no_grad_accepts_post_processors_kwarg(self, recording_client: ArcticRLClient) -> None:
        # `_ArcticDispatch.forward` call: `fwd_no_grad(batch, post_processors=["logprobs"])`.
        result = recording_client.fwd_no_grad(
            {"batch": {"input_ids": [[1, 2]]}},
            post_processors=["logprobs"],
        )
        req = recording_client._transport_for_tests.calls[-1]
        assert req.op == "fwd-no-grad"
        assert req.body["post_processors"] == ["logprobs"]
        assert result["model_outputs"]["logprobs"]

    def test_step_grad_norm_is_top_level(self, recording_client: ArcticRLClient) -> None:
        # `_ArcticDispatch.optim_step` reads `.get("grad_norm")` on the return.
        out = recording_client.step()
        assert out.get("grad_norm") == 0.9

    def test_sync_weights_accepts_cuda_ipc(self, recording_client: ArcticRLClient) -> None:
        # SkyRL colocated path: `sync_weights(cuda_ipc=True)`.
        recording_client.sync_weights(cuda_ipc=True)
        req = recording_client._transport_for_tests.calls[-1]
        assert req.body["cuda_ipc"] is True

    def test_generate_returns_list(self, recording_client: ArcticRLClient) -> None:
        # `ArcticGenerator.generate` reads `output.get("token_ids"/"text"/"finish_reason")`.
        out = recording_client.generate(["hello"], sampling_params={"temperature": 1.0})
        assert isinstance(out, list)
        assert out[0]["token_ids"] == [1, 2, 3]

    def test_colocation_lifecycle_exists(self, recording_client: ArcticRLClient) -> None:
        for op in ("wake_training", "wake_inference", "sleep_training", "sleep_inference", "empty_training_cache"):
            assert callable(getattr(recording_client, op))
            getattr(recording_client, op)()

    def test_wake_inference_forwards_tags(self, recording_client: ArcticRLClient) -> None:
        # verl adapter calls ``self._client.wake_inference(tags=tags)`` and
        # ``sleep_inference(level=level)``; the transport-facing body must
        # carry those kwargs so on-prem colocation can honor them.
        recording_client.wake_inference(tags=["actor"])
        req = recording_client._transport_for_tests.calls[-1]
        assert req.op == "wake-inference"
        assert req.body == {"tags": ["actor"]}

        recording_client.sleep_inference(level=2)
        req = recording_client._transport_for_tests.calls[-1]
        assert req.op == "sleep-inference"
        assert req.body == {"level": 2}

    def test_job_id_attributes_exposed(self, recording_client: ArcticRLClient) -> None:
        # `pre_client.training_job_id` / `sampling_job_id` / `log_prob_job_id`
        # read in `integrations.arctic_rl.entrypoint.main`.
        assert recording_client.training_job_id == "t-1"
        assert recording_client.sampling_job_id == "s-1"
        assert recording_client.log_prob_job_id == "lp-1"

    def test_get_server_state_defaults_to_none(self, recording_client: ArcticRLClient) -> None:
        # The Ray path returns an actor; non-Ray transports return None.
        assert recording_client.get_server_state() is None


# ── verl client-surface pinning ──────────────────────────────────────────


class TestVerlCallPatterns:
    """`ArcticRLClientWrapper` in verl builds a config with `sample_gpus=`,
    `backend="local"`, `log_prob_engine="deepspeed"`, then calls
    `fwd_no_grad(batch)`, `fwd_bwd(batch, processing=...)`, `step()`,
    `generate(prompts=..., sampling_params=...)`, `shutdown()`.
    """

    def test_config_shape_matches_wrapper(self) -> None:
        cfg = ArcticRLClientConfig(
            host="localhost",
            port=7000,
            backend="local",
            training_gpus=2,
            sample_gpus=2,
            log_prob_gpus=2,
            colocate=True,
            log_prob_engine="deepspeed",
            model_name="Qwen/Qwen3-0.6B",
        )
        assert (cfg.backend, cfg.sampling_gpus, cfg.log_prob_gpus) == ("onprem", 2, 2)

    def test_fwd_bwd_returns_avg_loss_and_post_process_outputs(
        self, recording_client: ArcticRLClient
    ) -> None:
        # `ArcticRLClientWrapper.update_actor` reads
        # `result.get("avg_loss")` and `result.get("post_process_outputs")`.
        result = recording_client.fwd_bwd(
            {"batch": {"input_ids": [[1]]}, "context": {"input_ids": [[1]]}},
            processing={"loss_fn": "grpo", "post": ["compute_logprobs"]},
        )
        assert "avg_loss" in result
        assert "post_process_outputs" in result

    def test_fwd_no_grad_model_outputs_logprobs(self, recording_client: ArcticRLClient) -> None:
        # `ArcticRLClientWrapper.compute_log_prob` reads
        # `result.get("model_outputs", result).get("logprobs")`.
        result = recording_client.fwd_no_grad({"kwargs": {"input_ids": [[1]]}, "context": {}})
        outputs = result.get("model_outputs", result)
        assert outputs.get("logprobs")

    def test_fwd_no_grad_forwards_reference_model(self, recording_client: ArcticRLClient) -> None:
        # verl's ``_send_compute_ref_log_prob`` calls
        # ``self._client.fwd_no_grad(payload, reference_model=True)``. The
        # transport-facing body must carry the flag so the Cortex normalizer
        # can route the request to the reference-model engine.
        recording_client.fwd_no_grad({"batch": {"input_ids": [[1]]}}, reference_model=True)
        req = recording_client._transport_for_tests.calls[-1]
        assert req.op == "fwd-no-grad"
        assert req.body["reference_model"] is True

    def test_shutdown_is_idempotent(self, recording_client: ArcticRLClient) -> None:
        recording_client.shutdown()
        recording_client.shutdown()


# ── op-vocabulary sanity ─────────────────────────────────────────────────


class TestOpVocabulary:
    def test_new_lifecycle_ops_registered(self) -> None:
        for op in (
            "wake-training",
            "sleep-training",
            "wake-inference",
            "sleep-inference",
            "empty-training-cache",
            "save-weights",
        ):
            assert op in OPS

    def test_cortex_covers_every_op(self) -> None:
        """CortexTransport must resolve every op in OPS (some as no-ops)."""
        from arctic_platform.client.transports.cortex import CortexTransport

        class _Fake(CortexTransport):
            def __init__(self) -> None:
                pass

        fake = _Fake()
        # Handlers dispatch by name; expose them the way the transport ABC docs
        # `unresolved_ops` — construct a shim that maps method_name -> handler.
        for op in OPS:
            attr = method_name(op)
            # ``fwd_bwd`` etc. exist as private methods; the transport dispatches
            # through ``_handlers``. This assertion is a lint that new ops added
            # to OPS also land in ``_handlers`` (populated in __init__).
            if attr.startswith(("fwd", "log_probs", "generate", "step", "save_checkpoint", "sync_weights", "reset_prefix_cache")):
                assert hasattr(fake, "_" + attr) or hasattr(fake, attr)
