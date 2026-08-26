# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""The verl-GRPO -> Cortex forward-backward lowering, and the response mirroring
that lets on-prem-shaped callers read Cortex results unchanged.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.client.cortex_batch import is_cortex_shaped
from arctic_platform.client.cortex_batch import lower_fwd_bwd_batch
from arctic_platform.client.transports.cortex import _mirror_metrics


def _verl_batch(**meta) -> dict:
    """A minimal verl-GRPO payload: the shape on-prem's fwd_bwd takes."""
    return {
        "batch": {
            "input_ids": torch.zeros((2, 10), dtype=torch.int64),
            "attention_mask": torch.ones((2, 10), dtype=torch.int64),
            "advantages": torch.zeros((2, 10)),
            "response_mask": torch.ones((2, 10), dtype=torch.int64),
        },
        "meta": dict(meta),
    }


class TestLowering:
    def test_matches_cortex_wire_shape(self):
        batch = _verl_batch()
        out = lower_fwd_bwd_batch(batch)

        assert out["args"] == ()
        assert torch.equal(out["kwargs"]["input_ids"], batch["batch"]["input_ids"])
        assert torch.equal(out["kwargs"]["attention_mask"], batch["batch"]["attention_mask"])
        # Server-side grpo reads these from context for its preflight.
        assert torch.equal(out["context"]["input_ids"], batch["batch"]["input_ids"])
        assert "advantages" in out["context"]
        assert "loss_mask" in out["context"]

    def test_old_log_probs_dropped(self):
        """π_old defaults to logprobs.detach() server-side, so shipping it is waste."""
        batch = _verl_batch()
        batch["batch"]["old_log_probs"] = torch.zeros((2, 10))
        out = lower_fwd_bwd_batch(batch)
        assert "old_log_probs" not in out["kwargs"]
        assert "old_log_probs" not in out["context"]

    def test_position_ids_and_labels_forwarded(self):
        batch = _verl_batch()
        batch["batch"]["position_ids"] = torch.arange(10).expand(2, 10)
        batch["batch"]["labels"] = torch.zeros((2, 10), dtype=torch.int64)
        out = lower_fwd_bwd_batch(batch)
        assert "position_ids" in out["kwargs"]
        assert "labels" in out["kwargs"]

    def test_loss_mask_preferred_over_response_mask(self):
        batch = _verl_batch()
        batch["batch"]["loss_mask"] = torch.zeros((2, 10), dtype=torch.int64)
        out = lower_fwd_bwd_batch(batch)
        # loss_mask wins, and is cast to bool for the server.
        assert out["context"]["loss_mask"].dtype == torch.bool
        assert not out["context"]["loss_mask"].any()

    def test_missing_mask_raises_rather_than_falling_back(self):
        """attention_mask would include prompt tokens and corrupt the gradient."""
        batch = _verl_batch()
        del batch["batch"]["response_mask"]
        with pytest.raises(ValueError, match="loss_mask"):
            lower_fwd_bwd_batch(batch)

    def test_missing_advantages_raises(self):
        batch = _verl_batch()
        del batch["batch"]["advantages"]
        with pytest.raises(ValueError, match="advantages"):
            lower_fwd_bwd_batch(batch)

    def test_missing_input_ids_raises(self):
        batch = _verl_batch()
        del batch["batch"]["input_ids"]
        with pytest.raises(ValueError, match="input_ids"):
            lower_fwd_bwd_batch(batch)


class TestProcessingBlock:
    def test_defaults_match_the_standalone_recipe(self):
        cfg = lower_fwd_bwd_batch(_verl_batch())["processing"]["config"]
        assert cfg["loss_agg_mode"] == "token-mean"
        assert cfg["entropy_coeff"] == 0.0
        assert cfg["eps_clip"] == 0.2
        # dp_size acts as an extra LR divisor at multi-GPU DP, so it must not leak.
        assert "dp_size" not in cfg

    def test_caller_config_wins(self):
        out = lower_fwd_bwd_batch(
            _verl_batch(),
            processing={"config": {"eps_clip": 0.3, "loss_agg_mode": "seq-mean-token-sum"}},
        )
        cfg = out["processing"]["config"]
        assert cfg["eps_clip"] == pytest.approx(0.3)
        assert cfg["loss_agg_mode"] == "seq-mean-token-sum"
        # Untouched defaults survive.
        assert cfg["entropy_coeff"] == 0.0

    def test_batch_sizing_lifted_from_meta(self):
        cfg = lower_fwd_bwd_batch(_verl_batch(global_batch_size=128, batch_num_tokens=4096))["processing"]["config"]
        assert cfg["global_batch_size"] == 128
        assert cfg["batch_num_tokens"] == 4096

    def test_loss_fn_and_post_are_pinned(self):
        """The frame carries advantages in `context`, which is `grpo`'s contract.

        verl asks for `verl_grpo`, whose meta contract we do not send, so the
        request must not be honoured. compute_logprobs is what populates
        per-token logprobs in the response.
        """
        out = lower_fwd_bwd_batch(_verl_batch(), processing={"loss_fn": "verl_grpo", "post": ["apply_temperature"]})
        assert out["processing"]["loss_fn"] == "grpo"
        assert out["processing"]["post"] == ["compute_logprobs"]


class TestPassThrough:
    def test_recipe_built_frames_are_detected(self):
        """The standalone recipes build the RPC frame themselves."""
        assert is_cortex_shaped({"args": (), "kwargs": {}, "context": {}})
        assert is_cortex_shaped({"kwargs": {"input_ids": 1}})
        assert not is_cortex_shaped({"batch": {}, "meta": {}})
        assert not is_cortex_shaped({"input_ids": 1, "attention_mask": 1})

    def test_flat_batch_without_wrapper_is_lowered(self):
        """SkyRL sends the tensors flat, with `context` alongside."""
        out = lower_fwd_bwd_batch(
            {
                "input_ids": torch.zeros((2, 10), dtype=torch.int64),
                "attention_mask": torch.ones((2, 10), dtype=torch.int64),
                "advantages": torch.zeros((2, 10)),
                "response_mask": torch.ones((2, 10), dtype=torch.int64),
                "context": {"global_batch_size": 64},
            }
        )
        assert out["processing"]["config"]["global_batch_size"] == 64
        assert "advantages" in out["context"]


class TestTransportWiring:
    """The lowering hangs off `_op_target`, which both the sync and async submit
    paths share, so callers holding a verl batch need no shim of their own."""

    @pytest.fixture
    def transport(self, monkeypatch):
        from arctic_platform.client import ArcticClientConfig
        from arctic_platform.client import CortexConfig
        from arctic_platform.client.transports.cortex import CortexTransport

        monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://mock")
        config = ArcticClientConfig(
            model_name="m", backend=CortexConfig.from_env(), training_gpus=1, sampling_gpus=1
        )
        return CortexTransport(config)

    def test_verl_batch_is_lowered(self, transport):
        from arctic_platform.client.transport import Request

        _, body = transport._op_target(Request("forward-backward", 1, _verl_batch()))
        assert set(body) == {"args", "kwargs", "context", "processing"}

    def test_recipe_frame_passes_through_untouched(self, transport):
        from arctic_platform.client.transport import Request

        frame = {"args": (), "kwargs": {"prompts": ["hi"]}, "context": {}, "processing": {"loss_fn": "grpo"}}
        _, body = transport._op_target(Request("forward-backward", 1, dict(frame)))
        assert body == frame

    def test_other_ops_untouched(self, transport):
        from arctic_platform.client.transport import Request

        _, body = transport._op_target(Request("step", 1, {"learning_rate": 1e-5}))
        assert body == {"learning_rate": 1e-5}


class TestMetricMirroring:
    """Cortex puts avg_loss / last_lr at the top level; on-prem callers read
    them out of `metrics`. Mirroring is additive so both shapes work."""

    def test_fwd_bwd_avg_loss_mirrored_to_loss(self):
        out = _mirror_metrics("forward-backward", {"avg_loss": 0.5, "metrics": {"kl": 0.01}})
        assert out["metrics"]["loss"] == pytest.approx(0.5)
        assert out["metrics"]["kl"] == pytest.approx(0.01)
        # Original key stays for callers reading it directly.
        assert out["avg_loss"] == pytest.approx(0.5)

    def test_step_last_lr_mirrored(self):
        out = _mirror_metrics("step", {"global_steps": 3, "last_lr": 1e-5})
        assert out["metrics"]["last_lr"] == pytest.approx(1e-5)
        assert out["metrics"]["global_steps"] == 3

    def test_step_without_metrics_key_gets_one(self):
        """verl does `step_response["metrics"].update(...)`, so the key must exist."""
        assert _mirror_metrics("step", {"global_steps": 1})["metrics"] == {"global_steps": 1}

    def test_existing_metric_not_overwritten(self):
        out = _mirror_metrics("forward-backward", {"avg_loss": 0.5, "metrics": {"loss": 0.9}})
        assert out["metrics"]["loss"] == pytest.approx(0.9)

    def test_untouched_ops_pass_through(self):
        assert _mirror_metrics("save", {"checkpoint_id": "c1"}) == {"checkpoint_id": "c1"}
