"""Unit tests for ``dss_client.neutrino_engine.NeutrinoTrainingEngine``.

All NeutrinoClient interactions are mocked -- no real HTTP calls or GPU needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(job_id: str = "job-1") -> MagicMock:
    """Return a mock NeutrinoClient with sensible defaults."""
    client = MagicMock()
    client.create_job.return_value = job_id
    client.wait_for_job.return_value = {"status": "running"}
    return client


def _make_sub_job() -> MagicMock:
    sub_job = MagicMock()
    sub_job.to_wire.return_value = {"job_type": "training", "model_name": "gpt2"}
    return sub_job


def _make_engine(client=None, sub_job=None):
    from dss_client.neutrino_engine import NeutrinoTrainingEngine

    client = client or _make_client()
    sub_job = sub_job or _make_sub_job()
    return NeutrinoTrainingEngine(client=client, sub_job=sub_job), client


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------


class TestInit:
    def test_calls_create_job_and_wait(self):
        client = _make_client("job-42")
        sub_job = _make_sub_job()
        engine, _ = _make_engine(client, sub_job)

        client.create_job.assert_called_once_with(sub_jobs=[sub_job])
        client.wait_for_job.assert_called_once_with("job-42")
        assert engine.job_id == "job-42"

    def test_compat_shims(self):
        engine, _ = _make_engine()
        assert engine.module is None
        assert engine.optimizer is None
        assert engine.training_dataloader is None
        assert engine.global_steps == 0
        assert engine.lr_scheduler.get_last_lr() == [-1.0]


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_stashes_batch(self):
        engine, _ = _make_engine()
        result = engine(input_ids=torch.tensor([1, 2, 3]), attention_mask=torch.tensor([1, 1, 1]))
        assert result is not None
        assert engine._input_batch is not None
        assert "input_ids" in engine._input_batch.kwargs

    def test_double_forward_raises(self):
        engine, _ = _make_engine()
        engine(input_ids=torch.tensor([1]))
        with pytest.raises(AssertionError, match="already set"):
            engine(input_ids=torch.tensor([2]))


# ---------------------------------------------------------------------------
# Backward
# ---------------------------------------------------------------------------


class TestBackward:
    def test_sends_batch_and_returns_loss(self):
        engine, client = _make_engine()
        client.forward_backward.return_value = "req-fb-1"
        client.poll_request.return_value = {"avg_loss": 0.75}

        engine(input_ids=torch.tensor([1, 2]))
        loss = engine.backward()

        client.forward_backward.assert_called_once()
        args = client.forward_backward.call_args
        assert args[0][0] == "job-1"  # job_id
        assert isinstance(args[0][1], bytes)  # serialized batch

        client.poll_request.assert_called_once_with("job-1", "req-fb-1")
        assert torch.is_tensor(loss)
        assert loss.item() == pytest.approx(0.75)

    def test_resets_input_batch_after(self):
        engine, client = _make_engine()
        client.forward_backward.return_value = "req-1"
        client.poll_request.return_value = {"avg_loss": 0.5}

        engine(input_ids=torch.tensor([1]))
        engine.backward()
        assert engine._input_batch is None

    def test_backward_without_forward_raises(self):
        engine, _ = _make_engine()
        with pytest.raises(AssertionError, match="not set"):
            engine.backward()


# ---------------------------------------------------------------------------
# Step
# ---------------------------------------------------------------------------


class TestStep:
    def test_updates_global_steps_and_lr(self):
        engine, client = _make_engine()
        client.step.return_value = "req-step-1"
        client.poll_request.return_value = {"global_steps": 7, "last_lr": [1e-4]}

        engine.step()

        client.step.assert_called_once_with("job-1")
        client.poll_request.assert_called_once_with("job-1", "req-step-1")
        assert engine.global_steps == 7
        assert engine.lr_scheduler.get_last_lr() == [1e-4]


# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------


class TestSaveCheckpoint:
    def test_calls_save_and_polls(self):
        engine, client = _make_engine()
        client.save.return_value = "req-save-1"
        client.poll_request.return_value = {"checkpoint_id": "cp-42"}

        engine.save_checkpoint()

        client.save.assert_called_once_with("job-1", checkpoint_type=None)
        client.poll_request.assert_called_once_with("job-1", "req-save-1")

    def test_forwards_checkpoint_type(self):
        engine, client = _make_engine()
        client.save.return_value = "req-save-1"
        client.poll_request.return_value = {"checkpoint_id": "cp-42"}

        engine.save_checkpoint(checkpoint_type="weights-only")

        client.save.assert_called_once_with(
            "job-1",
            checkpoint_type="weights-only",
        )

    def test_accepts_engine_compatibility_arguments(self):
        engine, client = _make_engine()
        client.save.return_value = "req-save-1"
        client.poll_request.return_value = {}

        engine.save_checkpoint("/unused/path", tag="unused")

        client.save.assert_called_once_with("job-1", checkpoint_type=None)

    def test_handles_no_checkpoint_id(self):
        engine, client = _make_engine()
        client.save.return_value = "req-save-2"
        client.poll_request.return_value = {}

        engine.save_checkpoint()  # should not raise


# ---------------------------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------------------------


class TestLoadCheckpoint:
    def test_calls_load_and_polls(self):
        engine, client = _make_engine()
        client.load.return_value = "req-load-1"
        client.poll_request.return_value = {"checkpoint_id": "cp-42"}

        result = engine.load_checkpoint(
            "cp-42",
            source_job_id="source-job",
        )

        client.load.assert_called_once_with(
            "job-1",
            checkpoint_id="cp-42",
            source_job_id="source-job",
        )
        client.poll_request.assert_called_once_with("job-1", "req-load-1")
        assert result == {"checkpoint_id": "cp-42"}


# ---------------------------------------------------------------------------
# Destroy
# ---------------------------------------------------------------------------


class TestDestroy:
    def test_calls_cancel_job(self):
        engine, client = _make_engine()
        engine.destroy()
        client.cancel_job.assert_called_once_with("job-1")


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_is_gradient_accumulation_boundary(self):
        engine, _ = _make_engine()
        assert engine.is_gradient_accumulation_boundary() is True

    def test_is_nn_module(self):
        engine, _ = _make_engine()
        assert isinstance(engine, torch.nn.Module)

    def test_train_eval_modes(self):
        engine, _ = _make_engine()
        engine.train()
        assert engine.training is True
        engine.eval()
        assert engine.training is False
