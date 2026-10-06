"""One collection, several optimizer steps.

Collection costs hours and an optimizer step costs seconds, so replaying a
batch is the only lever that changes the economics of a run. It is also the
only thing that puts the importance ratio away from 1, which is what makes the
clip -- and therefore the trust region -- exist at all. These tests pin the
loop shape: how many forward passes, how many optimizer steps, one weight sync,
and the refusal to replay a batch whose drift cannot be measured.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from arctic_platform.integrations.harbor.backend import (  # noqa: E402
    ArcticCortexBackend,
)


class _RecordingClient:
    """Counts the calls the train loop makes, in order."""

    def __init__(self) -> None:
        self.fwd_bwd_calls = 0
        self.step_calls = 0
        self.sync_calls = 0
        self.order: list[str] = []

    async def fwd_bwd(self, batch: dict, processing: dict | None = None) -> dict:
        self.fwd_bwd_calls += 1
        self.order.append("fwd_bwd")
        return {"metrics": {"trainable_logprob_count_all": 4.0, "entropy": 0.5}}

    async def step(self) -> dict:
        self.step_calls += 1
        self.order.append("step")
        return {"metrics": {"grad_norm": 1.0, "importance_weight": 1.0}}

    async def sync_weights(self) -> None:
        self.sync_calls += 1
        self.order.append("sync")


class _Cfg:
    micro_batch_size = 2
    train_gpus = 1
    sample_gpus = 1

    def __init__(self, mops: int = 1) -> None:
        self.max_off_policy_steps = mops


def _backend(mops: int = 1) -> tuple[ArcticCortexBackend, _RecordingClient]:
    b = ArcticCortexBackend.__new__(ArcticCortexBackend)
    client = _RecordingClient()
    b._client = client
    b.config = _Cfg(mops)
    return b, client


def _batch(rows: int = 4, width: int = 6, with_logprobs: bool = True) -> dict:
    batch = {
        "input_ids": torch.ones((rows, width), dtype=torch.long),
        "attention_mask": torch.ones((rows, width), dtype=torch.long),
        "loss_mask": torch.ones((rows, width), dtype=torch.long),
        "advantages": torch.full((rows, width), 0.5),
    }
    if with_logprobs:
        batch["old_log_probs"] = torch.full((rows, width), -1.0)
    return batch


async def _train(backend, batch, step: int = 0) -> dict:
    """Drive the half of train() that follows batch construction."""
    return await backend._train_batch(batch, step=step)


@pytest.mark.asyncio
async def test_on_policy_takes_exactly_one_optimizer_step():
    b, c = _backend(mops=1)
    await _train(b, _batch())
    assert c.step_calls == 1


@pytest.mark.asyncio
async def test_replay_takes_one_optimizer_step_per_off_policy_step():
    b, c = _backend(mops=8)
    await _train(b, _batch())
    assert c.step_calls == 8


@pytest.mark.asyncio
async def test_every_optimizer_step_sees_the_whole_batch_again():
    b, c = _backend(mops=4)
    await _train(b, _batch(rows=4))
    # 4 rows at micro_batch 2 is 2 forward passes, repeated 4 times.
    assert c.fwd_bwd_calls == 8


@pytest.mark.asyncio
async def test_weights_sync_once_at_the_end_not_per_inner_step():
    # The sampler only needs the policy the next collection runs against, and
    # a sync is a full weight transfer.
    b, c = _backend(mops=8)
    await _train(b, _batch())
    assert c.sync_calls == 1
    assert c.order[-1] == "sync"


@pytest.mark.asyncio
async def test_gradients_are_consumed_before_the_batch_is_replayed():
    # fwd_bwd accumulates; without an intervening step the second pass would
    # add to the first pass's gradient instead of starting clean.
    b, c = _backend(mops=2)
    await _train(b, _batch(rows=4))
    assert c.order == ["fwd_bwd", "fwd_bwd", "step", "fwd_bwd", "fwd_bwd", "step", "sync"]


@pytest.mark.asyncio
async def test_replay_is_refused_without_sampler_logprobs():
    # Without pi_old the loss pins the ratio at 1, so eight replayed steps
    # would be eight uncorrected on-policy steps.
    b, c = _backend(mops=8)
    await _train(b, _batch(with_logprobs=False))
    assert c.step_calls == 1


@pytest.mark.asyncio
async def test_pre_shifted_logprobs_also_authorise_replay():
    b, c = _backend(mops=3)
    batch = _batch(with_logprobs=False)
    batch["old_log_probs_shifted"] = torch.full((4, 6), -1.0)
    await _train(b, batch)
    assert c.step_calls == 3


@pytest.mark.asyncio
async def test_metrics_report_how_many_steps_the_batch_fed():
    b, _ = _backend(mops=8)
    metrics = await _train(b, _batch())
    assert metrics["off_policy_steps"] == 8.0
