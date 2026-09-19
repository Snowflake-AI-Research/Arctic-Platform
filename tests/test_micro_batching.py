"""Micro-batching must change only memory, never the update.

A step's worth of long agent rollouts does not fit in one ``fwd_bwd``, so the
batch is split. Two things then have to hold, and neither is visible from the
metrics: every rollout must still reach the trainer exactly once, and each
rollout must keep the advantage it was assigned by its *own* group rather than
one recomputed over whatever slice it landed in.

    /data-fast/ap-venv/bin/python -m pytest tests/test_micro_batching.py -q
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from arctic_platform.integrations.harbor.backend import ArcticCortexBackend  # noqa: E402
from arctic_platform.integrations.harbor.backend import _grpo_advantages  # noqa: E402
from arctic_platform.integrations.harbor.models import PostTrainingConfig  # noqa: E402
from arctic_platform.integrations.harbor.models import Rollout  # noqa: E402
from arctic_platform.integrations.harbor.models import RolloutDataset  # noqa: E402


class _RecordingClient:
    def __init__(self) -> None:
        self.batches: list[dict[str, Any]] = []

    async def fwd_bwd(self, batch: dict) -> dict:
        self.batches.append({k: v.clone() for k, v in batch.items()})
        return {"metrics": {"loss": 0.0}}

    async def step(self) -> dict:
        return {"metrics": {"lr": 1e-6}}

    async def sync_weights(self) -> None:
        return None


def _backend(micro: int, n: int, lengths: list[int]) -> tuple[Any, _RecordingClient]:
    cfg = PostTrainingConfig(
        base_model="m", max_seq_len=4096, micro_batch_size=micro,
        std_normalization=False, n_samples_per_prompt=2,
    )
    b = ArcticCortexBackend(cfg)
    client = _RecordingClient()
    b._client = client
    return b, client


def _dataset(lengths: list[int], rewards: list[float], groups: list[str]) -> RolloutDataset:
    rollouts = []
    for i, (n, rew, g) in enumerate(zip(lengths, rewards, groups)):
        prompt = list(range(1, n // 2 + 1))
        completion = list(range(100, 100 + (n - len(prompt))))
        rollouts.append(Rollout(
            prompt_token_ids=prompt,
            completion_token_ids=completion,
            reward=rew,
            group_id=g,
        ))
    return RolloutDataset(
        rollouts=rollouts, dataset_id="d", model_name="m", tokenizer_name="m",
    )


@pytest.mark.asyncio
async def test_every_rollout_is_sent_exactly_once():
    lengths = [10, 20, 30, 40, 50]
    b, client = _backend(micro=2, n=5, lengths=lengths)
    await b.train(_dataset(lengths, [1, 0, 1, 0, 1], ["g", "g", "g", "g", "g"]))

    assert len(client.batches) == 3  # 2 + 2 + 1
    sent = sum(batch["input_ids"].shape[0] for batch in client.batches)
    assert sent == 5


@pytest.mark.asyncio
async def test_advantages_are_group_relative_not_slice_relative():
    """The failure this guards against.

    Two groups, split so that a micro-batch can contain members of only one of
    them. If advantages were computed per slice, the all-successful group would
    get non-zero advantages from being compared against the other group's
    failures. Computed per group, it contributes exactly zero.
    """
    lengths = [10, 10, 10, 10]
    groups = ["win", "win", "lose", "lose"]
    rewards = [1.0, 1.0, 0.0, 0.0]
    b, client = _backend(micro=2, n=4, lengths=lengths)
    await b.train(_dataset(lengths, rewards, groups))

    seen = torch.cat([batch["advantages"].flatten() for batch in client.batches])
    assert torch.count_nonzero(seen) == 0, (
        "uniform groups must yield zero advantage; non-zero means advantages "
        "were normalised over a micro-batch instead of over the group"
    )


@pytest.mark.asyncio
async def test_micro_batches_are_trimmed_to_their_own_width():
    """One long rollout must not widen the others.

    Without length-sorted trimming every micro-batch is padded to the global
    maximum, so a single 4000-token rollout makes a batch of 10-token turns
    cost 400x its own size.
    """
    lengths = [4000, 10, 10, 10]
    b, client = _backend(micro=2, n=4, lengths=lengths)
    await b.train(_dataset(lengths, [1, 0, 1, 0], ["g", "g", "g", "g"]))

    widths = sorted(batch["input_ids"].shape[1] for batch in client.batches)
    assert widths[0] <= 12, f"short batch was padded to {widths[0]}"
    assert widths[-1] >= 4000, "the long rollout must not be truncated away"


@pytest.mark.asyncio
async def test_one_optimizer_step_per_train_call():
    """Gradients accumulate across micro-batches; stepping per micro-batch
    would turn one update into several at a fraction of the batch size."""
    lengths = [10] * 6
    b, client = _backend(micro=2, n=6, lengths=lengths)

    steps = 0
    original = client.step

    async def counting_step():
        nonlocal steps
        steps += 1
        return await original()

    client.step = counting_step
    await b.train(_dataset(lengths, [1, 0, 1, 0, 1, 0], ["g"] * 6))
    assert len(client.batches) == 3
    assert steps == 1


def test_std_normalization_off_keeps_advantages_proportional():
    """Mean-centred only: the reference std_normalization = false."""
    adv = _grpo_advantages([1.0, 0.0, 0.0, 0.0], ["g"] * 4, std_normalization=False)
    assert adv == pytest.approx([0.75, -0.25, -0.25, -0.25])


def test_std_normalization_on_rescales_by_group_std():
    adv = _grpo_advantages([1.0, 0.0, 0.0, 0.0], ["g"] * 4, std_normalization=True)
    # std of [1,0,0,0] is sqrt(3)/4; the lone success is pulled far out.
    assert adv[0] == pytest.approx(0.75 / (3 ** 0.5 / 4), rel=1e-4)
    assert adv[0] > 1.7


def test_uniform_group_is_zero_under_both_settings():
    for std in (True, False):
        adv = _grpo_advantages([1.0, 1.0, 1.0], ["g"] * 3, std_normalization=std)
        assert adv == pytest.approx([0.0, 0.0, 0.0])
