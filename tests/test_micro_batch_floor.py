"""Micro-batch slicing must never emit a request narrower than the training world.

Cortex shards a ``fwd_bwd`` batch across its training GPUs and rejects the
request outright if any shard would get no rows. That rejection arrives after
a step's rollouts have already been collected and paid for, which is the
expensive way to find out, so the shapes are asserted here instead.
"""

import asyncio

import pytest
import torch

from arctic_platform.integrations.harbor.backend import ArcticCortexBackend
from arctic_platform.integrations.harbor.models import (
    PostTrainingConfig,
    Rollout,
    RolloutDataset,
)


class _RecordingClient:
    """Stands in for Cortex, recording the shape of every request."""

    def __init__(self):
        self.row_counts: list[int] = []

    async def fwd_bwd(self, batch, processing=None):
        self.row_counts.append(int(batch["input_ids"].shape[0]))
        return {"metrics": {"loss": 0.0}}

    async def step(self):
        return {"metrics": {"update_successful": 1.0}}

    async def sync_weights(self):
        return None


def _dataset(n: int) -> RolloutDataset:
    # Alternating rewards keep every group's spread non-zero, and varied
    # lengths exercise the length-sorted ordering.
    return RolloutDataset(
        dataset_id="d",
        model_name="m",
        tokenizer_name="m",
        rollouts=[
            Rollout(
                prompt_token_ids=list(range(3 + i)),
                completion_token_ids=list(range(2 + i)),
                reward=float(i % 2),
                group_id="g",
                metadata={"traj_id": f"t{i}"},
            )
            for i in range(n)
        ],
    )


def _backend(train_gpus: int, micro: int) -> tuple[ArcticCortexBackend, _RecordingClient]:
    cfg = PostTrainingConfig(
        base_model="m",
        train_gpus=train_gpus,
        micro_batch_size=micro,
        max_seq_len=4096,
        cortex_host="h",
        cortex_database="d",
        cortex_schema="s",
    )
    backend = ArcticCortexBackend(cfg)
    client = _RecordingClient()
    backend._client = client
    return backend, client


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 9, 36, 37])
def test_no_request_is_narrower_than_the_training_world(n):
    backend, client = _backend(train_gpus=2, micro=1)
    asyncio.run(backend.train(_dataset(n), step=0))

    assert client.row_counts, "no fwd_bwd calls were made"
    assert min(client.row_counts) >= 2, client.row_counts
    assert sum(client.row_counts) == n, "every rollout must be trained on once"


def test_odd_tail_is_absorbed_not_left_alone():
    backend, client = _backend(train_gpus=2, micro=2)
    asyncio.run(backend.train(_dataset(7), step=0))

    assert client.row_counts == [2, 2, 3]
    assert sum(client.row_counts) == 7


def test_configured_micro_batch_is_respected_when_wide_enough():
    backend, client = _backend(train_gpus=2, micro=4)
    asyncio.run(backend.train(_dataset(8), step=0))

    assert client.row_counts == [4, 4]


def test_single_gpu_allows_single_row_batches():
    backend, client = _backend(train_gpus=1, micro=1)
    asyncio.run(backend.train(_dataset(3), step=0))

    assert client.row_counts == [1, 1, 1]


def test_four_gpus_raise_the_floor_to_four():
    backend, client = _backend(train_gpus=4, micro=1)
    asyncio.run(backend.train(_dataset(9), step=0))

    assert min(client.row_counts) >= 4, client.row_counts
    assert sum(client.row_counts) == 9


def test_batch_smaller_than_the_world_is_sent_whole():
    """Three rows on four GPUs cannot be split; one request is the best we can do."""
    backend, client = _backend(train_gpus=4, micro=1)
    asyncio.run(backend.train(_dataset(3), step=0))

    assert client.row_counts == [3]
