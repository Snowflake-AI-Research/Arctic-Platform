"""Two real processes on a gloo group: a drifting TP rank 1 ends every step with rank 0's drafts and samples."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.inference.vllm import tp_spec_decode_sync as sync

STEPS = 4


def _rank_values(rank: int, step: int) -> dict:
    """What this rank's own drafter and sampler produce; rank 1 drifts from rank 0 on every step."""
    drift = 100 * rank
    return dict(
        draft=torch.arange(8).reshape(4, 2) + 10 * step + drift,
        sampled=torch.tensor([[1, 2, 3], [4, 5, 6]]) + 10 * step + drift,
        num_sampled=torch.tensor([3, 1]) if rank == 0 else torch.tensor([1, 3]),
        num_rejected=torch.tensor([0, 2]) if rank == 0 else torch.tensor([2, 0]),
    )


class _Group:
    def __init__(self, rank: int, world_size: int):
        self.rank_in_group, self.world_size = rank, world_size

    def broadcast(self, t, src=0):
        dist.broadcast(t, src=src)
        return t


def _rank_main(rank: int, world_size: int, init_file: str, out_file: str, stats: bool) -> None:
    dist.init_process_group("gloo", init_method=f"file://{init_file}", world_size=world_size, rank=rank,
                            timeout=timedelta(seconds=60))  # a mismatched collective fails here instead of hanging
    group = _Group(rank, world_size)
    sync._tp_group = lambda: group
    sync.STATS_ENABLED = stats

    class Runner:
        def __init__(self):
            self.batch_sharder = None
            self.speculator = SimpleNamespace(propose=self._propose)
            self.step = 0

        def _propose(self):
            draft = _rank_values(rank, self.step)["draft"]
            # odd steps return a non-contiguous column slice, as some vLLM speculators do
            return draft[:, :1] if self.step % 2 else draft

        def sample(self, hidden_states, input_batch, grammar_output):
            v = _rank_values(rank, self.step)
            out = SimpleNamespace(sampled_token_ids=v["sampled"], num_sampled=v["num_sampled"],
                                  num_rejected=v["num_rejected"])  # fmt: skip
            return out, out.num_sampled, out.num_rejected

    sync.patch_runner(Runner)
    runner = Runner()
    seen = []
    for step in range(STEPS):
        runner.step = step
        draft = runner.speculator.propose()
        out, num_sampled, num_rejected = runner.sample(None, None, None)
        seen.append((draft.tolist(), out.sampled_token_ids.tolist(), num_sampled.tolist(), num_rejected.tolist()))
    stats = getattr(runner, "_arctic_tp_sync_stats", None)
    corrected = {k: int(v) for k, v in stats.corrected.items()} if stats else None
    torch.save({"seen": seen, "corrected": corrected}, out_file)
    dist.destroy_process_group()


def _expected_rank0() -> list:
    out = []
    for step in range(STEPS):
        v = _rank_values(0, step)
        draft = v["draft"][:, :1] if step % 2 else v["draft"]
        out.append((draft.tolist(), v["sampled"].tolist(), v["num_sampled"].tolist(), v["num_rejected"].tolist()))
    return out


@pytest.mark.parametrize("stats", [False, True])
def test_drifting_rank_ends_each_step_with_rank0_values(tmp_path, stats):
    out = str(tmp_path / "rank{}.pt")
    mp.spawn(_run_rank, args=(2, str(tmp_path / "group"), out, stats), nprocs=2, join=True)

    results = [torch.load(out.format(r)) for r in range(2)]
    assert results[0]["seen"] == results[1]["seen"] == _expected_rank0()
    if stats:
        assert results[0]["corrected"] == {"draft": 0, "sample": 0}
        assert results[1]["corrected"] == {"draft": STEPS, "sample": STEPS}
    else:
        assert results[0]["corrected"] is None and results[1]["corrected"] is None


def _run_rank(rank: int, world_size: int, init_file: str, out_pattern: str, stats: bool) -> None:
    _rank_main(rank, world_size, init_file, out_pattern.format(rank), stats)
