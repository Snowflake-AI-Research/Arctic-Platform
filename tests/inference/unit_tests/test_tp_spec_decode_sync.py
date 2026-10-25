import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.inference.vllm import tp_spec_decode_sync as sync


class _FakeTP:
    """This process is TP rank 1; broadcast copies rank 0's tensor of the same shape in place."""

    def __init__(self, rank0=None, world_size=2):
        self.world_size = world_size
        self.rank_in_group = 1
        self.rank0 = rank0 or {}
        self.calls = []

    def broadcast(self, t, src=0):
        assert src == 0 and t.is_contiguous()
        self.calls.append(tuple(t.shape))
        t.copy_(self.rank0[tuple(t.shape)])
        return t


@pytest.fixture
def tp(monkeypatch):
    fake = _FakeTP()
    monkeypatch.setattr(sync, "_tp_group", lambda: fake)
    return fake


def test_broadcast_overwrites_in_place_including_non_contiguous(tp):
    tp.rank0 = {(2, 2): torch.tensor([[1, 2], [3, 4]]), (2,): torch.tensor([2, 1])}
    a = torch.tensor([[1, 9], [3, 4]]).t()
    b = torch.tensor([2, 2])

    sync.broadcast_from_rank0([a, None, b])

    assert torch.equal(a, tp.rank0[(2, 2)])
    assert torch.equal(b, tp.rank0[(2,)])
    assert len(tp.calls) == 2


def test_single_rank_is_a_no_op(monkeypatch):
    fake = _FakeTP(world_size=1)
    monkeypatch.setattr(sync, "_tp_group", lambda: fake)
    t = torch.tensor([5])

    sync.broadcast_from_rank0([t])

    assert fake.calls == [] and t.item() == 5


def _runner_class(sampled, num_sampled, num_rejected, draft):
    class Runner:
        def __init__(self, speculate=True, sharded=False):
            self.batch_sharder = object() if sharded else None
            self.speculator = (
                SimpleNamespace(propose=lambda *a, **k: draft) if speculate else None
            )

        def sample(self, hidden_states, input_batch, grammar_output):
            out = SimpleNamespace(
                sampled_token_ids=sampled,
                num_sampled=num_sampled,
                num_rejected=num_rejected,
            )
            return out, out.num_sampled, out.num_rejected

    sync.patch_runner(Runner)
    sync.patch_runner(Runner)  # idempotent
    return Runner


def test_rank0_drafts_and_samples_win(tp):
    sampled = torch.tensor([[5, 6, 0], [7, 0, 0]])
    draft = torch.tensor([[11, 12], [13, 14]])
    tp.rank0 = {
        (2, 3): torch.tensor([[5, 6, 8], [7, 0, 0]]),
        (2,): torch.tensor([3, 1]),
        (2, 2): torch.tensor([[11, 12], [13, 99]]),
    }
    runner = _runner_class(sampled, torch.tensor([2, 1]), torch.tensor([1, 2]), draft)()

    assert runner.speculator.propose("batch") is draft and draft[1, 1] == 99
    out, num_sampled, num_rejected = runner.sample(None, None, None)

    assert torch.equal(out.sampled_token_ids, tp.rank0[(2, 3)])
    assert num_sampled is out.num_sampled and num_sampled.tolist() == [3, 1]
    assert num_rejected.tolist() == [3, 1]
    stats = runner._arctic_tp_sync_stats
    assert stats.steps == {"draft": 1, "sample": 1}
    assert {k: int(v) for k, v in stats.corrected.items()} == {"draft": 1, "sample": 1}


def test_no_sync_without_speculation_or_with_batch_sharded_sampling(tp):
    t = torch.tensor([[1]])
    runner_cls = _runner_class(t, torch.tensor([1]), torch.tensor([0]), t)

    runner_cls(speculate=False).sample(None, None, None)
    sharded = runner_cls(sharded=True)
    sharded.sample(None, None, None)
    sharded.speculator.propose()

    assert tp.calls == []


def test_unread_sample_slots_do_not_count_as_corrections():
    before = torch.tensor([[5, 6, 123], [7, 0, 456]])
    after = torch.tensor([[5, 6, 0], [7, 0, 0]])
    counts = torch.tensor([2, 1])

    assert not sync.samples_differ(before, counts, after, counts)
    assert sync.samples_differ(before, counts, after, torch.tensor([3, 1]))
    after[0, 1] = 9
    assert sync.samples_differ(before, counts, after, counts)


def test_disabled_by_env(monkeypatch):
    monkeypatch.setattr(sync, "_APPLIED", False)
    monkeypatch.setenv("ARCTIC_TP_SPEC_SYNC", "0")

    sync.ensure_tp_spec_decode_sync()

    assert sync._APPLIED is False


def test_vllm_model_runner_v2_matches_what_the_patch_assumes():
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from vllm.v1.worker.gpu.sample.output import SamplerOutput

    src = inspect.getsource(importlib.import_module("vllm.v1.worker.gpu.model_runner"))
    assert (
        "    def sample(\n        self,\n        hidden_states: torch.Tensor,\n"
        "        input_batch: InputBatch,\n        grammar_output: GrammarOutput | None,"
    ) in src
    assert (
        "return sampler_output, sampler_output.num_sampled, sampler_output.num_rejected"
        in src
    )
    assert "self.speculator = init_speculator(" in inspect.getsource(GPUModelRunner.__init__)
    assert src.count("self.speculator.propose(") == 2
    assert "self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens" in src
    assert {"sampled_token_ids", "num_sampled", "num_rejected"} <= set(
        SamplerOutput.__dataclass_fields__
    )
