import logging
import pickle

import pytest
import torch

from arctic_platform.inference.server.router_replay import (
    RouterReplayCacheRX,
    RouterReplayCacheTX,
    RouterReplayMissingError,
)
from arctic_platform.inference.server.router_replay.all2all import (
    RouterReplayGroup,
    _compute_plan,
    _missing_is_tolerated,
    _PerRankManifest,
)


def _sender(rank, held, shape=(4, 2, 2), supports_allow_missing=True):
    return _PerRankManifest(
        role="sender", rank=rank, held=list(held), shapes={sid: list(shape) for sid in held},
        supports_allow_missing=supports_allow_missing,
    )


def _receiver(rank, needed, allow_missing=False, supports_allow_missing=True):
    return _PerRankManifest(
        role="receiver", rank=rank, needed=list(needed), allow_missing=allow_missing,
        supports_allow_missing=supports_allow_missing,
    )


def _legacy(manifest):
    """The manifest as a rank on the pre-``allow_missing`` planner sends it: no handshake field."""
    del manifest.__dict__["allow_missing"]
    del manifest.__dict__["supports_allow_missing"]
    return pickle.loads(pickle.dumps(manifest))


class _FakePG:
    """``all_gather_obj`` over a fixed peer list, with the caller's manifest swapped in at its rank."""

    def __init__(self, peers):
        self._peers = list(peers)

    def all_gather_obj(self, mine):
        return [mine if m.rank == mine.rank else m for m in self._peers]


class _FakeNCCL:
    """Loopback NCCL: ``recv`` fills from ``recv_values`` in call order; ``send`` records destinations."""

    def __init__(self, recv_values=()):
        self._recv_values = list(recv_values)
        self.sent = []
        self.fuses = 0

    def group_start(self):
        self.fuses += 1

    def group_end(self):
        pass

    def send(self, tensor, dst):
        self.sent.append((dst, tensor.clone()))

    def recv(self, tensor, src):
        tensor.copy_(self._recv_values.pop(0))


def _group(role, rank, peers, nccl=None):
    return RouterReplayGroup(
        role=role,
        rank=rank,
        world_size=len(peers),
        pg=_FakePG(peers),
        nccl=nccl or _FakeNCCL(),
        device=torch.device("cpu"),
    )


@pytest.fixture(autouse=True)
def _cpu_stream(monkeypatch):
    # The recv path synchronizes the CUDA stream after the fuse; on CPU there is nothing to wait for.
    class _Stream:
        def synchronize(self):
            pass

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_a, **_k: _Stream())


def test_planner_tolerates_missing_only_when_every_needing_receiver_allows():
    senders = [_sender(2, ["rr1:a"])]
    allowing = [_receiver(0, ["rr1:a", "rr1:b"], allow_missing=True), _receiver(1, ["rr1:a"])]

    plan, missing = _compute_plan(senders + allowing)

    assert missing == {"rr1:b"}
    assert [(op.recv_rank, op.sample_id) for op in plan] == [(0, "rr1:a"), (1, "rr1:a")]
    # Rank 1 is strict but needs nothing missing, so it does not veto.
    assert _missing_is_tolerated(senders + allowing, missing) is True

    mixed = [_receiver(0, ["rr1:b"], allow_missing=True), _receiver(1, ["rr1:b"])]
    _, missing = _compute_plan(senders + mixed)
    assert _missing_is_tolerated(senders + mixed, missing) is False


def test_strict_recv_raises_missing_error_unchanged():
    peers = [_receiver(0, ["rr1:b", "rr1:a"]), _sender(1, ["rr1:a"])]
    group = _group("receiver", 0, peers)
    cache = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)

    with pytest.raises(RouterReplayMissingError) as exc_info:
        group.recv(cache, needed_sample_ids=["rr1:b", "rr1:a"])

    assert exc_info.value.missing_sample_ids == ["rr1:b"]
    assert str(exc_info.value) == "router-replay missing for 1 sample_id(s): ['rr1:b']"
    assert group.stats()["n_missing_raises"] == 1
    assert group.nccl.fuses == 0  # data collective skipped


def test_allow_missing_recv_drops_missing_and_receives_rest(caplog):
    caplog.set_level(logging.INFO, logger="arctic_platform.inference.server.router_replay.all2all")
    present = torch.arange(16, dtype=torch.uint8).reshape(4, 2, 2)
    needed = ["rr1:gone-1", "rr1:a", "rr1:gone-0"]
    peers = [_receiver(0, needed, allow_missing=True), _sender(1, ["rr1:a"])]
    group = _group("receiver", 0, peers, _FakeNCCL([present]))
    cache = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)

    result = group.recv(cache, needed_sample_ids=needed, allow_missing=True)

    assert result == {"tensors_recv": 1, "dropped_sample_ids": ["rr1:gone-1", "rr1:gone-0"]}
    assert torch.equal(cache.pop("rr1:a"), present)
    assert "rr1:gone-1" not in cache
    assert "dropped 2 of 3 needed sample_id(s)" in caplog.text


@pytest.mark.parametrize("held", [[], ["rr1:a", "rr1:b"]])
def test_allow_missing_recv_handles_all_missing_and_none_missing(held):
    needed = ["rr1:a", "rr1:b"]
    peers = [_receiver(0, needed, allow_missing=True), _sender(1, held)]
    values = [torch.full((4, 2, 2), i, dtype=torch.uint8) for i in range(len(held))]
    group = _group("receiver", 0, peers, _FakeNCCL(values))
    cache = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)

    result = group.recv(cache, needed_sample_ids=needed, allow_missing=True)

    expected_dropped = [] if held else needed
    assert result == {"tensors_recv": len(held), "dropped_sample_ids": expected_dropped}
    assert len(cache) == len(held)


def test_mixed_mode_receivers_raise_identically_on_every_rank():
    """A strict receiver needing a missing id vetoes tolerance on every rank, senders included."""
    peers = [
        _receiver(0, ["rr1:a", "rr1:gone"], allow_missing=True),
        _receiver(1, ["rr1:gone"]),
        _sender(2, ["rr1:a"]),
    ]
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    tx.put("rr1:a", torch.zeros(4, 2, 2, dtype=torch.uint8))
    rx = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)
    errors = []
    for rank, call in (
        (0, lambda g: g.recv(rx, needed_sample_ids=["rr1:a", "rr1:gone"], allow_missing=True)),
        (1, lambda g: g.recv(rx, needed_sample_ids=["rr1:gone"])),
        (2, lambda g: g.send(tx)),
    ):
        group = _group(peers[rank].role, rank, peers)
        with pytest.raises(RouterReplayMissingError) as exc_info:
            call(group)
        errors.append(str(exc_info.value))
        assert group.nccl.fuses == 0

    assert errors == [errors[0]] * 3
    assert "rr1:a" in tx  # the raise keeps the sender's entry for a retry


def test_sender_proceeds_when_missing_is_tolerated_and_discards_sent():
    peers = [
        _receiver(0, ["rr1:a", "rr1:gone"], allow_missing=True),
        _sender(1, ["rr1:a"]),
    ]
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    tx.put("rr1:a", torch.ones(4, 2, 2, dtype=torch.uint8))
    group = _group("sender", 1, peers)

    assert group.send(tx) == {"tensors_sent": 1}

    assert [dst for dst, _ in group.nccl.sent] == [0]
    assert "rr1:a" not in tx


def test_legacy_manifest_unpickles_without_the_handshake():
    legacy = _legacy(_sender(1, ["rr1:a"]))

    assert legacy.supports_allow_missing is False
    assert legacy.allow_missing is False


def test_legacy_sender_vetoes_tolerance_so_every_rank_raises(caplog):
    """An older sender raises on any missing sid; new receivers must raise too, not enter the data fuse."""
    caplog.set_level(logging.INFO, logger="arctic_platform.inference.server.router_replay.all2all")
    peers = [
        _receiver(0, ["rr1:a", "rr1:gone"], allow_missing=True),
        _legacy(_sender(1, ["rr1:a"])),
    ]
    _, missing = _compute_plan(peers)
    assert _missing_is_tolerated(peers, missing) is False

    group = _group("receiver", 0, peers)
    rx = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)
    with pytest.raises(RouterReplayMissingError) as exc_info:
        group.recv(rx, needed_sample_ids=["rr1:a", "rr1:gone"], allow_missing=True)

    assert exc_info.value.missing_sample_ids == ["rr1:gone"]
    assert group.nccl.fuses == 0
    assert "allow_missing refused; ranks [1]" in caplog.text


def test_legacy_sender_does_not_matter_when_nothing_is_missing():
    peers = [_receiver(0, ["rr1:a"], allow_missing=True), _legacy(_sender(1, ["rr1:a"]))]
    group = _group("receiver", 0, peers, _FakeNCCL([torch.zeros(4, 2, 2, dtype=torch.uint8)]))
    rx = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)

    assert group.recv(rx, needed_sample_ids=["rr1:a"], allow_missing=True) == {
        "tensors_recv": 1, "dropped_sample_ids": [],
    }


def test_uint16_ids_cross_as_bytes_and_land_exact():
    """vLLM captures uint16 ids above 256 experts; NCCL has no 16-bit integer type."""
    ids = torch.tensor([255, 256, 287, 511], dtype=torch.uint16).reshape(4, 1, 1)
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    tx.put("rr1:a", ids)
    sender = _group("sender", 1, [_receiver(0, ["rr1:a"]), _sender(1, [])])
    peers = [_receiver(0, ["rr1:a"]), sender._build_send_manifest(tx.snapshot())]
    sender.send(tx)
    [(dst, wire)] = sender.nccl.sent
    rx = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)

    _group("receiver", 0, peers, _FakeNCCL([wire])).recv(rx, needed_sample_ids=["rr1:a"])

    assert (dst, wire.dtype) == (0, torch.uint8)
    out = rx.pop("rr1:a")
    assert out.dtype == torch.uint16 and torch.equal(out, ids)
