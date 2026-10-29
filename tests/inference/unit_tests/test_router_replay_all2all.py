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
    _plan_rounds,
)


def _sender(rank, held, shape=(4, 2, 2), supports_allow_missing=True, staging_bytes=None):
    return _PerRankManifest(
        role="sender", rank=rank, held=list(held), shapes={sid: list(shape) for sid in held},
        supports_allow_missing=supports_allow_missing,
        staging_bytes=staging_bytes, supports_staged_rounds=True, supports_token_ranges=True,
    )


def _receiver(rank, needed, allow_missing=False, supports_allow_missing=True, token_ranges=None):
    return _PerRankManifest(
        role="receiver", rank=rank, needed=list(needed), allow_missing=allow_missing,
        supports_allow_missing=supports_allow_missing, supports_staged_rounds=True,
        token_ranges=token_ranges, supports_token_ranges=True,
    )


_PRE_ROUNDS_FIELDS = ("staging_bytes", "supports_staged_rounds")


def _legacy(
    manifest,
    fields=("allow_missing", "supports_allow_missing", "token_ranges", "supports_token_ranges", *_PRE_ROUNDS_FIELDS),
):
    """The manifest as a rank on an older planner sends it: by default the pre-``allow_missing`` one."""
    for name in fields:
        del manifest.__dict__[name]
    return pickle.loads(pickle.dumps(manifest))


def _pre_rounds(manifest):
    """The manifest as a rank whose planner predates staged rounds sends it."""
    return _legacy(manifest, fields=_PRE_ROUNDS_FIELDS)


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
        self.ops_per_fuse = []

    def group_start(self):
        self.fuses += 1
        self.ops_per_fuse.append([])

    def group_end(self):
        pass

    def send(self, tensor, dst):
        self.sent.append((dst, tensor.clone()))
        self.ops_per_fuse[-1].append(("send", dst, tensor.numel()))

    def recv(self, tensor, src):
        tensor.copy_(self._recv_values.pop(0))
        self.ops_per_fuse[-1].append(("recv", src, tensor.numel()))


def _group(role, rank, peers, nccl=None, device="cpu", staging_bytes=None):
    return RouterReplayGroup(
        role=role,
        rank=rank,
        world_size=len(peers),
        pg=_FakePG(peers),
        nccl=nccl or _FakeNCCL(),
        device=torch.device(device),
        staging_bytes=staging_bytes,
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


def _ops(rounds):
    return [[(op.sender_rank, op.recv_rank, op.sample_id) for op in ops] for ops in rounds]


def test_staged_rounds_cap_each_senders_bytes_and_keep_plan_order():
    # 16-byte captures and a 40-byte budget: two of a sender's captures per round.
    manifests = [
        _receiver(0, ["rr1:a", "rr1:b", "rr1:c"]),
        _receiver(1, ["rr1:d", "rr1:e"]),
        _sender(2, ["rr1:a", "rr1:b", "rr1:c", "rr1:e"], staging_bytes=40),
        _sender(3, ["rr1:d"]),
    ]
    plan, _ = _compute_plan(manifests)

    assert _ops(_plan_rounds(manifests, plan)) == [
        [(2, 0, "rr1:a"), (2, 0, "rr1:b"), (3, 1, "rr1:d")],
        [(2, 0, "rr1:c"), (2, 1, "rr1:e")],
    ]


def test_staged_rounds_use_the_smallest_budget_and_send_an_oversized_capture_alone():
    manifests = [
        _receiver(0, ["rr1:a", "rr1:b"]),
        _sender(1, ["rr1:a", "rr1:b"], staging_bytes=1 << 20),
        _sender(2, [], staging_bytes=8),
    ]
    plan, _ = _compute_plan(manifests)

    assert _ops(_plan_rounds(manifests, plan)) == [[(1, 0, "rr1:a")], [(1, 0, "rr1:b")]]


@pytest.mark.parametrize("staging_bytes,pre_rounds_rank", [(None, None), (8, 0), (8, 2)])
def test_plan_is_one_round_without_a_budget_or_with_a_peer_that_predates_rounds(
    staging_bytes, pre_rounds_rank,
):
    manifests = [
        _receiver(0, ["rr1:a", "rr1:b"]),
        _sender(1, ["rr1:a"], staging_bytes=staging_bytes),
        _sender(2, ["rr1:b"]),
    ]
    if pre_rounds_rank is not None:
        manifests[pre_rounds_rank] = _pre_rounds(manifests[pre_rounds_rank])
    plan, _ = _compute_plan(manifests)

    assert _plan_rounds(manifests, plan) == [plan]


def test_sender_advertises_staging_only_for_a_cache_off_the_nccl_device():
    group = _group("sender", 1, [], device="cuda:0", staging_bytes=1234)
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)

    host = group._build_send_manifest(tx.snapshot(), cache_device=tx.device)
    on_device = group._build_send_manifest(tx.snapshot(), cache_device=torch.device("cuda:0"))

    assert (host.staging_bytes, on_device.staging_bytes) == (1234, None)
    assert host.supports_staged_rounds and on_device.supports_staged_rounds


def test_staged_exchange_runs_one_fuse_per_round_on_both_sides_and_lands_exact(caplog):
    caplog.set_level(logging.INFO, logger="arctic_platform.inference.server.router_replay.all2all")
    captures = {f"rr1:{i}": torch.full((4, 2, 2), i, dtype=torch.uint8) for i in range(5)}
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    for sid, capture in captures.items():
        tx.put(sid, capture)
    needed = list(captures)
    # A host-memory peer's 40-byte budget sets the rounds for every rank: two 16-byte captures each.
    budget_peer = _sender(2, [], staging_bytes=40)
    sender = _group("sender", 1, [_receiver(0, needed), _sender(1, []), budget_peer])
    sender_manifest = sender._build_send_manifest(tx.snapshot(), cache_device=tx.device)

    assert sender.send(tx) == {"tensors_sent": 5}
    assert "sent 5 tensor(s) in 3 staged round(s)" in caplog.text

    receiver = _group(
        "receiver", 0, [_receiver(0, needed), sender_manifest, budget_peer],
        _FakeNCCL([wire for _, wire in sender.nccl.sent]),
    )
    rx = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)
    assert receiver.recv(rx, needed_sample_ids=needed) == {"tensors_recv": 5, "dropped_sample_ids": []}

    sent = [[(dst, n) for _, dst, n in ops] for ops in sender.nccl.ops_per_fuse]
    received = [[(src, n) for _, src, n in ops] for ops in receiver.nccl.ops_per_fuse]
    assert sent == [[(0, 16), (0, 16)], [(0, 16), (0, 16)], [(0, 16)]]
    assert received == [[(1, 16), (1, 16)], [(1, 16), (1, 16)], [(1, 16)]]
    assert sender.stats()["n_rounds"] == receiver.stats()["n_rounds"] == 3
    for sid, capture in captures.items():
        assert torch.equal(rx.pop(sid), capture)
    assert len(tx) == 0


def test_staging_budget_reads_the_environment_and_must_be_positive(monkeypatch):
    monkeypatch.setenv("ARCTIC_ROUTER_REPLAY_STAGING_BYTES", "4096")
    assert _group("sender", 1, []).staging_bytes == 4096

    with pytest.raises(ValueError, match="must be positive"):
        _group("sender", 1, [], staging_bytes=0)


def test_staged_rounds_count_each_slice_not_the_whole_capture():
    # 16-byte captures, but each receiver asks for half of each (8 bytes): a 16-byte budget fits two slices.
    halves = {"rr1:a": (0, 2), "rr1:b": (2, 4)}
    manifests = [
        _receiver(0, ["rr1:a", "rr1:b"], token_ranges=halves),
        _sender(1, ["rr1:a", "rr1:b"], staging_bytes=16),
    ]
    plan, _ = _compute_plan(manifests)

    assert [op.nbytes for op in plan] == [8, 8]
    assert _ops(_plan_rounds(manifests, plan)) == [[(1, 0, "rr1:a"), (1, 0, "rr1:b")]]


def test_token_ranges_move_only_each_receivers_slice():
    capture = torch.arange(24, dtype=torch.uint8).reshape(6, 2, 2)
    other = capture + 100
    peers = [
        _receiver(0, ["rr1:a", "rr1:b"], token_ranges={"rr1:a": (0, 4)}),
        _receiver(1, ["rr1:a", "rr1:b"], token_ranges={"rr1:a": (4, 9), "rr1:b": (6, 6)}),
        _sender(2, ["rr1:a", "rr1:b"], shape=(6, 2, 2)),
    ]
    tx = RouterReplayCacheTX(device=torch.device("cpu"), max_bytes=1 << 20)
    tx.put("rr1:a", capture)
    tx.put("rr1:b", other)
    sender = _group("sender", 2, peers)
    sender.send(tx)
    # A range past the capture is clipped to it; an empty one moves nothing.
    assert [(dst, sent.tolist()) for dst, sent in sender.nccl.sent] == [
        (0, capture[:4].tolist()), (0, other.tolist()), (1, capture[4:].tolist()),
    ]

    receiver = _group("receiver", 1, peers, _FakeNCCL([capture[4:]]))
    cache = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)
    result = receiver.recv(
        cache, needed_sample_ids=["rr1:a", "rr1:b"], token_ranges={"rr1:a": (4, 9), "rr1:b": (6, 6)},
    )
    assert result["capture_slices"] == {"rr1:a": (4, 6), "rr1:b": (6, 6)}  # (start, whole capture length)
    assert torch.equal(cache.pop("rr1:a"), capture[4:])
    assert cache.pop("rr1:b").shape == (0, 2, 2)


def test_a_peer_without_token_ranges_sends_whole_captures_to_every_rank():
    capture = torch.arange(24, dtype=torch.uint8).reshape(6, 2, 2)
    older_sender = _legacy(_sender(1, ["rr1:a"], shape=(6, 2, 2)), fields=("token_ranges", "supports_token_ranges"))
    peers = [_receiver(0, ["rr1:a"], token_ranges={"rr1:a": (2, 4)}), older_sender]
    receiver = _group("receiver", 0, peers, _FakeNCCL([capture]))
    cache = RouterReplayCacheRX(device=torch.device("cpu"), max_bytes=1 << 20)

    result = receiver.recv(cache, needed_sample_ids=["rr1:a"], token_ranges={"rr1:a": (2, 4)})

    assert result["capture_slices"] == {}
    assert torch.equal(cache.pop("rr1:a"), capture)
