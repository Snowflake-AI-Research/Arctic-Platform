"""Two-collective NCCL protocol for cross-zone router-replay transfer.

The exchange is many-to-many sparse: each sampling replica holds some
sample_ids; each training rank needs some, possibly from different replicas.
Run as two collectives with no out-of-band coordination:

    A (CPU, all_gather_obj)  ranks publish held/needed sample_ids.
    B (local, pure)          every rank derives the same plan; if any
                             needed sid has no sender, every rank raises
                             RouterReplayMissingError identically and the
                             data collective is skipped -- unless every
                             receiver needing a missing sid set
                             ``allow_missing``, in which case those sids are
                             dropped from the plan on every rank.
    C (GPU, group_start/send|recv/group_end)
                             one fused NCCL op transfers all (sid -> recv).

TCP/stateless bootstrap mirrors weight_sync's stateless_init_nccl; the only
new ingredient is the manifest-driven plan.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Literal

import torch

from arctic_platform.inference.server.router_replay.cache import ROUTER_REPLAY_CACHE_DTYPE
from arctic_platform.inference.server.router_replay.cache import RouterReplayCacheRX
from arctic_platform.inference.server.router_replay.cache import RouterReplayCacheTX
from arctic_platform.inference.server.router_replay.cache import RouterReplayDuplicateError
from arctic_platform.inference.server.router_replay.cache import RouterReplayMissingError
from arctic_platform.inference.server.router_replay.cache import is_exact_replay_id

logger = logging.getLogger(__name__)

# Dtype stored/transferred for routed_experts. Qwen3.6 MoE has 256 experts,
# so ids are in [0, 255] and fit in uint8. Training widens to int64 at the
# model boundary where torch gather/DeepEP dispatch require long indices.
_ROUTED_EXPERTS_DTYPE = ROUTER_REPLAY_CACHE_DTYPE

Role = Literal["sender", "receiver"]


@dataclass
class _PerRankManifest:
    """One rank's Phase-A contribution.

    Senders fill ``held`` + ``shapes``; receivers fill ``needed``. ``role``
    is in-band so the planner needs no out-of-band role table.
    ``allow_missing`` lets a receiver tolerate needed sids no sender holds.
    ``supports_allow_missing`` is the capability handshake: this code always
    sets it, while a manifest from a rank running an older planner (which
    raises on any missing sid) unpickles with the ``False`` class default.
    """

    role: Role
    rank: int
    held: list[str] = field(default_factory=list)
    needed: list[str] = field(default_factory=list)
    shapes: dict[str, list[int]] = field(default_factory=dict)
    discard: bool = True
    allow_missing: bool = False
    supports_allow_missing: bool = False


@dataclass(frozen=True)
class _TransferOp:
    sender_rank: int
    recv_rank: int
    sample_id: str
    shape: tuple[int, ...]
    discard: bool = True


def _compute_plan(
    manifests: Sequence[_PerRankManifest],
) -> tuple[list[_TransferOp], set[str]]:
    """Deterministically derive the per-pair transfer plan and missing set.

    Pure function of the manifest list (identical on every rank after
    all_gather_obj), so the plan is computed locally without coordination.
    Exact replay ids must have one owner. Legacy ids retain deterministic
    lowest-rank selection for compatibility, with an explicit warning.
    """
    owner: dict[str, int] = {}
    owners: dict[str, list[int]] = {}
    shape_by_sid: dict[str, tuple[int, ...]] = {}
    for m in sorted(manifests, key=lambda x: x.rank):
        if m.role != "sender":
            continue
        for sid in m.held:
            owners.setdefault(sid, []).append(m.rank)
            if sid in owner:
                continue  # earlier (lower-rank) sender already owns it
            owner[sid] = m.rank
            shape_by_sid[sid] = tuple(m.shapes[sid])

    duplicate_exact = sorted(
        sid for sid, sender_ranks in owners.items() if len(sender_ranks) > 1 and is_exact_replay_id(sid)
    )
    if duplicate_exact:
        raise RouterReplayDuplicateError(
            duplicate_exact,
            owners={sid: owners[sid] for sid in duplicate_exact},
        )
    duplicate_legacy = sorted(
        sid for sid, sender_ranks in owners.items() if len(sender_ranks) > 1 and not is_exact_replay_id(sid)
    )
    if duplicate_legacy:
        logger.warning(
            "router-replay legacy duplicate ids use lowest-rank owner count=%d ids_head=%s owners_head=%s",
            len(duplicate_legacy),
            duplicate_legacy[:8],
            {sid: owners[sid] for sid in duplicate_legacy[:8]},
        )

    plan: list[_TransferOp] = []
    missing: set[str] = set()
    for m in sorted(manifests, key=lambda x: x.rank):
        if m.role != "receiver":
            continue
        for sid in m.needed:
            sender_rank = owner.get(sid)
            if sender_rank is None:
                missing.add(sid)
                continue
            plan.append(
                _TransferOp(
                    sender_rank=sender_rank,
                    recv_rank=m.rank,
                    sample_id=sid,
                    shape=shape_by_sid[sid],
                    discard=m.discard,
                )
            )
    return plan, missing


def _missing_is_tolerated(
    manifests: Sequence[_PerRankManifest],
    missing: set[str],
) -> bool:
    """True iff every receiver needing a missing sid set ``allow_missing``
    and every rank runs a planner that tolerates missing sids.

    Pure function of the gathered manifests, so every rank -- senders
    included -- reaches the same verdict and either all proceed or all raise.
    A rank on an older planner always raises on a missing sid, so its peers
    must raise too or they would block in the data collective without it.
    """
    return all(m.supports_allow_missing for m in manifests) and all(
        m.allow_missing for m in manifests if m.role == "receiver" and not missing.isdisjoint(m.needed)
    )


class RouterReplayGroup:
    """One persistent NCCL group joining sampling + training workers.

    Lifecycle is per joint-job (bootstrap once after both jobs are RUNNING;
    tear down on destroy). Per-step calls are :meth:`send` / :meth:`recv`;
    every rank must call exactly one per step or the all_gather_obj hangs.

    Independent of weight_sync's group (no barrier interaction).
    """

    def __init__(
        self,
        *,
        role: Role,
        rank: int,
        world_size: int,
        pg: Any,
        nccl: Any,
        device: torch.device,
    ) -> None:
        self.role = role
        self.rank = rank
        self.world_size = world_size
        self.pg = pg
        self.nccl = nccl
        self.device = device
        self._closed = False
        # Lifetime stats for observability; reset only on close().
        self._n_exchanges = 0
        self._n_tensors_sent = 0
        self._n_tensors_recv = 0
        self._n_missing_raises = 0

    # ------------------------------------------------------------------
    # Per-step entrypoints
    # ------------------------------------------------------------------

    def send(
        self,
        cache_tx: RouterReplayCacheTX,
    ) -> dict[str, Any]:
        """Sender-side per-step call; pairs with :meth:`recv`.

        The evict set is derived from the gathered receiver manifests (the
        sids this rank just sent to receivers that requested discard) and is
        removed *after* a successful exchange, so a missing-data raise does
        not lose entries needed for a retry.
        """
        if self._closed:
            raise RuntimeError("RouterReplayGroup is closed")
        if self.role != "sender":
            raise RuntimeError(f"send() called on role={self.role!r}")
        snapshot = cache_tx.snapshot()
        manifest = self._build_send_manifest(snapshot)
        plan, _ = self._exchange_manifest(manifest)  # raises on untolerated missing
        n_sent = self._run_data_exchange_send(plan, snapshot)
        discard_sample_ids = [op.sample_id for op in plan if op.sender_rank == self.rank and op.discard]
        if discard_sample_ids:
            cache_tx.discard(discard_sample_ids)
        self._n_exchanges += 1
        self._n_tensors_sent += n_sent
        return {"tensors_sent": n_sent}

    def recv(
        self,
        cache_rx: RouterReplayCacheRX,
        *,
        needed_sample_ids: Sequence[str],
        discard: bool = True,
        allow_missing: bool = False,
    ) -> dict[str, Any]:
        """Receiver-side per-step call; pairs with :meth:`send`.

        Defensively clears ``cache_rx`` first to drop stale entries from a
        prior step that may have crashed between recv and fwd_bwd.

        ``discard`` is advertised in the Phase-A manifest so the owning
        sender evicts these ``needed_sample_ids`` from its TX cache after a
        successful send (consume-once); pass ``False`` to keep them for
        replay / multi-consumer scenarios.

        ``allow_missing`` is advertised in the same manifest: when every
        receiver needing a sid no sender holds set it, the exchange proceeds
        without those sids and they are returned as ``dropped_sample_ids``
        (this rank's, in ``needed_sample_ids`` order) instead of raising
        :class:`RouterReplayMissingError`.
        """
        if self._closed:
            raise RuntimeError("RouterReplayGroup is closed")
        if self.role != "receiver":
            raise RuntimeError(f"recv() called on role={self.role!r}")
        stale = cache_rx.clear()
        if stale:
            logger.warning(
                "router-replay: cleared %d stale RX entries before exchange "
                "(prior step likely crashed between recv and fwd_bwd)",
                stale,
            )
        manifest = self._build_recv_manifest(
            needed_sample_ids,
            discard=discard,
            allow_missing=allow_missing,
        )
        try:
            plan, missing = self._exchange_manifest(manifest)
        except RouterReplayMissingError:
            self._n_missing_raises += 1
            raise
        dropped = list(dict.fromkeys(sid for sid in needed_sample_ids if sid in missing))
        if len(dropped) > 0:
            logger.info(
                "router-replay: dropped %d of %d needed sample_id(s) with no sender (allow_missing) ids_head=%s",
                len(dropped),
                len(needed_sample_ids),
                dropped[:8],
            )
        n_recv = self._run_data_exchange_recv(plan, cache_rx)
        self._n_exchanges += 1
        self._n_tensors_recv += n_recv
        return {"tensors_recv": n_recv, "dropped_sample_ids": dropped}

    def close(self) -> None:
        """Tear down the NCCL comm + stateless PG. Idempotent.

        Safe to call concurrently with an in-flight exchange — the next
        send/recv sees ``self._closed`` and raises RuntimeError.
        """
        if self._closed:
            return
        self._closed = True
        # Setting to None lets the StatelessProcessGroup drop its TCPStore
        # promptly; NCCLLibrary teardown happens on GC of the comm.
        self.nccl = None
        self.pg = None

    def stats(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "rank": self.rank,
            "world_size": self.world_size,
            "n_exchanges": self._n_exchanges,
            "n_tensors_sent": self._n_tensors_sent,
            "n_tensors_recv": self._n_tensors_recv,
            "n_missing_raises": self._n_missing_raises,
            "closed": self._closed,
        }

    # ------------------------------------------------------------------
    # Phase A: manifest exchange
    # ------------------------------------------------------------------

    def _build_send_manifest(self, snapshot: dict[str, torch.Tensor]) -> _PerRankManifest:
        shapes = {}
        for sid, tensor in snapshot.items():
            shapes[sid] = list(tensor.shape)
        return _PerRankManifest(
            role="sender",
            rank=self.rank,
            held=list(shapes.keys()),
            shapes=shapes,
            supports_allow_missing=True,
        )

    def _build_recv_manifest(
        self,
        needed_sample_ids: Sequence[str],
        *,
        discard: bool = True,
        allow_missing: bool = False,
    ) -> _PerRankManifest:
        return _PerRankManifest(
            role="receiver",
            rank=self.rank,
            needed=list(needed_sample_ids),
            discard=discard,
            allow_missing=allow_missing,
            supports_allow_missing=True,
        )

    def _exchange_manifest(
        self,
        mine: _PerRankManifest,
    ) -> tuple[list[_TransferOp], set[str]]:
        """Phase A + Phase B: gather manifests, compute plan, raise if missing.

        Returns the plan and the tolerated missing set (empty unless every
        receiver needing a missing sid set ``allow_missing``).
        """
        all_manifests: list[_PerRankManifest] = self.pg.all_gather_obj(mine)
        plan, missing = _compute_plan(all_manifests)
        if len(missing) > 0 and not _missing_is_tolerated(all_manifests, missing):
            legacy_ranks = sorted(m.rank for m in all_manifests if not m.supports_allow_missing)
            if mine.allow_missing and len(legacy_ranks) > 0:
                logger.info(
                    "router-replay: allow_missing refused; ranks %s run a planner without missing-sid tolerance",
                    legacy_ranks[:16],
                )
            # Every rank raises identically (same input -> same missing set),
            # so the data collective never runs and no rank blocks.
            raise RouterReplayMissingError(sorted(missing))
        return plan, missing

    # ------------------------------------------------------------------
    # Phase C: data exchange
    # ------------------------------------------------------------------

    def _run_data_exchange_send(
        self,
        plan: list[_TransferOp],
        snapshot: dict[str, torch.Tensor],
    ) -> int:
        my_ops = [op for op in plan if op.sender_rank == self.rank]
        if not my_ops:
            # Empty group_start/group_end keeps every rank in step with the
            # NCCL barrier (PyNccl tolerates empty fuses).
            self.nccl.group_start()
            self.nccl.group_end()
            return 0
        # Resolve tensors BEFORE opening the fuse so a stale-manifest miss
        # surfaces a clear error without touching NCCL.
        tensors_by_op: list[tuple[_TransferOp, torch.Tensor]] = []
        for op in my_ops:
            try:
                t = snapshot[op.sample_id]
            except KeyError as e:
                raise RuntimeError(
                    f"router-replay: cache.get({op.sample_id!r}) failed during send "
                    "despite manifest claim; snapshot is inconsistent"
                ) from e
            tensors_by_op.append((op, t.contiguous()))
        self.nccl.group_start()
        for op, t in tensors_by_op:
            self.nccl.send(t, dst=op.recv_rank)
        self.nccl.group_end()
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        return len(tensors_by_op)

    def _run_data_exchange_recv(
        self,
        plan: list[_TransferOp],
        cache_rx: RouterReplayCacheRX,
    ) -> int:
        my_ops = [op for op in plan if op.recv_rank == self.rank]
        if not my_ops:
            self.nccl.group_start()
            self.nccl.group_end()
            return 0
        dests: list[tuple[_TransferOp, torch.Tensor]] = []
        for op in my_ops:
            dest = torch.empty(
                op.shape,
                dtype=_ROUTED_EXPERTS_DTYPE,
                device=self.device,
            )
            dests.append((op, dest))
        self.nccl.group_start()
        for op, dest in dests:
            self.nccl.recv(dest, src=op.sender_rank)
        self.nccl.group_end()
        # Sync so subsequent cache.put copies observe fully-landed data.
        torch.cuda.current_stream(self.device).synchronize()
        for op, dest in dests:
            cache_rx.put(op.sample_id, dest)
        return len(dests)


def init_router_replay_group(
    *,
    role: Role,
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
    device: torch.device | int | str,
    is_server: bool | None = None,
) -> RouterReplayGroup:
    """Bootstrap a per-joint-job NCCL communicator + stateless PG.

    Mirrors weight_sync's stateless_init_nccl (same TCPStore + PG;
    identical failure modes). Returns once the NCCL handshake completes.

    Gateway-set rank layout: receivers occupy ``0..num_training_workers - 1``
    and senders the rest. ``role`` here is just this rank's local role.
    """
    from arctic_platform.inference.server.weight_sync.utils import stateless_init_nccl

    if isinstance(device, (int, str)):
        device_t = torch.device(device if isinstance(device, str) else f"cuda:{device}")
    else:
        device_t = device

    nccl = stateless_init_nccl(
        master_addr,
        master_port,
        rank,
        world_size,
        device_t,
        is_server=is_server,
    )
    pg = nccl.group  # StatelessProcessGroup stored on the comm by PyNccl init
    if pg is None:
        raise RuntimeError(
            "router-replay: PyNcclCommunicator.group is None — stateless bootstrap did not attach a process group"
        )
    return RouterReplayGroup(
        role=role,
        rank=rank,
        world_size=world_size,
        pg=pg,
        nccl=nccl,
        device=device_t,
    )
