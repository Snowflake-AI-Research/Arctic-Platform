"""A follower node-partition of a semi-p engine that spans pods.

A pod-spanning engine is one vLLM engine whose ranks are split into one
node-partition per pod: two at TP=16, four at TP=32. The leader
``InferenceWorker`` owns node-partition 0 (ranks 0-7) and serves; one
``SemipNodeAgent`` per further pod owns the next block of ranks and never
serves. Every collective is issued by the leader, whose
``MultiprocExecutor`` spans every node-partition, so the agent's job is
only the part the leader cannot reach: this pod's CRIU image, its CUDA
checkpoint and restore, and its end of the message-queue plane.

The agent is the Ray-actor form of the follower in the out-of-repo experiment
driver ``exp2_tp16.py``, which drove the same sequence over a TCP socket. The
ordering below is that protocol, and it is not arbitrary:

  Dump      all node-partitions ``init`` together (they rendezvous over socket NCCL),
            the leader runs the generation and the staging steps, then the
            leader checkpoints CUDA and each follower does the same. The
            leader's ``criu_dump`` parks every rank's message queue as its
            last collective step, so a follower can only dump once its own
            ranks are parked -- hence ``criu_dump`` starts in
            ``wait_parked``.

  Restore   all ``criu_restore``; the leader binds a new broadcast writer and
            hands out a handle; each follower orders its ranks onto it and
            returns their response handles; the leader connects to those and
            swaps the plane in. Only then can any node-partition ``cuda_restore``,
            and only then can the leader re-init NCCL -- which is the first
            time EFA comes up, since the cold start ran on sockets.

Joint or nothing: a node-partition that cold-starts while another restores would
rendezvous with a group that does not exist. ``probe`` reports what this pod
has so the leader can require every node-partition to carry the same ``dump_id``.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import ray

logger = logging.getLogger(__name__)


@ray.remote
class SemipNodeAgent:
    """Owns the follower ``Instance`` for one pod of a pod-spanning engine.

    Every method is a plain blocking call: the leader sequences the node-partitions
    and has to know each step finished before it starts the next, so there is
    nothing to gain from returning early. The Ray call is the await point.
    """

    def __init__(self) -> None:
        from arctic_platform.inference.server.semip_engine import (
            _UNPRIVILEGED_ENV)
        # The leader sets this default in its own process only, and the mode
        # has to match across node-partitions: it decides the capability level
        # each node-partition's child records at init and the flags its CRIU runs with. As
        # there, a job's extra_env (this actor's runtime_env) overrides it.
        os.environ.setdefault(_UNPRIVILEGED_ENV, "1")
        self._inst = None
        self._model_dir: str | None = None
        self._node_rank: int | None = None
        self._gpus: list[int] = []
        self._last_error: str | None = None

    # -- discovery ---------------------------------------------------------

    def probe(self, model_dir: str) -> dict[str, Any]:
        """What this pod holds for *model_dir*, for the joint hit decision.

        ``dump_id`` is the identity the leader compares. Node-partitions that each
        hold an image prove nothing on their own: they could be from different
        dumps, and restoring mismatched node-partitions would deadlock in the first
        collective rather than fail cleanly.
        """
        meta_path = os.path.join(model_dir, "image", "meta.json")
        out: dict[str, Any] = {"model_dir": model_dir, "hit": False,
                               "dump_id": None, "node_rank": None}
        try:
            with open(meta_path) as handle:
                meta = json.load(handle)
        except (OSError, ValueError):
            return out
        out["hit"] = True
        out["dump_id"] = meta.get("dump_id")
        out["node_rank"] = meta.get("node_rank")
        out["nnodes"] = meta.get("nnodes")
        return out

    def config_digest(self, vllm_config: dict[str, Any]) -> str:
        """This pod's hash of the engine config it is about to build.

        The node-partitions profile their shapes independently at cold start and
        then meet in a collective. If they disagree -- a different
        ``max_num_batched_tokens``, a different KV dtype -- they deadlock
        there, silently, for the full gloo timeout. Comparing a digest before
        ``init`` turns that into an error with both values in it.
        """
        from arctic_platform.inference.server.semip_engine import _config_hash
        return _config_hash(vllm_config)

    def materialize(self, source_dir: str | None, model_dir: str,
                    weight_root: str | None, weight_hash: str | None,
                    verified_dir: str | None = None) -> bool:
        """Copy this pod's node-partition of a published skeleton into place."""
        from arctic_platform.inference.server.semip_engine import (
            _materialize_from_source)
        return bool(_materialize_from_source(
            source_dir, model_dir, weight_root, weight_hash,
            verified_dir=verified_dir, strict=True))

    # -- cold start --------------------------------------------------------

    def init(self, vllm_config: dict[str, Any], model_dir: str,
             gpus: list[int], node_rank: int, nnodes: int,
             master_addr: str, master_port: int, ifname: str) -> dict[str, Any]:
        """Cold-start this node-partition and block until its ranks have joined.

        The leader starts its own ``init`` at the same time; they all
        rendezvous inside vLLM. None returns until all have, so a slow
        node-partition shows up as a long call here rather than as a deadlock.
        """
        from arctic_platform.inference.semi_persistence import (
            Instance, MultiNode)
        from arctic_platform.inference.server.semip_engine import (
            _raise_pid_floor, _unprivileged_mode)
        # Per pod: the counter is this PID namespace's, and the leader's floor
        # does nothing for the ids this node-partition's image records.
        pid_floor = _raise_pid_floor()
        self._model_dir = model_dir
        self._node_rank = node_rank
        self._gpus = list(gpus)
        os.makedirs(model_dir, exist_ok=True)
        self._inst = Instance(
            vllm_config, model_dir,
            multinode=MultiNode(node_rank=node_rank, master_addr=master_addr,
                                master_port=int(master_port), ifname=ifname))
        # The leader's park clears the directory on its own pod only, and a
        # marker left here by an earlier dump of this key would satisfy
        # wait_parked before this dump's ranks have parked.
        for path in self._parked_markers().values():
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        self._inst.init(gpus=list(gpus)).wait()
        return {"ok": True, "node_rank": node_rank, "pid": self._inst.pid,
                "pid_floor": pid_floor, "unprivileged": _unprivileged_mode()}

    # -- dump --------------------------------------------------------------

    def cuda_checkpoint(self) -> dict[str, Any]:
        """Release this node-partition's GPU state.

        Runs after the leader's, because the leader's ``cuda_checkpoint``
        drops the graphs and tears NCCL down across every rank including
        these. Doing it first would checkpoint ranks that the leader then
        tries to reach.
        """
        self._require().cuda_checkpoint().wait()
        return {"ok": True}

    def wait_parked(self, timeout_s: float = 300.0) -> dict[str, Any]:
        """Block until this pod's ranks have parked their message queues.

        The leader's ``criu_dump`` parks every rank as its last collective
        step. Dumping this node-partition before that lands would capture a process
        still holding queue sockets, which is exactly what the image must not
        contain.
        """
        import time
        markers = self._parked_markers()
        deadline = time.monotonic() + float(timeout_s)
        while True:
            missing = [r for r, path in markers.items()
                       if not os.path.exists(path)]
            if not missing:
                return {"ok": True, "ranks": list(markers)}
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"semi_p agent node{self._node_rank}: ranks {missing} "
                    f"never parked within {timeout_s:.0f}s; the leader's "
                    "criu_dump parks them as its last collective step, so "
                    "this means the leader's dump did not get that far")
            time.sleep(0.2)

    def criu_dump(self, meta_extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dump this node-partition once its ranks are parked. Destructive: the child is
        gone afterwards.

        The wait is in here rather than a separate call because the actor runs
        calls concurrently, so nothing would order a separate call before this
        one.
        """
        from arctic_platform.inference.server.semip_engine import (
            _record_env_files)
        self.wait_parked()
        inst = self._require()
        inst.criu_dump(meta_extra=dict(meta_extra or {})).wait()
        info = (inst.last_info.get("criu_dump") or {})
        census = info.get("inet_census")
        if census:
            raise RuntimeError(
                f"semi_p agent node{self._node_rank}: the image still holds "
                f"{len(census)} inet socket(s) {census}; a restore in another "
                "pod would fail to rebind them")
        # The leader records only its own node-partition. Without this one, a
        # copy of this node-partition from the mirror comes back with every file at the sync's
        # mode, and CRIU refuses the first executable mapping.
        _record_env_files(self._model_dir)
        return {"ok": True, "inet_census": census}

    # -- restore -----------------------------------------------------------

    def criu_restore(self, vllm_config: dict[str, Any], model_dir: str,
                     gpus: list[int], node_rank: int) -> dict[str, Any]:
        """Rebuild this node-partition's process from its image."""
        from arctic_platform.inference.semi_persistence import Instance
        self._model_dir = model_dir
        self._node_rank = node_rank
        self._gpus = list(gpus)
        # No MultiNode here: a restored child carries the dump's state, and
        # the rendezvous it will use arrives with the leader's reinit_nccl.
        self._inst = Instance(vllm_config, model_dir)
        self._inst.criu_restore().wait()
        return {"ok": True}

    def mq_follower_unpark(self, handle: Any,
                           ranks: list[int]) -> dict[str, Any]:
        """Order this pod's ranks onto the leader's new broadcast writer.

        Returns their response handles, which the leader connects to in
        ``mq_finish_unpark``. The plane is only whole once both directions
        exist, so the leader cannot proceed on its own.

        The ranks bind their response writers to this pod's address, which
        only this pod can name.
        """
        from arctic_platform.inference.server.semip_engine import _leader_ip
        inst = self._require()
        inst.mq_follower_unpark(handle, list(ranks), _leader_ip()).wait()
        return {"handles": inst.last_info["mq_follower_unpark"]["handles"]}

    def cuda_restore(self) -> dict[str, Any]:
        """Put this node-partition's CUDA state back on its GPUs."""
        self._require().cuda_restore(gpus=list(self._gpus)).wait()
        return {"ok": True}

    # -- lifecycle ---------------------------------------------------------

    def teardown(self) -> dict[str, Any]:
        if self._inst is None:
            return {"ok": True, "note": "no instance"}
        try:
            self._inst.teardown().wait()
        except Exception as exc:  # noqa: BLE001
            # Teardown runs on the failure path too, where the instance may
            # already be half gone. Report rather than mask the original
            # failure the caller is unwinding from.
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("semi_p agent teardown failed: %s", self._last_error)
            return {"ok": False, "error": self._last_error}
        finally:
            self._inst = None
        return {"ok": True}

    def status(self) -> dict[str, Any]:
        return {
            "node_rank": self._node_rank,
            "model_dir": self._model_dir,
            "gpus": list(self._gpus),
            "alive": self._inst is not None,
            "last_error": self._last_error,
        }

    def _require(self):
        if self._inst is None:
            raise RuntimeError(
                "semi_p agent: no Instance in this pod yet; init() or "
                "criu_restore() has to run first")
        return self._inst

    def _parked_markers(self) -> dict[int, str]:
        """``rank -> path`` of the marker each of this pod's ranks writes
        once its message queues are parked."""
        inst = self._require()
        unpark_dir = inst._unpark_dir()
        local = int(inst.n_gpus)
        first = self._node_rank * local
        return {r: os.path.join(unpark_dir, f"rank{r}.parked")
                for r in range(first, first + local)}
