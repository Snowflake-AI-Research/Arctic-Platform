"""The follower half of a semi-p engine that spans nodes.

A TP=16 engine is one vLLM engine whose ranks live on two pods. The leader
``InferenceWorker`` owns ranks 0-7 and serves; a ``SemipNodeAgent`` owns ranks
8-15 on the other pod and never serves. Every collective is issued by the
leader, whose ``MultiprocExecutor`` spans both halves, so the agent's job is
only the part the leader cannot reach: this pod's CRIU image, its CUDA
checkpoint and restore, and its end of the message-queue plane.

The agent is the Ray-actor form of the follower in the out-of-repo experiment
driver ``exp2_tp16.py``, which drove the same sequence over a TCP socket. The
ordering below is that protocol, and it is not arbitrary:

  Dump      both halves ``init`` together (they rendezvous over socket NCCL),
            the leader runs the generation and the staging steps, then the
            leader checkpoints CUDA and the follower does the same. The
            leader's ``criu_dump`` parks every rank's message queue as its
            last collective step, so the follower can only dump once its own
            ranks are parked -- hence ``wait_parked`` between the two.

  Restore   both ``criu_restore``; the leader binds a new broadcast writer and
            hands out a handle; the follower orders its ranks onto it and
            returns their response handles; the leader connects to those and
            swaps the plane in. Only then can either half ``cuda_restore``,
            and only then can the leader re-init NCCL -- which is the first
            time EFA comes up, since the cold start ran on sockets.

Joint or nothing: a half that cold-starts while the other restores would
rendezvous with a group that does not exist. ``probe`` reports what this pod
has so the leader can require both halves to carry the same ``dump_id``.
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
    """Owns the follower ``Instance`` for one node of a multi-node engine.

    Every method is a plain blocking call: the leader sequences the two halves
    and has to know each step finished before it starts the next, so there is
    nothing to gain from returning early. The Ray call is the await point.
    """

    def __init__(self) -> None:
        self._inst = None
        self._model_dir: str | None = None
        self._node_rank: int | None = None
        self._gpus: list[int] = []
        self._last_error: str | None = None

    # -- discovery ---------------------------------------------------------

    def probe(self, model_dir: str) -> dict[str, Any]:
        """What this pod holds for *model_dir*, for the joint hit decision.

        ``dump_id`` is the identity the leader compares. Two halves that each
        hold an image prove nothing on their own: they could be from different
        dumps, and restoring mismatched halves would deadlock in the first
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

        The two halves profile their shapes independently at cold start and
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
        """Copy this node's half of a published skeleton into place."""
        from arctic_platform.inference.server.semip_engine import (
            _materialize_from_source)
        return bool(_materialize_from_source(
            source_dir, model_dir, weight_root, weight_hash,
            verified_dir=verified_dir, strict=True))

    # -- cold start --------------------------------------------------------

    def init(self, vllm_config: dict[str, Any], model_dir: str,
             gpus: list[int], node_rank: int, nnodes: int,
             master_addr: str, master_port: int, ifname: str) -> dict[str, Any]:
        """Cold-start this half and block until its ranks have joined.

        The leader starts its own ``init`` at the same time; the two
        rendezvous inside vLLM. Neither returns until both have, so a slow
        half shows up as a long call here rather than as a deadlock.
        """
        from arctic_platform.inference.semi_persistence import (
            Instance, MultiNode)
        self._model_dir = model_dir
        self._node_rank = node_rank
        self._gpus = list(gpus)
        os.makedirs(model_dir, exist_ok=True)
        self._inst = Instance(
            vllm_config, model_dir,
            multinode=MultiNode(node_rank=node_rank, master_addr=master_addr,
                                master_port=int(master_port), ifname=ifname))
        self._inst.init(gpus=list(gpus)).wait()
        return {"ok": True, "node_rank": node_rank, "pid": self._inst.pid}

    # -- dump --------------------------------------------------------------

    def cuda_checkpoint(self) -> dict[str, Any]:
        """Release this half's GPU state.

        Runs after the leader's, because the leader's ``cuda_checkpoint``
        drops the graphs and tears NCCL down across every rank including
        these. Doing it first would checkpoint ranks that the leader then
        tries to reach.
        """
        self._require().cuda_checkpoint().wait()
        return {"ok": True}

    def wait_parked(self, timeout_s: float = 300.0) -> dict[str, Any]:
        """Block until this node's ranks have parked their message queues.

        The leader's ``criu_dump`` parks every rank as its last collective
        step. Dumping this half before that lands would capture a process
        still holding queue sockets, which is exactly what the image must not
        contain.
        """
        import time
        inst = self._require()
        unpark_dir = inst._unpark_dir()
        local = int(getattr(inst, "n_gpus", 0) or 0)
        ranks = range(self._node_rank * local, (self._node_rank + 1) * local)
        deadline = time.monotonic() + float(timeout_s)
        while True:
            missing = [r for r in ranks if not os.path.exists(
                os.path.join(unpark_dir, f"rank{r}.parked"))]
            if not missing:
                return {"ok": True, "ranks": list(ranks)}
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"semi_p agent node{self._node_rank}: ranks {missing} "
                    f"never parked within {timeout_s:.0f}s; the leader's "
                    "criu_dump parks them as its last collective step, so "
                    "this means the leader's dump did not get that far")
            time.sleep(0.2)

    def criu_dump(self, meta_extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Dump this half. Destructive: the child is gone afterwards."""
        inst = self._require()
        inst.criu_dump(meta_extra=dict(meta_extra or {})).wait()
        info = (inst.last_info.get("criu_dump") or {})
        census = info.get("inet_census")
        if census:
            raise RuntimeError(
                f"semi_p agent node{self._node_rank}: the image still holds "
                f"{len(census)} inet socket(s) {census}; a restore on another "
                "node would fail to rebind them")
        return {"ok": True, "inet_census": census}

    # -- restore -----------------------------------------------------------

    def criu_restore(self, vllm_config: dict[str, Any], model_dir: str,
                     gpus: list[int], node_rank: int) -> dict[str, Any]:
        """Rebuild this half's process from its image."""
        from arctic_platform.inference.semi_persistence import Instance
        self._model_dir = model_dir
        self._node_rank = node_rank
        self._gpus = list(gpus)
        # No MultiNode here: a restored child carries the dump's state, and
        # the rendezvous it will use arrives with the leader's reinit_nccl.
        self._inst = Instance(vllm_config, model_dir)
        self._inst.criu_restore().wait()
        return {"ok": True}

    def mq_follower_unpark(self, handle: Any, ranks: list[int],
                           connect_ip: str) -> dict[str, Any]:
        """Order this node's ranks onto the leader's new broadcast writer.

        Returns their response handles, which the leader connects to in
        ``mq_finish_unpark``. The plane is only whole once both directions
        exist, so the leader cannot proceed on its own.
        """
        inst = self._require()
        inst.mq_follower_unpark(handle, list(ranks), connect_ip).wait()
        return {"handles": inst.last_info["mq_follower_unpark"]["handles"]}

    def cuda_restore(self) -> dict[str, Any]:
        """Put this half's CUDA state back on its GPUs."""
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
                "semi_p agent: no Instance on this node yet; init() or "
                "criu_restore() has to run first")
        return self._inst
