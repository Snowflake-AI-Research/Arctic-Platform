"""Node identity for a tensor-parallel group that spans machines.

One ``Instance`` per node drives ``tensor_parallel_size / nnodes`` local GPUs,
and the node-partitions form a single TP group.  The split itself is topology --
``nnodes`` lives in ``vllm_config``, is hashed into the image cache key, and
changes the image.  *Which* node this is, and where the rendezvous lives, is
not: ``node_rank``, ``master_addr``, ``master_port`` and the interface change
on every restore and must never reach the key.

That separation is the whole point of this module.  The experiment driver put
all of it in ``vllm_config``, where it was recorded in ``meta.json``, compared
at ``criu_restore`` and hashed into ``cfg12`` -- so the node-partitions of one job
hashed differently and a restore onto a different set of pods could not match
its own image.  Keeping node identity in a separate parameter means every
node-partition derives the same key, and a restored engine is free to
rendezvous somewhere new.

A ``MultiNode`` of ``None`` is the single-node case, which is every TP <= 8
deployment today: no cross-node rendezvous, custom all-reduce stays on, and the
graphs are preserved through the dump and rebound rather than recaptured.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MultiNode:
    """Where this node sits in a multi-node TP group.

    ``nnodes`` is deliberately absent: it belongs to ``vllm_config`` because it
    is hashed, and duplicating it here would let the two disagree.  Use
    ``Instance.nnodes``, which reads the config.

    ``ifname`` is the interface the group rendezvouses and runs NCCL on.  It
    overrides whatever the job's ``extra_env`` set (dss pins
    ``NCCL_SOCKET_IFNAME=^lo`` for multi-node jobs), because a semi-p cold
    start has to bind sockets that the dump can account for and close.
    """

    node_rank: int
    master_addr: str
    master_port: int
    ifname: str = "eth0"

    def __post_init__(self):
        if self.node_rank < 0:
            raise ValueError(f"node_rank must be >= 0, got {self.node_rank}")
        if not 0 < int(self.master_port) < 65536:
            raise ValueError(
                f"master_port out of range: {self.master_port}")
        if not self.master_addr:
            raise ValueError("master_addr is required")
        if not self.ifname:
            raise ValueError("ifname is required")

    @property
    def is_leader(self) -> bool:
        """Rank 0 owns the engine and drives every collective."""
        return self.node_rank == 0

    def as_init_kwargs(self) -> dict:
        """What ``Instance.init`` sends the child.

        A plain dict rather than the dataclass, because it crosses a
        multiprocessing queue into a spawned child and travels as a command
        kwarg -- not into ``vllm_config``.
        """
        return {
            "node_rank": self.node_rank,
            "master_addr": self.master_addr,
            "master_port": int(self.master_port),
            "ifname": self.ifname,
        }


# The values aws-ofi-nccl 1.21.1 exports when it initializes on EFA.
#
# NCCL caches its parameters at the first NCCL init in a process.  A semi-p
# multi-node cold start runs on NCCL_NET=Socket, so without these it caches
# NCCL's socket-era defaults, and the plugin's EFA values arrive at the
# restore's re-init -- too late.  The restored engine then runs with
# P2P_NET_CHUNKSIZE 131072 instead of 524288 and a different ring/tree channel
# order, and computes a different (still deterministic) answer than a cold
# start would.  Pinning them at the cold start is what makes a restored engine
# bit-identical to an EFA cold start; that is measured, not assumed.
#
# NCCL_TOPO_FILE is deliberately not here: it is fd-backed, so the plugin
# supplies it at every init and a stale value would point at a closed fd.
PINNED_OFI_ENV = {
    "NCCL_BUFFSIZE": "8388608",
    "NCCL_P2P_NET_CHUNKSIZE": "524288",
    "NCCL_NVLS_CHUNKSIZE": "524288",
    "NCCL_NVLSTREE_MAX_CHUNKSIZE": "524288",
    "NCCL_NET_FORCE_FLUSH": "0",
    "NCCL_NETDEVS_POLICY": "max:1",
}

# Set on the child's own environment at a multi-node cold start.
#
# NCCL_NET=Socket because the cold start must not bring EFA up: EFA state does
# not survive CRIU, and the restore's re-init is the first EFA bring-up.
# GIN and RAS are background services that hold sockets the dump would have to
# account for.
MULTINODE_COLD_START_ENV = {
    "NCCL_NET": "Socket",
    "NCCL_GIN_ENABLE": "0",
    "NCCL_RAS_ENABLE": "0",
}
