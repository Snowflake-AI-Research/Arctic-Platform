"""Ray placement-group helpers for colocated RL training/inference.

The colocated mode pins training, sampling, and log-prob actors to the same
physical GPU bundles via Ray placement groups (PGs).  Splitting the cluster
into **per-node STRICT_PACK** PGs (instead of a single PACK PG spanning all
nodes) lets us guarantee that any TP=tp group whose `tp` consecutive global
bundles fit inside `gpus_per_node` lives on a single physical node.  Without
this, vLLM's RayDistributedExecutor IP-uniqueness check fails when a TP
group accidentally straddles nodes.

This module centralizes:

* :func:`detect_gpus_per_node`     — query Ray for the (homogeneous) per-node
  GPU count.
* :func:`create_colocate_placement` — build the per-node STRICT_PACK PGs.
* :class:`ColocatePlacement`       — holds the PG list plus helpers to map a
  *global* bundle index to ``(pg, local_idx)`` and to lay out a TP group of
  replicas across the PGs.
* :func:`pg_scheduling_options`    — Ray ``options(...)`` kwargs for an actor
  pinned to a global bundle, with a fractional GPU claim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import ray
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

logger = logging.getLogger(__name__)

DEFAULT_BUNDLE_RESOURCES: dict = {"GPU": 1, "CPU": 4}


def detect_gpus_per_node() -> int:
    """Return the GPU count per Ray node, assuming homogeneous GPU nodes.

    Used by colocate placement to size per-node STRICT_PACK placement groups
    so that any TP group whose size divides ``gpus_per_node`` is guaranteed
    to fit on a single physical node.
    """
    counts: list[int] = []
    for n in ray.nodes():
        if not n.get("Alive", False):
            continue
        g = int(n.get("Resources", {}).get("GPU", 0))
        if g > 0:
            counts.append(g)
    if not counts:
        raise RuntimeError(
            "Could not detect any alive Ray nodes with GPUs while creating "
            "colocate placement groups"
        )
    if len(set(counts)) > 1:
        logger.warning(
            "Heterogeneous GPU counts per node detected: %s; using max=%d",
            counts, max(counts),
        )
    return max(counts)


@dataclass
class ColocatePlacement:
    """Per-node STRICT_PACK placement groups plus bundle resolution helpers.

    Bundles are addressed by a single *global* index in
    ``[0, n_bundles)``; the global index maps to
    ``(placement_groups[g // gpus_per_node],  g % gpus_per_node)``.
    """

    placement_groups: list[PlacementGroup] = field(default_factory=list)
    gpus_per_node: int = 0
    n_bundles: int = 0

    def __bool__(self) -> bool:
        return bool(self.placement_groups)

    def resolve(self, global_idx: int) -> tuple[PlacementGroup, int]:
        """Map a global bundle index to ``(placement_group, local_idx)``."""
        if self.gpus_per_node <= 0 or not self.placement_groups:
            raise RuntimeError("ColocatePlacement is not configured")
        pg_idx, local_idx = divmod(global_idx, self.gpus_per_node)
        if pg_idx >= len(self.placement_groups):
            raise IndexError(
                f"Global bundle {global_idx} maps to PG {pg_idx} but only "
                f"{len(self.placement_groups)} placement groups exist "
                f"(gpus_per_node={self.gpus_per_node})"
            )
        return self.placement_groups[pg_idx], local_idx

    def tp_layout(
        self,
        num_replicas: int,
        tp: int,
        bundle_offset: int = 0,
    ) -> tuple[list[PlacementGroup], list[int]]:
        """Lay out ``num_replicas`` TP=``tp`` groups across the per-node PGs.

        Replica ``r`` is assumed to span the ``tp`` consecutive global bundles
        ``[bundle_offset + r*tp .. bundle_offset + r*tp + tp - 1]``.  These
        all fall inside one per-node PG when ``gpus_per_node % tp == 0``.

        Returns:
            (per_replica_pgs, bundle_indices) suitable for
            ``ReplicaPool.initialize(placement_groups=..., bundle_indices=...)``.
            ``bundle_indices[r]`` is the TP-group index within that replica's
            PG, so the vLLM TP workers occupy local bundles
            ``[bundle_indices[r]*tp .. *tp + tp - 1]``.
        """
        if self.gpus_per_node % tp != 0:
            raise ValueError(
                f"TP={tp} must divide gpus_per_node={self.gpus_per_node} "
                f"so that each TP group fits on a single node"
            )
        per_replica_pgs: list[PlacementGroup] = []
        bundle_indices: list[int] = []
        for r in range(num_replicas):
            pg, local_start = self.resolve(bundle_offset + r * tp)
            per_replica_pgs.append(pg)
            bundle_indices.append(local_start // tp)
        return per_replica_pgs, bundle_indices


def create_colocate_placement(
    n_bundles: int,
    gpus_per_node: int | None = None,
    bundle_resources: dict | None = None,
) -> ColocatePlacement:
    """Build per-node STRICT_PACK placement groups for colocated RL.

    Creates ``ceil(n_bundles / gpus_per_node)`` STRICT_PACK groups (1 group if
    ``n_bundles <= gpus_per_node``).  Requires ``n_bundles`` to be a multiple
    of ``gpus_per_node`` when multiple groups are needed; otherwise the last
    group would be smaller and break the global indexing.

    Args:
        n_bundles: Total number of GPU bundles required.
        gpus_per_node: GPUs per physical node (autodetected via
            :func:`detect_gpus_per_node` if ``None``).
        bundle_resources: Per-bundle resource spec; defaults to one GPU and
            four CPUs.

    Returns:
        A :class:`ColocatePlacement` whose PGs are ready (``pg.ready()`` has
        been awaited).
    """
    if gpus_per_node is None:
        gpus_per_node = detect_gpus_per_node()
    if gpus_per_node <= 0:
        raise ValueError(f"gpus_per_node must be > 0, got {gpus_per_node}")

    resources = bundle_resources or DEFAULT_BUNDLE_RESOURCES

    if n_bundles <= gpus_per_node:
        pg_sizes: list[int] = [n_bundles]
    else:
        if n_bundles % gpus_per_node != 0:
            raise ValueError(
                f"colocate placement requires n_bundles ({n_bundles}) to be a "
                f"multiple of gpus_per_node ({gpus_per_node})"
            )
        pg_sizes = [gpus_per_node] * (n_bundles // gpus_per_node)

    pgs = [
        placement_group([dict(resources)] * sz, strategy="STRICT_PACK")
        for sz in pg_sizes
    ]
    ray.get([pg.ready() for pg in pgs])
    logger.info(
        "Created colocate placement: %d PG(s) STRICT_PACK, sizes=%s, "
        "gpus_per_node=%d, n_bundles=%d",
        len(pgs), pg_sizes, gpus_per_node, n_bundles,
    )
    return ColocatePlacement(
        placement_groups=pgs,
        gpus_per_node=gpus_per_node,
        n_bundles=n_bundles,
    )


def pg_scheduling_options(
    placement: ColocatePlacement,
    global_bundle_index: int,
    num_gpus: float | int,
) -> dict:
    """Return ``ray.remote().options(...)`` kwargs pinning an actor to a bundle.

    Pairs a fractional or whole GPU claim with a
    :class:`PlacementGroupSchedulingStrategy` aimed at the per-node PG that
    owns ``global_bundle_index``.
    """
    pg, local_idx = placement.resolve(global_bundle_index)
    return dict(
        num_gpus=num_gpus,
        scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=pg,
            placement_group_bundle_index=local_idx,
        ),
    )


__all__ = [
    "ColocatePlacement",
    "DEFAULT_BUNDLE_RESOURCES",
    "create_colocate_placement",
    "detect_gpus_per_node",
    "pg_scheduling_options",
]
