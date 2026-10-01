"""Router replay caches and NCCL collectives for DSS-internal router replay.

Public API
----------
- :class:`RouterReplayCacheTX`     — sampling-side, overwrite-on-put
- :class:`RouterReplayCacheRX`     — training-side, pop-on-read
- :class:`RouterReplayMissingError` — raised identically on every rank when
  the training side needs a sample_id no sampling rank holds

Layout mirrors :mod:`arctic_inference.server.weight_sync`. Tensors live on
GPU and never leave the worker over HTTP; the cross-zone transport is the
two-collective NCCL protocol in :mod:`.all2all` (added in a later commit).
"""

from arctic_inference.server.router_replay.all2all import (
    RouterReplayGroup,
    init_router_replay_group,
)
from arctic_inference.server.router_replay.cache import (
    RouterReplayCacheRX,
    RouterReplayCacheTX,
    RouterReplayDuplicateError,
    RouterReplayMissingError,
)

__all__ = [
    "RouterReplayCacheRX",
    "RouterReplayCacheTX",
    "RouterReplayDuplicateError",
    "RouterReplayGroup",
    "RouterReplayMissingError",
    "init_router_replay_group",
]
