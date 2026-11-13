from typing import Any


STRICT_REPLAY_KEY = "dss_router_replay_strict"
PREFIX_REPLICA_KEY = "dss_router_replay_prefix_replica"


class RouterReplayPrefixMissingError(RuntimeError):
    """A strict continuation cannot reuse its named prefix capture."""


class RouterReplayCacheFull(RuntimeError):
    """Adding a capture would exceed the cache's byte limit."""


def strict_replay_options(params: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    extra = params.get("extra_args") or {}
    strict = extra.get(STRICT_REPLAY_KEY, False)
    if not isinstance(strict, bool):
        raise ValueError(f"{STRICT_REPLAY_KEY} must be a boolean")
    prefix_id = extra.get("dss_router_replay_prefix_id")
    owner = extra.get(PREFIX_REPLICA_KEY)
    if owner is not None and not strict:
        raise ValueError(f"{PREFIX_REPLICA_KEY} requires {STRICT_REPLAY_KEY}")
    if strict:
        replay_id = params.get("dss_router_replay_id")
        if not isinstance(replay_id, str) or not replay_id.startswith("rr1:") or not params.get("dss_sample_id"):
            raise ValueError("strict router replay requires dss_sample_id and an exact rr1: replay id")
        if prefix_id is not None:
            if not isinstance(prefix_id, str) or not prefix_id.startswith("rr1:"):
                raise ValueError("strict router replay requires an exact rr1: prefix id")
            if prefix_id == replay_id:
                raise ValueError("strict router replay requires a new replay id for each attempt")
            if not isinstance(owner, str) or not owner:
                raise ValueError(f"strict router replay prefix requires {PREFIX_REPLICA_KEY}")
        elif owner is not None:
            raise ValueError(f"{PREFIX_REPLICA_KEY} requires dss_router_replay_prefix_id")
    return strict, prefix_id, owner
