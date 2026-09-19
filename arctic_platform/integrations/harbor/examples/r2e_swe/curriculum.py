"""Difficulty-filtered instance pool with easy/hard eviction.

GRPO's gradient comes from spread *within* a group: if all G rollouts of an
instance score the same, its advantages are identically zero and the step is
wasted no matter how much compute it cost. On raw R2E almost every instance is
uniformly unsolved for a 4B policy, which is why an unfiltered run sits at zero
and looks like a broken loop rather than a badly chosen task mix.

So instances earn their place. Each one carries a running solve tally; an
instance that is never solved is evicted as too hard, one that is always solved
as too easy, and training draws from what is left. This is the same shape as
the curriculum behind the run we are matching, whose headline metric is
eviction-adjusted precisely because the pool is not static.

The tally is persisted so the filtering survives the ~35-minute idle timeout on
a Cortex job: restarting the driver must not throw away the pool it paid for.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass
class Record:
    attempts: int = 0
    solved: int = 0
    # Kept per instance rather than derived, so a decision made under an
    # earlier policy can be re-examined instead of being permanent.
    evicted: str | None = None

    @property
    def rate(self) -> float:
        return self.solved / self.attempts if self.attempts else 0.0


@dataclass
class Curriculum:
    path: Path
    # How many attempts before a verdict. One group of 4 all failing is weak
    # evidence; the reference setup re-samples before giving up on an instance.
    min_attempts: int = 8
    records: dict[str, Record] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, min_attempts: int = 8) -> Curriculum:
        p = Path(path)
        c = cls(path=p, min_attempts=min_attempts)
        if p.exists():
            raw = json.loads(p.read_text())
            c.records = {k: Record(**v) for k, v in raw.items()}
        return c

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {k: asdict(v) for k, v in self.records.items()}, indent=2
        ))
        tmp.replace(self.path)

    def observe(self, instance_id: str, rewards: list[float]) -> None:
        """Fold one group's outcomes into the tally and re-check eviction."""
        rec = self.records.setdefault(instance_id, Record())
        rec.attempts += len(rewards)
        rec.solved += sum(1 for r in rewards if r > 0)
        if rec.attempts >= self.min_attempts:
            if rec.solved == 0:
                rec.evicted = "too_hard"
            elif rec.solved == rec.attempts:
                rec.evicted = "too_easy"
            else:
                # Spread appeared after an earlier verdict: the policy moved,
                # so the instance is trainable again.
                rec.evicted = None

    def is_live(self, instance_id: str) -> bool:
        rec = self.records.get(instance_id)
        return rec is None or rec.evicted is None

    def pick(self, instances: list[dict], n: int, rng: random.Random) -> list[dict]:
        """Prefer instances already known to have spread, then unmeasured ones.

        Known-spread instances are the only ones guaranteed to produce a
        gradient, but drawing exclusively from them would freeze the pool and
        let the policy overfit a handful of repos, so unmeasured instances are
        mixed in to keep replenishing it.
        """
        live = [r for r in instances if self.is_live(r["instance_id"])]
        spread, fresh = [], []
        for r in live:
            rec = self.records.get(r["instance_id"])
            (spread if rec and 0 < rec.solved < rec.attempts else fresh).append(r)
        rng.shuffle(spread)
        rng.shuffle(fresh)
        want_spread = min(len(spread), max(1, n // 2)) if spread else 0
        chosen = spread[:want_spread] + fresh[: n - want_spread]
        # Pool exhausted of fresh instances: top up from spread rather than
        # returning a short step.
        if len(chosen) < n:
            chosen += spread[want_spread : want_spread + (n - len(chosen))]
        return chosen[:n]

    def summary(self) -> str:
        n = len(self.records)
        hard = sum(1 for r in self.records.values() if r.evicted == "too_hard")
        easy = sum(1 for r in self.records.values() if r.evicted == "too_easy")
        spread = sum(1 for r in self.records.values() if 0 < r.solved < r.attempts)
        return (f"pool: {n} measured, {spread} with spread, "
                f"{hard} evicted too_hard, {easy} evicted too_easy")
