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

    def pick(
        self,
        instances: list[dict],
        n: int,
        rng: random.Random,
        explore_frac: float = 0.25,
    ) -> list[dict]:
        """Draw a step's instances, preferring ones that will yield a gradient.

        Three tiers, in order of what a step gets back for its compute:

        ``spread``
            Already shown to be solved sometimes and failed sometimes. These
            are the only instances *guaranteed* to produce a nonzero advantage,
            so they get the bulk of the step.
        ``pending``
            Seen, but with fewer than ``min_attempts`` attempts, so no verdict
            yet. Re-sampling these is what ``min_attempts`` assumes: patience
            of three groups only means anything if the same instance is
            actually revisited. Draw once and move on and nothing is ever
            evicted, the pool fills with dead weight, and exploration
            degenerates into a uniform draw over the whole dataset.
        ``unseen``
            Never attempted. Genuinely new information, and the only way the
            pool grows, but on raw R2E most are uniformly unsolved for a small
            policy, so a step spent here usually buys a zero-gradient group.

        ``explore_frac`` bounds the share given to the last two tiers, which is
        the exploration/exploitation dial: too low and the policy grinds a
        handful of repos, too high and most groups are flat.
        """
        live = [r for r in instances if self.is_live(r["instance_id"])]
        spread: list[dict] = []
        pending: list[dict] = []
        unseen: list[dict] = []
        for r in live:
            rec = self.records.get(r["instance_id"])
            if rec is None or rec.attempts == 0:
                unseen.append(r)
            elif 0 < rec.solved < rec.attempts:
                spread.append(r)
            elif rec.attempts < self.min_attempts:
                pending.append(r)
            # Anything else is at quota without spread, which ``observe`` has
            # already evicted, so it is not live.
        for bucket in (spread, pending, unseen):
            rng.shuffle(bucket)

        n_exploit = n - min(n, round(n * explore_frac))
        chosen = spread[:n_exploit]
        # Finish verdicts before opening new ones, so the pool converges
        # instead of accumulating half-measured instances. Spread appears
        # again last, holding only what the exploit slice did not take, so a
        # short explore pool tops up from it without repeating an instance.
        for bucket in (pending, unseen, spread[n_exploit:]):
            if len(chosen) >= n:
                break
            chosen += bucket[: n - len(chosen)]
        return chosen[:n]

    def summary(self) -> str:
        n = len(self.records)
        hard = sum(1 for r in self.records.values() if r.evicted == "too_hard")
        easy = sum(1 for r in self.records.values() if r.evicted == "too_easy")
        spread = sum(1 for r in self.records.values() if 0 < r.solved < r.attempts)
        return (f"pool: {n} measured, {spread} with spread, "
                f"{hard} evicted too_hard, {easy} evicted too_easy")
