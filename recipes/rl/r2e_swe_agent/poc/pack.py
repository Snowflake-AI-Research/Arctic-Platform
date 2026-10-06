"""Pack a multi-turn trajectory into one masked sequence.

Training each assistant turn as its own sequence is correct but wasteful: an
agent that takes twenty turns re-sends nineteen copies of a growing prefix, so
a 128-trajectory step becomes a couple of thousand sequences and the step time
is dominated by round trips rather than by arithmetic. The reference recipe
trains one sequence per trajectory instead, masked to the assistant tokens.

Doing that safely turns on a property worth stating explicitly: each turn's
prompt should be the previous turn's prompt plus the previous turn's sampled
completion, *in exactly the ids that were sampled*. It need not hold in
general -- a chat template is free to re-render earlier messages, and a
tokeniser is free to merge across a boundary -- so this module checks rather
than assumes, and reports precisely where a trajectory diverges. When the
property holds, the last turn's prompt already contains every earlier turn
verbatim, and packing is a matter of marking spans inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PackError:
    """Where a trajectory stopped being an append-only extension of itself."""

    turn: int
    kind: str  # "prefix" | "completion" | "shrink"
    detail: str

    def __str__(self) -> str:  # pragma: no cover - diagnostic sugar
        return f"turn {self.turn}: {self.kind}: {self.detail}"


@dataclass(frozen=True)
class Packed:
    input_ids: list[int]
    loss_mask: list[float]
    # Sampler log-probs aligned to ``input_ids``, or None when any turn was
    # missing them. All-or-nothing on purpose: a half-filled array reads as
    # log-prob 0, i.e. pi_old = 1, which silently inflates the importance
    # ratio on the missing spans instead of falling back to on-policy.
    logprobs: list[float] | None = None

    @property
    def num_trained_tokens(self) -> int:
        return int(sum(self.loss_mask))


def _first_divergence(a: list[int], b: list[int]) -> int:
    """Index of the first differing element, for a readable error."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def pack_trajectory(
    turns: list[tuple[list[int], list[int]]],
) -> tuple[Packed | None, PackError | None]:
    """Fold ``[(prompt, completion), ...]`` into one sequence plus a loss mask.

    Returns ``(packed, None)`` when every turn sits verbatim inside the final
    turn's context, and ``(None, error)`` at the first turn where it does not.
    The caller decides what to do with a divergent trajectory; silently packing
    one would train on token positions that were never sampled.
    """
    if not turns:
        return None, PackError(0, "shrink", "no turns")

    final = list(turns[-1][0]) + list(turns[-1][1])
    mask = [0.0] * len(final)

    for k, (prompt, completion) in enumerate(turns):
        start, end = len(prompt), len(prompt) + len(completion)

        if end > len(final):
            return None, PackError(
                k, "shrink",
                f"turn ends at {end} but the final sequence is {len(final)} long",
            )
        if final[:start] != list(prompt):
            i = _first_divergence(final[:start], list(prompt))
            return None, PackError(
                k, "prefix",
                f"prompt diverges from the final context at token {i}",
            )
        if final[start:end] != list(completion):
            i = _first_divergence(final[start:end], list(completion))
            return None, PackError(
                k, "completion",
                f"completion not preserved in the final context at token {i}",
            )

        for i in range(start, end):
            mask[i] = 1.0

    return Packed(final, mask), None


class _Tokenizer(Protocol):  # pragma: no cover - structural typing only
    def decode(self, ids: list[int], **kw) -> str: ...
    def encode(self, text: str, **kw) -> list[int]: ...


def pack_trajectory_exact(
    turns: list[tuple[list[int], list[int]]],
    tokenizer: _Tokenizer,
    turn_logprobs: list[list[float] | None] | None = None,
) -> tuple[Packed | None, PackError | None]:
    """Pack in text space, keeping every trained token exactly as sampled.

    ``pack_trajectory`` asks whether the final turn's ids contain the earlier
    turns' ids. In practice they usually do not, and for an uninteresting
    reason: tokenising a 60k-token context merges across boundaries that
    incremental sampling left split, so the same text comes back as different
    ids. Measured over a full step, every divergence was of that kind and none
    changed the text.

    Rather than re-tokenise and hope, this builds the sequence itself. Each
    turn contributes its sampled completion ids verbatim, and the tool output
    between turns is tokenised on its own. The asymmetry is the point: the
    spans we train on are exactly the ids the sampler produced, so the
    log-probs are attributable, while the spans that only provide context are
    masked and their segmentation cannot affect the loss.

    The append-only requirement does not go away, it just moves to where it
    actually holds -- text rather than ids.

    ``turn_logprobs`` carries the sampler's per-token log-probs through the
    same splice, so a packed trajectory can be replayed off-policy. Only the
    trained spans have log-probs to carry; the interleaved tool output is
    context the sampler never generated, and its entries are filled with zero
    and masked out. If any turn is missing them the whole trajectory comes back
    without them, since a partial array is worse than none.
    """
    if not turns:
        return None, PackError(0, "shrink", "no turns")

    def decode(ids: list[int]) -> str:
        return tokenizer.decode(list(ids), skip_special_tokens=False)

    have_lp = (
        turn_logprobs is not None
        and len(turn_logprobs) == len(turns)
        and all(
            lp is not None and len(lp) == len(completion)
            for lp, (_, completion) in zip(turn_logprobs, turns)
        )
    )

    ids: list[int] = []
    mask: list[float] = []
    lps: list[float] | None = [] if have_lp else None

    def extend(
        chunk: list[int], trained: bool, chunk_lp: list[float] | None = None
    ) -> None:
        ids.extend(chunk)
        mask.extend([1.0 if trained else 0.0] * len(chunk))
        if lps is not None:
            lps.extend(chunk_lp if chunk_lp is not None else [0.0] * len(chunk))

    seen = ""
    for k, (prompt, completion) in enumerate(turns):
        prompt_text = decode(prompt)

        if k == 0:
            extend(list(prompt), trained=False)
        else:
            if not prompt_text.startswith(seen):
                i = next(
                    (
                        j
                        for j in range(min(len(prompt_text), len(seen)))
                        if prompt_text[j] != seen[j]
                    ),
                    min(len(prompt_text), len(seen)),
                )
                return None, PackError(
                    k, "prefix", f"prompt text diverges from the trajectory at char {i}"
                )
            delta = prompt_text[len(seen) :]
            # add_special_tokens=False suppresses a second BOS; literal
            # <|im_start|> markers inside the delta still map to their own ids.
            extend(tokenizer.encode(delta, add_special_tokens=False), trained=False)

        extend(
            list(completion),
            trained=True,
            chunk_lp=list(turn_logprobs[k]) if have_lp else None,  # type: ignore[index,arg-type]
        )
        seen = prompt_text + decode(list(completion))

    return Packed(ids, mask, lps), None
