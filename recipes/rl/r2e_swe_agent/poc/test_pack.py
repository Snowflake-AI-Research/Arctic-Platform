"""Packing a trajectory into one masked sequence, and refusing to when unsafe."""

import pytest

from pack import PackError, pack_trajectory, pack_trajectory_exact


def conversation(*turns):
    """Build an append-only trajectory from (completion, next_tool_output) pairs.

    Mirrors how a turn actually grows: the next prompt is everything so far
    plus the tool output the harness appended.
    """
    prompt = [1, 2, 3]  # system + task
    out = []
    for completion, tool_output in turns:
        out.append((list(prompt), list(completion)))
        prompt = prompt + list(completion) + list(tool_output)
    return out


def test_single_turn_is_prompt_plus_completion():
    packed, err = pack_trajectory([([1, 2, 3], [4, 5])])
    assert err is None
    assert packed.input_ids == [1, 2, 3, 4, 5]
    assert packed.loss_mask == [0, 0, 0, 1, 1]


def test_only_assistant_tokens_are_trained():
    turns = conversation(([10, 11], [90]), ([12], [91, 92]), ([13, 14], []))
    packed, err = pack_trajectory(turns)

    assert err is None
    # system+task, turn 1, tool, turn 2, tool, turn 3
    assert packed.input_ids == [1, 2, 3, 10, 11, 90, 12, 91, 92, 13, 14]
    assert packed.loss_mask == [0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    assert packed.num_trained_tokens == 5


def test_trained_tokens_equal_the_sum_of_completions():
    turns = conversation(([10, 11], [90]), ([12], [91, 92]), ([13, 14, 15], [93]))
    packed, err = pack_trajectory(turns)

    assert err is None
    assert packed.num_trained_tokens == sum(len(c) for _, c in turns)


def test_packed_length_matches_the_final_turn():
    turns = conversation(([10], [90, 91]), ([11, 12], [92]))
    packed, _ = pack_trajectory(turns)
    last_prompt, last_completion = turns[-1]
    assert len(packed.input_ids) == len(last_prompt) + len(last_completion)


def test_masked_spans_recover_each_completion():
    """The mask must select exactly the sampled tokens, in order."""
    turns = conversation(([10, 11], [90]), ([12], [91]), ([13], []))
    packed, _ = pack_trajectory(turns)

    trained = [t for t, m in zip(packed.input_ids, packed.loss_mask) if m]
    assert trained == [10, 11, 12, 13]


def test_rewritten_history_is_refused():
    """A template that re-renders an earlier turn must not be packed silently."""
    turns = conversation(([10, 11], [90]), ([12], [91]))
    # The second turn's prompt no longer contains what was actually sampled.
    prompt, completion = turns[1]
    turns[1] = (prompt[:3] + [99, 11, 90], completion)

    packed, err = pack_trajectory(turns)
    assert packed is None
    assert isinstance(err, PackError)
    assert err.kind == "completion"


def test_dropped_prefix_is_refused():
    """Context trimming breaks the invariant and is reported, not packed."""
    turns = conversation(([10, 11], [90]), ([12], [91]), ([13], []))
    prompt, completion = turns[2]
    turns[2] = (prompt[2:], completion)  # simulate a trimmed window

    packed, err = pack_trajectory(turns)
    assert packed is None
    assert err.kind in {"prefix", "completion"}


def test_shorter_final_context_is_refused():
    turns = [([1, 2, 3], [4, 5, 6]), ([1], [7])]
    packed, err = pack_trajectory(turns)
    assert packed is None
    assert err.kind == "shrink"


def test_empty_trajectory_is_refused():
    packed, err = pack_trajectory([])
    assert packed is None
    assert err.turn == 0


def test_error_points_at_the_offending_turn():
    """Corrupt a middle turn: the last turn defines the reference sequence, so
    only an earlier one can be caught disagreeing with it."""
    turns = conversation(([10], [90]), ([11], [91]), ([12], [92]))
    prompt, completion = turns[1]
    turns[1] = (prompt[:-1] + [77], completion)

    _, err = pack_trajectory(turns)
    assert err is not None and err.turn == 1


@pytest.mark.parametrize("n", [1, 2, 8, 40])
def test_holds_at_realistic_turn_counts(n):
    turns = conversation(*[([100 + i, 200 + i], [300 + i]) for i in range(n)])
    packed, err = pack_trajectory(turns)

    assert err is None
    assert packed.num_trained_tokens == 2 * n


class MergingTokenizer:
    """A tokenizer that merges greedily, the way a real BPE one does.

    ``"a"`` and ``"b"`` have ids of their own, but so does ``"ab"``, so text
    sampled as two tokens comes back as one when it is encoded again. That is
    the entire reason the id-space packer rejects healthy trajectories, so the
    text-space packer has to be tested against a tokenizer that does it.
    """

    VOCAB = {1: "<sys>", 2: "task ", 3: "a", 4: "b", 5: "ab", 6: "<out>", 7: "c"}

    def __init__(self):
        self._by_text = sorted(self.VOCAB.items(), key=lambda kv: -len(kv[1]))

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.VOCAB[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        out, i = [], 0
        while i < len(text):
            for tid, s in self._by_text:
                if text.startswith(s, i):
                    out.append(tid)
                    i += len(s)
                    break
            else:
                raise AssertionError(f"cannot encode at {i}: {text[i:]!r}")
        return out


@pytest.fixture
def tok():
    return MergingTokenizer()


def test_exact_packs_what_id_space_packing_rejects(tok):
    """Sampling "a" then "b" separately, but re-encoding them as "ab"."""
    turns = [
        ([1, 2], [3, 4]),          # sampled as two tokens
        ([1, 2, 5, 6], [7]),       # the same text, re-encoded as one
    ]
    assert pack_trajectory(turns)[0] is None

    packed, err = pack_trajectory_exact(turns, tok)
    assert err is None
    assert packed is not None


def test_exact_keeps_trained_tokens_bit_identical(tok):
    """The trained spans must be the sampled ids, not a re-encoding of them."""
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok)

    trained = [t for t, m in zip(packed.input_ids, packed.loss_mask) if m]
    assert trained == [3, 4, 7]  # never [5, ...]


def test_exact_preserves_the_full_text(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok)

    last = turns[-1]
    assert tok.decode(packed.input_ids) == tok.decode(last[0] + last[1])


def test_exact_masks_the_tool_output_between_turns(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok)

    untrained = [t for t, m in zip(packed.input_ids, packed.loss_mask) if not m]
    assert 6 in untrained  # the <out> delta
    assert len(packed.loss_mask) == len(packed.input_ids)


def test_exact_agrees_with_id_packing_when_ids_do_line_up(tok):
    turns = conversation(([3], [6]), ([7], [6]), ([4], []))
    by_ids, err = pack_trajectory(turns)
    assert err is None

    by_text, err = pack_trajectory_exact(turns, tok)
    assert err is None
    assert by_text.input_ids == by_ids.input_ids
    assert by_text.loss_mask == by_ids.loss_mask


def test_exact_still_refuses_genuinely_rewritten_history(tok):
    """Text-space packing is more permissive, not unconditional."""
    turns = [([1, 2], [3, 4]), ([1, 2, 7, 6], [7])]  # "ab" became "c"

    packed, err = pack_trajectory_exact(turns, tok)
    assert packed is None
    assert err.kind == "prefix"


def test_exact_refuses_an_empty_trajectory(tok):
    packed, err = pack_trajectory_exact([], tok)
    assert packed is None
    assert err.turn == 0


def test_exact_trains_every_completion_token_once(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7]), ([1, 2, 5, 6, 7, 6], [3])]
    packed, err = pack_trajectory_exact(turns, tok)

    assert err is None
    assert packed.num_trained_tokens == sum(len(c) for _, c in turns)


# ── sampler log-probs carried through the splice ──────────────────────────
# Off-policy replay needs pi_old for every trained token. The packer is the
# only place that knows which positions in the packed sequence came from the
# sampler, so it has to carry the log-probs across the same splice it applies
# to the ids.


def test_logprobs_align_with_the_trained_positions(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    lps = [[-0.1, -0.2], [-0.3]]
    packed, err = pack_trajectory_exact(turns, tok, lps)

    assert err is None
    trained = [lp for lp, m in zip(packed.logprobs, packed.loss_mask) if m]
    assert trained == [-0.1, -0.2, -0.3]


def test_logprobs_are_the_same_length_as_the_ids(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok, [[-0.1, -0.2], [-0.3]])
    assert len(packed.logprobs) == len(packed.input_ids)


def test_context_positions_carry_zero(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok, [[-0.1, -0.2], [-0.3]])
    untrained = [lp for lp, m in zip(packed.logprobs, packed.loss_mask) if not m]
    assert set(untrained) == {0.0}


def test_absent_logprobs_leave_the_field_unset(tok):
    packed, _ = pack_trajectory_exact([([1, 2], [3, 4])], tok)
    assert packed.logprobs is None


def test_one_missing_turn_discards_all_of_them(tok):
    # A partial array reads as pi_old = 1 on the gaps, which inflates the
    # importance ratio instead of degrading to on-policy.
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok, [[-0.1, -0.2], None])
    assert packed.logprobs is None


def test_mismatched_logprob_length_is_rejected(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    packed, _ = pack_trajectory_exact(turns, tok, [[-0.1], [-0.3]])
    assert packed.logprobs is None


def test_logprobs_do_not_disturb_the_ids_or_mask(tok):
    turns = [([1, 2], [3, 4]), ([1, 2, 5, 6], [7])]
    plain, _ = pack_trajectory_exact(turns, tok)
    withlp, _ = pack_trajectory_exact(turns, tok, [[-0.1, -0.2], [-0.3]])
    assert withlp.input_ids == plain.input_ids
    assert withlp.loss_mask == plain.loss_mask
