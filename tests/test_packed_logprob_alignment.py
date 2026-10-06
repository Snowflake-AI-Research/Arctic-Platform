"""The sampler's log-probs must land on the tokens they actually scored.

This is the composition the two halves of the change meet at: the packer
splices per-turn log-probs into a whole-trajectory array, the driver hands the
backend the slice that starts at the prompt boundary, and the backend lays that
slice back down from the same boundary. Each half is individually plausible and
an off-by-one between them is invisible -- it shows up only as an importance
ratio that is quietly wrong, which is exactly the failure mode replay exists to
protect against. So the composition is tested, not just the pieces.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_POC = Path(__file__).resolve().parents[2] / "poc"
if str(_POC) not in sys.path:
    sys.path.insert(0, str(_POC))

from pack import pack_trajectory_exact  # noqa: E402

from arctic_platform.integrations.harbor.backend import (  # noqa: E402
    ArcticCortexBackend,
)
from arctic_platform.integrations.harbor.models import (  # noqa: E402
    Rollout,
    RolloutDataset,
)


class MergingTokenizer:
    """BPE-like: "a" and "b" have ids, and so does "ab"."""

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


class _Cfg:
    """Only the fields _build_grpo_batch reads."""

    max_seq_len = 4096
    std_normalization = False


def _dataset_from_trajectories(trajs, rewards):
    """Mimic exactly what r2e_driver does on the packed path."""
    tok = MergingTokenizer()
    rollouts, packs = [], []
    for i, (turns, lps) in enumerate(trajs):
        packed, err = pack_trajectory_exact(turns, tok, lps)
        assert err is None
        head = len(turns[0][0])
        packs.append(packed)
        rollouts.append(
            Rollout(
                metadata={"traj_id": f"t{i}", "traj_stats": {}},
                prompt_token_ids=packed.input_ids[:head],
                completion_token_ids=packed.input_ids[head:],
                loss_mask=packed.loss_mask,
                logprobs=(
                    packed.logprobs[head:] if packed.logprobs is not None else None
                ),
                reward=rewards[i],
                group_id="g0",
            )
        )
    return RolloutDataset(
        rollouts=rollouts, dataset_id="d", model_name="m", tokenizer_name="m"
    ), packs


def _backend():
    b = ArcticCortexBackend.__new__(ArcticCortexBackend)
    b.config = _Cfg()
    return b


# A two-turn trajectory whose second prompt re-encodes "a"+"b" as "ab", so the
# packed sequence genuinely differs from any turn's own ids.
TRAJ = ([([1, 2], [3, 4]), ([1, 2, 5, 6], [7])], [[-0.1, -0.2], [-0.3]])
TRAJ2 = ([([1, 2], [3, 4]), ([1, 2, 5, 6], [7])], [[-0.7, -0.8], [-0.9]])


def test_logprobs_land_exactly_on_the_trained_tokens():
    ds, packs = _dataset_from_trajectories([TRAJ, TRAJ2], [1.0, 0.0])
    batch = _backend()._build_grpo_batch(ds)

    for row, packed in enumerate(packs):
        n = len(packed.input_ids)
        got = batch["old_log_probs"][row, :n].tolist()
        assert got == pytest.approx(packed.logprobs)


def test_no_trained_token_is_left_without_a_logprob():
    # A zero here would read as pi_old = 1 and inflate that token's ratio.
    ds, _ = _dataset_from_trajectories([TRAJ, TRAJ2], [1.0, 0.0])
    batch = _backend()._build_grpo_batch(ds)

    trained = batch["loss_mask"] == 1
    assert torch.all(batch["old_log_probs"][trained] != 0.0)


def test_context_tokens_carry_no_logprob():
    ds, _ = _dataset_from_trajectories([TRAJ, TRAJ2], [1.0, 0.0])
    batch = _backend()._build_grpo_batch(ds)

    context = batch["loss_mask"] == 0
    assert torch.all(batch["old_log_probs"][context] == 0.0)


def test_the_key_is_sent_unshifted_for_the_shim_to_roll():
    # The dispatch shim owns the roll onto old_log_probs_shifted, including
    # zeroing the wrapped tail. Rolling here as well would double-shift.
    ds, _ = _dataset_from_trajectories([TRAJ, TRAJ2], [1.0, 0.0])
    batch = _backend()._build_grpo_batch(ds)
    assert "old_log_probs" in batch
    assert "old_log_probs_shifted" not in batch


def test_one_trajectory_without_logprobs_suppresses_the_key_entirely():
    bare = ([([1, 2], [3, 4]), ([1, 2, 5, 6], [7])], None)
    ds, _ = _dataset_from_trajectories([TRAJ, bare], [1.0, 0.0])
    batch = _backend()._build_grpo_batch(ds)
    assert "old_log_probs" not in batch
