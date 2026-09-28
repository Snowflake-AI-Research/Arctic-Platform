"""Opt-in fixes for two sequence reductions that depend on layout: packed ``seq-mean-*`` aggregation and padded
sequence-level IS advantages. The invariant is layout independence — the same rows give the same loss and
gradient packed or padded — with the default path left exactly as it was."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deepspeed.utils import groups

from arctic_platform.rl.processors.grpo import grpo_loss

# Two rows of lengths 1 and 3 with different per-token advantages, so per-row and per-pack means differ.
LENGTHS = (1, 3)
OLD = torch.tensor([-1.0, -1.2, -0.8, -1.1])
LOGPROBS = torch.tensor([-0.9, -1.0, -0.9, -1.0])
ADVANTAGES = torch.tensor([1.0, -1.0, -2.0, -3.0])


def _run(layout, config, *, loss_mask=None, advantages=ADVANTAGES):
    """One grpo call on the rows above, packed ([T] + cu_seqlens) or padded ([B, S])."""
    mask = torch.ones(4, dtype=torch.bool) if loss_mask is None else loss_mask
    if layout == "packed":
        leaf = LOGPROBS.clone().requires_grad_(True)
        model = dict(logprobs=leaf)
        context = dict(old_log_probs_shifted=OLD, advantages=advantages, loss_mask=mask,
                       cu_seqlens=torch.tensor([0, 1, 4], dtype=torch.int32))
    else:
        index = torch.tensor([[0, -1, -1], [1, 2, 3]])
        valid = index >= 0
        pad = lambda t, fill: torch.where(valid, t[index.clamp(min=0)], torch.full_like(t[index.clamp(min=0)], fill))
        leaf = LOGPROBS.clone().requires_grad_(True)
        model = dict(logprobs=pad(leaf, 0.0))
        context = dict(old_log_probs_shifted=pad(OLD, 0.0), advantages=pad(advantages, 9.0),
                       loss_mask=pad(mask, False) & valid)
    loss, _ = grpo_loss(model, context, dict(use_cispo_loss=True, is_weight_clip_max=5.0, **config), "cpu")
    (grad,) = torch.autograd.grad(loss, leaf)
    return loss.detach(), grad


@pytest.mark.parametrize("mode", ["seq-mean-token-mean", "seq-mean-token-sum"])
def test_packed_seq_mean_matches_padded_only_with_the_fix(mode):
    padded = _run("padded", dict(loss_agg_mode=mode))
    packed_default = _run("packed", dict(loss_agg_mode=mode))
    packed_fixed = _run("packed", dict(loss_agg_mode=mode, seq_mean_per_packed_sequence=True))
    # The default treats the pack as one sequence; that is the behaviour this key opts out of.
    assert not torch.allclose(packed_default[0], padded[0])
    torch.testing.assert_close(packed_fixed[0], padded[0])
    torch.testing.assert_close(packed_fixed[1], padded[1])
    # The key does not touch the padded layout, whose reduction is already per row.
    padded_fixed = _run("padded", dict(loss_agg_mode=mode, seq_mean_per_packed_sequence=True))
    torch.testing.assert_close(padded_fixed[0], padded[0], rtol=0, atol=0)


def test_padded_sequence_is_advantages_match_packed_only_with_the_fix():
    # The last token is masked out; its advantage must not enter its sequence's mean.
    mask = torch.tensor([True, True, True, False])
    config = dict(importance_sampling_level="sequence", loss_agg_mode="token-mean")
    packed = _run("packed", config, loss_mask=mask)
    padded_default = _run("padded", config, loss_mask=mask)
    padded_fixed = _run("padded", dict(config, sequence_is_masked_advantages=True), loss_mask=mask)
    assert not torch.allclose(padded_default[1], packed[1])
    torch.testing.assert_close(padded_fixed[0], packed[0])
    torch.testing.assert_close(padded_fixed[1], packed[1])


def test_fix_keys_must_be_bools():
    with pytest.raises(ValueError, match="seq_mean_per_packed_sequence must be a bool"):
        _run("packed", dict(loss_agg_mode="seq-mean-token-mean", seq_mean_per_packed_sequence=1))


# --- sequence parallelism: the fixed packed seq-mean is a whole-sequence mean across windows ------------------

SP_CU_SEQLENS = torch.tensor([0, 5, 12, 16], dtype=torch.int32)
SP_TOKENS = 16


def _sp_frame():
    positions = torch.arange(SP_TOKENS, dtype=torch.float32)
    old = -1.0 - 0.01 * positions
    logprobs = old + 0.02 * torch.sin(positions)
    advantages = torch.where(positions.long() % 3 == 0, 1.0, -0.5)
    mask = torch.ones(SP_TOKENS, dtype=torch.bool)
    mask[[0, 5, 12]] = False
    return logprobs, old, advantages, mask


def _sp_loss(logprobs, old, advantages, mask, cu_seqlens):
    context = dict(old_log_probs_shifted=old, advantages=advantages, loss_mask=mask, cu_seqlens=cu_seqlens)
    config = dict(use_cispo_loss=True, is_weight_clip_max=5.0, loss_agg_mode="seq-mean-token-mean",
                  seq_mean_per_packed_sequence=True, global_batch_size=3, dp_size=1)
    return grpo_loss(dict(logprobs=logprobs), context, config, "cpu")[0]


def _sp_worker(rank, world_size, init_method):
    torch.set_num_threads(1)
    logprobs, old, advantages, mask = _sp_frame()
    whole = logprobs.clone().requires_grad_(True)
    expected = _sp_loss(whole, old, advantages, mask, SP_CU_SEQLENS)
    (expected_grad,) = torch.autograd.grad(expected, whole)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=60)
    )
    try:
        width = SP_TOKENS // world_size
        start, end = rank * width, (rank + 1) * width
        window = (SP_CU_SEQLENS.clamp(min=start, max=end) - start).to(torch.int32)
        window[-1] = end - start
        with (
            patch.object(groups, "_get_sequence_parallel_world_size", return_value=world_size),
            patch.object(groups, "_get_sequence_parallel_group", return_value=dist.group.WORLD),
        ):
            local = logprobs[start:end].clone().requires_grad_(True)
            loss = _sp_loss(local, old[start:end], advantages[start:end], mask[start:end], window)
            (grad,) = torch.autograd.grad(loss, local)
        total = loss.detach().clone()
        dist.all_reduce(total)
        torch.testing.assert_close(total, expected.detach())
        torch.testing.assert_close(grad, expected_grad[start:end])
    finally:
        dist.destroy_process_group()


def test_packed_seq_mean_fix_under_sequence_parallelism(tmp_path):
    mp.spawn(_sp_worker, args=(2, (tmp_path / "gloo").as_uri()), nprocs=2, join=True)
