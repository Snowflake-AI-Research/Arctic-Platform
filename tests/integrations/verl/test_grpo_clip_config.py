"""verl_grpo clip settings come from ``actor_config`` instead of fixed constants."""

from __future__ import annotations

import importlib

import pytest
import torch

from arctic_platform.rl.processors.verl_grpo import verl_grpo_loss

grpo_loss = importlib.import_module(verl_grpo_loss.__module__)
compute_policy_loss_vanilla = grpo_loss.compute_policy_loss_vanilla

GLOBAL_BATCH_INFO = {"dp_size": 1, "batch_num_tokens": 8, "global_batch_size": 2, "loss_scale_factor": None}


def _inputs():
    # ratio = exp(log_probs - old) spans [0.5, 1.5] so ε=0.2 and ε=0.4 clip different tokens.
    ratios = torch.tensor([[0.5, 0.7, 0.9, 1.1], [1.15, 1.3, 1.5, 1.0]])
    old = torch.full_like(ratios, -1.0)
    log_probs = old + ratios.log()
    advantages = torch.tensor([[1.0, -1.0, 1.0, -1.0], [1.0, 1.0, 1.0, -1.0]])
    mask = torch.ones_like(ratios, dtype=torch.bool)
    return old, log_probs, advantages, mask


def _run(**clip):
    old, log_probs, advantages, mask = _inputs()
    return compute_policy_loss_vanilla(
        old_log_probs=old,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=mask,
        global_batch_info=GLOBAL_BATCH_INFO,
        **clip,
    )


def test_defaults_match_the_former_constants_bitwise():
    loss_default, metrics_default = _run()
    loss_explicit, metrics_explicit = _run(clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0)
    assert torch.equal(loss_default, loss_explicit)
    assert metrics_default == metrics_explicit


def test_wider_clip_range_clips_fewer_tokens_and_changes_the_loss():
    loss_narrow, metrics_narrow = _run(clip_ratio_low=0.2, clip_ratio_high=0.2)
    loss_wide, metrics_wide = _run(clip_ratio_low=0.4, clip_ratio_high=0.4)
    assert metrics_wide["actor/pg_clipfrac.sum"] < metrics_narrow["actor/pg_clipfrac.sum"]
    assert not torch.equal(loss_narrow, loss_wide)


def test_low_and_high_fall_back_to_clip_ratio_when_unset():
    loss_fallback, metrics_fallback = _run(clip_ratio=0.4, clip_ratio_low=None, clip_ratio_high=None)
    loss_explicit, metrics_explicit = _run(clip_ratio=0.4, clip_ratio_low=0.4, clip_ratio_high=0.4)
    assert torch.equal(loss_fallback, loss_explicit)
    assert metrics_fallback == metrics_explicit


def test_clip_ratio_c_bounds_the_negative_advantage_loss():
    _, metrics_loose = _run(clip_ratio_c=3.0)
    _, metrics_tight = _run(clip_ratio_c=1.05)
    assert metrics_loose["actor/pg_clipfrac_lower.sum"] == 0.0
    assert metrics_tight["actor/pg_clipfrac_lower.sum"] > 0.0


def test_clip_ratio_c_must_exceed_one():
    with pytest.raises(AssertionError, match="clip_ratio_c"):
        _run(clip_ratio_c=1.0)


def test_verl_grpo_loss_forwards_actor_config_clip_settings(monkeypatch):
    seen = {}
    real = grpo_loss.compute_policy_loss_vanilla

    def spy(**kwargs):
        seen.update({k: kwargs[k] for k in ("clip_ratio", "clip_ratio_low", "clip_ratio_high", "clip_ratio_c")})
        return real(**kwargs)

    monkeypatch.setattr(grpo_loss, "compute_policy_loss_vanilla", spy)
    monkeypatch.setattr(grpo_loss, "shift_log_probs_left", lambda log_probs, batch: log_probs)

    old, log_probs, advantages, mask = _inputs()
    meta = {
        "actor_config": {"clip_ratio": 0.3, "clip_ratio_low": 0.25, "clip_ratio_high": 0.28, "clip_ratio_c": 2.5},
        "policy_loss_config": {},
        "zorro_train_enable": True,
        "dp_size": 1,
        "batch_num_tokens": 8,
        "global_batch_size": 2,
    }
    batch = {"response_mask": mask, "old_log_probs": old, "advantages": advantages}
    grpo_loss.verl_grpo_loss({"logprobs": log_probs.unsqueeze(0)}, batch, meta, {}, "cpu")
    assert seen == {"clip_ratio": 0.3, "clip_ratio_low": 0.25, "clip_ratio_high": 0.28, "clip_ratio_c": 2.5}
