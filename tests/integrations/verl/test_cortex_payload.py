# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Log-prob alignment on the verl -> Cortex ``fwd_bwd`` wire.

The zone's ``compute_logprobs`` post-processor scores
``labels = roll(input_ids, -1)``, so slot ``i`` of its ``logprobs`` holds
``log P(input_ids[i + 1])``. verl keeps ``old_log_probs`` / ``advantages`` /
``response_mask`` aligned to the token they belong to. Shipping verl's layout
into ``old_log_probs_shifted`` unconverted makes the importance ratio
``exp(log P(tok i+1) - log P(tok i))``; a 250-step GSM8K run logged
``actor/importance_weight`` between 1.86 and 496 where a single-epoch
on-policy update must produce exactly 1.0.

These tests pin the index math, so a regression shows up here rather than as
a silently flat reward curve.
"""

from __future__ import annotations

import torch

from arctic_platform.integrations.verl.cortex_payload import to_cortex_fwd_bwd_payload

PAD = 0


def _zone_logprobs(input_ids: torch.Tensor, true_logprobs: torch.Tensor) -> torch.Tensor:
    """What the zone computes: slot ``i`` holds ``log P(input_ids[i + 1])``.

    ``true_logprobs[i]`` is the log-prob of the token sitting at slot ``i``,
    which is the convention verl uses. Reimplemented here (one line) rather
    than importing the real pipeline so the test stays CPU- and torch-only.
    """
    del input_ids
    shifted = torch.zeros_like(true_logprobs)
    shifted[..., :-1] = true_logprobs[..., 1:]
    return shifted


def test_response_aligned_tensors_move_one_slot_left():
    """prompt_len=3, resp_len=4, S=10: response at 3..6 -> shifted to 2..5."""
    input_ids = torch.tensor([[10, 11, 12, 30, 31, 32, 33, PAD, PAD, PAD]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    # verl layout: value for the response token at slot p lives at slot p.
    old_log_probs = torch.tensor([[0.0, 0.0, 0.0, -1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0]])
    advantages = torch.tensor([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0]])
    response_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]])

    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_log_probs": old_log_probs,
                "advantages": advantages,
                "response_mask": response_mask,
            }
        }
    )
    ctx = out["context"]

    assert torch.equal(
        ctx["old_log_probs_shifted"],
        torch.tensor([[0.0, 0.0, -1.0, -2.0, -3.0, -4.0, 0.0, 0.0, 0.0, 0.0]]),
    )
    assert torch.equal(
        ctx["advantages"],
        torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]),
    )
    # loss_mask must follow: unshifted it scores a pad target and drops the
    # final response token.
    assert ctx["loss_mask"].nonzero()[:, 1].tolist() == [2, 3, 4, 5]
    # input_ids are NOT shifted -- the zone rolls them itself.
    assert torch.equal(out["kwargs"]["input_ids"], input_ids)


def test_importance_ratio_is_exactly_one_on_policy():
    """The property that matters: ratio == 1 when nothing has changed yet.

    With ``ppo_epochs=1`` and ``ppo_mini_batch_size == train_batch_size`` there
    is one ``fwd_bwd`` per step against the very weights that produced
    ``old_log_probs``, so ``exp(new - old)`` must be exactly 1.0.
    """
    torch.manual_seed(0)
    input_ids = torch.tensor([[10, 11, 12, 30, 31, 32, 33, PAD, PAD, PAD]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0]])
    response_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0, 0]])

    # Ground truth: log-prob of the token at each slot.
    true_logprobs = -torch.rand(1, 10)
    old_log_probs = true_logprobs * response_mask  # verl only keeps response slots

    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_log_probs": old_log_probs,
                "advantages": torch.zeros(1, 10),
                "response_mask": response_mask,
            }
        }
    )
    ctx = out["context"]

    new_logprobs = _zone_logprobs(input_ids, true_logprobs)
    mask = ctx["loss_mask"].bool()
    ratio = torch.exp(new_logprobs - ctx["old_log_probs_shifted"])[mask]

    assert torch.equal(ratio, torch.ones_like(ratio)), f"ratio must be exactly 1, got {ratio}"

    # Tripwire: verl's unshifted layout is what produced importance_weight >> 1.
    bad = torch.exp(new_logprobs - old_log_probs)[mask]
    assert not torch.allclose(bad, torch.ones_like(bad)), "unshifted layout should NOT give ratio 1"


def test_left_aligned_rows_keep_alignment_with_uneven_prompts():
    """Production shape: verl left-pads prompts, so ``_left_align`` compacts.

    Row A: prompt 3 + response 3 (no padding). Row B: prompt 2 + response 2,
    so its row arrives as ``[PAD, p, p, r, r, PAD]``.
    """
    input_ids = torch.tensor(
        [
            [10, 11, 12, 30, 31, 32],
            [PAD, 20, 21, 40, 41, PAD],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 0],
        ]
    )
    # Adapter left-pads the response-only [B, R] tensors by S - R = 3, which
    # lines them up with the response block because prompts are left-padded.
    old_log_probs = torch.tensor(
        [
            [0.0, 0.0, 0.0, -1.0, -2.0, -3.0],
            [0.0, 0.0, 0.0, -7.0, -8.0, 0.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ]
    )

    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_log_probs": old_log_probs,
                "advantages": response_mask.float(),
                "response_mask": response_mask,
            }
        }
    )
    ctx = out["context"]

    # Row B compacts to [p, p, r, r, 0, 0]: response tokens at 2,3 -> slots 1,2.
    assert torch.equal(out["kwargs"]["input_ids"][1], torch.tensor([20, 21, 40, 41, PAD, PAD]))
    assert ctx["loss_mask"][0].nonzero().flatten().tolist() == [2, 3, 4]
    assert ctx["loss_mask"][1].nonzero().flatten().tolist() == [1, 2]
    assert torch.equal(ctx["old_log_probs_shifted"][1], torch.tensor([0.0, -7.0, -8.0, 0.0, 0.0, 0.0]))


def test_full_row_drops_the_last_slot_without_wrapping():
    """``true_len == S``: the shift must not wrap slot 0 into the last column."""
    input_ids = torch.tensor([[10, 11, 30, 31, 32, 33]])
    attention_mask = torch.ones(1, 6, dtype=torch.long)
    old_log_probs = torch.tensor([[0.0, 0.0, -1.0, -2.0, -3.0, -4.0]])
    response_mask = torch.tensor([[0, 0, 1, 1, 1, 1]])

    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_log_probs": old_log_probs,
                "advantages": response_mask.float(),
                "response_mask": response_mask,
            }
        }
    )
    ctx = out["context"]

    assert torch.equal(ctx["old_log_probs_shifted"], torch.tensor([[0.0, -1.0, -2.0, -3.0, -4.0, 0.0]]))
    # The final response token has no predict-next slot inside the row.
    assert ctx["loss_mask"].nonzero()[:, 1].tolist() == [1, 2, 3, 4]
    assert bool(ctx["loss_mask"][0, -1]) is False


def test_drop_env_omits_old_log_probs(monkeypatch):
    """A/B control: no ``old_log_probs_shifted`` -> zone pins the ratio at 1."""
    monkeypatch.setenv("CORTEX_VERL_DROP_OLD_LOGPROBS", "1")

    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": torch.tensor([[10, 11, 30, 31]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "old_log_probs": torch.tensor([[0.0, 0.0, -1.0, -2.0]]),
                "advantages": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "response_mask": torch.tensor([[0, 0, 1, 1]]),
            }
        }
    )

    assert "old_log_probs_shifted" not in out["context"]
    assert out["processing"]["loss_fn"] == "grpo"


def test_zone_grpo_config_matches_verl_dual_clip():
    """AP cannot register verl_grpo on the zone; it can send dual-clip knobs."""
    out = to_cortex_fwd_bwd_payload(
        {
            "batch": {
                "input_ids": torch.tensor([[10, 11, 30, 31]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "old_log_probs": torch.tensor([[0.0, 0.0, -1.0, -2.0]]),
                "advantages": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "response_mask": torch.tensor([[0, 0, 1, 1]]),
            }
        }
    )
    cfg = out["processing"]["config"]
    assert cfg["eps_clip"] == 0.2
    assert cfg["eps_clip_higher"] == 0.2
    assert cfg["c_clip"] == 3.0
    assert cfg["loss_agg_mode"] == "token-mean"
    assert cfg["entropy_coeff"] == 0.0
    assert "dp_size" not in cfg

