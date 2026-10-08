# Copyright 2025 Snowflake Inc.
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

"""The replay, the held-out request's alignment, what the verdict gates on, and the optimizer state.

None of it needs a GPU. The two trajectories are replaced by lists of numbers, the two packers by
recorders, and the optimizer by a two-parameter linear layer on the CPU, so each property is asserted on
the code that carries it rather than on a run that would also exercise a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import e2e_sft_train_validate as check
from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness.batches import load
from arctic_platform.correctness.harness.dss_driver import pack
from arctic_platform.correctness.harness.dss_driver import pack_logit_aligned


def _assert_nested_equal(left, right):
    import torch

    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


from arctic_platform.correctness.harness.registry import TestOutcome as Outcome  # noqa: E402
from arctic_platform.correctness.reference.adamw_trajectory import AdamWTrajectory  # noqa: E402
from arctic_platform.correctness.reference.optimizer_step import run_adamw_step  # noqa: E402

STEPS = 4
ROWS_PER_STEP = 2

ADAMW = {"name": "AdamW", "lr": 2e-05, "weight_decay": 0.0, "betas": [0.9, 0.999], "eps": 1e-08}

# Disagreements are sized against the gate the check holds rather than written as literals, so a case
# built to fail keeps failing and one built to pass keeps passing if the stated criterion moves.
OVER_GATE = check.TOLERANCE * 3


def examples(count: int, *, first: int = 0):
    """Rows of unequal length, each with a distinct token stream, so no two steps share a batch."""
    return [
        gsm8k.Example(
            question=f"q{first + index}",
            prompt="Question: q\nAnswer:",
            completion=" a",
            input_ids=tuple(range(10 * (first + index) + 1, 10 * (first + index) + 4 + index % 3)),
            n_prompt=2,
        )
        for index in range(count)
    ]


def a_replay(tmp_path: Path, steps: int = STEPS) -> check.Replay:
    rows = examples(steps * ROWS_PER_STEP)
    return check.materialize_replay(
        check.replay_batches(rows, ROWS_PER_STEP, steps, pad_id=0),
        gsm8k.build_batch("validation", examples(3, first=100), pad_id=0),
        tmp_path / "replay",
    )


def test_mixer_packing_is_selected_from_checkpoint_layers(tmp_path) -> None:
    from arctic_platform.correctness.reference.model_features import uses_mixer_packing

    (tmp_path / "config.json").write_text(json.dumps({"layer_types": ["linear_attention", "full_attention"]}))
    assert uses_mixer_packing(tmp_path) is True

    (tmp_path / "config.json").write_text(json.dumps({"layer_types": ["full_attention"]}))
    assert uses_mixer_packing(tmp_path) is False


def test_mixer_packing_keeps_each_sequence_in_its_own_reference_call() -> None:
    from arctic_platform.correctness.reference.hf_single_gpu import reference_row_groups

    lengths = [4, 4]
    assert reference_row_groups(lengths, 4, 64, mixer_packing=True) == [[0], [1]]
    assert reference_row_groups(lengths, 4, 64, mixer_packing=False) == [[0, 1]]


def test_every_step_gets_its_own_batch_from_the_training_slice() -> None:
    rows = examples(STEPS * ROWS_PER_STEP)

    batches = check.replay_batches(rows, ROWS_PER_STEP, STEPS, pad_id=0)

    assert len(batches) == STEPS
    assert all(batch.rows == ROWS_PER_STEP for batch in batches)
    seen = [tuple(row) for batch in batches for row in batch.input_ids.tolist()]
    assert len(set(seen)) == len(rows)


def test_a_slice_that_does_not_fill_the_steps_is_refused() -> None:
    with pytest.raises(ValueError, match="need 8 examples"):
        check.replay_batches(examples(7), ROWS_PER_STEP, STEPS, pad_id=0)


def test_both_engines_are_handed_the_same_replay_files_in_the_same_order(tmp_path) -> None:
    """The one property that makes the comparison a comparison: one list of files, read by both sides."""
    replay = a_replay(tmp_path)

    manifest = json.loads(check.write_replay_manifest(replay, tmp_path / "replay.json").read_text())
    training_bodies, _ = check.request_bodies(replay, model_provider="huggingface")

    # The reference loads exactly the paths the Arctic Platform bodies were packed from, in step order.
    assert manifest["train_batches"] == [str(path) for path in replay.train_batch_paths]
    assert manifest["validation_batch"] == str(replay.validation_batch_path)
    assert len(training_bodies) == replay.steps
    expected = [pack(load(path), model_provider="huggingface") for path in replay.train_batch_paths]
    _assert_nested_equal(training_bodies, expected)
    # And no step's request repeats another's, so a run that stalled on one batch cannot pass.
    assert (
        len({body["batch"]["input_ids"].detach().cpu().numpy().tobytes() for body in training_bodies}) == replay.steps
    )


def test_the_held_out_request_carries_logit_aligned_labels(tmp_path) -> None:
    import torch

    """The forward-only route shifts nothing, so the same bytes ``/fwd-bwd`` takes would score one off."""
    replay = a_replay(tmp_path)
    validation_batch = load(replay.validation_batch_path)

    _, validation_body = check.request_bodies(replay, model_provider="huggingface")

    _assert_nested_equal(validation_body, pack_logit_aligned(validation_batch))
    assert not torch.equal(
        validation_body["batch"]["labels"],
        pack(validation_batch, model_provider="huggingface")["batch"]["labels"],
    )


def test_the_training_requests_carry_the_convention_their_dispatch_expects(tmp_path) -> None:
    """``/fwd-bwd`` shifts HuggingFace-convention labels itself; pre-shifting them would double it."""
    replay = a_replay(tmp_path)
    packed = {"train": [], "validation": []}

    def record_train(batch, *, model_provider):
        packed["train"].append(batch.name)
        return f"train:{batch.name}".encode()

    def record_validation(batch):
        packed["validation"].append(batch.name)
        return f"validation:{batch.name}".encode()

    original = check.pack, check.pack_logit_aligned
    check.pack, check.pack_logit_aligned = record_train, record_validation
    try:
        training_bodies, validation_body = check.request_bodies(replay, model_provider="huggingface")
    finally:
        check.pack, check.pack_logit_aligned = original

    assert packed["train"] == [f"train-step-{step:03d}" for step in range(1, replay.steps + 1)]
    assert packed["validation"] == ["validation"]
    assert validation_body == b"validation:validation"
    assert len(training_bodies) == replay.steps


def a_verdict(reference, target, reference_validation=1.5, target_validation=1.5):
    return check.verdict(
        SimpleNamespace(config_id="example"), reference, target, reference_validation, target_validation
    )


def test_the_verdict_gates_on_the_final_training_loss_and_the_validation_loss() -> None:
    assert check.gated_training_steps(100) == (100,)

    result = a_verdict([2.5] * 100, [2.5] * 100)

    assert result.metrics["gated_quantities"] == 2.0
    assert result.metrics["train_steps_recorded"] == 100.0


def test_two_identical_trajectories_pass_and_still_record_every_step() -> None:
    losses = [2.5 - 0.01 * step for step in range(10)]

    result = a_verdict(losses, list(losses))

    assert result.outcome is Outcome.PASS
    assert result.mismatches == []
    assert result.reason == ""
    assert result.metrics["stated_criterion_abs"] == check.TOLERANCE
    assert result.metrics["first_disagreeing_train_step"] == 0.0
    # The trajectory is the diagnostic, so it is recorded whether or not anything failed.
    assert [f"train_loss_abs_diff_step_{step:03d}" in result.metrics for step in range(1, 11)] == [True] * 10


def test_a_mid_run_disagreement_that_closes_by_the_final_step_does_not_decide_the_outcome() -> None:
    """The trajectory is reported and not gated, so the widest step does not reach ``max_abs_diff``."""
    reference = [2.5] * 10
    target = [2.5, 2.5, 2.5 + OVER_GATE * 2, 2.5 + OVER_GATE] + [2.5] * 6

    result = a_verdict(reference, target)

    assert result.outcome is Outcome.PASS
    assert result.metrics["final_train_loss_abs_diff"] == pytest.approx(0.0)
    assert result.metrics["max_abs_diff"] == pytest.approx(0.0)
    # Recorded in full, all the same: two steps drifted, the widest of them at step 3.
    assert result.metrics["trajectory_steps_over_gate"] == 2.0
    assert result.metrics["worst_train_loss_step"] == 3.0
    assert result.metrics["worst_train_loss_abs_diff"] == pytest.approx(OVER_GATE * 2)
    assert result.metrics["first_disagreeing_train_step"] == 3.0


def test_a_disagreement_at_step_one_is_read_as_the_forward_and_not_as_accumulation() -> None:
    result = a_verdict([2.5] * 10, [2.5 + OVER_GATE] * 10)

    assert result.outcome is Outcome.FAIL
    assert result.metrics["first_disagreeing_train_step"] == 1.0
    assert "step 1" in result.reason
    assert "not one that accumulated" in result.reason


def test_a_disagreement_that_appears_later_is_read_as_accumulation() -> None:
    """The distinction the trajectory exists for: same final magnitude, opposite cause."""
    result = a_verdict([2.5] * 10, [2.5] * 5 + [2.5 + OVER_GATE] * 5)

    assert result.outcome is Outcome.FAIL
    assert result.metrics["first_disagreeing_train_step"] == 6.0
    assert "step 6 of 10" in result.reason
    assert "accumulated over 5 optimizer steps" in result.reason


def test_an_onset_at_the_second_step_counts_one_elapsed_optimizer_step() -> None:
    """The sentence counts optimizer steps, and one of them is a step rather than steps."""
    result = a_verdict([2.5] * 10, [2.5] + [2.5 + OVER_GATE] * 9)

    assert result.outcome is Outcome.FAIL
    assert result.metrics["first_disagreeing_train_step"] == 2.0
    assert "first disagree at step 2 of 10" in result.reason
    assert "accumulated over 1 optimizer step rather" in result.reason


def test_the_final_training_loss_fails_on_its_own() -> None:
    result = a_verdict([2.5] * 10, [2.5] * 9 + [2.5 + OVER_GATE])

    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [check.TRAIN_LOSS_QUANTITY.format(step=10)]
    assert result.metrics["final_train_loss_abs_diff"] == pytest.approx(OVER_GATE)


def test_a_validation_disagreement_fails_on_its_own() -> None:
    result = a_verdict([2.5] * 10, [2.5] * 10, reference_validation=1.75, target_validation=1.75 + OVER_GATE)

    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [check.VALIDATION_QUANTITY]
    # No training step drifted, so there is no onset to explain.
    assert result.reason == ""
    assert result.metrics["validation_abs_diff"] == pytest.approx(OVER_GATE)


def test_the_first_and_widest_step_are_reported_whatever_the_verdict() -> None:
    """A passing run still states where the two runs started and how far apart they ever got."""
    under = check.TOLERANCE / 2
    result = a_verdict([2.5] * 10, [2.5] + [2.5 + under] * 8 + [2.5])

    assert result.outcome is Outcome.PASS
    # Step 1 and the widest step locate the onset without a second threshold deciding what counts.
    assert result.metrics["first_train_loss_abs_diff"] == 0.0
    assert result.metrics["worst_train_loss_abs_diff"] == pytest.approx(under)
    assert "step 1 0.000e-03" in result.summary
    assert "neither gated" in result.summary


def test_the_stated_gate_decides_the_outcome() -> None:
    """The stated gate, asserted where it acts: either side of ``TOLERANCE`` decides the outcome."""
    inside, outside = check.TOLERANCE * 0.9, check.TOLERANCE * 1.1
    assert a_verdict([2.5] * 10, [2.5] * 9 + [2.5 + inside]).outcome is Outcome.PASS
    assert a_verdict([2.5] * 10, [2.5] * 9 + [2.5 + outside]).outcome is Outcome.FAIL
    assert a_verdict([2.5] * 10, [2.5] * 10, 1.75, 1.75 + inside).outcome is Outcome.PASS
    assert a_verdict([2.5] * 10, [2.5] * 10, 1.75, 1.75 + outside).outcome is Outcome.FAIL


def test_reference_fused_cross_entropy_matches_product_selection(tmp_path) -> None:
    from arctic_platform.correctness.harness.config import LoadedConfig

    def cfg(training):
        return LoadedConfig(
            "example",
            tmp_path / "example.config",
            {
                "model_name": "example",
                "training_config": training,
            },
        )

    assert cfg({"model_provider": "prime_rl"}).fused_cross_entropy == "liger"
    assert cfg({"model_provider": "prime_rl", "fused_cross_entropy": False}).fused_cross_entropy is False
    assert cfg({"model_provider": "huggingface"}).fused_cross_entropy is False


def test_reference_optimizer_backend_matches_product_selection() -> None:
    def cfg(optimizer):
        return SimpleNamespace(training={"optimizer": optimizer})

    assert check.reference_optimizer_backend(cfg({"name": "AdamW"})) == "fused_adam"
    assert check.reference_optimizer_backend(cfg({"name": "FusedAdam"})) == "fused_adam"
    assert check.reference_optimizer_backend(cfg({"name": "AdamW", "fused": False})) == "torch"
    assert check.reference_optimizer_backend(cfg({"name": "torch_adamw"})) == "torch"


def test_trajectories_of_different_lengths_are_refused() -> None:
    with pytest.raises(ValueError, match="one loss per step"):
        a_verdict([2.5] * 5, [2.5] * 4)


def _model() -> torch.nn.Module:
    """A weight and a bias, so the no-decay group is populated as it is for a real model."""
    torch.manual_seed(0)
    return torch.nn.Linear(4, 3, bias=True)


def _gradients(model, value: float = 0.1) -> dict:
    return {name: torch.full_like(parameter, value) for name, parameter in model.named_parameters()}


def _trajectory(model) -> AdamWTrajectory:
    return AdamWTrajectory(model, ADAMW, learning_rate=1e-2, gradient_clipping=None, optimizer_dtype="float32")


def test_the_optimizer_moments_persist_across_steps() -> None:
    """Adam's step counter and both moments carry forward; a rebuilt optimizer would report step 1."""
    model = _model()
    trajectory = _trajectory(model)
    gradients = _gradients(model)

    trajectory.step(gradients)
    after_one = trajectory.state_norms()
    trajectory.step(gradients)
    after_two = trajectory.state_norms()

    assert trajectory.steps == 2
    assert {entry["step"] for entry in after_one.values()} == {1.0}
    assert {entry["step"] for entry in after_two.values()} == {2.0}
    # The second moment of a constant gradient grows with every step it has seen.
    assert after_two["weight"]["exp_avg_sq"] > after_one["weight"]["exp_avg_sq"]


def test_a_step_reads_the_moments_the_step_before_it_wrote() -> None:
    """One gradient, two optimizers: the update differs by exactly the history one of them carries.

    An Adam update at zero weight decay depends on the gradient and on the moments, not on the parameter
    value, so the two deltas below differ for one reason only. A trajectory that rebuilt its optimizer per
    step, or cleared its state, would produce the fresh delta at every step of a hundred-step run.
    """
    carried = _model()
    trajectory = _trajectory(carried)
    trajectory.step(_gradients(carried, 0.1))
    before = carried.weight.detach().clone()
    trajectory.step(_gradients(carried, 0.5))
    carried_delta = carried.weight.detach() - before

    fresh = _model()
    start = fresh.weight.detach().clone()
    _trajectory(fresh).step(_gradients(fresh, 0.5))
    fresh_delta = fresh.weight.detach() - start

    assert not torch.allclose(carried_delta, fresh_delta, atol=1e-9)
    assert carried_delta.abs().max().item() < fresh_delta.abs().max().item()


def test_the_step_is_written_back_into_the_model() -> None:
    """The next step's forward reads the model, so a step that only moved the master copy is lost."""
    model = _model()
    trajectory = _trajectory(model)

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    trajectory.step(_gradients(model))

    for name, parameter in model.named_parameters():
        assert not torch.equal(parameter.detach(), before[name])


def test_the_single_step_reference_cannot_be_looped(tmp_path) -> None:
    """Why the trajectory exists: ``run_adamw_step`` clears its state, so twice is once, twice over."""
    model = _model()
    gradients = _gradients(model)

    updates = []
    manifests = []
    for attempt in (1, 2):
        artifact = run_adamw_step(
            model,
            gradients,
            ADAMW,
            learning_rate=1e-2,
            gradient_clipping=None,
            optimizer_dtype="float32",
            output_dir=tmp_path / f"attempt-{attempt}",
        )
        manifests.append(json.loads(artifact.manifest_path.read_text()))
        updates.append({entry["name"]: entry["parameter_update_norm"] for entry in manifests[-1]["parameters"]})

    # Identical updates from two consecutive calls: the second is the first one again, because the state
    # was cleared and the model was never written back.
    assert updates[0] == updates[1]
    assert [manifest["step"] for manifest in manifests] == [1, 1]


def test_a_gradient_missing_for_an_owned_parameter_is_refused() -> None:
    model = _model()
    trajectory = _trajectory(model)
    gradients = _gradients(model)
    trajectory.step(gradients)

    with pytest.raises(ValueError, match="the trainable set changed mid-run"):
        trajectory.step({"weight": gradients["weight"]})


def test_an_optimizer_that_is_not_adamw_is_refused() -> None:
    with pytest.raises(ValueError, match="AdamW semantics"):
        AdamWTrajectory(
            _model(), {"name": "SGD"}, learning_rate=1e-2, gradient_clipping=None, optimizer_dtype="float32"
        )
