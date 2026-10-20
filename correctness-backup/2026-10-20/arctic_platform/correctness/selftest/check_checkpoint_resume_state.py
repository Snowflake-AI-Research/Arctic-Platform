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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Keep checkpoint state observations faithful to the frozen checkpoint workload."""

from __future__ import annotations

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.diagnostics import checkpoint_resume_state as probe


def _inputs(case="gas4"):
    arm = SimpleNamespace(name=case, global_batch_size=16, max_seq_len=64, dss_microbatches=4)
    cfg = SimpleNamespace(
        config_id="qwen3-8b-h200-train-sft-full-4gpus-64k",
        training={
            "n_gpus": 4,
            "ds_config": {
                "train_micro_batch_size_per_gpu": 1,
                "gradient_accumulation_steps": 1,
                "train_batch_size": 4,
            },
        },
        n_gpus=4,
        attention_implementation="flash_attention_3",
    )
    spec = SimpleNamespace(model=SimpleNamespace(cache_path="/models/reduced"), arms=[arm])
    ctx = SimpleNamespace(cfg=cfg, config_id=cfg.config_id, vocab_size=128)
    return cfg, spec, arm, checkpoint_resume._iteration_bodies(ctx, arm)


@pytest.fixture
def observed_jobs(monkeypatch):
    payloads, steps, snapshots, loads, saves = [], [], [], [], []

    @contextlib.contextmanager
    def gateway(*args, **kwargs):
        yield SimpleNamespace()

    @contextlib.contextmanager
    def running_job(session, payload):
        index = len(payloads)
        payloads.append(payload)

        def load_checkpoint(*, path):
            loads.append(path)
            return {"global_step": checkpoint_resume.CHECKPOINT_ITERATION}

        def save_checkpoint():
            saves.append((index, sum(job_index == index for job_index, body, rate in steps)))
            return {"path": "/saved/step10"}

        yield SimpleNamespace(
            index=index,
            client=SimpleNamespace(save_checkpoint=save_checkpoint, load_checkpoint=load_checkpoint),
        )

    def step(session, job, body, *, learning_rate):
        steps.append((job.index, body, learning_rate))
        return SimpleNamespace(avg_loss=2.0)

    monkeypatch.setattr(probe, "gateway", gateway)
    monkeypatch.setattr(probe, "optimizer_capture_worker", contextlib.nullcontext)
    monkeypatch.setattr(probe, "running_job", running_job)
    monkeypatch.setattr(probe, "fwd_bwd_step", step)
    monkeypatch.setattr(probe, "_snapshot", lambda source, target: snapshots.append((source, target)))
    monkeypatch.setattr(probe, "_compare_iteration", lambda *args, **kwargs: [])
    return SimpleNamespace(payloads=payloads, steps=steps, snapshots=snapshots, loads=loads, saves=saves)


def _assert_payloads(payloads, cfg, arm):
    assert payloads
    for payload in payloads:
        training = payload["training_config"]
        assert training["ds_config"]["gradient_accumulation_steps"] == 1
        assert training["ds_config"]["train_batch_size"] == cfg.training["ds_config"]["train_batch_size"]
        assert training["attn_implementation"] == cfg.attention_implementation
        assert training["debug"]["full_determinism"] is True
        assert training["debug"]["full_determinism_must_comply"] is False
        assert training["debug"]["optimizer_state_output_dir"]
    assert cfg.training["ds_config"]["gradient_accumulation_steps"] == 1
    assert cfg.training["ds_config"]["train_batch_size"] == 4


def test_checkpoint_probe_matches_accumulation_for_all_three_jobs(tmp_path, observed_jobs):
    cfg, spec, arm, bodies = _inputs()

    left, right = probe._run_pair(11, tmp_path, cfg, spec, arm, bodies)

    assert left == right == {11: 2.0}
    assert len(observed_jobs.payloads) == 3
    _assert_payloads(observed_jobs.payloads, cfg, arm)
    assert observed_jobs.loads == ["/saved/step10"]
    assert observed_jobs.saves == [(1, 10)]
    for index, expected in enumerate((bodies[:11], bodies[:10], bodies[10:11])):
        actual = [body for job_index, body, rate in observed_jobs.steps if job_index == index]
        assert len(actual) == len(expected)
        assert all(body is expected_body for body, expected_body in zip(actual, expected, strict=True))
    assert all(rate == checkpoint_resume.LEARNING_RATE for index, body, rate in observed_jobs.steps)


def test_independent_probe_matches_accumulation_for_both_jobs(tmp_path, observed_jobs):
    cfg, spec, arm, bodies = _inputs()

    assert probe._run_independent(2, tmp_path / "pair", cfg, spec, bodies, arm=arm) is None

    assert len(observed_jobs.payloads) == 2
    _assert_payloads(observed_jobs.payloads, cfg, arm)
    assert observed_jobs.loads == []
    for index in range(2):
        actual = [body for job_index, body, rate in observed_jobs.steps if job_index == index]
        assert all(body is expected for body, expected in zip(actual, bodies[:2], strict=True))


def test_shared_prefix_state_probe_compares_only_post_boundary_steps(tmp_path, observed_jobs):
    cfg, spec, arm, bodies = _inputs()

    left, right = probe._run_shared_prefix_pair(11, tmp_path, cfg, spec, arm, bodies)

    assert left == right == {11: 2.0}
    assert len(observed_jobs.payloads) == 2
    _assert_payloads(observed_jobs.payloads, cfg, arm)
    assert observed_jobs.loads == ["/saved/step10"]
    assert observed_jobs.saves == [(0, 10)]
    for index, expected in enumerate((bodies[:11], bodies[10:11])):
        actual = [body for job_index, body, rate in observed_jobs.steps if job_index == index]
        assert len(actual) == len(expected)
        assert all(body is expected_body for body, expected_body in zip(actual, expected, strict=True))
    assert all(rate == checkpoint_resume.LEARNING_RATE for index, body, rate in observed_jobs.steps)
    assert [target.relative_to(tmp_path).as_posix() for source, target in observed_jobs.snapshots] == [
        "uninterrupted/step-11",
        "resumed/step-11",
    ]


def test_shared_prefix_state_probe_rejects_wrong_restored_step(tmp_path, observed_jobs, monkeypatch):
    cfg, spec, arm, bodies = _inputs()
    original_running_job = probe.running_job

    @contextlib.contextmanager
    def running_job(session, payload):
        with original_running_job(session, payload) as job:
            if job.index == 1:
                job.client.load_checkpoint = lambda **kwargs: {"global_step": 0}
            yield job

    monkeypatch.setattr(probe, "running_job", running_job)

    with pytest.raises(RuntimeError, match="restored step 0.*expected 10"):
        probe._run_shared_prefix_pair(11, tmp_path, cfg, spec, arm, bodies)

    assert all(index == 0 for index, body, rate in observed_jobs.steps)


def test_shared_prefix_model_manifests_are_separate_from_step_snapshots(tmp_path, observed_jobs):
    cfg, spec, arm, bodies = _inputs()

    probe._run_shared_prefix_pair(11, tmp_path, cfg, spec, arm, bodies, model_state=True)

    for payload in observed_jobs.payloads:
        debug = payload["training_config"]["debug"]
        assert debug["model_state_output_dir"] == str(tmp_path / "model-state")
        assert debug["model_state_output_dir"] != debug["optimizer_state_output_dir"]
    assert observed_jobs.saves == [(0, 10)]
    assert observed_jobs.loads == ["/saved/step10"]


@pytest.mark.parametrize("limit", [0, 10, 21])
def test_shared_prefix_state_probe_rejects_invalid_horizon(tmp_path, observed_jobs, limit):
    cfg, spec, arm, bodies = _inputs()

    with pytest.raises(ValueError, match="requires post-checkpoint batches"):
        probe._run_shared_prefix_pair(limit, tmp_path, cfg, spec, arm, bodies)

    assert observed_jobs.payloads == []


def test_main_dispatches_shared_prefix_state_probe(tmp_path, monkeypatch, capsys):
    from transformers import AutoConfig

    cfg, spec, arm, _ = _inputs()
    observed = []

    def shared_prefix(limit, work, loaded_cfg, loaded_spec, loaded_arm, bodies):
        observed.append((limit, work, loaded_cfg, loaded_spec, loaded_arm, bodies))
        return {11: 2.0}, {11: 2.001}

    monkeypatch.setattr(probe, "load_config", lambda path: cfg)
    monkeypatch.setattr(probe.TestSpec, "read", lambda path: spec)
    monkeypatch.setattr(probe, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(probe, "_run_shared_prefix_pair", shared_prefix, raising=False)
    monkeypatch.setattr(
        probe, "_run_independent", lambda *args, **kwargs: pytest.fail("state probe must share its prefix")
    )
    monkeypatch.setattr(
        probe, "_first_difference", lambda *args: (11, probe.Comparison("fp32_masters", False, 0.01, 0.01, "test"))
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=128))

    assert probe.main(["--case", "gas4", "--checkpoint-state"]) == 0

    assert len(observed) == 1
    limit, work, loaded_cfg, loaded_spec, loaded_arm, bodies = observed[0]
    assert limit == 11 and work == tmp_path
    assert loaded_cfg is cfg and loaded_spec is spec and loaded_arm is arm
    assert len(bodies) == checkpoint_resume.ITERATIONS
    assert "FIRST_DIFFERENCE iteration=11 component=fp32_masters" in capsys.readouterr().out


def test_main_dispatches_per_rank_model_state_probe(tmp_path, monkeypatch, capsys):
    from transformers import AutoConfig

    cfg, spec, arm, _ = _inputs()
    observed = []

    def shared_prefix(*args, **kwargs):
        observed.append(kwargs)
        (tmp_path / "model-state").mkdir()
        return {11: 2.0}, {11: 2.0}

    def compare(left, right, *, ranks):
        assert left == tmp_path / "model-state/before-save"
        assert right == tmp_path / "model-state/after-load"
        assert ranks == cfg.n_gpus
        return {"exact": True, "differences": []}

    monkeypatch.setattr(probe, "load_config", lambda path: cfg)
    monkeypatch.setattr(probe.TestSpec, "read", lambda path: spec)
    monkeypatch.setattr(probe, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(probe, "_run_shared_prefix_pair", shared_prefix)
    monkeypatch.setattr(probe, "compare_model_state", compare)
    monkeypatch.setattr(probe, "_first_difference", lambda *args: None)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=128))

    assert probe.main(["--case", "gas4", "--checkpoint-model-state"]) == 0
    assert observed == [{"model_state": True}]
    assert (tmp_path / "model-state/comparison.json").is_file()
    assert "MODEL_STATE_COMPARISON" in capsys.readouterr().out


def test_loss_control_runs_twenty_steps_without_checkpoint_or_capture(tmp_path, monkeypatch):
    cfg, spec, arm, bodies = _inputs()
    spec.tolerance_for = lambda test_id: 0.002
    payloads, steps = [], []

    @contextlib.contextmanager
    def running_job(session, payload):
        index = len(payloads)
        payloads.append(payload)
        yield SimpleNamespace(index=index)

    def step(session, job, body, *, learning_rate):
        steps.append((job.index, body, learning_rate))
        iteration = sum(index == job.index for index, batch, rate in steps)
        delta = 0.003 if job.index == 1 and iteration == checkpoint_resume.ITERATIONS else 0.0
        return SimpleNamespace(avg_loss=2.0 + delta)

    from arctic_platform.correctness.harness import dss_driver

    monkeypatch.setattr(dss_driver, "running_job", running_job)
    monkeypatch.setattr(dss_driver, "fwd_bwd_step", step)
    monkeypatch.setattr(
        probe,
        "optimizer_capture_worker",
        lambda: pytest.fail("loss control must not install optimizer-capture hooks"),
    )

    result = probe._run_loss_control(tmp_path, cfg, spec, arm, bodies)

    assert len(payloads) == 2
    for payload in payloads:
        training = payload["training_config"]
        assert training["ds_config"]["gradient_accumulation_steps"] == 1
        assert training["ds_config"]["train_batch_size"] == cfg.training["ds_config"]["train_batch_size"]
        assert training["debug"]["full_determinism"] is True
        assert training["debug"]["full_determinism_must_comply"] is False
        assert "optimizer_state_output_dir" not in training["debug"]
    assert cfg.training["ds_config"]["gradient_accumulation_steps"] == 1
    assert len(steps) == 2 * checkpoint_resume.ITERATIONS
    for index in range(2):
        actual = [body for job_index, body, rate in steps if job_index == index]
        assert all(body is expected for body, expected in zip(actual, bodies, strict=True))
    assert all(rate == checkpoint_resume.LEARNING_RATE for index, body, rate in steps)
    assert result["case"] == "gas4"
    assert result["gate"] == 0.002
    assert [item["iteration"] for item in result["comparisons"]] == [10, 20]
    assert result["comparisons"][0]["abs_diff"] == 0.0
    assert result["comparisons"][0]["within_gate"] is True
    assert result["comparisons"][1]["abs_diff"] == pytest.approx(0.003)
    assert result["comparisons"][1]["within_gate"] is False


def test_main_dispatches_loss_control_without_state_capture(tmp_path, monkeypatch, capsys):
    from transformers import AutoConfig

    cfg, spec, arm, _ = _inputs()
    observed = []

    def loss_control(work, loaded_cfg, loaded_spec, loaded_arm, bodies):
        observed.append((work, loaded_cfg, loaded_spec, loaded_arm, bodies))
        return {"case": arm.name, "gate": 0.002, "comparisons": []}

    monkeypatch.setattr(probe, "load_config", lambda path: cfg)
    monkeypatch.setattr(probe.TestSpec, "read", lambda path: spec)
    monkeypatch.setattr(probe, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(probe, "_run_loss_control", loss_control, raising=False)
    monkeypatch.setattr(
        probe, "_run_independent", lambda *args, **kwargs: pytest.fail("loss control must not capture worker state")
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=128))

    assert probe.main(["--case", "gas4", "--loss-control"]) == 0

    assert len(observed) == 1
    work, loaded_cfg, loaded_spec, loaded_arm, bodies = observed[0]
    assert work == tmp_path
    assert loaded_cfg is cfg and loaded_spec is spec and loaded_arm is arm
    assert len(bodies) == checkpoint_resume.ITERATIONS
    assert "UNINTERRUPTED_LOSS_CONTROL" in capsys.readouterr().out
    assert (tmp_path / "loss-control.json").is_file()


def test_checkpoint_control_shares_prefix_and_resumes_exact_saved_checkpoint(tmp_path, monkeypatch):
    from arctic_platform.correctness.harness import dss_driver

    cfg, spec, arm, bodies = _inputs()
    spec.tolerance_for = lambda test_id: 0.002
    payloads, steps, events = [], [], []

    @contextlib.contextmanager
    def running_job(session, payload):
        index = len(payloads)
        payloads.append(payload)

        def save_checkpoint():
            assert index == 0
            assert len(steps) == checkpoint_resume.CHECKPOINT_ITERATION
            events.append(("save", len(steps)))
            return {"path": "/saved/shared-prefix-step10"}

        def load_checkpoint(*, path):
            assert index == 1
            assert len(steps) == checkpoint_resume.ITERATIONS
            events.append(("load", path))
            return {"global_step": checkpoint_resume.CHECKPOINT_ITERATION}

        yield SimpleNamespace(
            index=index, client=SimpleNamespace(save_checkpoint=save_checkpoint, load_checkpoint=load_checkpoint)
        )
        events.append(("close", index))

    def step(session, job, body, *, learning_rate):
        steps.append((job.index, body, learning_rate))
        iteration = next(index for index, candidate in enumerate(bodies, start=1) if candidate is body)
        return SimpleNamespace(avg_loss=float(iteration) + (0.003 if job.index == 1 else 0.0))

    monkeypatch.setattr(dss_driver, "running_job", running_job)
    monkeypatch.setattr(dss_driver, "fwd_bwd_step", step)
    monkeypatch.setattr(
        probe,
        "optimizer_capture_worker",
        lambda: pytest.fail("checkpoint control must not install optimizer-capture hooks"),
    )

    result = probe._run_loss_control(tmp_path, cfg, spec, arm, bodies, checkpoint_control=True)

    assert len(payloads) == 2
    assert events == [
        ("save", 10),
        ("close", 0),
        ("load", "/saved/shared-prefix-step10"),
        ("close", 1),
    ]
    for index, expected in enumerate((bodies, bodies[10:])):
        actual = [body for job_index, body, rate in steps if job_index == index]
        assert all(body is expected_body for body, expected_body in zip(actual, expected, strict=True))
    assert all(rate == checkpoint_resume.LEARNING_RATE for index, body, rate in steps)
    for payload in payloads:
        training = payload["training_config"]
        assert training["ds_config"]["gradient_accumulation_steps"] == 1
        assert training["debug"]["full_determinism"] is True
        assert training["debug"]["full_determinism_must_comply"] is False
        assert "optimizer_state_output_dir" not in training["debug"]
    assert cfg.training["ds_config"]["gradient_accumulation_steps"] == 1
    assert result["comparisons"][0]["abs_diff"] == 0.0
    assert result["comparisons"][0]["within_gate"] is True
    assert result["comparisons"][1]["abs_diff"] == pytest.approx(0.003)
    assert result["comparisons"][1]["within_gate"] is False


def test_main_dispatches_checkpoint_control_without_state_capture(tmp_path, monkeypatch, capsys):
    from transformers import AutoConfig

    cfg, spec, arm, _ = _inputs()
    observed = []

    def loss_control(work, loaded_cfg, loaded_spec, loaded_arm, bodies, **kwargs):
        observed.append((work, loaded_cfg, loaded_spec, loaded_arm, bodies, kwargs))
        return {"case": arm.name, "gate": 0.002, "comparisons": []}

    monkeypatch.setattr(probe, "load_config", lambda path: cfg)
    monkeypatch.setattr(probe.TestSpec, "read", lambda path: spec)
    monkeypatch.setattr(probe, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(probe, "_run_loss_control", loss_control)
    monkeypatch.setattr(
        probe,
        "_run_independent",
        lambda *args, **kwargs: pytest.fail("checkpoint control must not capture worker state"),
    )
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=128))

    assert probe.main(["--case", "gas4", "--checkpoint-control"]) == 0

    assert len(observed) == 1
    work, loaded_cfg, loaded_spec, loaded_arm, bodies, kwargs = observed[0]
    assert loaded_cfg is cfg and loaded_spec is spec and loaded_arm is arm
    assert len(bodies) == checkpoint_resume.ITERATIONS
    assert kwargs == {"checkpoint_control": True}
    assert "SHARED_PREFIX_CHECKPOINT_CONTROL" in capsys.readouterr().out
    assert (tmp_path / "checkpoint-control.json").is_file()


@pytest.mark.parametrize(
    "flags",
    [
        ["--loss-control", "--checkpoint-control"],
        ["--loss-control", "--checkpoint-state"],
        ["--checkpoint-control", "--checkpoint-state"],
        ["--checkpoint-state", "--checkpoint-model-state"],
    ],
)
def test_main_rejects_conflicting_control_modes(flags):
    with pytest.raises(SystemExit) as raised:
        probe.main(flags)
    assert raised.value.code == 2


@pytest.mark.parametrize("case,provider", [("gas1", "prime_rl"), ("gas4", "huggingface")])
def test_main_uses_current_checkpoint_iteration_bodies(tmp_path, monkeypatch, case, provider):
    from transformers import AutoConfig

    cfg, spec, arm, _ = _inputs(case=case)
    cfg.training["model_provider"] = provider
    observed = []

    def independent(limit, work, loaded_cfg, loaded_spec, bodies, **kwargs):
        observed.extend(bodies)
        return 1, probe.Comparison("gradient", False, 1.0, 1.0, "example")

    monkeypatch.setattr(probe, "load_config", lambda path: cfg)
    monkeypatch.setattr(probe.TestSpec, "read", lambda path: spec)
    monkeypatch.setattr(probe, "correctness_workdir", lambda prefix: tmp_path)
    monkeypatch.setattr(probe, "_run_independent", independent)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(vocab_size=128))
    monkeypatch.setattr(sys, "argv", ["checkpoint_resume_state", "--case", case])

    assert probe.main() == 0

    ctx = SimpleNamespace(cfg=cfg, config_id=cfg.config_id, vocab_size=128)
    expected = checkpoint_resume._iteration_bodies(ctx, arm)
    assert len(observed) == len(expected) == checkpoint_resume.ITERATIONS
    for actual_body, expected_body in zip(observed, expected, strict=True):
        assert actual_body["processing"] == expected_body["processing"]
        assert isinstance(actual_body["batch"], dict)
        assert actual_body["batch"]["input_ids"].shape == (arm.global_batch_size, arm.max_seq_len)
        for field in ("input_ids", "position_ids", "labels"):
            assert torch.equal(actual_body["batch"][field], expected_body["batch"][field])
