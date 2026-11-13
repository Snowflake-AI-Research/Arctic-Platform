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

import json
from pathlib import Path

import pytest

from arctic_platform.catalog_correctness import calibrate
from arctic_platform.catalog_correctness import checks
from arctic_platform.catalog_correctness import suite
from arctic_platform.catalog_correctness.reference import _tensor_hash
from arctic_platform.testing_utils import torch_assert_close


class Tokenizer:
    all_special_ids = [0, 7]

    def get_vocab(self):
        return {str(i): i for i in range(8)}


def recipe(kind, peft=None):
    key = "training_config" if kind == "training" else "inference_config"
    return {
        "hardware": "H200",
        "sub_job_configs": [
            {
                "job_type": kind,
                "model_name": "example/model",
                key: {"peft_config": peft},
            }
        ],
    }


def test_recipe_validation_requires_matching_full_models(tmp_path: Path):
    training = recipe("training")
    sampling = recipe("sampling")
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(training))
    loaded, digest = checks.read_recipe(path, "training")
    checks.validate_recipes(loaded, sampling)
    assert len(digest) == 64
    assert len(loaded["sub_job_configs"]) == 1
    sampling["sub_job_configs"][0]["model_name"] = "different/model"
    with pytest.raises(ValueError, match="same model"):
        checks.validate_recipes(loaded, sampling)
    sampling["sub_job_configs"][0]["model_name"] = "example/model"
    sampling["sub_job_configs"][0]["inference_config"]["peft_config"] = {"r": 8}
    with pytest.raises(ValueError, match="full-parameter"):
        checks.validate_recipes(loaded, sampling)


def test_synthetic_batches_are_reproducible_and_exclude_special_tokens():
    first = checks.batches(Tokenizer(), rows=4, width=32, count=2, seed=3)
    assert first == checks.batches(Tokenizer(), rows=4, width=32, count=2, seed=3)
    assert {len(row) for row in first[0]} == {20, 24, 28, 32}
    assert all(token not in {0, 7} for batch in first for row in batch for token in row)


def test_comparison_uses_token_weighted_loss_and_checks_coverage():
    target, reference = [[-1.0, -1.0], [-4.0]], [[-1.0, -2.0], [-3.0]]
    result = checks.compare_logprobs(target, reference, 0.0, 1.0)
    assert result["status"] == "pass"
    assert result["tokens"] == 3
    assert result["mean_token_abs_diff"] == pytest.approx(2 / 3)
    assert checks.compare_logprobs(target, reference, 0.0, 0.1)["status"] == "fail"
    assert checks.compare_logprobs(target, reference, 0.0)["status"] == "measured"
    assert checks.compare_logprobs(target, reference, None)["status"] == "measured"
    with pytest.raises(ValueError, match="coverage"):
        checks.compare_logprobs([[-1.0]], reference, 1)
    with pytest.raises(ValueError, match="nonpositive"):
        checks.compare_logprobs([[float("nan")]], [[-1.0]], 1)


def test_named_measurements_require_complete_finite_coverage():
    assert suite._comparison({"a": 1.1}, {"a": 1.0}, 0.11)["status"] == "pass"
    assert suite._comparison({"a": 1.1}, {"a": 1.0}, 0.01)["status"] == "fail"
    assert suite._comparison({"a": 1.1}, {"a": 1.0}, None)["status"] == "measured"
    with pytest.raises(ValueError, match="coverage"):
        suite._comparison({"a": 1.0}, {"b": 1.0}, 1)
    with pytest.raises(ValueError, match="nonfinite"):
        suite._comparison({"a": float("nan")}, {"a": 1.0}, 1)


def test_initial_identity_checks_tied_weights_and_tensor_contents(monkeypatch, tmp_path):
    import deepspeed
    import torch

    from arctic_platform.correctness.harness import optimizer

    weight = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    expected = {
        "dtype": "bfloat16",
        "gas4": {"gradient_accumulation_dtype": "bfloat16"},
        "initial_parameters": {"embedding": {"shape": [2], "sha256": _tensor_hash(weight)}},
    }
    checkpoint = {
        "global_steps": 0,
        "ds_version": deepspeed.__version__,
        "ds_config": {},
        "module": {"embedding": weight, "head": weight.clone()},
        "shared_params": {"head": "embedding"},
    }
    monkeypatch.setattr(optimizer, "load_dense_checkpoint_model", lambda _: checkpoint)
    suite._initial_identity(tmp_path, expected)
    checkpoint["module"]["head"][0] = 4
    with pytest.raises(ValueError, match="shared parameter differs"):
        suite._initial_identity(tmp_path, expected)
    checkpoint["global_steps"] = 1
    with pytest.raises(ValueError, match="step zero"):
        suite._initial_identity(tmp_path, expected)


def test_reference_accumulation_dtype_tracks_recipe():
    from arctic_platform.catalog_correctness.reference import _accumulation_dtype

    assert _accumulation_dtype({}, "bfloat16") == "bfloat16"
    assert _accumulation_dtype({"data_types": {"grad_accum_dtype": "fp32"}}, "bfloat16") == "float32"
    with pytest.raises(ValueError, match="unsupported gradient"):
        _accumulation_dtype({"data_types": {"grad_accum_dtype": "float16"}}, "bfloat16")


def test_checkpoint_external_optimizer_settings_are_explicit():
    from arctic_platform.correctness.harness.optimizer import _config
    from arctic_platform.correctness.harness.optimizer import _snapshot

    config = {"zero_optimization": {"stage": 2}}
    with pytest.raises(ValueError, match="requires optimizer"):
        _config(config, "checkpoint")
    _config(config, "checkpoint", allow_external_optimizer=True)
    with pytest.raises(ValueError, match="zero-weight-decay AdamW"):
        _snapshot(
            Path("unused"),
            expected_optimizer_config={"name": "AdamW", "weight_decay": 0.1},
            expected_learning_rate=None,
            allow_external_optimizer_config=True,
        )
    with pytest.raises(ValueError, match="requires optimizer"):
        _config({**config, "optimizer": {"type": "SGD"}}, "checkpoint", allow_external_optimizer=True)


def test_optimizer_output_root_rejects_paths_outside_scratch(monkeypatch, tmp_path):
    from arctic_platform.correctness.harness import optimizer

    def reached_snapshot(*_args, **_kwargs):
        raise RuntimeError("snapshot reached")

    monkeypatch.setattr(optimizer, "_snapshot", reached_snapshot)
    with pytest.raises(RuntimeError, match="snapshot reached"):
        optimizer.decode_optimizer_capture(
            tmp_path / "before", tmp_path / "after", tmp_path / "output", output_root=tmp_path
        )
    with pytest.raises(ValueError, match="inside the supplied output root"):
        optimizer.decode_optimizer_capture(
            tmp_path / "before", tmp_path / "after", tmp_path.parent / "outside", output_root=tmp_path
        )


def test_reference_calibration_uses_same_parameter_ranges(tmp_path):
    import hashlib

    state = {"parameter_update_residual_norm": 0.0, "exp_avg_norm": 0.1, "exp_avg_sq_norm": 0.01}
    case = {"gradient_norms": {"weight": 1.0}, "optimizer_step": {"weight": state}}
    record = {
        "schema": "catalog-hf-reference-v2",
        "model": "example",
        "recipe_sha256": "recipe",
        "inputs_sha256": "inputs",
        "model_files_sha256": {"weights": "weights"},
        "reference_source_sha256": "source",
        "initial_parameters": {"weight": {"shape": [1], "sha256": "test"}},
        **case,
        "gas4": {**case, "gradient_accumulation_dtype": "bfloat16"},
    }
    paths = []
    for index in range(16):
        path = tmp_path / f"repeat-{index}.json"
        path.write_text(json.dumps(record))
        paths.append(path)
    changed = json.loads(paths[0].read_text())
    changed["gradient_norms"]["weight"] = 1.00026
    paths[0].write_text(json.dumps(changed))
    calibration = calibrate.calibrate(paths)
    gates = calibration["gates"]
    assert gates["gradient"]["absolute_tolerance"] == 0.002
    assert gates["optimizer"]["absolute_tolerance"] == 0.001
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    digest = hashlib.sha256(paths[0].read_bytes()).hexdigest()
    assert suite._read_calibration(calibration_path, changed, digest) == {"gradient": 0.002, "optimizer": 0.001}
    with pytest.raises(ValueError, match="does not match"):
        suite._read_calibration(calibration_path, changed, "wrong-digest")
    changed["model"] = "different"
    paths[0].write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="same workload"):
        calibrate.calibrate(paths)


def test_optimizer_comparison_detects_update_not_explained_by_moments(tmp_path):
    import torch

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": [{"name": "weight", "shape": [1], "file": "state.pt"}]}))
    state = {
        "parameter_update": torch.tensor([-1.0]),
        "exp_avg": torch.tensor([1.0]),
        "exp_avg_sq": torch.tensor([1.0]),
    }
    reference = {
        "optimizer": {"name": "AdamW", "lr": 1.0, "betas": [0.0, 0.0], "eps": 1e-8, "weight_decay": 0.0},
        "initial_parameters": {"weight": {"shape": [1]}},
        "optimizer_step": {
            "weight": {"parameter_update_residual_norm": 0.0, "exp_avg_norm": 1.0, "exp_avg_sq_norm": 1.0}
        },
    }
    torch.save(state, tmp_path / "state.pt")
    assert suite._optimizer_manifest_comparison(manifest, reference, 1e-6)["status"] == "pass"
    state["parameter_update"] = torch.tensor([0.0])
    torch.save(state, tmp_path / "state.pt")
    result = suite._optimizer_manifest_comparison(manifest, reference, 1e-6)
    assert result["status"] == "fail"
    assert result["measurements"]["parameter_update_residual_norm"]["max_absolute_difference"] == pytest.approx(1.0)


class ScoringClient:
    def generate(self, job, prompts, sampling_params, sub_job_id):
        assert sub_job_id == f"{job}:sampling:0"
        assert sampling_params["prompt_logprobs"] == 0
        self.prompts = prompts
        return "request"

    def poll_request(self, job, request):
        return {
            "results": [
                {"prompt_logprobs": [None] + [{str(token): {"logprob": -1.0}} for token in row[1:]]}
                for row in self.prompts
            ]
        }


def test_sampler_checks_every_requested_token():
    client = ScoringClient()
    assert checks.sampling_logprobs(client, "job", [[1, 2, 3], [4, 5]]) == [
        [-1.0, -1.0],
        [-1.0],
    ]


def test_sampler_rejects_missing_token():
    client = ScoringClient()
    client.poll_request = lambda job, request: {"results": [{"prompt_logprobs": [None, {}]}]}
    with pytest.raises(ValueError, match="omitted"):
        checks.sampling_logprobs(client, "job", [[1, 2]])


def test_training_step_encodes_global_token_weights():
    pytest.importorskip("cortex_training")
    import torch
    from cortex_training import wire

    class Client:
        def forward_backward(self, job, payload):
            frame = wire.loads(payload)
            assert frame["processing"]["loss_fn"] == "weighted_cross_entropy"
            weights = frame["context"]["logprob_weights_shifted"]
            torch_assert_close(weights, torch.tensor([[1 / 3, 1 / 3, 0], [1 / 3, 0, 0]]), atol=0, rtol=0)
            return "backward"

        def step(self, job):
            return "step"

        def poll_request(self, job, request):
            if request == "backward":
                return {
                    "metrics": {"rl_model_calls": 4},
                    "post_process_outputs": {"logprobs": torch.tensor([[-1.0, -2.0, 0], [-3.0, 0, 0]])},
                }
            return {"grad_norm": 1.0, "global_steps": 1}

    value, metrics = checks.train_step(Client(), "job", [[1, 2, 3], [1, 2]])
    assert value == pytest.approx(2.0)
    assert metrics["grad_norm"] == 1.0
    checks.train_step(Client(), "job", [[1, 2, 3], [1, 2]], expected_model_calls=4)
    with pytest.raises(ValueError, match="expected number of model calls"):
        checks.train_step(Client(), "job", [[1, 2, 3], [1, 2]], expected_model_calls=3)


def test_four_call_probe_preserves_backend_and_requires_reported_calls(monkeypatch):
    class Jobs:
        def start(self, body):
            self.body = body
            return "job"

    jobs = Jobs()
    body = recipe("training")
    body["sub_job_configs"][0]["training_config"]["attn_implementation"] = "model_backend"
    checkpoint_ids = iter(["initial", "step1"])
    monkeypatch.setattr(suite, "_save", lambda *_args: next(checkpoint_ids))

    def step(_client, _job, rows, *, expected_model_calls):
        assert expected_model_calls == 4
        assert rows == [[1, 2], [3, 4], [5, 6], [7, 8]]
        return 1.0, {"global_steps": 1, "gradient_norms_per_param": {"weight": 2.0}}

    monkeypatch.setattr(checks, "train_step", step)
    result = suite._gas4(None, jobs, body, [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]])
    assert result["initial"] == "initial" and result["step1"] == "step1"
    assert jobs.body["sub_job_configs"][0]["training_config"]["attn_implementation"] == "model_backend"
    assert jobs.body["sub_job_configs"][0]["training_config"]["mb_spec"]["max_tokens_per_mb"] == 2
    assert "mb_spec" not in body["sub_job_configs"][0]["training_config"]


def test_training_jobs_must_use_the_same_identified_image():
    class Jobs:
        runtime = {"first": {"image_tag": "image-a"}, "second": {"image_tag": "image-a"}}

    jobs = Jobs()
    suite._same_image(jobs, "first", "second")
    jobs.runtime["second"]["image_tag"] = "image-b"
    with pytest.raises(ValueError, match="same service image"):
        suite._same_image(jobs, "first", "second")


def test_sampler_job_uses_saved_checkpoint(monkeypatch):
    class Jobs:
        def start(self, body):
            self.body = body
            return "sample-job"

    jobs = Jobs()
    body = recipe("sampling")
    monkeypatch.setattr(checks, "sampling_logprobs", lambda *_args: [[-1.0, -2.0]])
    result = suite._sample(None, jobs, body, "train-job", "checkpoint", [[1, 2, 3]], [[-1.0, -2.0]], 0.0)
    assert result["status"] == "pass"
    assert jobs.body["sub_job_configs"][0]["source_checkpoint_info"] == {
        "source_job_id": "train-job",
        "checkpoint_id": "checkpoint",
    }
    assert "source_checkpoint_info" not in body["sub_job_configs"][0]


def test_full_resume_checks_control_restore_and_post_update(monkeypatch):
    class Client:
        step = 0
        restored_id = "checkpoint"

        def load(self, job, checkpoint, source_job_id):
            assert (job, checkpoint, source_job_id) == ("control", "checkpoint", "source")
            self.step = 10
            return "restore"

        def poll_request(self, job, request):
            return {"checkpoint_id": self.restored_id}

    class Jobs:
        runtime = {"source": {"image_tag": "image"}, "control": {"image_tag": "image"}}

        def start(self, body):
            return "control"

    client = Client()
    inputs = [[[index + 1, 0]] for index in range(20)]

    def step(client, job, batch):
        client.step += 1
        return float(batch[0][0]), {"global_steps": client.step}

    monkeypatch.setattr(checks, "train_step", step)
    monkeypatch.setattr(checks, "training_logprobs", lambda *_args: [[-2.0]])
    result = suite._resume(
        client, Jobs(), recipe("training"), "source", "checkpoint", inputs, list(range(1, 21)), 1.0, 0.0
    )
    assert result["status"] == "fail"
    assert result["post_update_difference"] == 1.0
    client.step = 0
    client.restored_id = "wrong"
    with pytest.raises(ValueError, match="different checkpoint"):
        suite._resume(client, Jobs(), recipe("training"), "source", "checkpoint", inputs, list(range(1, 21)), 1.0, 0.0)


@pytest.mark.parametrize("step", [None, 0, -1, 1.5, float("nan"), True])
def test_step_requires_a_finite_positive_integer(step):
    with pytest.raises(ValueError, match="optimizer step"):
        checks.step_number({"global_steps": step})


def test_job_is_tracked_when_startup_fails():
    class Client:
        def create_job_from_body(self, body):
            return {"job_id": "job"}

        def wait_for_job(self, job):
            raise RuntimeError("startup failed")

        def cancel_job(self, job):
            pass

        def get_job(self, job):
            return {"status": "cancelled"}

    jobs = checks.Jobs(Client(), lambda: None)
    with pytest.raises(RuntimeError, match="startup failed"):
        jobs.start(recipe("training"))
    assert jobs.active == ["job"]
    assert jobs.cleanup() == []
    assert jobs.active == []


def test_job_records_client_normalization_without_changing_the_recipe():
    class Client:
        def create_job_from_body(self, body):
            body["sub_job_configs"][0]["training_config"]["model_provider"] = "prime_rl"
            return {"job_id": "job"}

        def wait_for_job(self, job):
            return {"image_tag": "image"}

    body = recipe("training")
    body["sub_job_configs"][0]["training_config"]["model_provider"] = "arctic"
    jobs = checks.Jobs(Client(), lambda: None)
    jobs.start(body)
    assert body["sub_job_configs"][0]["training_config"]["model_provider"] == "arctic"
    assert jobs.runtime["job"]["client_normalized_fields"] == ["sub_job_configs[0].training_config.model_provider"]
    assert len(jobs.runtime["job"]["submitted_config_sha256"]) == 64
