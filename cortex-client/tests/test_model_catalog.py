from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

CLIENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CLIENT_ROOT.parent
sys.path.insert(0, str(CLIENT_ROOT / "scripts"))

import smoke_test_model_catalog as smoke_catalog  # noqa: E402
from smoke_test_model_catalog import build_profile_request  # noqa: E402
from smoke_test_model_catalog import build_forward_backward_probe_spec  # noqa: E402
from smoke_test_model_catalog import run_data_plane_probes  # noqa: E402
from validate_model_catalog import (  # noqa: E402
    CatalogValidationError,
    load_catalog,
    load_and_validate,
    validate_catalog,
)

CONFIG_DIR = CLIENT_ROOT / "configs" / "cortex-training"


def test_checked_in_catalog_is_valid():
    summary = load_and_validate(CONFIG_DIR, REPO_ROOT)

    assert summary == {
        "schemaVersion": 1,
        "lastReviewed": "2026-08-19",
        "models": 10,
        "profiles": 11,
    }


def test_catalog_context_limits_match_upstream_models():
    models_doc, _ = load_catalog(CONFIG_DIR)
    expected_limits = {
        "Qwen/Qwen3-0.6B": 32768,
        "Qwen/Qwen3-1.7B": 32768,
        "Qwen/Qwen3-8B": 32768,
        "Qwen/Qwen3.5-4B": 262144,
        "Qwen/Qwen3.6-35B-A3B": 262144,
        "Qwen/Qwen3.8-27B": 262144,
        "deepseek-ai/DeepSeek-V4-Flash-0731": 1048576,
        "openai/gpt-oss-120b": 131072,
        "zai-org/GLM-5.2": 1048576,
        "zai-org/GLM-5.2-FP8": 1048576,
    }

    for model in models_doc["models"]:
        expected = expected_limits[model["modelId"]]
        capabilities = model["capabilities"]
        assert capabilities["inference"]["maxContextTokens"] == expected
        if capabilities["training"]["supported"]:
            assert {
                profile["maxContextTokens"]
                for profile in capabilities["training"]["profiles"].values()
            } == {expected}


def test_qwen_38_live_smoke_uses_catalog_rl_lora_profile():
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.8-27B",
        "rlLora",
    )
    wire = request["sub_job_configs"]

    assert [sub_job["job_type"] for sub_job in wire] == ["training", "sampling"]
    assert {sub_job["model_name"] for sub_job in wire} == {"Qwen/Qwen3.8-27B"}
    assert [
        wire[0]["training_config"]["n_gpus"],
        wire[1]["inference_config"]["n_gpus"],
    ] == [8, 8]
    assert "peft_config" in wire[0]["training_config"]
    assert "peft_config" in wire[1]["inference_config"]


@pytest.mark.parametrize(
    ("model_id", "profile_key", "expected_max_context"),
    [
        ("Qwen/Qwen3-0.6B", "inference", 32768),
        ("Qwen/Qwen3.8-27B", "sftFull", 262144),
        ("deepseek-ai/DeepSeek-V4-Flash-0731", "inference", 1048576),
        ("openai/gpt-oss-120b", "inference", 131072),
        ("zai-org/GLM-5.2", "inference", 1048576),
    ],
)
def test_max_context_smoke_uses_advertised_model_limit(
    model_id,
    profile_key,
    expected_max_context,
):
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        model_id,
        profile_key,
        max_context=True,
    )

    assert {
        (
            sub_job.get("training_config") or sub_job.get("inference_config")
        )["max_seq_len"]
        for sub_job in request["sub_job_configs"]
    } == {expected_max_context}


@pytest.mark.parametrize("profile_key", ["sftFull", "rlFull"])
def test_dense_full_recommendations_use_zero_stage_two(profile_key):
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3.8-27B",
        profile_key,
    )
    training = next(
        sub_job["training_config"]
        for sub_job in request["sub_job_configs"]
        if sub_job["job_type"] == "training"
    )

    assert training["ds_config"]["zero_optimization"]["stage"] == 2


def test_forward_backward_probe_matches_catalog_training_batch_size():
    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3-0.6B",
        "sftLora",
    )

    spec = build_forward_backward_probe_spec(request)
    payload = spec["payload"]

    assert len(payload["input_ids"]) == 8
    assert payload["input_ids"][0] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert payload["position_ids"] == "arange"
    assert payload["labels"] == {
        "strategy": "next_token",
        "mask_padding": False,
    }


@pytest.mark.parametrize(
    ("profile_key", "expected_operations"),
    [
        ("inference", ["generate"]),
        ("sftLora", ["forward-backward"]),
        ("rlFull", ["generate", "forward-backward"]),
    ],
)
def test_live_smoke_executes_workflow_data_plane_probes(
    monkeypatch,
    profile_key,
    expected_operations,
):
    class FakeClient:
        def __init__(self):
            self.calls = []

        def generate(self, job_id, prompts, sampling_params):
            self.calls.append(("generate", job_id, prompts, sampling_params))
            return "generate-request"

        def forward_backward(self, job_id, payload):
            self.calls.append(("forward-backward", job_id, payload))
            return "forward-backward-request"

        def poll_request(self, job_id, request_id):
            self.calls.append(("poll", job_id, request_id))
            if request_id == "generate-request":
                return {"results": [{"text": "OK"}]}
            return {"avg_loss": 0.25}

    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3-0.6B",
        profile_key,
    )
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda _: b"forward-backward-payload",
    )
    client = FakeClient()

    probes = run_data_plane_probes(client, "job-1", profile_key, request)

    assert [probe["operation"] for probe in probes] == expected_operations
    if "generate" in expected_operations:
        assert (
            "generate",
            "job-1",
            ["Reply with OK."],
            {"max_tokens": 1, "temperature": 0.0},
        ) in client.calls
        assert ("poll", "job-1", "generate-request") in client.calls
    if "forward-backward" in expected_operations:
        assert (
            "forward-backward",
            "job-1",
            b"forward-backward-payload",
        ) in client.calls
        assert ("poll", "job-1", "forward-backward-request") in client.calls


def test_live_smoke_rejects_non_finite_forward_backward_loss(monkeypatch):
    class FakeClient:
        def forward_backward(self, job_id, payload):
            return "forward-backward-request"

        def poll_request(self, job_id, request_id):
            return {"avg_loss": float("nan")}

    request = build_profile_request(
        CONFIG_DIR,
        REPO_ROOT,
        "Qwen/Qwen3-0.6B",
        "sftFull",
    )
    monkeypatch.setattr(
        smoke_catalog,
        "_build_forward_backward_probe_payload",
        lambda _: b"forward-backward-payload",
    )

    with pytest.raises(RuntimeError, match="non-finite avg_loss"):
        run_data_plane_probes(FakeClient(), "job-1", "sftFull", request)


def test_duplicate_model_id_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"].append(copy.deepcopy(models_doc["models"][0]))

    with pytest.raises(CatalogValidationError, match="duplicate model id"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_missing_recommended_profile_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    models_doc["models"][0]["capabilities"]["inference"][
        "recommendedProfileId"
    ] = "missing"

    with pytest.raises(CatalogValidationError, match="unknown profile missing"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_profile_context_cannot_exceed_capability_limit():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "dense-sft-full-8gpu")
    profile["subJobs"][0]["args"]["max_seq_len"] = 32769

    with pytest.raises(CatalogValidationError, match="exceeds max context 32768"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_profile_gpu_count_must_be_a_multiple_of_eight():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "inference-8gpu")
    profile["subJobs"][0]["args"]["n_gpus"] = 4

    with pytest.raises(CatalogValidationError, match="n_gpus must be a multiple of 8"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_zero_stage_three_is_rejected():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    profile = next(item for item in profiles if item["id"] == "dense-sft-full-8gpu")
    zero_optimization = profile["subJobs"][0]["args"]["extra_training"]["ds_config"][
        "zero_optimization"
    ]
    zero_optimization["stage"] = 3

    with pytest.raises(CatalogValidationError, match="ZeRO stage 3 is unsupported"):
        validate_catalog(models_doc, profiles, REPO_ROOT)


def test_unsupported_capability_cannot_carry_configuration():
    models_doc, profiles = load_catalog(CONFIG_DIR)
    capability = models_doc["models"][-1]["capabilities"]["training"]
    capability["profiles"] = copy.deepcopy(
        models_doc["models"][0]["capabilities"]["training"]["profiles"]
    )

    with pytest.raises(
        CatalogValidationError,
        match="unsupported capabilities cannot define profiles",
    ):
        validate_catalog(models_doc, profiles, REPO_ROOT)
