"""Unit tests for ``dss_client.neutrino_client``.

Two suites:

- Pure-logic tests for the typed config dataclasses (``JobType``,
  ``TrainingConfig``, ``InferenceConfig``, ``SubJobConfig``): construction,
  validation (mirrors the Control Plane validators), and ``to_wire``
  serialization (matches the GS yaml shape).

- HTTP-surface tests for ``NeutrinoClient`` that stub ``client._session`` with
  a :class:`unittest.mock.MagicMock`, so every test runs offline.
"""

from __future__ import annotations

import base64
import importlib
import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

# Import the module directly so we don't pull dss_client/__init__.py (which
# imports torch via engine.py).
spec = importlib.util.spec_from_file_location(
    "neutrino_client_under_test",
    str(__import__("pathlib").Path(__file__).resolve().parent.parent
        / "dss_client" / "neutrino_client.py"),
)
nc = importlib.util.module_from_spec(spec)
sys.modules["neutrino_client_under_test"] = nc
spec.loader.exec_module(nc)

JobType = nc.JobType
TrainingConfig = nc.TrainingConfig
InferenceConfig = nc.InferenceConfig
SubJobConfig = nc.SubJobConfig
NeutrinoClient = nc.NeutrinoClient


def _wire_load(data: bytes):
    from dss_client import wire

    return wire.loads(data)


# ─── Helpers ────────────────────────────────────────────────────────────


def _ok_training() -> TrainingConfig:
    return TrainingConfig(
        optimizer={"type": "adamw", "lr": 1e-5},
        max_seq_len=128,
        train_batch_size=1,
        n_gpus=2,
    )


def _ok_inference() -> InferenceConfig:
    return InferenceConfig(max_seq_len=2048, n_gpus=1)


def _make_response(json_body=None, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    resp.raise_for_status = MagicMock()
    return resp


def _make_client(post_json=None, get_json=None) -> NeutrinoClient:
    client = NeutrinoClient(
        base_url="http://test.local",
        database="DB",
        schema="SCH",
    )
    client._session = MagicMock()
    client._session.post.return_value = _make_response(post_json)
    client._session.get.return_value = _make_response(get_json)
    return client


# ─── JobType ────────────────────────────────────────────────────────────


class TestJobType:
    def test_string_values(self):
        assert JobType.TRAINING.value == "training"
        assert JobType.SAMPLING.value == "sampling"
        assert JobType.LOG_PROBABILITY.value == "log_probability"

    def test_enum_is_str_subclass(self):
        # JSON serialization treats it as the literal string.
        assert JobType.TRAINING == "training"


# ─── TrainingConfig ─────────────────────────────────────────────────────


class TestTrainingConfig:
    def test_validate_ok(self):
        _ok_training().validate()

    def test_validate_rejects_zero_max_seq_len(self):
        tc = _ok_training()
        tc.max_seq_len = 0
        with pytest.raises(ValueError, match="max_seq_len"):
            tc.validate()

    def test_validate_rejects_zero_train_batch_size(self):
        tc = _ok_training()
        tc.train_batch_size = 0
        with pytest.raises(ValueError, match="train_batch_size"):
            tc.validate()

    def test_validate_rejects_empty_optimizer(self):
        tc = _ok_training()
        tc.optimizer = {}
        with pytest.raises(ValueError, match="optimizer"):
            tc.validate()

    def test_validate_accepts_nested_primerl_fused_ce_false(self):
        tc = _ok_training()
        tc.extra = {
            "fp32_lm_head": True,
            "fused_lm_head_token_chunk_size": 8192,
            "prime_rl": {
                "fused_cross_entropy": False,
                "fp32_lm_head": True,
                "fused_lm_head_token_chunk_size": 8192,
            },
        }
        tc.validate()

    def test_validate_nested_primerl_overrides_top_level_fused_ce(self):
        tc = _ok_training()
        tc.extra = {
            "fused_cross_entropy": "liger",
            "fp32_lm_head": True,
            "prime_rl": {
                "fused_cross_entropy": False,
                "fp32_lm_head": True,
            },
        }
        tc.validate()

    def test_validate_rejects_default_fused_ce_with_chunked_lm_head(self):
        tc = _ok_training()
        tc.extra = {"fused_lm_head_token_chunk_size": 8192}
        with pytest.raises(ValueError, match="cannot combine fused_cross_entropy"):
            tc.validate()

    def test_validate_rejects_nested_primerl_fused_ce_with_fp32_lm_head(self):
        tc = _ok_training()
        tc.extra = {
            "fused_cross_entropy": False,
            "prime_rl": {
                "fused_cross_entropy": "liger",
                "fp32_lm_head": True,
            },
        }
        with pytest.raises(ValueError, match="cannot combine fused_cross_entropy"):
            tc.validate()

    def test_to_wire_required_fields(self):
        wire = _ok_training().to_wire()
        assert wire == {
            "optimizer": {"type": "adamw", "lr": 1e-5},
            "max_seq_len": 128,
            "train_batch_size": 1,
            "n_gpus": 2,
        }

    def test_to_wire_includes_gradient_clipping_when_set(self):
        tc = _ok_training()
        tc.gradient_clipping = 1.5
        assert tc.to_wire()["gradient_clipping"] == 1.5

    def test_to_wire_omits_gradient_clipping_when_none(self):
        assert "gradient_clipping" not in _ok_training().to_wire()

    def test_to_wire_always_includes_n_gpus(self):
        assert _ok_training().to_wire()["n_gpus"] == 2

    def test_validate_rejects_zero_n_gpus(self):
        tc = _ok_training()
        tc.n_gpus = 0
        with pytest.raises(ValueError, match="n_gpus"):
            tc.validate()

    def test_to_wire_includes_multiplex_job_id_when_set(self):
        tc = _ok_training()
        tc.multiplex_job_id = "train-1"
        assert tc.to_wire()["multiplex_job_id"] == "train-1"

    def test_to_wire_omits_multiplex_job_id_when_none(self):
        assert "multiplex_job_id" not in _ok_training().to_wire()

    def test_to_wire_omits_load_optimizer_states_when_none(self):
        assert "load_optimizer_states" not in _ok_training().to_wire()

    def test_to_wire_includes_load_optimizer_states_when_false(self):
        tc = _ok_training()
        tc.load_optimizer_states = False
        assert tc.to_wire()["load_optimizer_states"] is False

    def test_to_wire_includes_load_optimizer_states_when_true(self):
        tc = _ok_training()
        tc.load_optimizer_states = True
        assert tc.to_wire()["load_optimizer_states"] is True

    def test_to_wire_merges_extra_passthrough(self):
        tc = _ok_training()
        tc.extra = {"fp16": {"enabled": True}, "zero_stage": 2}
        wire = tc.to_wire()
        assert wire["fp16"] == {"enabled": True}
        assert wire["zero_stage"] == 2

    def test_to_wire_extra_does_not_override_required(self):
        # If a caller stuffs a required field name into `extra`, the typed
        # required value wins. Mirrors setdefault semantics.
        tc = _ok_training()
        tc.extra = {"max_seq_len": 999, "optimizer": {"type": "sgd"}}
        wire = tc.to_wire()
        assert wire["max_seq_len"] == 128
        assert wire["optimizer"] == {"type": "adamw", "lr": 1e-5}


# ─── InferenceConfig ────────────────────────────────────────────────────


class TestInferenceConfig:
    def test_validate_ok(self):
        _ok_inference().validate()

    def test_validate_rejects_zero(self):
        with pytest.raises(ValueError, match="max_seq_len"):
            InferenceConfig(max_seq_len=0, n_gpus=1).validate()

    def test_validate_rejects_zero_n_gpus(self):
        with pytest.raises(ValueError, match="n_gpus"):
            InferenceConfig(max_seq_len=2048, n_gpus=0).validate()

    def test_to_wire_includes_extra(self):
        ic = InferenceConfig(max_seq_len=4096, n_gpus=1, extra={"gpu_memory_utilization": 0.9})
        assert ic.to_wire() == {"max_seq_len": 4096, "n_gpus": 1, "gpu_memory_utilization": 0.9}

    def test_to_wire_always_includes_n_gpus(self):
        assert _ok_inference().to_wire()["n_gpus"] == 1

    def test_to_wire_includes_multiplex_job_id_when_set(self):
        ic = InferenceConfig(max_seq_len=4096, n_gpus=1, multiplex_job_id="train-1")
        assert ic.to_wire()["multiplex_job_id"] == "train-1"

    def test_to_wire_omits_multiplex_job_id_when_none(self):
        assert "multiplex_job_id" not in _ok_inference().to_wire()

    def test_to_wire_extra_does_not_override_required(self):
        ic = InferenceConfig(max_seq_len=4096, n_gpus=1, extra={"max_seq_len": 1})
        assert ic.to_wire()["max_seq_len"] == 4096


# ─── SubJobConfig — factories ───────────────────────────────────────────


class TestSubJobConfigFactories:
    def test_training_job_factory_minimal(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert sub.job_type == JobType.TRAINING
        assert sub.model_name == "gpt2"
        assert isinstance(sub.training, TrainingConfig)
        assert sub.sampling is None
        assert sub.training.extra == {}

    def test_training_job_factory_full(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=4,
            gradient_clipping=1.0,
            multiplex_job_id="train-1",
            extra_training={"fp16": {"enabled": True}},
            global_batch_size=4,
            dtype="bf16",
            seed=42,
            model_post_init=["init_a", "init_b"],
            load_optimizer_states=False,
            source_checkpoint_info={"checkpoint_id": "cp_1", "source_job_id": "job-a"},
        )
        assert sub.training.gradient_clipping == 1.0
        assert sub.training.n_gpus == 4
        assert sub.training.multiplex_job_id == "train-1"
        assert sub.training.load_optimizer_states is False
        assert sub.training.to_wire()["load_optimizer_states"] is False
        assert sub.training.extra == {"fp16": {"enabled": True}}
        assert sub.global_batch_size == 4
        assert sub.dtype == "bf16"
        assert sub.seed == 42
        assert sub.model_post_init == ["init_a", "init_b"]
        assert sub.source_checkpoint_info == {"checkpoint_id": "cp_1", "source_job_id": "job-a"}

    def test_sampling_job_factory_default_type(self):
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        assert sub.job_type == JobType.SAMPLING
        assert isinstance(sub.sampling, InferenceConfig)
        assert sub.training is None

    def test_sampling_job_factory_optional_sizing(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=2,
            multiplex_job_id="train-1",
        )
        assert sub.sampling.n_gpus == 2
        assert sub.sampling.multiplex_job_id == "train-1"

    def test_sampling_job_factory_source_checkpoint(self):
        source = {"checkpoint_id": "cp_1", "source_job_id": "job-a"}
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=1,
            source_checkpoint_info=source,
        )
        assert sub.source_checkpoint_info == source

    def test_sampling_job_factory_log_probability(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2", max_seq_len=128, n_gpus=1, job_type=JobType.LOG_PROBABILITY,
        )
        assert sub.job_type == JobType.LOG_PROBABILITY

    def test_sampling_job_factory_rejects_training_type(self):
        with pytest.raises(ValueError, match="SAMPLING or LOG_PROBABILITY"):
            SubJobConfig.sampling_job(
                model_name="gpt2", max_seq_len=128, n_gpus=1, job_type=JobType.TRAINING,
            )

    def test_extra_dicts_are_copied(self):
        # Mutating the caller's dict after construction must not leak into the
        # built sub-job (defensive copy).
        extra = {"fp16": {"enabled": True}}
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            extra_training=extra,
        )
        extra["leaked"] = "yes"
        assert "leaked" not in sub.training.extra


# ─── SubJobConfig — validate ────────────────────────────────────────────


class TestSubJobConfigValidate:
    def test_ok_training(self):
        SubJobConfig(
            job_type=JobType.TRAINING, model_name="gpt2", training=_ok_training(),
        ).validate()

    def test_ok_sampling(self):
        SubJobConfig(
            job_type=JobType.SAMPLING, model_name="gpt2", sampling=_ok_inference(),
        ).validate()

    def test_ok_log_probability_uses_sampling_block(self):
        SubJobConfig(
            job_type=JobType.LOG_PROBABILITY,
            model_name="gpt2",
            sampling=_ok_inference(),
        ).validate()

    def test_rejects_empty_model_name(self):
        with pytest.raises(ValueError, match="model_name"):
            SubJobConfig(
                job_type=JobType.TRAINING, model_name="", training=_ok_training(),
            ).validate()

    def test_training_without_training_block(self):
        with pytest.raises(ValueError, match="training sub-job requires"):
            SubJobConfig(job_type=JobType.TRAINING, model_name="gpt2").validate()

    def test_sampling_without_sampling_block(self):
        with pytest.raises(ValueError, match="sampling sub-job requires"):
            SubJobConfig(job_type=JobType.SAMPLING, model_name="gpt2").validate()

    def test_log_probability_without_sampling_block(self):
        with pytest.raises(ValueError, match="log_probability sub-job requires"):
            SubJobConfig(
                job_type=JobType.LOG_PROBABILITY, model_name="gpt2",
            ).validate()

    def test_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            SubJobConfig(
                job_type=JobType.TRAINING,
                model_name="gpt2",
                training=_ok_training(),
                sampling=_ok_inference(),
            ).validate()

    def test_propagates_nested_validation_failure(self):
        bad = SubJobConfig(
            job_type=JobType.TRAINING,
            model_name="gpt2",
            training=TrainingConfig(
                optimizer={"type": "adamw"}, max_seq_len=128, train_batch_size=0, n_gpus=2,
            ),
        )
        with pytest.raises(ValueError, match="train_batch_size"):
            bad.validate()


# ─── SubJobConfig — to_wire ─────────────────────────────────────────────


class TestSubJobConfigToWire:
    def test_training_minimal(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        assert sub.to_wire() == {
            "job_type": "training",
            "model_name": "gpt2",
            "training_config": {
                "optimizer": {"type": "adamw"},
                "max_seq_len": 128,
                "train_batch_size": 1,
                "n_gpus": 2,
            },
        }

    def test_training_with_optionals(self):
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
            global_batch_size=4,
            dtype="bf16",
            seed=42,
            model_post_init=["init"],
            source_checkpoint_info={"checkpoint_id": "cp_1", "source_job_id": "job-a"},
        )
        wire = sub.to_wire()
        assert wire["global_batch_size"] == 4
        assert wire["model_post_init"] == ["init"]
        assert wire["dtype"] == "bf16"
        assert wire["seed"] == 42
        assert wire["source_checkpoint_info"] == {"checkpoint_id": "cp_1", "source_job_id": "job-a"}

    def test_sampling_block_used_for_log_probability(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2", max_seq_len=128, n_gpus=1, job_type=JobType.LOG_PROBABILITY,
        )
        wire = sub.to_wire()
        assert wire["job_type"] == "log_probability"
        assert wire["inference_config"] == {"max_seq_len": 128, "n_gpus": 1}
        assert "training_config" not in wire

    def test_sampling_with_source_checkpoint(self):
        sub = SubJobConfig.sampling_job(
            model_name="gpt2",
            max_seq_len=128,
            n_gpus=1,
            source_checkpoint_info={
                "checkpoint_id": "cp_1",
                "source_job_id": "job-a",
            },
        )
        assert sub.to_wire()["source_checkpoint_info"] == {
            "checkpoint_id": "cp_1",
            "source_job_id": "job-a",
        }

    def test_omits_unset_optional_fields(self):
        sub = SubJobConfig(
            job_type=JobType.TRAINING, model_name="gpt2", training=_ok_training(),
        )
        wire = sub.to_wire()
        for absent in ("global_batch_size", "source_checkpoint_info",
                       "dtype", "seed", "model_post_init", "inference_config"):
            assert absent not in wire


# ─── NeutrinoClient — auth & URL composition ────────────────────────────


class TestClientConstruction:
    def test_init_strips_trailing_slash(self):
        c = NeutrinoClient(base_url="http://x.test/", database="DB", schema="SCH")
        assert c.base_url == "http://x.test"

    def test_default_polling_config(self):
        c = NeutrinoClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c.poll_interval == 0.5
        assert c.poll_timeout == 1800.0
        assert c.poll_backoff_multiplier == 1.25
        assert c.poll_max_interval == 6.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"poll_interval": 0}, "poll_interval"),
            ({"poll_timeout": 0}, "poll_timeout"),
            ({"poll_backoff_multiplier": 0.5}, "poll_backoff_multiplier"),
            ({"poll_interval": 2, "poll_max_interval": 1}, "poll_max_interval"),
        ],
    )
    def test_rejects_invalid_polling_config(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            NeutrinoClient(
                base_url="http://x.test",
                database="DB",
                schema="SCH",
                **kwargs,
            )

    def test_default_endpoint_is_cortex_training(self):
        c = NeutrinoClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c.endpoint == "cortex-training"

    def test_prefix_url(self):
        c = NeutrinoClient(base_url="http://x.test", database="DB", schema="SCH")
        assert c._prefix == "http://x.test/api/v2/databases/DB/schemas/SCH/cortex-training"

    def test_from_pat_sets_headers_and_verify(self):
        c = NeutrinoClient.from_pat(
            host="x.test", pat="tok-xyz", database="DB", schema="SCH",
            verify_ssl=False,
        )
        assert c.base_url == "https://x.test"
        assert c._session.headers["Authorization"] == "Bearer tok-xyz"
        assert c._session.headers["X-Snowflake-Authorization-Token-Type"] == "PROGRAMMATIC_ACCESS_TOKEN"
        assert c._session.verify is False

    def test_no_legacy_auth_methods(self):
        # Production client only supports PAT.
        assert not hasattr(NeutrinoClient, "from_snowflake_login")
        # And no legacy builder helpers.
        assert not hasattr(NeutrinoClient, "build_training_sub_job")
        assert not hasattr(NeutrinoClient, "build_sampling_sub_job")
        assert not hasattr(NeutrinoClient, "create_training_engine")


# ─── NeutrinoClient — create_job ────────────────────────────────────────


class TestCreateJob:
    def test_empty_list_raises(self):
        c = _make_client()
        with pytest.raises(ValueError, match="non-empty"):
            c.create_job(sub_jobs=[])

    def test_validates_each_sub_job_before_post(self):
        c = _make_client()
        bad = SubJobConfig(job_type=JobType.TRAINING, model_name="gpt2")  # no training block
        with pytest.raises(ValueError):
            c.create_job(sub_jobs=[bad])
        c._session.post.assert_not_called()

    def test_posts_in_flight_yaml_shape(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"type": "adamw"},
            max_seq_len=128,
            train_batch_size=1,
            n_gpus=2,
        )
        job_id = c.create_job(sub_jobs=[sub])
        assert job_id == "srv-1"
        url, kwargs = c._session.post.call_args
        assert url[0] == c._prefix
        assert kwargs["json"] == {
            "sub_job_configs": [{
                "job_type": "training",
                "model_name": "gpt2",
                "training_config": {
                    "optimizer": {"type": "adamw"},
                    "max_seq_len": 128,
                    "train_batch_size": 1,
                    "n_gpus": 2,
                },
            }],
        }

    def test_includes_job_id_when_given(self):
        c = _make_client(post_json={"job_id": "client-chosen"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], job_id="client-chosen")
        body = c._session.post.call_args.kwargs["json"]
        assert body["job_id"] == "client-chosen"

    def test_omits_job_id_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "job_id" not in body

    def test_includes_experiment_name_when_given(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub], experiment_name="my-experiment")
        body = c._session.post.call_args.kwargs["json"]
        assert body["experiment_name"] == "my-experiment"

    def test_omits_experiment_name_when_none(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        sub = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[sub])
        body = c._session.post.call_args.kwargs["json"]
        assert "experiment_name" not in body

    def test_supports_multiple_sub_jobs(self):
        c = _make_client(post_json={"job_id": "srv-1"})
        train = SubJobConfig.training_job(
            model_name="gpt2",
            optimizer={"a": 1}, max_seq_len=128, train_batch_size=1, n_gpus=2,
        )
        samp = SubJobConfig.sampling_job(model_name="gpt2", max_seq_len=128, n_gpus=1)
        c.create_job(sub_jobs=[train, samp])
        body = c._session.post.call_args.kwargs["json"]
        assert len(body["sub_job_configs"]) == 2
        assert body["sub_job_configs"][0]["job_type"] == "training"
        assert body["sub_job_configs"][1]["job_type"] == "sampling"

    def test_create_job_from_body_posts_raw_body(self):
        c = _make_client(post_json={"job_id": "srv-raw"})
        body = {
            "job_id": "client-raw",
            "sub_job_configs": [{
                "job_type": "sampling",
                "model_name": "gpt2",
                "inference_config": {"max_seq_len": 128, "n_gpus": 1},
            }],
        }
        assert c.create_job_from_body(body) == {"job_id": "srv-raw"}
        c._session.post.assert_called_once_with(c._prefix, json=body)

    def test_create_job_from_body_requires_object(self):
        c = _make_client()
        with pytest.raises(ValueError, match="JSON object"):
            c.create_job_from_body([])

    def test_create_job_from_body_requires_sub_job_configs(self):
        c = _make_client()
        with pytest.raises(ValueError, match="sub_job_configs"):
            c.create_job_from_body({"sub_job_configs": []})

    def test_create_job_from_body_rejects_debug_without_env(self, monkeypatch):
        monkeypatch.delenv(nc.DEBUG_OPTIONS_ENV, raising=False)
        c = _make_client(post_json={"job_id": "srv"})
        body = {
            "sub_job_configs": [{
                "job_type": "training",
                "model_name": "gpt2",
                "training_config": {"n_gpus": 1},
            }],
            "debug": {"job": {"image_tag": "release_internal"}},
        }
        with pytest.raises(ValueError, match=nc.DEBUG_OPTIONS_ENV):
            c.create_job_from_body(body)
        # The gate must short-circuit before any request is sent.
        c._session.post.assert_not_called()

    def test_create_job_from_body_allows_debug_when_env_set(self, monkeypatch):
        monkeypatch.setenv(nc.DEBUG_OPTIONS_ENV, "1")
        c = _make_client(post_json={"job_id": "srv-dbg"})
        body = {
            "sub_job_configs": [{
                "job_type": "training",
                "model_name": "gpt2",
                "training_config": {"n_gpus": 1},
            }],
            "debug": {"job": {"image_tag": "release_internal"}},
        }
        assert c.create_job_from_body(body) == {"job_id": "srv-dbg"}
        c._session.post.assert_called_once_with(c._prefix, json=body)

    def test_create_job_from_body_allows_empty_debug_without_env(self, monkeypatch):
        # A falsy/empty debug block carries no directives, so it is not gated.
        monkeypatch.delenv(nc.DEBUG_OPTIONS_ENV, raising=False)
        c = _make_client(post_json={"job_id": "srv"})
        body = {
            "sub_job_configs": [{
                "job_type": "training",
                "model_name": "gpt2",
                "training_config": {"n_gpus": 1},
            }],
            "debug": {},
        }
        assert c.create_job_from_body(body) == {"job_id": "srv"}


# ─── NeutrinoClient — read & control endpoints ─────────────────────────


class TestReadAndControl:
    def test_normalize_job_status(self):
        assert NeutrinoClient._normalize_job_status("JOB_STATE_RUNNING") == "running"
        assert NeutrinoClient._normalize_job_status("running") == "running"
        assert NeutrinoClient._normalize_job_status("JOB_STATE_FAILED") == "failed"

    def test_normalize_job_status_tolerates_unknown_and_non_string(self):
        # A CP ahead of GS can surface a JobState enum GS doesn't know, which GS
        # renders as a raw integer. Must not crash on int.lower(); an unknown
        # value normalizes to a harmless non-terminal string.
        assert NeutrinoClient._normalize_job_status(12) == "12"
        assert NeutrinoClient._normalize_job_status(None) == ""
        assert NeutrinoClient._normalize_job_status("JOB_STATE_INITIALIZING") == "initializing"

    def test_get_job(self):
        c = _make_client(get_json={"job_id": "j1", "status": "running"})
        out = c.get_job("j1")
        assert out["job_id"] == "j1"
        c._session.get.assert_called_once_with(f"{c._prefix}/j1")

    def test_list_jobs_no_filter(self):
        c = _make_client(get_json={"jobs": [{"job_id": "a"}, {"job_id": "b"}]})
        jobs = c.list_jobs()
        assert [j["job_id"] for j in jobs] == ["a", "b"]
        c._session.get.assert_called_once_with(c._prefix, params={})

    def test_list_jobs_with_status(self):
        c = _make_client(get_json={"jobs": []})
        c.list_jobs(status="running")
        c._session.get.assert_called_once_with(c._prefix, params={"status": "running"})

    def test_cancel_job_uses_colon_action(self):
        c = _make_client(post_json={})
        c.cancel_job("j1")
        c._session.post.assert_called_once_with(f"{c._prefix}/j1:cancel")

    def test_get_capacity_reserved_account(self):
        c = _make_client(get_json={
            "has_reservation": True,
            "reserved_gpus": 64,
            "in_use_gpus": 8,
            "available_gpus": 56,
        })
        cap = c.get_capacity()
        assert cap == {
            "has_reservation": True,
            "reserved_gpus": 64,
            "in_use_gpus": 8,
            "available_gpus": 56,
        }
        c._session.get.assert_called_once_with(f"{c._prefix}/capacity")

    def test_get_capacity_fills_proto3_omitted_defaults(self):
        # proto3 JSON omits zero/false fields; an unreserved account is `{}`.
        c = _make_client(get_json={})
        cap = c.get_capacity()
        assert cap == {
            "has_reservation": False,
            "reserved_gpus": 0,
            "in_use_gpus": 0,
            "available_gpus": 0,
        }

    def test_get_capacity_fills_partial_omitted_fields(self):
        # Fully-drained reservation: only the non-zero reserved_gpus is present.
        c = _make_client(get_json={"has_reservation": True, "reserved_gpus": 8})
        cap = c.get_capacity()
        assert cap == {
            "has_reservation": True,
            "reserved_gpus": 8,
            "in_use_gpus": 0,
            "available_gpus": 0,
        }


# ─── NeutrinoClient — wait_for_job ─────────────────────────────────────


class TestWaitForJob:
    def test_returns_when_running(self, monkeypatch):
        c = _make_client(get_json={"status": "JOB_STATE_RUNNING"})
        sleep = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", sleep)
        assert c.wait_for_job("j1")["status"] == "JOB_STATE_RUNNING"
        sleep.assert_not_called()

    def test_uses_exponential_backoff_until_running(self, monkeypatch):
        c = _make_client()
        c.poll_interval = 1.0
        c.poll_backoff_multiplier = 2.0
        c.poll_max_interval = 3.0
        c._session.get.side_effect = [
            _make_response({"status": "pending"}),
            _make_response({"status": "pending"}),
            _make_response({"status": "pending"}),
            _make_response({"status": "running"}),
        ]
        sleeps = []
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)

        assert c.wait_for_job("j1")["status"] == "running"
        assert sleeps == [1.0, 2.0, 3.0]

    def test_backoff_sleep_does_not_exceed_remaining_timeout(self, monkeypatch):
        c = _make_client()
        c.poll_backoff_multiplier = 2.0
        c.poll_max_interval = 30.0
        sleeps = []
        monkeypatch.setattr(nc.time, "monotonic", lambda: 95.0)
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)

        assert c._sleep_with_backoff(10.0, deadline=100.0) == 20.0
        assert sleeps == [5.0]

    def test_raises_on_terminal(self, monkeypatch):
        c = _make_client(get_json={"status": "failed", "reason": "boom"})
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(RuntimeError, match="boom"):
            c.wait_for_job("j1")

    def test_times_out(self, monkeypatch):
        c = _make_client(get_json={"status": "pending"})
        c.poll_timeout = 0.01
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(TimeoutError):
            c.wait_for_job("j1")

    def test_polls_through_unknown_numeric_status(self, monkeypatch):
        # GS running behind a CP that emits a new JobState renders it as a raw
        # integer (e.g. 12 for a state GS doesn't know). wait_for_job must treat
        # it as still-in-progress and keep polling until RUNNING — not crash on
        # int.lower() and not mistake it for a terminal state.
        c = _make_client()
        c._session.get.side_effect = [
            _make_response({"status": 12}),  # unknown enum, rendered as int
            _make_response({"status": "running"}),
        ]
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.wait_for_job("j1")["status"] == "running"


# ─── NeutrinoClient — data-plane ───────────────────────────────────────


class TestForwardBackwardPayloadHelpers:
    def test_build_forward_backward_payload_from_tensor_kwargs(self):
        body = nc.build_forward_backward_payload({
            "payload": {
                "kwargs": {
                    "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
                    "labels": {"data": [[2, 3, -100]], "dtype": "long"},
                },
            },
        })

        loaded = _wire_load(body)
        assert loaded["args"] == ()
        assert loaded["kwargs"]["input_ids"].tolist() == [[1, 2, 3]]
        assert loaded["kwargs"]["labels"].tolist() == [[2, 3, -100]]

    def test_tokenized_payload_builds_prime_rl_style_kwargs(self, monkeypatch):
        import torch

        class FakeTokenizer:
            pad_token_id = None
            eos_token = "<eos>"

            def __call__(self, texts, **kwargs):
                assert texts == ["first", "second"]
                assert kwargs["padding"] == "max_length"
                assert kwargs["max_length"] == 4
                return {
                    "input_ids": torch.tensor([
                        [10, 11, 0, 0],
                        [20, 21, 22, 0],
                    ]),
                    "attention_mask": torch.tensor([
                        [1, 1, 0, 0],
                        [1, 1, 1, 0],
                    ]),
                }

        fake_transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(
                from_pretrained=lambda model_name, **kwargs: FakeTokenizer()
            )
        )
        monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

        kwargs = nc.build_forward_backward_kwargs({
            "tokenizer": {"model_name": "fake-model"},
            "texts": ["first", "second"],
            "batch_size": 2,
            "max_length": 4,
            "position_ids": "arange",
            "labels": {"strategy": "next_token", "ignore_index": -100},
        })

        assert kwargs["input_ids"].tolist() == [
            [10, 11, 0, 0],
            [20, 21, 22, 0],
        ]
        assert kwargs["position_ids"].tolist() == [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ]
        assert kwargs["labels"].tolist() == [
            [11, -100, -100, -100],
            [21, 22, -100, -100],
        ]
        assert "attention_mask" not in kwargs


class TestDataPlane:
    def test_forward_backward(self):
        c = _make_client(post_json={"request_id": "r1"})
        rid = c.forward_backward("j1", b"\x01\x02")
        assert rid == "r1"
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/forward-backward"
        assert kwargs["data"] == b"\x01\x02"
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"

    def test_forward_backward_chunks_oversized_payload(self):
        c = _make_client()
        c._MAX_FWD_BWD_BYTES = 64 * 1024
        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            _make_response({"request_id": "r1"}),
        ]
        oversized = b"x" * (70 * 1024)

        assert c.forward_backward("j1", oversized) == "r1"
        assert c._session.post.call_count == 2
        first = c._session.post.call_args_list[0].kwargs["data"]
        from dss_client import wire

        assert wire.read_byte_chunk_metadata(first)["operation"] == "fwd-bwd"

    def test_generate_minimal_omits_optionals(self):
        c = _make_client(post_json={"request_id": "g1"})
        rid = c.generate("j1", prompts=["hello"])
        assert rid == "g1"
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/generate"
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
        body = _wire_load(kwargs["data"])
        assert body == {"prompts": ["hello"]}

    def test_generate_full_payload(self):
        c = _make_client(post_json={"request_id": "g2"})
        rid = c.generate(
            "j1",
            prompts=["a", "b"],
            sampling_params={"max_tokens": 4, "temperature": 0.7},
            routing_key="rk-1",
            strict=True,
        )
        assert rid == "g2"
        kwargs = c._session.post.call_args.kwargs
        body = _wire_load(kwargs["data"])
        assert body == {
            "prompts": ["a", "b"],
            "sampling_params": {"max_tokens": 4, "temperature": 0.7},
            "routing_key": "rk-1",
            "strict": True,
        }
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"

    def test_generate_accepts_per_prompt_sampling_params(self):
        c = _make_client(post_json={"request_id": "g-list"})
        rid = c.generate(
            "j1",
            prompts=["a", "b"],
            sampling_params=[
                {"max_tokens": 4, "temperature": 0.7},
                None,
            ],
            routing_key=["rk-1", None],
        )
        assert rid == "g-list"
        body = _wire_load(c._session.post.call_args.kwargs["data"])
        assert body == {
            "prompts": ["a", "b"],
            "sampling_params": [
                {"max_tokens": 4, "temperature": 0.7},
                None,
            ],
            "routing_key": ["rk-1", None],
        }

    def test_generate_strict_false_is_sent(self):
        c = _make_client(post_json={"request_id": "g3"})
        c.generate("j1", prompts=["x"], strict=False)
        body = _wire_load(c._session.post.call_args.kwargs["data"])
        assert body == {"prompts": ["x"], "strict": False}

    def test_generate_stream_full_payload(self):
        c = _make_client(post_json={"request_id": "s1", "count": 2})
        out = c.generate_stream(
            "j1",
            prompts=[[1, 2], [3, 4]],
            sampling_params=[
                {"max_tokens": 4, "temperature": 0.7},
                {"max_tokens": 2, "temperature": 0.3},
            ],
            routing_key=["rk-1", None],
            strict=True,
        )
        assert out == {"request_id": "s1", "count": 2}
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/generate-stream"
        assert kwargs["headers"]["Content-Type"] == "application/octet-stream"
        import json as _json
        body = _json.loads(kwargs["data"])
        assert body == {
            "prompts": [[1, 2], [3, 4]],
            "sampling_params": [
                {"max_tokens": 4, "temperature": 0.7},
                {"max_tokens": 2, "temperature": 0.3},
            ],
            "routing_key": ["rk-1", None],
            "strict": True,
        }

    def test_generate_chunks_oversized_payload(self):
        c = _make_client()
        c._MAX_GENERATE_BYTES = 64 * 1024
        c._session.post.side_effect = [
            _make_response({"chunk_cached": True}),
            _make_response({"request_id": "g4"}),
        ]
        oversized_prompt = "x" * (70 * 1024)

        assert c.generate("j1", prompts=[oversized_prompt]) == "g4"
        assert c._session.post.call_count == 2
        first = c._session.post.call_args_list[0].kwargs["data"]
        from dss_client import wire

        assert wire.read_byte_chunk_metadata(first)["operation"] == "generate"

    def test_generate_stream_rejects_oversized_payload(self):
        c = _make_client(post_json={"request_id": "s2", "count": 1})
        oversized_prompt = "x" * (NeutrinoClient._MAX_GENERATE_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            c.generate_stream("j1", prompts=[oversized_prompt])
        c._session.post.assert_not_called()

    def test_generate_rejects_overlong_tokenized_prompt(self):
        # Sampling sub-job exposes max_seq_len=4 (proto numbers arrive as floats).
        c = _make_client(
            post_json={"request_id": "g5"},
            get_json={"sub_jobs": [{"inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        # 4 tokens == max_seq_len leaves no room for output -> rejected.
        with pytest.raises(ValueError, match="does not fit the sampling job's max_seq_len of 4"):
            c.generate("j1", prompts=[[1, 2, 3, 4]])
        c._session.post.assert_not_called()

    def test_generate_allows_tokenized_prompt_under_limit(self):
        c = _make_client(
            post_json={"request_id": "g6"},
            get_json={"sub_jobs": [{"inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        # 3 tokens < max_seq_len=4 -> accepted.
        assert c.generate("j1", prompts=[[1, 2, 3]]) == "g6"
        c._session.post.assert_called_once()

    def test_generate_caches_max_seq_len_across_calls(self):
        c = _make_client(
            post_json={"request_id": "g7"},
            get_json={"sub_jobs": [{"inference_config": {"max_seq_len": 8.0, "n_gpus": 1.0}}]},
        )
        c.generate("j1", prompts=[[1, 2]])
        c.generate("j1", prompts=[[3, 4]])
        # Only the first call should have hit get_job; the value is cached.
        assert c._session.get.call_count == 1

    def test_generate_skips_validation_for_string_prompts(self):
        # String prompts are not validated client-side, so get_job is not called.
        c = _make_client(post_json={"request_id": "g8"})
        assert c.generate("j1", prompts=["a very long prompt"]) == "g8"
        c._session.get.assert_not_called()

    def test_generate_skips_when_no_sampling_config(self):
        # A job without an inference_config (e.g. training-only) disables the
        # check; the request is sent through unvalidated.
        c = _make_client(
            post_json={"request_id": "g9"},
            get_json={"sub_jobs": [{"training_config": {"max_seq_len": 2.0}}]},
        )
        assert c.generate("j1", prompts=[[1, 2, 3, 4, 5]]) == "g9"
        c._session.post.assert_called_once()

    def test_generate_skips_when_get_job_fails(self):
        # A transient get_job failure must not break generate, and must not be
        # cached as a permanent miss.
        c = _make_client(post_json={"request_id": "g10"})
        c._session.get.side_effect = RuntimeError("boom")
        assert c.generate("j1", prompts=[[1, 2, 3]]) == "g10"
        assert "j1" not in c._sampling_max_seq_len

    def test_generate_stream_rejects_overlong_tokenized_prompt(self):
        c = _make_client(
            post_json={"request_id": "s3", "count": 1},
            get_json={"sub_jobs": [{"inference_config": {"max_seq_len": 4.0, "n_gpus": 1.0}}]},
        )
        with pytest.raises(ValueError, match="does not fit the sampling job's max_seq_len of 4"):
            c.generate_stream("j1", prompts=[[1, 2, 3, 4, 5]])
        c._session.post.assert_not_called()

    def test_step_no_lr(self):
        c = _make_client(post_json={"request_id": "r2"})
        c.step("j1")
        kwargs = c._session.post.call_args.kwargs
        assert kwargs["json"] == {}

    def test_step_with_lr(self):
        c = _make_client(post_json={"request_id": "r2"})
        c.step("j1", learning_rate=2e-5)
        assert c._session.post.call_args.kwargs["json"] == {"learning_rate": 2e-5}

    def test_save_omits_optionals(self):
        c = _make_client(post_json={"request_id": "r3"})
        c.save("j1")
        assert c._session.post.call_args.kwargs["json"] == {}

    def test_save_with_optionals(self):
        c = _make_client(post_json={"request_id": "r3"})
        c.save("j1", checkpoint_id="cp", checkpoint_type="WEIGHTS-ONLY")
        assert c._session.post.call_args.kwargs["json"] == {
            "checkpoint_id": "cp", "checkpoint_type": "weights-only",
        }

    def test_save_rejects_unknown_checkpoint_type(self):
        c = _make_client(post_json={"request_id": "r3"})
        with pytest.raises(ValueError, match="resumable.*weights-only"):
            c.save("j1", checkpoint_type="full")
        c._session.post.assert_not_called()

    def test_load_minimal(self):
        c = _make_client(post_json={"request_id": "r-load"})
        rid = c.load("j1", checkpoint_id="cp")

        assert rid == "r-load"
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/load", json={"checkpoint_id": "cp"}
        )

    def test_load_with_source_job(self):
        c = _make_client(post_json={"request_id": "r-load"})
        c.load(
            "j1",
            checkpoint_id="cp",
            source_job_id="source-job",
        )

        assert c._session.post.call_args.kwargs["json"] == {
            "checkpoint_id": "cp",
            "source_job_id": "source-job",
        }

    def test_weight_sync(self):
        c = _make_client(post_json={"request_id": "r4"})
        rid = c.weight_sync(
            "j1",
            source_sub_job_id="j1:training:0",
            target_sub_job_ids=["j1:sampling:0", "j1:sampling:1"],
        )
        assert rid == "r4"
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/operation"
        assert kwargs["json"] == {
            "operation_type": "weight-sync",
            "sub_job_id": "j1:training:0",
            "sub_job_type": "training",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_ids": ["j1:sampling:0", "j1:sampling:1"],
            },
        }

    def test_weight_sync_accepts_explicit_operation_route(self):
        c = _make_client(post_json={"request_id": "r4"})
        c.weight_sync(
            "j1",
            source_sub_job_id="j1:training:0",
            target_sub_job_ids=["j1:sampling:0"],
            sub_job_id="j1:training:1",
            sub_job_type="training",
        )
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "weight-sync",
            "sub_job_id": "j1:training:1",
            "sub_job_type": "training",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_ids": ["j1:sampling:0"],
            },
        }

    def test_forward_operation(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.forward(
            "j1",
            {"tokens": [1, 2, 3]},
            sub_job_id="j1:training:0",
            sub_job_type="training",
        )
        assert out == {"request_id": "r-forward"}
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/operation",
            json={
                "operation_type": "forward",
                "sub_job_id": "j1:training:0",
                "sub_job_type": "training",
                "payload": {"tokens": [1, 2, 3]},
            },
        )

    def test_forward_operation_omits_none_fields(self):
        c = _make_client(post_json={"ok": True})
        c.forward("j1")
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/operation",
            json={"operation_type": "forward"},
        )

    def test_forward_wraps_bytes_payload(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.forward("j1", b"\x03\x04", sub_job_id="j1:training:0")
        assert out == {"request_id": "r-forward"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_id": "j1:training:0",
            "payload": {
                "payload_b64": base64.b64encode(b"\x03\x04").decode("ascii"),
                "content_type": "application/octet-stream",
            },
        }

    def test_fwd_alias_uses_forward_operation(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.fwd("j1", b"\x05\x06", sub_job_id="j1:training:0")
        assert out == {"request_id": "r-forward"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_id": "j1:training:0",
            "payload": {
                "payload_b64": base64.b64encode(b"\x05\x06").decode("ascii"),
                "content_type": "application/octet-stream",
            },
        }

    def test_fwd_no_grad_alias_uses_forward_operation(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        out = c.fwd_no_grad("j1", b"\x07\x08", sub_job_type="training")
        assert out == {"request_id": "r-forward"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "forward",
            "sub_job_type": "training",
            "payload": {
                "payload_b64": base64.b64encode(b"\x07\x08").decode("ascii"),
                "content_type": "application/octet-stream",
            },
        }

    def test_forward_rejects_oversized_bytes_payload(self):
        c = _make_client(post_json={"request_id": "r-forward"})
        oversized = b"\x00" * (NeutrinoClient._MAX_FWD_BWD_BYTES + 1)
        with pytest.raises(ValueError, match="exceeds the maximum allowed size"):
            c.forward("j1", oversized)
        c._session.post.assert_not_called()

    def test_bootstrap_router_replay(self):
        c = _make_client(post_json={"request_id": "r-bootstrap"})
        out = c.bootstrap_router_replay(
            "j1",
            "j1:training:0",
            "j1:sampling:0",
            max_cache_bytes=4096,
            sub_job_id="j1:sampling:0",
            sub_job_type="sampling",
        )
        assert out == {"request_id": "r-bootstrap"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "bootstrap-router-replay",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {
                "source_sub_job_id": "j1:training:0",
                "target_sub_job_id": "j1:sampling:0",
                "max_cache_bytes": 4096,
            },
        }

    def test_router_replay_discard_sends_operation(self):
        c = _make_client(post_json={"status": "discarded"})
        out = c.router_replay_discard(
            "j1",
            ["sample-1", "sample-2"],
            sub_job_id="j1:sampling:0",
            sub_job_type="sampling",
        )
        assert out == {"status": "discarded"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "router-replay-discard",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {"sample_ids": ["sample-1", "sample-2"]},
        }

    def test_router_replay_discard_accepts_extra_payload(self):
        c = _make_client(post_json={"status": "discarded"})
        out = c.router_replay_discard(
            "j1",
            sub_job_type="sampling",
            extra={"sample_ids": ["sample-extra"], "reason": "test"},
        )
        assert out == {"status": "discarded"}
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "router-replay-discard",
            "sub_job_type": "sampling",
            "payload": {"sample_ids": ["sample-extra"], "reason": "test"},
        }

    def test_reset_prefix_cache(self):
        c = _make_client(post_json={"reset_ok": True})
        out = c.reset_prefix_cache(
            "j1",
            sub_job_id="j1:sampling:0",
            sub_job_type="sampling",
            drain=False,
            timeout_s=12.5,
            retry_interval_s=0.25,
        )
        assert out == {"reset_ok": True}
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/operation"
        assert kwargs["json"] == {
            "operation_type": "reset-prefix-cache",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {
                "drain": False,
                "timeout_s": 12.5,
                "retry_interval_s": 0.25,
            },
        }

    def test_reset_prefix_cache_preserves_extra_payload(self):
        c = _make_client(post_json={"request_id": "r-reset"})
        c.reset_prefix_cache(
            "j1",
            sub_job_id="j1:sampling:0",
            extra={"target_sub_job_ids": ["j1:sampling"], "drain": False},
        )
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "reset-prefix-cache",
            "sub_job_id": "j1:sampling:0",
            "sub_job_type": "sampling",
            "payload": {
                "target_sub_job_ids": ["j1:sampling"],
                "drain": False,
                "timeout_s": 60.0,
                "retry_interval_s": 0.1,
            },
        }


# ─── NeutrinoClient — request polling ──────────────────────────────────


class TestRequestPolling:
    def test_get_request_status(self):
        c = _make_client(get_json={"request_id": "r1", "status": "running"})
        out = c.get_request_status("j1", "r1")
        assert out["status"] == "running"
        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1/requests/r1", params=None,
        )

    def test_get_request_status_with_stream_max_events(self):
        c = _make_client(get_json={"request_id": "r1", "status": "streaming"})
        c.get_request_status("j1", "r1", max_events=64)
        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1/requests/r1",
            params={"max_events": 64},
        )

    def test_get_request_status_with_cursor(self):
        c = _make_client(get_json={"request_id": "r1", "status": "streaming"})
        c.get_request_status("j1", "r1", max_events=64, cursor="2")
        c._session.get.assert_called_once_with(
            f"{c._prefix}/j1/requests/r1",
            params={"max_events": 64, "cursor": "2"},
        )

    def test_get_request_status_decodes_stream_dssst1_result_events(self):
        from dss_client import wire

        payload = wire.dumps(
            {
                "text": "ok",
                "token_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
                "logprobs": torch.tensor([0.125, -0.5], dtype=torch.float32),
            }
        )
        c = _make_client(
            get_json={
                "request_id": "r1",
                "status": "streaming",
                "events": [
                    {
                        "type": "result",
                        "index": 0,
                        "result": {
                            "content_type": "application/octet-stream",
                            "encoding": "base64",
                            "wire_format": "DSSST1",
                            "payload_b64": base64.b64encode(payload).decode("ascii"),
                        },
                    },
                    {"type": "done", "completed": 1, "failed": 0},
                ],
            }
        )

        assert c.get_request_status("j1", "r1")["events"] == [
            {
                "type": "result",
                "index": 0,
                "result": {
                    "text": "ok",
                    "token_ids": [1, 2, 3],
                    "logprobs": [0.125, -0.5],
                },
            },
            {"type": "done", "completed": 1, "failed": 0},
        ]

    def test_cancel_request(self):
        c = _make_client(post_json={"request_id": "r1"})
        out = c.cancel_request("j1", "r1", sub_job_id="j1:sampling:0")
        assert out["request_id"] == "r1"
        c._session.post.assert_called_once_with(
            f"{c._prefix}/j1/operation",
            json={
                "operation_type": "cancel-request",
                "sub_job_id": "j1:sampling:0",
                "payload": {"request_id": "r1"},
            },
        )

    def test_cancel_request_fans_out_without_sub_job_hint(self):
        c = _make_client(post_json={"request_id": "r1"})
        c.cancel_request("j1", "r1")
        assert c._session.post.call_args.kwargs["json"] == {
            "operation_type": "cancel-request",
            "payload": {"request_id": "r1"},
        }

    def test_poll_request_done(self, monkeypatch):
        c = _make_client(get_json={"status": "done", "result": {"avg_loss": 0.5}})
        sleep = MagicMock()
        monkeypatch.setattr(nc.time, "sleep", sleep)
        assert c.poll_request("j1", "r1") == {"avg_loss": 0.5}
        sleep.assert_not_called()

    def test_poll_request_done_no_result_returns_empty(self, monkeypatch):
        c = _make_client(get_json={"status": "done"})
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.poll_request("j1", "r1") == {}

    def test_poll_request_decodes_dssst1_result_payload(self, monkeypatch):
        from dss_client import wire

        payload = wire.dumps({"avg_loss": 0.5})
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        assert c.poll_request("j1", "r1") == {"avg_loss": 0.5}

    def test_poll_request_keeps_non_generate_dssst1_tensors(self, monkeypatch):
        from dss_client import wire

        payload = wire.dumps({"logprobs": torch.tensor([0.125, -0.5], dtype=torch.float32)})
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        result = c.poll_request("j1", "r1")
        assert torch.equal(result["logprobs"], torch.tensor([0.125, -0.5]))

    def test_poll_request_restores_generate_dssst1_tensors_to_lists(self, monkeypatch):
        from dss_client import wire

        payload = wire.dumps(
            {
                "job_id": "j1",
                "results": [
                    {
                        "text": "ok",
                        "token_ids": torch.tensor([1, 2, 3], dtype=torch.int64),
                        "logprobs": torch.tensor([0.125, -0.5], dtype=torch.float32),
                        "action_masks": torch.tensor([True, False], dtype=torch.bool),
                    }
                ],
            }
        )
        c = _make_client(
            get_json={
                "status": "done",
                "result": {
                    "content_type": "application/octet-stream",
                    "encoding": "base64",
                    "wire_format": "DSSST1",
                    "payload_b64": base64.b64encode(payload).decode("ascii"),
                },
            }
        )
        c._generate_request_ids.add("r-generate")
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        assert c.poll_request("j1", "r-generate") == {
            "job_id": "j1",
            "results": [
                {
                    "text": "ok",
                    "token_ids": [1, 2, 3],
                    "logprobs": [0.125, -0.5],
                    "action_masks": [True, False],
                }
            ],
        }
        assert "r-generate" not in c._generate_request_ids

    def test_poll_request_decodes_chunked_dssst1_result(self, monkeypatch):
        from dss_client import wire

        chunks = wire.encode_result_chunks({"text": "x" * 20_000}, max_bytes=12_000)
        assert len(chunks) > 1
        responses = []
        for idx, chunk in enumerate(chunks):
            event = {
                "type": "result_chunk",
                "payload_b64": base64.b64encode(chunk).decode("ascii"),
                "payload_sha256": __import__("hashlib").sha256(chunk).hexdigest(),
            }
            body = {"status": "running", "events": [event], "next_cursor": str(idx + 1)}
            if idx == len(chunks) - 1:
                body = {"status": "done", "events": [event]}
            responses.append(_make_response(body))

        c = _make_client()
        c._session.get.side_effect = responses
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)

        assert c.poll_request("j1", "r1") == {"text": "x" * 20_000}
        assert c._session.get.call_args_list[1].kwargs["params"] == {"cursor": "1"}

    def test_poll_request_failed(self, monkeypatch):
        c = _make_client(get_json={"status": "failed", "error": "oom"})
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(RuntimeError, match="oom"):
            c.poll_request("j1", "r1")

    def test_poll_request_uses_exponential_backoff_until_done(self, monkeypatch):
        c = _make_client()
        c.poll_interval = 1.0
        c.poll_backoff_multiplier = 2.0
        c.poll_max_interval = 3.0
        c._session.get.side_effect = [
            _make_response({"status": "running"}),
            _make_response({"status": "running"}),
            _make_response({"status": "running"}),
            _make_response({"status": "done", "result": {"ok": True}}),
        ]
        sleeps = []
        monkeypatch.setattr(nc.time, "sleep", sleeps.append)

        assert c.poll_request("j1", "r1") == {"ok": True}
        assert sleeps == [1.0, 2.0, 3.0]

    def test_poll_request_times_out(self, monkeypatch):
        c = _make_client(get_json={"status": "running"})
        c.poll_timeout = 0.01
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        with pytest.raises(TimeoutError):
            c.poll_request("j1", "r1")


# ─── NeutrinoClient — checkpoints ──────────────────────────────────────


class TestCheckpoints:
    def test_list_checkpoints(self):
        c = _make_client(get_json={"checkpoints": [{"checkpoint_id": "cp1"}]})
        out = c.list_checkpoints("j1")
        assert out == [{"checkpoint_id": "cp1"}]
        c._session.get.assert_called_once_with(f"{c._prefix}/j1/checkpoints")

    def test_export_checkpoint(self):
        c = _make_client(post_json={"checkpoint_id": "cp1", "files": []})
        out = c.export_checkpoint("j1", "cp1")
        assert out["checkpoint_id"] == "cp1"
        c._session.post.assert_called_once_with(f"{c._prefix}/j1/checkpoints/cp1:export")

    def test_delete_checkpoint(self):
        c = _make_client()
        # GS returns 204 No Content; delete_checkpoint returns None.
        c._session.delete.return_value = _make_response(status_code=204)
        assert c.delete_checkpoint("j1", "cp1") is None
        c._session.delete.assert_called_once_with(f"{c._prefix}/j1/checkpoints/cp1")

    def test_delete_checkpoint_raises_on_error(self):
        # delete_checkpoint discards the response, so pin the inherited
        # raise_for_status() contract: a non-2xx status propagates rather than
        # returning None. Models re-deleting an already-absent checkpoint (404).
        c = _make_client()
        c.max_retries = 0  # 404 is transient/retried; skip the backoff here
        resp = _make_response(status_code=404)
        resp.raise_for_status.side_effect = nc.requests.exceptions.HTTPError(response=resp)
        c._session.delete.return_value = resp
        with pytest.raises(nc.requests.exceptions.HTTPError):
            c.delete_checkpoint("j1", "missing")


# ─── NeutrinoClient — logs (read-only) ─────────────────────────────────


class TestLogs:
    def test_tail_logs_builds_payload(self):
        c = _make_client(post_json={"entries": [], "next_cursor": "c1", "eof": True})
        c.tail_logs("j1", cursor="c0", max_lines=50, sub_job_id="sj-0")
        body = c._session.post.call_args.kwargs["json"]
        assert body["operation_type"] == "tail-logs"
        assert body["sub_job_id"] == "sj-0"
        assert body["payload"] == {"cursor": "c0", "max_lines": 50}

    def test_tail_logs_omits_unset_payload_fields(self):
        c = _make_client(post_json={"entries": [], "next_cursor": "c1", "eof": True})
        c.tail_logs("j1", sub_job_id="sj-0")
        body = c._session.post.call_args.kwargs["json"]
        assert body["payload"] == {}  # nothing set → empty payload

    def test_stream_logs_drains_then_stops_on_eof(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"entries": [{"msg": "a"}, {"msg": "b"}], "next_cursor": "c1", "eof": False}),
            _make_response({"entries": [{"msg": "c"}], "next_cursor": "c2", "eof": False}),
            _make_response({"entries": [], "next_cursor": "c2", "eof": True}),
        ]
        got = [e["msg"] for e in c.stream_logs("j1", sub_job_id="sj-0", follow=False)]
        assert got == ["a", "b", "c"]
        # Cursor advanced across polls: the last call carried the prior next_cursor.
        assert c._session.post.call_args.kwargs["json"]["payload"]["cursor"] == "c2"

    def test_stream_logs_follow_keeps_yielding(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client(
            post_json={"entries": [{"msg": "a"}, {"msg": "b"}], "next_cursor": "c1", "eof": False}
        )
        got = []
        for entry in c.stream_logs("j1", sub_job_id="sj-0", follow=True):
            got.append(entry["msg"])
            if len(got) == 2:
                break  # an infinite tail; caller stops by breaking
        assert got == ["a", "b"]


# ─── NeutrinoClient — ZMD events (read-only) ───────────────────────────


class TestEvents:
    def test_tail_events_builds_payload(self):
        c = _make_client(post_json={"events": [], "next_cursor": "3", "eof": True})
        c.tail_events("j1", cursor="2", max_events=10, sub_job_id="sj-0")
        url, kwargs = c._session.post.call_args
        assert url[0] == f"{c._prefix}/j1/operation"
        body = kwargs["json"]
        assert body["operation_type"] == "zmd-events"
        assert body["sub_job_id"] == "sj-0"
        assert body["payload"] == {"cursor": "2", "max_events": 10}

    def test_tail_events_omits_empty_payload(self):
        c = _make_client(post_json={"events": [], "next_cursor": "0", "eof": True})
        c.tail_events("j1")
        body = c._session.post.call_args.kwargs["json"]
        assert body["operation_type"] == "zmd-events"
        assert "payload" not in body  # nothing to send

    def test_stream_events_drains_to_eof(self, monkeypatch):
        monkeypatch.setattr(nc.time, "sleep", lambda *_: None)
        c = _make_client()
        c._session.post.side_effect = [
            _make_response({"events": [{"seq": 1}, {"seq": 2}], "next_cursor": "2", "eof": False}),
            _make_response({"events": [{"seq": 3}], "next_cursor": "3", "eof": False}),
            _make_response({"events": [], "next_cursor": "3", "eof": True}),
        ]
        seqs = [e["seq"] for e in c.stream_events("j1", follow=False)]
        assert seqs == [1, 2, 3]
        assert c._session.post.call_args.kwargs["json"]["payload"]["cursor"] == "3"


class TestExecutionLogDownload:
    @staticmethod
    def _make_creds_json():
        return json.dumps(
            {
                "locationType": "S3",
                "location": "s3://bucket/stage/abc/",
                "region": "us-west-2",
                "creds": {
                    "AWS_KEY_ID": "key",
                    "AWS_SECRET_KEY": "secret",
                    "AWS_TOKEN": "token",
                },
            }
        )

    def test_parse_s3_stage_credentials_extracts_documented_keys(self):
        creds = nc._parse_s3_stage_credentials(self._make_creds_json())
        assert creds == {
            "bucket": "bucket",
            "prefix": "stage/abc",
            "region": "us-west-2",
            "access_key_id": "key",
            "secret_access_key": "secret",
            "session_token": "token",
        }

    def test_parse_s3_stage_credentials_rejects_non_s3(self):
        with pytest.raises(NotImplementedError):
            nc._parse_s3_stage_credentials({"locationType": "AZURE"})

    def test_get_experiment_run_calls_endpoint(self):
        c = _make_client(
            get_json={
                "experiment_name": "DB.SCH.EXP",
                "experiment_run_name": "RUN_ABC",
            }
        )
        out = c.get_experiment_run("job-1")
        assert out == {"experiment_name": "DB.SCH.EXP", "experiment_run_name": "RUN_ABC"}
        c._session.get.assert_called_once_with(f"{c._prefix}/job-1/experiment-run")

    def test_fetch_execution_logs_lists_and_downloads_every_file_under_logs(self, monkeypatch):
        c = _make_client(
            get_json={
                "experiment_name": "DB.SCH.EXP",
                "experiment_run_name": "RUN_ABC",
            }
        )
        sql_calls: list[str] = []

        def fake_scalar(statement):
            sql_calls.append(statement)
            return self._make_creds_json()

        monkeypatch.setattr(c, "_query_sql_scalar", fake_scalar)

        listed_prefixes: list[tuple[str, str]] = []
        get_calls: list[tuple[str, str]] = []
        keys_in_stage = [
            # Two siblings under the same sub_job (mixed extensions): both kept.
            "stage/abc/versions/v1/checkpoints/_logs/job-1:training:0/execution.jsonl",
            "stage/abc/versions/v1/checkpoints/_logs/job-1:training:0/server.log",
            # Different sub_job: still kept.
            "stage/abc/versions/v1/checkpoints/_logs/job-1:sampling:0/execution.jsonl",
            # _logs subtree without the checkpoints/ ancestor: still kept.
            "stage/abc/versions/v1/_logs/job-1:eval:0/execution.jsonl",
            # Non-_logs entry: filtered out.
            "stage/abc/versions/v1/checkpoints/model.bin",
        ]
        bodies = {
            keys_in_stage[0]: b'{"a":1}\n',
            keys_in_stage[1]: b"server line\n",
            keys_in_stage[2]: b'{"b":2}\n',
            keys_in_stage[3]: b'{"c":3}\n',
        }

        class FakePaginator:
            def paginate(self, *, Bucket, Prefix):
                listed_prefixes.append((Bucket, Prefix))
                return iter([{"Contents": [{"Key": k} for k in keys_in_stage]}])

        class FakeS3:
            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return FakePaginator()

            def get_object(self, *, Bucket, Key):
                get_calls.append((Bucket, Key))
                return {"Body": SimpleNamespace(read=lambda: bodies[Key])}

        monkeypatch.setitem(
            sys.modules, "boto3", SimpleNamespace(client=lambda *a, **kw: FakeS3())
        )

        out = c.fetch_execution_logs("job-1")

        assert sql_calls == [
            "SELECT SYSTEM$GET_VSTAGE_WRITE_CREDS("
            "'snow://experiment/DB.SCH.EXP/versions/RUN_ABC/')"
        ]
        assert listed_prefixes == [("bucket", "stage/abc/")]
        assert get_calls == [
            ("bucket", keys_in_stage[0]),
            ("bucket", keys_in_stage[1]),
            ("bucket", keys_in_stage[2]),
            ("bucket", keys_in_stage[3]),
        ]
        assert out == [
            {
                "sub_job_id": "job-1:training:0",
                "filename": "execution.jsonl",
                "s3_uri": f"s3://bucket/{keys_in_stage[0]}",
                "content": '{"a":1}\n',
            },
            {
                "sub_job_id": "job-1:training:0",
                "filename": "server.log",
                "s3_uri": f"s3://bucket/{keys_in_stage[1]}",
                "content": "server line\n",
            },
            {
                "sub_job_id": "job-1:sampling:0",
                "filename": "execution.jsonl",
                "s3_uri": f"s3://bucket/{keys_in_stage[2]}",
                "content": '{"b":2}\n',
            },
            {
                "sub_job_id": "job-1:eval:0",
                "filename": "execution.jsonl",
                "s3_uri": f"s3://bucket/{keys_in_stage[3]}",
                "content": '{"c":3}\n',
            },
        ]

    def test_fetch_execution_logs_errors_when_experiment_run_missing_fields(self):
        c = _make_client(get_json={"experiment_name": "DB.SCH.EXP"})
        with pytest.raises(ValueError, match="experiment_run_name"):
            c.fetch_execution_logs("job-1")
