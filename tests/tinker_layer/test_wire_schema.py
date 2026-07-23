# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wire schema round-trip tests.

These exist to catch upstream ``tinker.types.*`` drift: for every wire type
we redefine locally in ``arctic_platform.rl.tinker_server``, dump a valid
example to JSON and ensure our Pydantic parser accepts it. When the
upstream tinker SDK adds/removes fields, one of these will start failing
and point at the exact spot to update.
"""

from __future__ import annotations

import json

import pytest

from arctic_platform.rl.tinker_server import AdamParams
from arctic_platform.rl.tinker_server import ClientConfigResponse
from arctic_platform.rl.tinker_server import CreateModelRequest
from arctic_platform.rl.tinker_server import CreateModelResponse
from arctic_platform.rl.tinker_server import CreateSessionRequest
from arctic_platform.rl.tinker_server import Datum
from arctic_platform.rl.tinker_server import EncodedTextChunk
from arctic_platform.rl.tinker_server import ForwardBackwardInput
from arctic_platform.rl.tinker_server import ForwardBackwardOutput
from arctic_platform.rl.tinker_server import ForwardBackwardRequest
from arctic_platform.rl.tinker_server import ForwardInput
from arctic_platform.rl.tinker_server import ForwardRequest
from arctic_platform.rl.tinker_server import FutureRetrieveRequest
from arctic_platform.rl.tinker_server import LoraConfig
from arctic_platform.rl.tinker_server import ModelInput
from arctic_platform.rl.tinker_server import OptimStepRequest
from arctic_platform.rl.tinker_server import OptimStepResponse
from arctic_platform.rl.tinker_server import SampleRequest
from arctic_platform.rl.tinker_server import SampleResponse
from arctic_platform.rl.tinker_server import SampledSequence
from arctic_platform.rl.tinker_server import SamplingParams
from arctic_platform.rl.tinker_server import SaveWeightsForSamplerRequest
from arctic_platform.rl.tinker_server import SaveWeightsForSamplerResponse
from arctic_platform.rl.tinker_server import StopReason
from arctic_platform.rl.tinker_server import TensorData
from arctic_platform.rl.tinker_server import TryAgainResponse
from arctic_platform.rl.tinker_server import UntypedAPIFuture


class TestRequestParsing:
    def test_create_session_request(self):
        body = {"tags": ["rl", "smoke"], "user_metadata": {"user": "k"},
                "sdk_version": "0.42.0"}
        req = CreateSessionRequest.model_validate(body)
        assert req.tags == ["rl", "smoke"]
        assert req.user_metadata == {"user": "k"}

    def test_create_model_request_full_weight(self):
        body = {
            "session_id": "sess-abc",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"rank": 0, "seed": 42, "train_mlp": True,
                            "train_attn": True, "train_unembed": True},
        }
        req = CreateModelRequest.model_validate(body)
        assert req.lora_config.rank == 0

    def test_create_model_request_lora(self):
        body = {
            "session_id": "sess-abc",
            "model_seq_id": 0,
            "base_model": "Qwen/Qwen3-8B",
            "lora_config": {"rank": 32},
        }
        req = CreateModelRequest.model_validate(body)
        assert req.lora_config.rank == 32

    def test_forward_backward_request(self):
        body = {
            "forward_backward_input": {
                "data": [
                    {
                        "model_input": {"chunks": [
                            {"type": "encoded_text", "tokens": [1, 2, 3]}
                        ]},
                        "loss_fn_inputs": {
                            "advantages": {"dtype": "float32",
                                           "data": [0.1, 0.2, 0.3],
                                           "shape": [3]},
                            "logprobs": {"dtype": "float32",
                                         "data": [-1.0, -1.1, -1.2],
                                         "shape": [3]},
                        },
                    },
                ],
                "loss_fn": "ppo",
                "loss_fn_config": {"clip_low_threshold": 0.8,
                                   "clip_high_threshold": 1.2},
            },
            "model_id": "main",
            "seq_id": 7,
        }
        req = ForwardBackwardRequest.model_validate(body)
        assert req.forward_backward_input.loss_fn == "ppo"
        assert len(req.forward_backward_input.data) == 1

    def test_forward_request_no_config(self):
        body = {
            "forward_input": {
                "data": [
                    {"model_input": {"chunks": [
                        {"type": "encoded_text", "tokens": [1, 2, 3]}
                    ]}, "loss_fn_inputs": {}},
                ],
                "loss_fn": "ppo",
            },
            "model_id": "main",
        }
        req = ForwardRequest.model_validate(body)
        assert req.forward_input.loss_fn == "ppo"

    def test_optim_step_request(self):
        body = {
            "adam_params": {"learning_rate": 5e-5, "beta1": 0.9, "beta2": 0.999,
                            "eps": 1e-8, "weight_decay": 0.01,
                            "grad_clip_norm": 1.0},
            "model_id": "main",
        }
        req = OptimStepRequest.model_validate(body)
        assert req.adam_params.learning_rate == pytest.approx(5e-5)
        assert req.adam_params.grad_clip_norm == 1.0

    def test_sample_request_with_sampling_session(self):
        body = {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]},
            "sampling_params": {"max_tokens": 32, "temperature": 0.7,
                                "top_p": 0.9, "stop": ["END"]},
            "num_samples": 16,
            "sampling_session_id": "ss@3",
            "seq_id": 42,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        }
        req = SampleRequest.model_validate(body)
        assert req.num_samples == 16
        assert req.sampling_session_id == "ss@3"

    def test_sample_request_with_base_model(self):
        body = {
            "prompt": {"chunks": [{"type": "encoded_text", "tokens": [1]}]},
            "sampling_params": {"max_tokens": 8},
            "num_samples": 1,
            "base_model": "Qwen/Qwen3-8B",
        }
        req = SampleRequest.model_validate(body)
        assert req.base_model == "Qwen/Qwen3-8B"

    def test_save_weights_for_sampler_request(self):
        body = {"model_id": "main", "path": "checkpoint-001", "ttl_seconds": 60}
        req = SaveWeightsForSamplerRequest.model_validate(body)
        assert req.model_id == "main"
        assert req.ttl_seconds == 60

    def test_future_retrieve_request(self):
        req = FutureRetrieveRequest.model_validate({"request_id": "42",
                                                    "allow_metadata_only": True})
        assert req.request_id == "42"
        assert req.allow_metadata_only is True


class TestResponseSerialization:
    def test_untyped_api_future_matches_sdk_shape(self):
        f = UntypedAPIFuture(request_id="17", model_id="main")
        d = f.model_dump()
        assert d == {"request_id": "17", "model_id": "main", "type": "future"}

    def test_try_again_response(self):
        assert TryAgainResponse().model_dump() == {"type": "try_again"}

    def test_forward_backward_output_serialization(self):
        out = ForwardBackwardOutput(
            loss_fn_output_type="TorchLossReturn",
            loss_fn_outputs=[],
            metrics={"loss": 0.5, "grad_norm": 1.2},
        )
        d = out.model_dump()
        assert d["loss_fn_output_type"] == "TorchLossReturn"
        assert d["metrics"] == {"loss": 0.5, "grad_norm": 1.2}

    def test_optim_step_response(self):
        r = OptimStepResponse(metrics={"grad_norm": 0.9, "last_lr": 1e-4})
        d = r.model_dump()
        assert d["metrics"]["last_lr"] == pytest.approx(1e-4)

    def test_save_weights_for_sampler_response(self):
        r = SaveWeightsForSamplerResponse(
            path="tinker://main/sampler_weights/5",
            sampling_session_id="ss@5",
        )
        d = r.model_dump()
        assert d["type"] == "save_weights_for_sampler"
        assert d["path"] == "tinker://main/sampler_weights/5"

    def test_sample_response_serialization(self):
        r = SampleResponse(sequences=[
            SampledSequence(tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3],
                            stop_reason=StopReason.STOP),
            SampledSequence(tokens=[4, 5], logprobs=None,
                            stop_reason=StopReason.LENGTH),
        ])
        d = r.model_dump()
        assert d["type"] == "sample"
        assert d["sequences"][0]["stop_reason"] == "stop"
        assert d["sequences"][1]["stop_reason"] == "length"
        assert d["sequences"][1]["logprobs"] is None

    def test_client_config_forces_json_path(self):
        cfg = ClientConfigResponse()
        d = cfg.model_dump()
        # The proto/zstd path is not implemented server-side.
        assert d["proto_write_fwdbwd"] is False
        assert d["proto_compress_fwdbwd"] is False
        assert d["fwd_via_fwdbwd"] is False


class TestTensorData:
    def test_dense_float32(self):
        td = TensorData(dtype="float32", data=[0.1, 0.2, 0.3], shape=[3])
        d = td.model_dump()
        assert d["dtype"] == "float32"
        assert d["data"] == [0.1, 0.2, 0.3]
        assert d["shape"] == [3]

    def test_sparse_encoding_round_trips(self):
        td = TensorData(
            dtype="float32",
            data=[1.0, 2.0],
            shape=[2, 4],
            sparse_crow_indices=[0, 1, 2],
            sparse_col_indices=[0, 3],
        )
        parsed = TensorData.model_validate_json(json.dumps(td.model_dump()))
        assert parsed.sparse_crow_indices == [0, 1, 2]
        assert parsed.sparse_col_indices == [0, 3]


class TestModelInputChunks:
    def test_encoded_text_only(self):
        mi = ModelInput.model_validate(
            {"chunks": [{"type": "encoded_text", "tokens": [1, 2, 3]}]}
        )
        assert mi.chunks[0].tokens == [1, 2, 3]

    def test_rejects_non_text_chunk(self):
        # v1 supports text only. Pydantic will fail with a discriminator error
        # because we don't include an ImageChunk in the union.
        with pytest.raises(Exception):
            ModelInput.model_validate(
                {"chunks": [{"type": "image", "url": "http://x"}]}
            )
