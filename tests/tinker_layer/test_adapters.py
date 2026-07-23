# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tinker_server adapters (Datum→batch, AdamParams→overrides,
SamplingParams→vLLM, loss_fn_config→actor_config)."""

from __future__ import annotations

import numpy as np
import pytest

from arctic_platform.rl.tinker_server import AdamParams
from arctic_platform.rl.tinker_server import Datum
from arctic_platform.rl.tinker_server import EncodedTextChunk
from arctic_platform.rl.tinker_server import ModelInput
from arctic_platform.rl.tinker_server import SamplingParams
from arctic_platform.rl.tinker_server import TensorData
from arctic_platform.rl.tinker_server import _loss_fn_config_to_actor_config
from arctic_platform.rl.tinker_server import adam_params_to_optim_overrides
from arctic_platform.rl.tinker_server import datum_list_to_arctic_batch
from arctic_platform.rl.tinker_server import sampling_params_tinker_to_vllm


def _mk_datum(tokens, advantages, logprobs, mask=None):
    inputs = {
        "advantages": TensorData(dtype="float32", data=advantages, shape=[len(advantages)]),
        "logprobs": TensorData(dtype="float32", data=logprobs, shape=[len(logprobs)]),
    }
    if mask is not None:
        inputs["mask"] = TensorData(dtype="float32", data=mask, shape=[len(mask)])
    return Datum(
        model_input=ModelInput(chunks=[EncodedTextChunk(tokens=tokens)]),
        loss_fn_inputs=inputs,
    )


class TestDatumAdapter:
    def test_pads_to_config_max(self):
        """Every batch row is padded to (max_prompt + max_response), not
        batch-local — this is the ZoRRo invariant fix."""
        datum = _mk_datum([1, 2, 3], [0.1, 0.2, 0.3], [-1.0, -1.1, -1.2])
        out = datum_list_to_arctic_batch(
            [datum], "ppo", None,
            max_prompt_length=8, max_response_length=4, pad_token_id=0,
        )
        assert out["batch"]["input_ids"].shape == (1, 12)
        assert out["batch"]["attention_mask"].shape == (1, 12)
        assert out["batch"]["advantages"].shape == (1, 12)

    def test_pad_token_id_used(self):
        out = datum_list_to_arctic_batch(
            [_mk_datum([7, 8], [0.5, 0.5], [-2.0, -2.0])],
            "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=42,
        )
        assert (out["batch"]["input_ids"][0, 2:] == 42).all()

    def test_multiple_datums_stack_row_wise(self):
        d1 = _mk_datum([1, 2], [0.5, 0.5], [-1.0, -1.1])
        d2 = _mk_datum([3, 4, 5], [0.1, 0.2, 0.3], [-2.0, -2.1, -2.2])
        out = datum_list_to_arctic_batch(
            [d1, d2], "ppo", None,
            max_prompt_length=8, max_response_length=4, pad_token_id=0,
        )
        assert out["batch"]["input_ids"].shape == (2, 12)
        assert out["batch"]["input_ids"][0, 0] == 1
        assert out["batch"]["input_ids"][1, 2] == 5
        # d1's advantages are packed to the front, then padded with 0.
        np.testing.assert_allclose(out["batch"]["advantages"][0, :2], [0.5, 0.5])
        assert out["batch"]["advantages"][0, 2] == 0.0

    def test_attention_mask_marks_real_tokens(self):
        out = datum_list_to_arctic_batch(
            [_mk_datum([1, 2, 3], [0.1] * 3, [-1.0] * 3)],
            "ppo", None,
            max_prompt_length=8, max_response_length=4, pad_token_id=0,
        )
        assert out["batch"]["attention_mask"][0, :3].tolist() == [1, 1, 1]
        assert out["batch"]["attention_mask"][0, 3:].tolist() == [0] * 9

    def test_mask_key_becomes_loss_mask(self):
        d = _mk_datum([1, 2, 3], [0.1] * 3, [-1.0] * 3, mask=[0.0, 1.0, 1.0])
        out = datum_list_to_arctic_batch(
            [d], "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
        )
        assert out["batch"]["loss_mask"][0, :3].tolist() == [0.0, 1.0, 1.0]

    def test_weights_key_also_maps(self):
        """Datum's spec uses ``weights`` but cookbook writes ``mask`` — both
        must resolve to loss_mask."""
        inputs = {
            "advantages": TensorData(dtype="float32", data=[0.5, 0.5, 0.5]),
            "logprobs": TensorData(dtype="float32", data=[-1.0, -1.0, -1.0]),
            "weights": TensorData(dtype="float32", data=[1.0, 0.5, 1.0]),
        }
        d = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            loss_fn_inputs=inputs,
        )
        out = datum_list_to_arctic_batch(
            [d], "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
        )
        assert out["batch"]["loss_mask"][0, :3].tolist() == [1.0, 0.5, 1.0]

    def test_processing_carries_loss_fn(self):
        out = datum_list_to_arctic_batch(
            [_mk_datum([1], [0.1], [-1.0])],
            "importance_sampling", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
        )
        assert out["processing"]["loss_fn"] == "importance_sampling"

    def test_forward_only_flag_present(self):
        out = datum_list_to_arctic_batch(
            [_mk_datum([1], [0.1], [-1.0])],
            "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
            forward_only=True,
        )
        assert out["meta"]["forward_only"] is True
        # actor_config is empty on forward_only (no loss reduction on the server).
        assert out["meta"]["actor_config"] == {}

    def test_max_response_len_threaded(self):
        out = datum_list_to_arctic_batch(
            [_mk_datum([1], [0.1], [-1.0])],
            "ppo", None,
            max_prompt_length=4, max_response_length=7, pad_token_id=0,
        )
        assert out["meta"]["max_response_len"] == 7


class TestLossFnConfigMapping:
    def test_ppo_default_clip(self):
        cfg = _loss_fn_config_to_actor_config("ppo", None)
        assert cfg["eps_clip"] == pytest.approx(0.2)
        assert cfg["eps_clip_higher"] == pytest.approx(0.2)

    def test_ppo_tight_clip(self):
        cfg = _loss_fn_config_to_actor_config(
            "ppo", {"clip_low_threshold": 0.9, "clip_high_threshold": 1.1}
        )
        assert cfg["eps_clip"] == pytest.approx(0.1)
        assert cfg["eps_clip_higher"] == pytest.approx(0.1)

    def test_importance_sampling_disables_clip(self):
        cfg = _loss_fn_config_to_actor_config("importance_sampling", None)
        assert cfg["eps_clip"] > 1e6
        assert cfg["eps_clip_higher"] > 1e6

    def test_kl_coef_maps(self):
        cfg = _loss_fn_config_to_actor_config("ppo", {"kl_coef": 0.05})
        assert cfg["kl_loss_coef"] == pytest.approx(0.05)
        assert cfg["use_kl_loss"] is True

    def test_kl_coef_zero_disables_kl(self):
        cfg = _loss_fn_config_to_actor_config("ppo", {"kl_coef": 0.0})
        assert cfg["use_kl_loss"] is False

    def test_entropy_coef_maps(self):
        cfg = _loss_fn_config_to_actor_config("ppo", {"entropy_coef": 1e-3})
        assert cfg["entropy_coeff"] == pytest.approx(1e-3)


class TestAdamParams:
    def test_defaults_round_trip(self):
        p = AdamParams()
        ov = adam_params_to_optim_overrides(p)
        assert ov["lr"] == pytest.approx(1e-4)
        assert ov["betas"] == (0.9, 0.95)
        assert ov["eps"] == pytest.approx(1e-12)
        assert ov["weight_decay"] == 0.0

    def test_all_fields_flow_through(self):
        p = AdamParams(learning_rate=1e-3, beta1=0.85, beta2=0.99,
                       eps=1e-8, weight_decay=0.05)
        ov = adam_params_to_optim_overrides(p)
        assert ov["lr"] == pytest.approx(1e-3)
        assert ov["betas"] == (0.85, 0.99)
        assert ov["eps"] == pytest.approx(1e-8)
        assert ov["weight_decay"] == pytest.approx(0.05)


class TestSamplingParams:
    def test_defaults(self):
        p = SamplingParams()
        v = sampling_params_tinker_to_vllm(p, num_samples=2)
        assert v["n"] == 2
        assert v["temperature"] == 1.0
        assert v["top_p"] == 1.0
        assert v["top_k"] == -1
        # logprobs is always forced to 1 for RL loops.
        assert v["logprobs"] == 1
        # Optional fields omitted when None.
        assert "max_tokens" not in v
        assert "seed" not in v
        assert "stop" not in v and "stop_token_ids" not in v

    def test_stop_string(self):
        p = SamplingParams(stop="END")
        v = sampling_params_tinker_to_vllm(p, num_samples=1)
        assert v["stop"] == "END"

    def test_stop_int_seq_becomes_stop_token_ids(self):
        p = SamplingParams(stop=[151643, 151645])
        v = sampling_params_tinker_to_vllm(p, num_samples=1)
        assert v["stop_token_ids"] == [151643, 151645]
        assert "stop" not in v

    def test_stop_string_seq_becomes_stop(self):
        p = SamplingParams(stop=["END", "STOP"])
        v = sampling_params_tinker_to_vllm(p, num_samples=1)
        assert v["stop"] == ["END", "STOP"]

    def test_max_tokens_and_seed(self):
        p = SamplingParams(max_tokens=64, seed=7)
        v = sampling_params_tinker_to_vllm(p, num_samples=3)
        assert v["max_tokens"] == 64
        assert v["seed"] == 7
        assert v["n"] == 3
