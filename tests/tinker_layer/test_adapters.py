# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for tinker_router adapters (Datum→batch, AdamParams→overrides,
SamplingParams→vLLM, loss_fn_config→actor_config)."""

from __future__ import annotations

import numpy as np
import pytest

from arctic_platform.rl.tinker_router import AdamParams
from arctic_platform.rl.tinker_router import Datum
from arctic_platform.rl.tinker_router import EncodedTextChunk
from arctic_platform.rl.tinker_router import ModelInput
from arctic_platform.rl.tinker_router import SamplingParams
from arctic_platform.rl.tinker_router import TensorData
from arctic_platform.rl.tinker_router import _loss_fn_config_to_actor_config
from arctic_platform.rl.tinker_router import adam_params_to_optim_overrides
from arctic_platform.rl.tinker_router import datum_list_to_arctic_batch
from arctic_platform.rl.tinker_router import sampling_params_tinker_to_vllm


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
        batch-local — this is the ZoRRo invariant fix. All per-token
        signals share the same ``[B, seq_len]`` shape so the packing
        pass flattens them together."""
        datum = _mk_datum([1, 2, 3], [0.1, 0.2, 0.3], [-1.0, -1.1, -1.2])
        out, _ = datum_list_to_arctic_batch(
            [datum], "ppo", None,
            max_prompt_length=8, max_response_length=4, pad_token_id=0,
        )
        assert out["batch"]["input_ids"].shape == (1, 12)
        assert out["batch"]["attention_mask"].shape == (1, 12)
        assert out["batch"]["response_mask"].shape == (1, 12)
        assert out["batch"]["advantages"].shape == (1, 12)
        assert out["batch"]["old_log_probs"].shape == (1, 12)

    def test_pad_token_id_used(self):
        # Explicit prompt-only mask (all zeros) → all tokens treated as
        # prompt, left-padded to mpl=4. Response window (columns 4:) is
        # all pad_token_id=42.
        out, _ = datum_list_to_arctic_batch(
            [_mk_datum([7, 8], [0.0, 0.0], [-2.0, -2.0], mask=[0, 0])],
            "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=42,
        )
        assert (out["batch"]["input_ids"][0, 4:] == 42).all()
        assert out["batch"]["input_ids"][0, 2] == 7
        assert out["batch"]["input_ids"][0, 3] == 8

    def test_multiple_datums_stack_row_wise(self):
        # d1: prompt=[1], response=[2]        (mask marks position 1 as resp)
        # d2: prompt=[3, 4], response=[5]     (mask marks position 2 as resp)
        d1 = _mk_datum([1, 2], [0.0, 0.5], [-1.0, -1.1], mask=[0, 1])
        d2 = _mk_datum([3, 4, 5], [0.0, 0.0, 0.3], [-2.0, -2.1, -2.2],
                       mask=[0, 0, 1])
        out, _ = datum_list_to_arctic_batch(
            [d1, d2], "ppo", None,
            max_prompt_length=8, max_response_length=4, pad_token_id=0,
        )
        assert out["batch"]["input_ids"].shape == (2, 12)
        # d1's prompt token 1 lands at column mpl-1 = 7; response at column 8.
        assert out["batch"]["input_ids"][0, 7] == 1
        assert out["batch"]["input_ids"][0, 8] == 2
        # d2's response token 5 lands at column mpl = 8.
        assert out["batch"]["input_ids"][1, 8] == 5
        # d1's response advantage sits at column mpl of the full-length tensor.
        np.testing.assert_allclose(out["batch"]["advantages"][0, 8], 0.5)

    def test_attention_mask_marks_real_tokens(self):
        # tokens=[1, 2, 3] with mask=[0, 0, 1] → prompt=[1, 2], response=[3].
        # Left-padded to mpl=8: real cols are [6, 7] (prompt), 8 (response).
        d = _mk_datum([1, 2, 3], [0.0, 0.0, 0.1], [-1.0] * 3, mask=[0, 0, 1])
        out, _ = datum_list_to_arctic_batch(
            [d], "ppo", None,
            max_prompt_length=8, max_response_length=4, pad_token_id=0,
        )
        assert out["batch"]["attention_mask"][0, 6:9].tolist() == [1, 1, 1]
        assert out["batch"]["attention_mask"][0, :6].tolist() == [0] * 6
        assert out["batch"]["attention_mask"][0, 9:].tolist() == [0, 0, 0]

    def test_response_mask_reflects_response_span(self):
        # mask=[0, 1, 1] → prompt=[1], response=[2, 3]. response_mask has
        # ones at the response columns (mpl, mpl+1) of the full [B, seq_len]
        # tensor and 0 elsewhere.
        d = _mk_datum([1, 2, 3], [0.0, 0.5, 0.5], [-1.0] * 3, mask=[0, 1, 1])
        out, _ = datum_list_to_arctic_batch(
            [d], "ppo", None,
            max_prompt_length=4, max_response_length=4, pad_token_id=0,
        )
        # Response span is columns [mpl=4, mpl+1=5]; everything else is 0.
        assert out["batch"]["response_mask"][0].tolist() == [0, 0, 0, 0, 1, 1, 0, 0]
        assert out["batch"]["prompts"][0].tolist() == [0, 0, 0, 1]
        assert out["batch"]["responses"][0].tolist() == [2, 3, 0, 0]

    def test_weights_key_also_maps(self):
        # Datum spec uses ``weights`` but cookbook writes ``mask`` — both
        # must trigger the response-split for prompts/responses layout.
        inputs = {
            "advantages": TensorData(dtype="float32", data=[0.0, 0.5, 0.5]),
            "logprobs": TensorData(dtype="float32", data=[-1.0, -1.0, -1.0]),
            "weights": TensorData(dtype="float32", data=[0.0, 1.0, 1.0]),
        }
        d = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            loss_fn_inputs=inputs,
        )
        out, _ = datum_list_to_arctic_batch(
            [d], "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
        )
        # prompt=[1] left-padded to length 4, response=[2, 3] right-padded to 2.
        assert out["batch"]["prompts"][0].tolist() == [0, 0, 0, 1]
        assert out["batch"]["responses"][0].tolist() == [2, 3]
        # response_mask lives on the full [B, seq_len=6] tensor; response
        # columns [mpl=4, mpl+1=5] are ones.
        assert out["batch"]["response_mask"][0].tolist() == [0, 0, 0, 0, 1, 1]

    def test_advantages_infer_response_boundary(self):
        # SkyRL-tx's tinker_cookbook rl_loop.py doesn't set ``weights``; the
        # prompt/response boundary is inferred from ``advantages`` (zero on
        # prompt, non-zero on response). Cross-check tokens=[1, 2, 3] with
        # advantages=[0, 0.5, 0.5] → prompt=[1], response=[2, 3].
        inputs = {
            "advantages": TensorData(dtype="float32", data=[0.0, 0.5, 0.5]),
            "logprobs": TensorData(dtype="float32", data=[-1.0, -1.0, -1.0]),
        }
        d = Datum(
            model_input=ModelInput(chunks=[EncodedTextChunk(tokens=[1, 2, 3])]),
            loss_fn_inputs=inputs,
        )
        out, _ = datum_list_to_arctic_batch(
            [d], "importance_sampling", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
        )
        assert out["batch"]["prompts"][0].tolist() == [0, 0, 0, 1]
        assert out["batch"]["responses"][0].tolist() == [2, 3]
        assert out["batch"]["response_mask"][0].tolist() == [0, 0, 0, 0, 1, 1]

    def test_processing_carries_loss_fn(self):
        # Arctic's LOSS_FNS ships one PPO-shaped loss (``verl_grpo``); the
        # Tinker adapter maps any RL loss_fn to it and threads ``ppo`` /
        # ``importance_sampling`` semantics through ``actor_config``.
        out, _ = datum_list_to_arctic_batch(
            [_mk_datum([1], [0.1], [-1.0])],
            "importance_sampling", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
        )
        assert out["processing"]["loss_fn"] == "verl_grpo"

    def test_forward_only_flag_present(self):
        out, _ = datum_list_to_arctic_batch(
            [_mk_datum([1], [0.1], [-1.0])],
            "ppo", None,
            max_prompt_length=4, max_response_length=2, pad_token_id=0,
            forward_only=True,
        )
        assert out["meta"]["forward_only"] is True
        # actor_config is empty on forward_only (no loss reduction on the server).
        assert out["meta"]["actor_config"] == {}

    def test_max_response_len_threaded(self):
        out, _ = datum_list_to_arctic_batch(
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
