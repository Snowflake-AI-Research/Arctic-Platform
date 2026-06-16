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

"""Arctic RL log-prob engine tests: the DeepSpeed reference engine and the text ``log_probs`` API.

Every other RL test runs with log_prob_gpus=0, so the log-prob job branch (ray_server / http_server initialize for
"log_prob"), the forward-only DeepSpeed engine, fwd_no_grad(reference_model=True) -- the KL-reference path -- and
the client ``log_probs`` text API never run; the e2e loop only hits the "no_log_prob_job" no-op. This stands up a
log-prob-only topology (log_prob_gpus=1, log_prob_engine="deepspeed") and covers two entry points:

  * ``test_reference_log_prob``: wake the engine, run fwd_no_grad(reference_model=True), offload it again. The
    reference per-token log-probs must match an independent per-row unpadded HF reference (same correctness bar as
    the training-engine forward in test_train_engine) and be finite. ``ray`` runs both ZoRRO on/off -- the only
    forward path this entry adds; http there is pure serialization plumbing already covered by test_train_engine /
    test_e2e, so it is omitted.
  * ``test_text_log_probs``: the high-level ``client.log_probs(prompts, completions)`` path -- server-side
    tokenization + the DeepSpeed worker's full-sequence ``compute_log_probs``. This is the one place that path runs,
    and its server wiring differs per transport (ray_server vs http_server build/split the batch separately), so it
    runs over both ``ray`` and ``http``.

Shared infra lives in ``rl_harness``. Heavyweight GPU test::

    pytest tests/rl/test_log_prob_engine.py -s
"""

from __future__ import annotations

import asyncio

import pytest
import torch
from parameterized import parameterized

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import torch_assert_close
from rl_harness import arctic_rl_client_session
from rl_harness import assert_finite_logprobs
from rl_harness import build_compute_log_prob_payload
from rl_harness import cached_padded_batch_and_reference
from rl_harness import cell_tag
from rl_harness import parameterized_custom_name_func
from rl_harness import response_region
from rl_harness import skip_if_unsupported

model_name = "Qwen/Qwen3-0.6B"
attn_implementation = "flash_attention_2"
num_unique_prompts = 2
rollout_n = 2
prompt_len = 8
response_len = 8

# Reference-only topology: a single forward-only DeepSpeed engine on its own GPU, no training or sampling job. This
# is the one place fwd_no_grad(reference_model=True) and the log_prob job branch execute.
training_gpus = 0
sampling_gpus = 0
log_prob_gpus = 1

# atol vs the reference (rtol=0): bf16 kernel/reduction noise is small while a misaligned return is off by several
# nats, so this absorbs bf16 jitter yet rejects garbage.
LOGPROB_ATOL = 0.25

# ray covers both reference-engine forward paths (ZoRRO on/off). http is pure serialization plumbing, independent of
# the forward path and already covered by test_train_engine's http cell + test_e2e, so it's dropped to save a spin-up.
log_prob_params = [("ray", True), ("ray", False)]

# Short prompt+completion texts for the log_probs API; the combined token count stays well under the engine's
# max_length (prompt_len + response_len). The server tokenizes these, so geometry above doesn't constrain them.
text_prompts = ["The capital of France is", "2 + 2 ="]
text_completions = [" Paris", " 4"]

# log_probs server wiring differs per transport (ray_server vs http_server), so cover both.
text_log_prob_params = [("ray",), ("http",)]


@require_torch_gpu
@pytest.mark.gpu_serial
@pytest.mark.xdist_group("arctic_rl_train")  # group with the other DeepSpeed-only tests; share one worker / GPU lock
class TestLogProbEngine(TestCasePlus):
    @staticmethod
    async def _drive(client, cl_payload: dict) -> dict:
        # Wake the offloaded reference engine, run the forward-only ref pass, then offload it again.
        await client.wake_log_prob()
        response = await client.fwd_no_grad(cl_payload, reference_model=True)
        await client.sleep_log_prob()
        return response

    @parameterized.expand(log_prob_params, name_func=parameterized_custom_name_func)
    def test_reference_log_prob(self, comm_protocol, zorro_enable):
        """Reference-engine fwd_no_grad(reference_model=True) log-probs match the independent HF reference."""
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus)
        batch, _, response_lens, ref, valid = cached_padded_batch_and_reference(
            model_name, attn_implementation, num_unique_prompts, rollout_n, prompt_len, response_len
        )
        cl_payload = build_compute_log_prob_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)
        tag = cell_tag(comm_protocol, zorro_enable)
        with arctic_rl_client_session(
            comm_protocol, zorro_enable, model_name, attn_implementation, prompt_len, response_len, rollout_n,
            training_gpus, sampling_gpus, log_prob_gpus,
        ) as client:
            logprob_response = asyncio.run(self._drive(client, cl_payload))

        logprobs = assert_finite_logprobs(logprob_response, batch)
        got_resp = response_region(logprobs, zorro_enable, prompt_len, response_len)
        print(f"[log-prob-engine] {tag} response_lens={response_lens}")
        torch_assert_close(got_resp[valid], ref[valid], rtol=0, atol=LOGPROB_ATOL, msg=f"{tag} reference tokens")

    @staticmethod
    async def _drive_text_log_probs(client) -> dict:
        # Wake the offloaded engine, score the text, then offload it again.
        await client.wake_log_prob()
        response = await client.log_probs(text_prompts, text_completions)
        await client.sleep_log_prob()
        return response

    @parameterized.expand(text_log_prob_params, name_func=parameterized_custom_name_func)
    def test_text_log_probs(self, comm_protocol):
        """client.log_probs(prompts, completions) scores text through the DeepSpeed engine over both transports."""
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus)
        with arctic_rl_client_session(
            comm_protocol, False, model_name, attn_implementation, prompt_len, response_len, rollout_n,
            training_gpus, sampling_gpus, log_prob_gpus,
        ) as client:
            response = asyncio.run(self._drive_text_log_probs(client))

        self.assertIn("results", response, f"[{comm_protocol}] log_probs response missing 'results': {list(response)}")
        results = response["results"]
        self.assertTrue(torch.is_tensor(results), f"[{comm_protocol}] expected results tensor, got {type(results)}")
        self.assertEqual(results.ndim, 2, f"[{comm_protocol}] expected [B, S-1] results, got {tuple(results.shape)}")
        self.assertEqual(results.shape[0], len(text_prompts), f"[{comm_protocol}] results batch dim mismatch")
        self.assertTrue(torch.isfinite(results).all(), f"[{comm_protocol}] log_probs contain non-finite values")
        print(f"[log-prob-engine] text log_probs {comm_protocol}: results shape={tuple(results.shape)}")
