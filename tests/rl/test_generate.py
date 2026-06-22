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

"""Arctic RL ``generate`` (sampling / vLLM) test.

Companion to ``test_train_engine.py``, which exercises the training engine on fake data; this one drives the
sampling engine: ``client.generate`` on a standalone sampling-only topology (``sampling_gpus=2``, no training job).
It runs the ``ray`` transport only -- the http generate-serialization path is already covered by ``test_e2e``'s
http cell, so a second vLLM spin-up here would be pure plumbing. Asserts each prompt round-trips to a non-empty text
completion. Heavyweight GPU test; shared infra lives in ``rl_harness``.

Tagged ``@pytest.mark.vllm`` + ``xdist_group("arctic_rl_vllm")``: under ``--dist loadgroup`` it shares a worker
with the other vLLM tests (never scheduled against them); ``-m "not vllm"`` drops it from a parallel pool.
"""

from __future__ import annotations

import asyncio

import pytest
from parameterized import parameterized
from rl_harness import arctic_rl_client_session
from rl_harness import assert_generations
from rl_harness import parameterized_custom_name_func
from rl_harness import skip_if_unsupported

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_multi_gpu

model_name = "Qwen/Qwen3-0.6B"
attn_implementation = "flash_attention_2"
# generate only needs the sampling (vLLM) job; no training job. prompt/response lengths size the vLLM max_model_len;
# rollout_n is irrelevant here.
training_gpus = 0
sampling_gpus = 2
log_prob_gpus = 0
prompt_len = 64
response_len = 64
rollout_n = 1

# ray only: this is the standalone sampling-only smoke; the http generate-serialization path is covered by test_e2e.
comm_params = [("ray",)]

prompts = ["The capital of France is", "2 + 2 ="]
sampling_params = {"temperature": 0.0, "max_tokens": 16}


async def _send_generate(client, prompts: list[str], sampling_params: dict) -> list[dict]:
    return await client.generate(prompts, sampling_params)


def _run_generate(comm_protocol: str) -> list[dict]:
    with arctic_rl_client_session(
        comm_protocol,
        False,
        model_name,
        attn_implementation,
        prompt_len,
        response_len,
        rollout_n,
        training_gpus,
        sampling_gpus,
        log_prob_gpus,
    ) as client:
        return asyncio.run(_send_generate(client, prompts, sampling_params))


@require_torch_multi_gpu
@pytest.mark.gpu_serial
@pytest.mark.vllm
@pytest.mark.xdist_group("arctic_rl_vllm")
class TestGenerate(TestCasePlus):
    @parameterized.expand(comm_params, name_func=parameterized_custom_name_func)
    def test_generate(self, comm_protocol):
        """Each transport round-trips prompts through the vLLM sampling engine."""
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus)
        results = _run_generate(comm_protocol)
        texts = assert_generations(results, len(prompts), comm_protocol)
        for prompt, text in zip(prompts, texts):
            print(f"[generate] {comm_protocol}: {prompt!r} -> {text[:80]!r}")
