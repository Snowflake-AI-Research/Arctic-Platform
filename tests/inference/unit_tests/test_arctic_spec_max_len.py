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

from __future__ import annotations

import pytest

import vllm
from vllm import LLM, SamplingParams

MAX_MODEL_LEN = 256

ARCTIC_BASE_MODEL = "Qwen/Qwen2.5-32B-Instruct"
ARCTIC_SPEC_MODEL = "Snowflake/Arctic-LSTM-Speculator-Qwen2.5-32B-Instruct"
SUFFIX_BASE_MODEL = "Qwen/Qwen3-0.6B"

ARCTIC_SPEC_CONFIG = {
    "method": "arctic",
    "model": ARCTIC_SPEC_MODEL,
    "num_speculative_tokens": 3,
    "disable_by_batch_size": 64,
    "enable_suffix_decoding": True,
}

SUFFIX_SPEC_CONFIG = {
    "method": "suffix",
    "disable_by_batch_size": 64,
}


@pytest.fixture
def test_prompts():
    return ["Hello"]


@pytest.fixture
def sampling_configs():
    return [
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN,
                       ignore_eos=True),
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN - 1,
                       ignore_eos=True),
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN - 2,
                       ignore_eos=True),
        SamplingParams(temperature=0,
                       max_tokens=MAX_MODEL_LEN - 3,
                       ignore_eos=True)
    ]


@pytest.mark.parametrize(
    "model_name, spec_config",
    [
        (ARCTIC_BASE_MODEL, ARCTIC_SPEC_CONFIG),
        (SUFFIX_BASE_MODEL, SUFFIX_SPEC_CONFIG),
    ],
    ids=["arctic", "suffix"],
)
def test_speculative_decoding(
    monkeypatch: pytest.MonkeyPatch,
    test_prompts: list[str],
    sampling_configs: list[SamplingParams],
    model_name: str,
    spec_config: dict,
):
    '''Spec decoding at max_model_len must not raise.'''
    with monkeypatch.context() as m:
        m.setenv("ARCTIC_INFERENCE_ENABLED", "1")
        m.setenv("VLLM_PLUGINS", "arctic_inference")
        m.setenv("VLLM_USE_V1", "1")

        vllm.plugins.load_general_plugins()

        spec_llm = LLM(
            model=model_name,
            tensor_parallel_size=2,
            speculative_config=spec_config,
            max_model_len=MAX_MODEL_LEN,
            enforce_eager=True,
            trust_remote_code=True,
        )

        for sampling_config in sampling_configs:
            try:
                spec_llm.generate(test_prompts, sampling_config)
            except Exception as e:
                method = spec_config.get('method', 'unknown')
                pytest.fail(
                    f"Speculative decoding with method '{method}' failed with error: {e}"
                )

        del spec_llm
