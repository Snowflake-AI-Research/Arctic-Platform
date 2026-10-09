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

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from arctic_platform.inference.server.config import ModelConfig
from arctic_platform.inference.server.worker import InferenceWorker
from arctic_platform.inference.server.worker import WorkerLifecycleState


def test_yarn_engine_channel_preserves_additional_config():
    base = ModelConfig(model="tiny", extra_engine_kwargs={"additional_config": {"other": 7}})
    assert base.to_engine_kwargs()["additional_config"] == {"other": 7}
    active = base.model_copy(update={"yarn_factors": [1.0, 1.5, 2.0]})
    assert active.to_engine_kwargs()["additional_config"] == {
        "other": 7,
        "yarn_factors": [1.0, 1.5, 2.0],
    }
    assert "yarn_factors" not in active.model_dump()


def test_generate_factor_validation_slots_and_cache_salt():
    from transformers import Qwen3_5TextConfig

    from arctic_platform.common.yarn_factors import build_yarn_factors

    worker = InferenceWorker.__ray_metadata__.modified_class()
    worker.state = WorkerLifecycleState.READY
    worker._yarn_tables = build_yarn_factors(
        Qwen3_5TextConfig(
            rope_parameters={
                "rope_type": "yarn",
                "factor": 1.0,
                "original_max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.25,
            },
            head_dim=256,
            max_position_embeddings=128,
        ),
        [1.0, 1.5, 2.0],
    )
    from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
    from vllm.v1.core.kv_cache_utils import hash_block_tokens
    from vllm.v1.request import Request

    calls, hashes = [], []

    class Engine:
        async def generate(self, prompt, params, **kwargs):
            calls.append((prompt, params))
            yield SimpleNamespace(outputs=[], prompt_token_ids=[1, 2])

    worker.llm = Engine()
    for factor, expected in [(None, None), (1.0, None), (1.5, "yarn=1.5"), (2.0, "yarn=2.0")]:
        params = {"max_tokens": 1, "extra_args": {"other": 9}}
        if factor is not None:
            params["yarn_factor"] = factor
        original = dict(params)
        asyncio.run(worker.generate([1, 2], params))
        prompt, sampling = calls[-1]
        assert prompt.get("cache_salt") == expected
        slot = {1.5: 1, 2.0: 2}.get(factor)
        assert sampling.extra_args == ({"other": 9, "yarn_factor_slot": slot} if slot else {"other": 9})
        assert params == original
        request = Request(
            "test",
            prompt["prompt_token_ids"],
            sampling,
            None,
            cache_salt=prompt.get("cache_salt"),
        )
        keys, _ = generate_block_hash_extra_keys(request, 0, 2, 0)
        hashes.append(
            hash_block_tokens(
                lambda x: hashlib.sha256(repr(x).encode()).digest(),
                b"parent",
                [1, 2],
                keys,
            )
        )
    assert hashes[0] == hashes[1]
    assert len(set(hashes)) == 3
    for value in [1.25, None, float("nan")]:
        with pytest.raises(ValueError, match="yarn_factor"):
            asyncio.run(worker.generate([1, 2], {"yarn_factor": value}))
    with pytest.raises(ValueError, match="yarn_factor_slot"):
        asyncio.run(worker.generate([1, 2], {"extra_args": {"yarn_factor_slot": 1}}))
    worker._yarn_tables = None
    with pytest.raises(ValueError, match="yarn_factor"):
        asyncio.run(worker.generate([1, 2], {"yarn_factor": 1.0}))
    asyncio.run(worker.generate([1, 2], {"max_tokens": 1}))
    assert calls[-1][0] == {"prompt_token_ids": [1, 2]}
    assert calls[-1][1].extra_args is None
