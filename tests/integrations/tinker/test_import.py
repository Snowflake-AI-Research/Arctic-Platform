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

"""The in-process client registers itself as ``tinker`` before a recipe imports it.

Importing ``arctic_platform.integrations.tinker`` inside this process would replace
``sys.modules["tinker"]`` for the rest of the suite, so these checks run in a
child interpreter.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("tinker")


def test_import_registers_the_tinker_module() -> None:
    script = """
import arctic_platform.integrations.tinker
import tinker
from tinker.types import LossFnType
assert tinker.ServiceClient.__module__ == "arctic_platform.integrations.tinker", tinker.ServiceClient
assert "importance_sampling" in LossFnType.__args__
tinker.configure(training_gpus=1, sampling_gpus=1)
try:
    tinker.configure(training_gpus=0, sampling_gpus=1)
except ValueError:
    pass
else:
    raise SystemExit("configure accepted zero training GPUs")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_teacher_logprobs_use_the_prompt_tokens() -> None:
    script = """
import asyncio
import arctic_platform.integrations.tinker as tinker

seen = {}

class Session:
    async def generate(self, tokens, params):
        seen["tokens"] = list(tokens)
        seen["params"] = dict(params)
        return {
            "outputs": [{"token_ids": [7], "logprobs": [-0.1], "finish_reason": "length"}],
            "prompt_logprobs": [None, -0.4, -0.2],
        }

prompt = tinker.types.ModelInput.from_ints([11, 12, 13])
logprobs = asyncio.run(tinker.SamplingClient(Session()).compute_logprobs_async(prompt))
assert seen["tokens"] == [11, 12, 13], seen
assert seen["params"]["prompt_logprobs"] == 0, seen
assert seen["params"]["max_tokens"] == 1, seen
assert logprobs == [None, -0.4, -0.2], logprobs
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_sampler_save_syncs_the_live_weights() -> None:
    script = """
import asyncio
import arctic_platform.integrations.tinker as tinker

synced = {}

async def sync():
    synced["ok"] = True

client = tinker.TrainingClient(None, {"sync_weights_handler": sync}, 0, 8, 8)

async def main():
    saved = await client.save_weights_for_sampler_async("step20")
    return await saved.result_async()

result = asyncio.run(main())
assert synced.get("ok") is True
assert result.path == "cortex://session/step20/sampler"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_cookbook_client_contract() -> None:
    script = """
import asyncio
import arctic_platform.integrations.tinker as tinker

synced = {}

async def sync():
    synced["n"] = synced.get("n", 0) + 1

client = tinker.TrainingClient(None, {"sync_weights_handler": sync}, 0, 8, 8)

async def main():
    saved = await client.save_weights_for_sampler_async("000020")
    result = await saved
    sampler = client.create_sampling_client(result.path)
    assert sampler._session is client
    try:
        client.create_sampling_client("cortex://session/other/sampler")
    except RuntimeError as exc:
        assert "just saved" in str(exc)
    else:
        raise SystemExit("accepted a sampler path from another save")
    return result.path

path = asyncio.run(main())
assert path == "cortex://session/000020/sampler"
assert synced["n"] == 1

service = tinker.ServiceClient(training_gpus=1, sampling_gpus=1)
service._session = client
service._student_spec = ("Qwen/Qwen3.5-4B", 32, True, True, True)
sentinel = tinker.TrainingClient(None, {}, 0, 8, 8)
service._open_teacher = lambda model: sentinel
base = service.create_sampling_client(base_model="Qwen/Qwen3.5-4B")
assert base._session is sentinel

try:
    service.create_sampling_client()
except ValueError as exc:
    assert "base_model" in str(exc)
else:
    raise SystemExit("create_sampling_client accepted no model")

try:
    service.create_sampling_client(model_path="tinker://run/sampler")
except RuntimeError as exc:
    assert "not supported" in str(exc)
else:
    raise SystemExit("checkpoint model_path was accepted")

try:
    asyncio.run(service.create_training_client_from_state_async(path))
except RuntimeError as exc:
    assert "cannot load" in str(exc)
else:
    raise SystemExit("resume was accepted")

try:
    service.create_rest_client()
except ValueError as exc:
    assert "not available" in str(exc)
else:
    raise SystemExit("rest client was created")

try:
    asyncio.run(service.create_lora_training_client_async("other/model"))
except RuntimeError as exc:
    assert "already opened" in str(exc)
else:
    raise SystemExit("a second model was accepted")

class Session:
    async def generate(self, tokens, params):
        raise SystemExit("temperature should be refused before generate")

prompt = tinker.types.ModelInput.from_ints([1])
try:
    asyncio.run(
        tinker.SamplingClient(Session()).sample_async(
            prompt, sampling_params=tinker.SamplingParams(temperature=0.0)
        )
    )
except RuntimeError as exc:
    assert "temperature" in str(exc)
else:
    raise SystemExit("temperature 0 was accepted")
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_optim_step_refuses_adam_the_job_cannot_apply() -> None:
    script = """
import asyncio
import arctic_platform.integrations.tinker as tinker

called = {}

async def step(overrides):
    called["overrides"] = overrides
    return {"metrics": {}}

client = tinker.TrainingClient(
    None,
    {"step_handler": step},
    0,
    8,
    8,
    fixed_adam={"beta1": 0.9, "beta2": 0.95, "eps": 1e-8, "weight_decay": 0.0, "grad_clip_norm": 0.0},
)

async def main():
    client._have_grad = True
    try:
        await (await client.optim_step_async(tinker.AdamParams(learning_rate=1e-4, eps=1e-12))).result_async()
    except ValueError as exc:
        assert "learning rate" in str(exc)
    else:
        raise SystemExit("a different Adam eps was accepted")
    assert "overrides" not in called
    client._have_grad = True
    await (
        await client.optim_step_async(
            tinker.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.0, grad_clip_norm=0.0)
        )
    ).result_async()
    assert called["overrides"]["lr"] == 1e-4

asyncio.run(main())
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_launcher_requires_gpu_counts() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "arctic_platform.integrations.tinker.run", "tinker_cookbook.recipes.math_rl.train"],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "--training-gpus" in completed.stderr + completed.stdout
