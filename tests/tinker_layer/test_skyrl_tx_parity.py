# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Parity tests borrowed from ``SkyRL/tests/tinker/test_api.py``.

These are the SkyRL-tx integration tests that drive the *real* upstream
``tinker`` Python SDK against a running HTTP server. SkyRL-tx uses them to
assert wire compatibility with Thinking Machines' hosted Tinker service;
we point them at Arctic so we can make the same claim.

Fixture ``arctic_server`` boots ``arctic_platform.rl.http_server`` as a
subprocess, provisions a training + sampling job via Arctic's native
``POST /initialize``, and finalises the Tinker layer via
``POST /tinker/bind``. Tests below the fixture are effectively unchanged
from upstream — the only edits are:

* ``LORA_RANK = 0`` (v1's full-weight training convention; ``rank>0``
  is rejected upstream in ``/create_model``).
* The ``test_training_workflow`` port swaps ``cross_entropy`` for the
  ``importance_sampling`` loss and drops the checkpoint-listing REST
  extras (``list_checkpoints`` / ``list_training_runs`` /
  ``get_checkpoint_archive_url``) that live outside the core Tinker
  protocol.
* ``test_sample`` runs the ``base_model`` branch only (LoRA is E1).

Requires GPUs. Skip with ``pytest -m 'not gpu'`` on CPU-only runs."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest


BASE_MODEL = os.environ.get("ARCTIC_TINKER_PARITY_MODEL", "Qwen/Qwen3-0.6B")
TEST_SERVER_PORT = int(os.environ.get("ARCTIC_TINKER_PARITY_PORT", "8100"))
TINKER_API_KEY = "tml-dummy"
MAX_PROMPT_LEN = 512
MAX_RESPONSE_LEN = 128
LORA_RANK = 0  # v1 FFT convention (SkyRL-tx uses rank=32; Arctic rejects rank>0 in /create_model)

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1200)]


# ---------------------------------------------------------------------------
# arctic_server fixture: spawn http_server, provision two jobs, bind Tinker.
# ---------------------------------------------------------------------------


def _get(url: str, timeout: float = 5.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode() if e.fp else ""
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        return 0, ""


def _post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _wait_until(pred, timeout_sec: float, poll_sec: float = 1.0) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(poll_sec)
    return False


@contextmanager
def _spawn_arctic_server(log_path: Path, port: int) -> Iterator[subprocess.Popen]:
    """Fork ``arctic_platform.rl.http_server`` and clean it up on exit."""
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    # Keep Ray tempdir off /tmp defaults so parallel local dev servers don't collide.
    env.setdefault("RAY_TMPDIR", str(log_path.parent / "ray_tmp"))
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "arctic_platform.rl.http_server",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--training-gpus", "1",
        "--sampling-gpus", "1",
        "--colocate",
    ]
    with open(log_path, "w") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group for kill(-pgid)
            env=env,
        )
    try:
        yield proc
    finally:
        # SIGTERM the whole process group so Ray/vLLM subprocesses go too.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def arctic_server(tmp_path_factory) -> Iterator[str]:
    """Boot Arctic on 1+1 GPU, provision jobs, bind Tinker, yield ``base_url``.

    Module-scoped: the ~90 s cold-start (Ray + DeepSpeed + vLLM) is amortised
    over every test in this file. Tests must remain order-independent.
    """
    workdir = tmp_path_factory.mktemp("arctic_parity")
    server_log = workdir / "server.log"
    base_url = f"http://127.0.0.1:{TEST_SERVER_PORT}"

    with _spawn_arctic_server(server_log, TEST_SERVER_PORT) as proc:
        # 1. Wait for /status to come up (uvicorn + Ray init).
        def status_up() -> bool:
            if proc.poll() is not None:
                pytest.fail(f"Arctic server died during boot:\n{server_log.read_text()[-3000:]}")
            return _get(f"{base_url}/status", timeout=2)[0] == 200
        if not _wait_until(status_up, timeout_sec=180):
            pytest.fail(f"Arctic /status did not respond in 180s:\n{server_log.read_text()[-3000:]}")

        # 2. Provision + bind via serve.sh — the same script the real recipes use.
        #    Keeps this fixture out of the business of duplicating Arctic's job schema.
        serve_sh = Path(__file__).resolve().parents[2] / "recipes" / "rl" / "tinker" / "serve.sh"
        assert serve_sh.exists(), f"serve.sh not found at {serve_sh}"
        serve_log = workdir / "serve.log"
        env = os.environ.copy()
        env.update(
            URL=base_url,
            MODEL=BASE_MODEL,
            MAX_PROMPT=str(MAX_PROMPT_LEN),
            MAX_RESPONSE=str(MAX_RESPONSE_LEN),
            ZORRO_ENABLE="0",
            CKPT_DIR=str(workdir / "ckpt"),
        )
        with open(serve_log, "w") as f:
            rc = subprocess.call(["bash", str(serve_sh)], stdout=f, stderr=subprocess.STDOUT, env=env, timeout=900)
        if rc != 0:
            pytest.fail(f"serve.sh exit={rc}\n--- serve.log tail ---\n{serve_log.read_text()[-2000:]}\n"
                        f"--- server.log tail ---\n{server_log.read_text()[-2000:]}")

        # 3. Readiness gate: /api/v1/healthz must report bound=True.
        def bound() -> bool:
            status, body = _get(f"{base_url}/api/v1/healthz", timeout=2)
            return status == 200 and json.loads(body or "{}").get("bound") is True
        if not _wait_until(bound, timeout_sec=30):
            pytest.fail(f"tinker layer never reported bound=True:\n{server_log.read_text()[-3000:]}")

        yield base_url


@pytest.fixture(scope="module")
def service_client(arctic_server):
    """Public Tinker SDK client rooted at Arctic. Same call shape SkyRL-tx uses."""
    import tinker

    return tinker.ServiceClient(base_url=arctic_server + "/", api_key=TINKER_API_KEY)


# ---------------------------------------------------------------------------
# Ported tests. See file docstring for the (small) list of edits.
# ---------------------------------------------------------------------------


def _rl_datum(tokenizer, prompt: str, completion: str, advantage: float = 1.0):
    """RL-flavoured datum for ``importance_sampling`` loss.

    Arctic v1 does not expose ``cross_entropy`` (see ``_V1_UNSUPPORTED_LOSSES``
    in ``tinker_router.py``); this is the RL-loss analogue of SkyRL-tx's
    upstream ``make_datum`` helper — same wire shape, ``advantages`` instead
    of ``weights``.
    """
    from tinker import types
    from tinker.types.tensor_data import TensorData
    import torch

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    completion_tokens = tokenizer.encode(f"{completion}\n\n", add_special_tokens=False)
    all_tokens = prompt_tokens + completion_tokens
    n_prompt, n_resp = len(prompt_tokens), len(completion_tokens)
    return types.Datum(
        model_input=types.ModelInput.from_ints(all_tokens),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor([0] * n_prompt + completion_tokens, dtype=torch.int64)),
            "advantages": TensorData.from_torch(torch.tensor([0.0] * n_prompt + [advantage] * n_resp, dtype=torch.float32)),
            "logprobs": TensorData.from_torch(torch.tensor([0.0] * (n_prompt + n_resp), dtype=torch.float32)),
        },
    )


def test_capabilities(service_client):
    """Verbatim from SkyRL-tx: bound base model appears in server capabilities."""
    caps = service_client.get_server_capabilities()
    assert BASE_MODEL in [item.model_name for item in caps.supported_models]


def test_training_workflow_core(service_client):
    """Adapted from SkyRL-tx ``test_training_workflow`` — drops REST-only
    checkpoint-listing extras that live outside core Tinker, and swaps
    ``cross_entropy`` for ``importance_sampling`` (Arctic v1's RL-focused
    supported-loss set).

    Assertions kept: forward_backward returns per-Datum loss_fn_outputs of
    the right length, optim_step returns metrics, weight sync survives a
    round-trip."""
    from tinker import types

    tc = service_client.create_lora_training_client(base_model=BASE_MODEL, rank=LORA_RANK)
    tokenizer = tc.get_tokenizer()

    data = [
        _rl_datum(tokenizer, "Question: What is 2+2?\nAnswer:", " 4", advantage=1.0),
        _rl_datum(tokenizer, "Question: What color is the sky?\nAnswer:", " Blue", advantage=0.5),
        _rl_datum(tokenizer, "Question: What is 3+3?\nAnswer:", " 6", advantage=0.0),
    ]

    tc.save_weights_for_sampler(name="pre_train").result()

    fwd_bwd = tc.forward_backward(data, "importance_sampling").result()
    optim = tc.optim_step(types.AdamParams(learning_rate=1e-6)).result()

    assert fwd_bwd is not None
    assert len(fwd_bwd.loss_fn_outputs) == len(data)
    assert optim is not None

    # Weight sync must survive a full forward_backward + step round-trip.
    tc.save_weights_for_sampler(name="post_train").result()


def test_sample_base_model(service_client):
    """Ported from SkyRL-tx ``test_sample[base_model]`` — LoRA branch omitted."""
    from tinker import types
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)

    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))
    num_samples_per_request = [1, 2]
    max_tokens_per_request = [20, 10]

    requests = [
        sampling_client.sample(
            prompt=prompt,
            sampling_params=types.SamplingParams(temperature=0.0, max_tokens=mt, seed=42),
            num_samples=n,
        )
        for n, mt in zip(num_samples_per_request, max_tokens_per_request)
    ]
    for request, n, mt in zip(requests, num_samples_per_request, max_tokens_per_request):
        result = request.result()
        assert len(result.sequences) == n
        assert len(result.sequences[0].tokens) == mt


def test_sample_top_k(service_client):
    """Verbatim from SkyRL-tx: top_k=1 collapses variance, top_k=-1 keeps it."""
    from tinker import types
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))

    def sample_with_top_k(top_k: int, num_runs: int = 3):
        return [
            sampling_client.sample(
                prompt=prompt,
                sampling_params=types.SamplingParams(temperature=1.0, max_tokens=5, seed=42 + i, top_k=top_k),
                num_samples=1,
            ).result().sequences[0].tokens
            for i in range(num_runs)
        ]

    results_top_1 = sample_with_top_k(top_k=1)
    assert all(r == results_top_1[0] for r in results_top_1), "top_k=1 should be deterministic across seeds"

    results_no_top_k = sample_with_top_k(top_k=-1)
    assert not all(r == results_no_top_k[0] for r in results_no_top_k), "top_k=-1 should vary across seeds"


def test_sample_num_samples_diversity(service_client):
    """Verbatim from SkyRL-tx: num_samples>1 gives diverse, reproducible-per-seed sequences."""
    from tinker import types
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    sampling_client = service_client.create_sampling_client(base_model=BASE_MODEL)
    prompt = types.ModelInput.from_ints(tokenizer.encode("Hello, how are you doing today? ", add_special_tokens=True))

    num_samples = 3
    params = types.SamplingParams(temperature=1.0, max_tokens=10, seed=42)

    result1 = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=num_samples).result()
    assert len(result1.sequences) == num_samples
    tokens1 = [seq.tokens for seq in result1.sequences]
    assert len({tuple(t) for t in tokens1}) > 1, "num_samples>1 should produce diverse sequences"

    result2 = sampling_client.sample(prompt=prompt, sampling_params=params, num_samples=num_samples).result()
    tokens2 = [seq.tokens for seq in result2.sequences]
    assert tokens1 == tokens2, "same seed should reproduce identical results"

    result3 = sampling_client.sample(
        prompt=prompt,
        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=10, seed=999),
        num_samples=num_samples,
    ).result()
    tokens3 = [seq.tokens for seq in result3.sequences]
    assert tokens1 != tokens3, "different seed should produce different results"
