# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end acceptance test: real ``tinker.ServiceClient`` drives our
router through a live HTTP transport, running a full 10-step RL loop.

This is the conformance test — every wire verb we implement must be hit at
least once, and the round-trip against the upstream SDK's own Pydantic
models must round-trip cleanly. The Arctic backend is mocked (returns
deterministic tokens/logprobs) so the test runs CPU-only.

Skipped when the ``tinker`` SDK is not installed. Install with
``pip install tinker`` to run.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import threading
import time

import pytest
import uvicorn

tinker = pytest.importorskip("tinker")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _UvicornInThread:
    """Run a uvicorn server on a background thread, tearing it down on exit.

    ``uvicorn.Server.serve`` runs on its own asyncio loop so it stays independent
    of pytest-asyncio's per-test loop; this is important for the SDK, which
    creates a separate internal event-loop thread on its side.
    """

    def __init__(self, app, host: str = "127.0.0.1", port: int | None = None) -> None:
        self.host = host
        self.port = port or _free_port()
        config = uvicorn.Config(app, host=host, port=self.port, log_level="warning",
                                access_log=False)
        self.server = uvicorn.Server(config)
        self.thread: threading.Thread | None = None

    def __enter__(self):
        def _run():
            asyncio.run(self.server.serve())

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        # Wait for the server to bind.
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                with socket.create_connection((self.host, self.port), timeout=0.2):
                    return self
            except OSError:
                time.sleep(0.05)
        raise RuntimeError(f"uvicorn failed to start on {self.host}:{self.port}")

    def __exit__(self, exc_type, exc, tb):
        self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout=5)


@pytest.fixture
def mocked_app():
    """FastAPI app with the Tinker router mounted + deterministic mock backend."""
    from fastapi import FastAPI

    from arctic_platform.rl.tinker_router import init_tinker_state
    from arctic_platform.rl.tinker_router import router as tinker_router

    calls: dict[str, list] = {"fwd_bwd": [], "step": [], "sync": [], "gen": []}

    async def fwd_bwd(batch):
        calls["fwd_bwd"].append(batch)
        return {"metrics": {"loss": 0.4}, "avg_loss": 0.4}

    async def fwd_no_grad(batch):
        import numpy as np
        bsz, seqlen = batch["batch"]["input_ids"].shape
        return {"batch": {"logprobs": np.zeros((bsz, seqlen), dtype=np.float32)},
                "metrics": {}}

    async def step(overrides):
        calls["step"].append(overrides)
        return {"metrics": {"last_lr": overrides["lr"] if overrides else 1e-4,
                            "grad_norm": 1.0}}

    async def sync():
        calls["sync"].append(True)
        return {"ok": True}

    async def generate(prompt_tokens, sampling_params):
        calls["gen"].append((list(prompt_tokens), dict(sampling_params)))
        n = sampling_params.get("n", 1)
        max_tokens = sampling_params.get("max_tokens", 4)
        return {"outputs": [{
            "token_ids": list(range(200, 200 + max_tokens)),
            "logprobs": [-0.5] * max_tokens,
            "finish_reason": "stop",
        } for _ in range(n)]}

    app = FastAPI()
    app.include_router(tinker_router)
    init_tinker_state(
        app,
        base_model="Qwen/Qwen3-0.6B",
        max_prompt_length=16,
        max_response_length=8,
        pad_token_id=0,
        fwd_bwd_handler=fwd_bwd,
        fwd_no_grad_handler=fwd_no_grad,
        step_handler=step,
        sync_weights_handler=sync,
        generate_handler=generate,
    )
    return app, calls


def test_e2e_service_client_bootstrap(mocked_app, monkeypatch):
    """Just spinning up a ``ServiceClient`` must succeed against our server:
    exercises ``/client/config``, ``/create_session``, ``/get_server_capabilities``."""
    app, calls = mocked_app
    with _UvicornInThread(app) as server:
        monkeypatch.setenv("TINKER_BASE_URL", f"http://{server.host}:{server.port}")
        monkeypatch.setenv("TINKER_API_KEY", "tml-dummy")
        client = tinker.ServiceClient()
        caps = client.get_server_capabilities()
        assert any(m.model_name == "Qwen/Qwen3-0.6B" for m in caps.supported_models)


def test_e2e_create_lora_training_client_full_weight(mocked_app, monkeypatch):
    """``create_lora_training_client(rank=0)`` = FFT convention must succeed."""
    app, calls = mocked_app
    with _UvicornInThread(app) as server:
        monkeypatch.setenv("TINKER_BASE_URL", f"http://{server.host}:{server.port}")
        monkeypatch.setenv("TINKER_API_KEY", "tml-dummy")
        client = tinker.ServiceClient()
        tc = client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=0)
        assert tc is not None
        assert tc.model_id == "main"


def test_e2e_create_lora_training_client_rank_positive_rejected(mocked_app, monkeypatch):
    """Rank > 0 must surface as a client-visible error."""
    app, _ = mocked_app
    with _UvicornInThread(app) as server:
        monkeypatch.setenv("TINKER_BASE_URL", f"http://{server.host}:{server.port}")
        monkeypatch.setenv("TINKER_API_KEY", "tml-dummy")
        client = tinker.ServiceClient()
        with pytest.raises(Exception):
            client.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=32)


def test_e2e_rl_step(mocked_app, monkeypatch):
    """Single fwd_bwd + optim_step + save_weights + sample round-trip."""
    from tinker import types

    app, calls = mocked_app
    with _UvicornInThread(app) as server:
        monkeypatch.setenv("TINKER_BASE_URL", f"http://{server.host}:{server.port}")
        monkeypatch.setenv("TINKER_API_KEY", "tml-dummy")
        service = tinker.ServiceClient()
        tc = service.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=0)

        # 1) sync weights + get a sampling client
        sc = tc.save_weights_and_get_sampling_client()
        assert sc is not None
        assert len(calls["sync"]) == 1

        # 2) sample: prompt (a small token sequence) → 2 completions
        prompt = types.ModelInput.from_ints([10, 11, 12])
        params = types.SamplingParams(max_tokens=4, temperature=0.7)
        sample_fut = sc.sample(prompt=prompt, num_samples=2, sampling_params=params)
        result = sample_fut.result()
        assert len(result.sequences) == 2
        assert result.sequences[0].tokens[:4] == [200, 201, 202, 203]
        assert result.sequences[0].stop_reason == "stop"

        # 3) forward_backward: PPO with clip config
        datum = types.Datum(
            model_input=prompt,
            loss_fn_inputs={
                "target_tokens": [10, 11, 12],
                "advantages": [0.5, 0.5, 0.5],
                "logprobs": [-0.5, -0.5, -0.5],
                "weights": [1.0, 1.0, 1.0],
            },
        )
        fbwd_fut = tc.forward_backward(
            data=[datum],
            loss_fn="ppo",
            loss_fn_config={"clip_low_threshold": 0.9,
                            "clip_high_threshold": 1.1},
        )
        fbwd = fbwd_fut.result()
        # Metric names carry a ``:reduction`` suffix as required by
        # Tinker's ``combine_fwd_bwd_output_results``.
        assert fbwd.metrics["loss:mean"] == pytest.approx(0.4)

        # 4) optim_step
        adam = types.AdamParams(learning_rate=1e-4)
        step_fut = tc.optim_step(adam)
        step_result = step_fut.result()
        # OptimStepResponse doesn't guarantee any particular metric name across
        # SDK versions — just confirm the round-trip landed.
        assert step_result is not None
        assert len(calls["step"]) == 1
        assert calls["step"][-1]["lr"] == pytest.approx(1e-4)


def test_e2e_ten_step_rl_loop(mocked_app, monkeypatch):
    """Full 10-step RL loop — every wire verb hit, deterministic mock rewards."""
    from tinker import types

    app, calls = mocked_app
    with _UvicornInThread(app) as server:
        monkeypatch.setenv("TINKER_BASE_URL", f"http://{server.host}:{server.port}")
        monkeypatch.setenv("TINKER_API_KEY", "tml-dummy")
        service = tinker.ServiceClient()
        tc = service.create_lora_training_client(base_model="Qwen/Qwen3-0.6B", rank=0)

        prompt = types.ModelInput.from_ints([10, 11, 12])
        params = types.SamplingParams(max_tokens=4, temperature=0.7)
        adam = types.AdamParams(learning_rate=1e-4)

        for step_i in range(10):
            sc = tc.save_weights_and_get_sampling_client()
            fut = sc.sample(prompt=prompt, num_samples=2, sampling_params=params)
            samples = fut.result()
            assert len(samples.sequences) == 2

            # Advantage = 1.0 if the rollout finished on stop, else 0.
            batch = []
            for seq in samples.sequences:
                adv = 1.0 if seq.stop_reason == "stop" else 0.0
                batch.append(types.Datum(
                    model_input=types.ModelInput.from_ints(
                        list(prompt.to_ints()) + list(seq.tokens)
                    ),
                    loss_fn_inputs={
                        "target_tokens": list(prompt.to_ints()) + list(seq.tokens),
                        "advantages": [adv] * (len(prompt.to_ints()) + len(seq.tokens)),
                        "logprobs": ([0.0] * len(prompt.to_ints())
                                     + list(seq.logprobs or [0.0] * len(seq.tokens))),
                        "weights": ([0.0] * len(prompt.to_ints())
                                    + [1.0] * len(seq.tokens)),
                    },
                ))

            tc.forward_backward(data=batch, loss_fn="ppo",
                                loss_fn_config={"clip_low_threshold": 0.8,
                                                "clip_high_threshold": 1.2}).result()
            tc.optim_step(adam).result()

        assert len(calls["sync"]) == 10
        assert len(calls["gen"]) == 10
        assert len(calls["fwd_bwd"]) == 10
        assert len(calls["step"]) == 10

        # Weight generation was bumped every step.
        assert app.state.tinker_weight_gen == 10
