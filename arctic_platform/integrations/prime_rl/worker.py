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

"""The Cortex worker: the trainer and the policy inference server of a Cortex run.

The orchestrator and env servers run unchanged. Towards them this process looks like
a vLLM server (``/inference/v1/generate``, ``/v1/chat/completions`` and the admin
routes) and like a trainer: it consumes the orchestrator's batches and offers every
policy version through the filesystem broadcast handshake. Behind it, generations run
on the Cortex sampling sub-job, and each step is one forward-backward plus an
optimizer step on the training sub-job, followed by a weight sync to the sampler.

A weight sync pauses the sampler's in-flight generations, swaps the weights and resumes
them, keeping the prefix cache, as the native inference server's pause/update/resume
does. The orchestrator's own pause/update/resume sequence is only a barrier here: its
``/update_weights`` returns once the offered version is live on the sampler.
"""

import asyncio
import os
import re
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import orjson
from aiohttp import web
from arctic_platform.client import ArcticClientConfig, ArcticRLClient, AsyncArcticRLClient
from arctic_platform.client import CortexConfig as ArcticCortexConfig
from arctic_platform.client.requests import sync_weights_request
from renderers import Renderer, ToolCallParseStatus, create_renderer
from renderers.base import load_tokenizer

from prime_rl import monitors
from prime_rl.transports.batch import MicroBatch, setup_batch_receiver
from prime_rl.transports.weights.base import (
    FINISHED_MARKER,
    RECEIVER_READY_MARKER,
    SENDER_READY_MARKER,
    STARTED_MARKER,
)
from prime_rl.utils.config import cli
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.pathing import get_all_ckpt_steps, get_broadcast_dir, get_step_path
from prime_rl.utils.process import set_proc_title

from .config import CortexWorkerConfig
from .jobs import sub_job_id
from .loss import grpo_processing
from .payload import build_fwd_bwd_payload

SESSION_ID_HEADER = "X-Session-ID"
WEIGHT_DIR_PATTERN = re.compile(r"^step_(\d+)$")
# Sampling-param keys of vLLM's token generate route that are request options, not
# SamplingParams fields; Cortex forwards sampling params to SamplingParams verbatim.
DROPPED_SAMPLING_KEYS = frozenset(
    {"return_token_ids", "return_tokens_as_token_ids", "routed_experts_prompt_start", "cache_salt", "priority"}
)
CHAT_SAMPLING_KEYS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "max_tokens",
    "max_completion_tokens",
)
KEEPALIVE_MAX_FAILURES = 3
KEEPALIVE_RETRY_S = 60.0


def completion_token_ids(result: dict[str, Any]) -> list[int]:
    token_ids = result.get("token_ids", result.get("output_token_ids"))
    if not isinstance(token_ids, list):
        raise RuntimeError(f"Cortex generate result carries no completion token ids (keys: {sorted(result)})")
    return [int(token_id) for token_id in token_ids]


def sampled_logprobs(result: dict[str, Any], token_ids: list[int]) -> list[float]:
    """The sampled token's log-prob at every completion position. Cortex reports a
    position as a float, as one log-prob record, or as records keyed by token id."""
    positions = result.get("logprobs")
    if not isinstance(positions, list) or len(positions) != len(token_ids):
        raise RuntimeError(
            f"Cortex generate returned {len(positions) if isinstance(positions, list) else 'no'} log-probs "
            f"for {len(token_ids)} completion tokens"
        )
    logprobs = []
    for position, token_id in zip(positions, token_ids):
        if isinstance(position, dict) and "logprob" in position:
            position = position["logprob"]
        elif isinstance(position, dict):
            entry = position.get(token_id, position.get(str(token_id)))
            if entry is None:
                raise RuntimeError(f"Cortex log-probs at a position omit the sampled token {token_id}")
            position = entry["logprob"] if isinstance(entry, dict) else entry
        logprobs.append(float(position))
    return logprobs


def finish_reason(result: dict[str, Any]) -> str:
    reason = result.get("finish_reason")
    if not reason:
        raise RuntimeError("Cortex generate result carries no finish_reason")
    return str(reason)


def sanitize_sampling_params(sampling_params: dict[str, Any], *, prompt_len: int, max_model_len: int) -> dict:
    params = dict(sampling_params)
    params = {**(params.pop("extra_body", None) or {}), **params}
    for key in DROPPED_SAMPLING_KEYS:
        params.pop(key, None)
    max_completion_tokens = params.pop("max_completion_tokens", None)
    max_tokens = params.get("max_tokens")
    if max_tokens is None:
        max_tokens = max_completion_tokens
    # Like vLLM's serving layer: completions are bounded by the remaining context.
    budget = max_model_len - prompt_len
    params["max_tokens"] = budget if max_tokens is None else min(int(max_tokens), budget)
    return params


class CortexWorker:
    def __init__(self, config: CortexWorkerConfig) -> None:
        self.config = config
        self.cortex = config.cortex
        self.trainer = config.trainer
        self.model_name = self.trainer.model.name
        self.output_dir = self.trainer.output_dir
        self.logger = setup_logger(self.trainer.log.level, json_logging=self.trainer.log.json_logging)

        self.training_lock = asyncio.Lock()
        self.synced = asyncio.Condition()
        self.last_synced_step = 0
        self.requests_in_flight = 0
        self.last_request = time.monotonic()
        self.renderer_lock = asyncio.Lock()

    def client_config(self) -> ArcticClientConfig:
        connection = self.cortex.connection
        return ArcticClientConfig(
            model_name=self.model_name,
            max_seq_len=self.config.max_model_len,
            training_gpus=self.cortex.training.num_gpus,
            sampling_gpus=self.cortex.sampling.num_gpus,
            backend=ArcticCortexConfig(
                host=connection.host,
                database=connection.database,
                schema_=connection.schema_name,
                endpoint=connection.endpoint,
                pat=os.environ[connection.pat_env_var],
                max_retries=connection.max_retries,
                # The per-request / poll / job-ready timeouts live on the backend config.
                request_timeout=self.cortex.request_timeout,
                poll_timeout=self.cortex.poll_timeout,
                job_ready_timeout=self.cortex.job_ready_timeout,
            ),
            training_job_id=sub_job_id(self.config.job_id, "training"),
            sampling_job_id=sub_job_id(self.config.job_id, "sampling"),
        )

    @asynccontextmanager
    async def cortex_request(self) -> AsyncIterator[None]:
        self.requests_in_flight += 1
        try:
            yield
        finally:
            self.requests_in_flight -= 1
            self.last_request = time.monotonic()

    # ── sampling ─────────────────────────────────────────────────────────

    async def generate(self, prompt_ids: list[int], sampling_params: dict[str, Any], routing_key: str | None) -> dict:
        if len(prompt_ids) >= self.config.max_model_len:
            raise web.HTTPBadRequest(
                text=f"Prompt of {len(prompt_ids)} tokens leaves no room under max_model_len {self.config.max_model_len}"
            )
        params = sanitize_sampling_params(
            sampling_params, prompt_len=len(prompt_ids), max_model_len=self.config.max_model_len
        )
        # One prompt per request: the sampler schedules prompts individually, so a request carrying
        # several would hold its finished generations until the last one is scheduled and done.
        async with self.cortex_request():
            results = await self.sampler.generate([prompt_ids], sampling_params=[params], routing_key=[routing_key])
        if len(results) != 1:
            raise RuntimeError(f"Cortex generate returned {len(results)} results for one prompt")
        return results[0]

    async def handle_generate(self, request: web.Request) -> web.Response:
        body = await request.json(loads=orjson.loads)
        if body.get("features") is not None or body.get("content_parts") is not None:
            raise web.HTTPBadRequest(text="Cortex sampling serves text-only prompts.")
        prompt_ids = [int(token_id) for token_id in body["token_ids"]]
        result = await self.generate(prompt_ids, body.get("sampling_params") or {}, request.headers.get(SESSION_ID_HEADER))
        token_ids = completion_token_ids(result)
        logprobs = sampled_logprobs(result, token_ids)
        response = {
            "request_id": f"cortex-{uuid.uuid4().hex}",
            "choices": [
                {
                    "index": 0,
                    "token_ids": token_ids,
                    "logprobs": {
                        "content": [
                            {"token": f"token_id:{token_id}", "logprob": logprob}
                            for token_id, logprob in zip(token_ids, logprobs)
                        ]
                    },
                    "finish_reason": finish_reason(result),
                    "routed_experts": None,
                    "sampling_mask": None,
                }
            ],
            "prompt_token_ids": None,
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(token_ids),
                "total_tokens": len(prompt_ids) + len(token_ids),
            },
        }
        return web.Response(body=orjson.dumps(response), content_type="application/json")

    async def handle_chat_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json(loads=orjson.loads)
        if body.get("n", 1) != 1:
            raise web.HTTPBadRequest(text="Cortex chat completions serve n=1.")
        messages, tools = body["messages"], body.get("tools")
        sampling_params = {key: body[key] for key in CHAT_SAMPLING_KEYS if body.get(key) is not None}
        sampling_params.update(body.get("extra_body") or {})
        renderer = self.renderer_for(sampling_params.pop("chat_template_kwargs", None) or body.get("chat_template_kwargs"))
        async with self.renderer_lock:
            rendered = await asyncio.to_thread(renderer.render, messages, tools=tools, add_generation_prompt=True)
        prompt_ids = list(rendered.token_ids)
        sampling_params.update(stop_token_ids=renderer.get_stop_token_ids(), skip_special_tokens=False)
        result = await self.generate(prompt_ids, sampling_params, request.headers.get(SESSION_ID_HEADER))
        token_ids = completion_token_ids(result)
        async with self.renderer_lock:
            parsed = await asyncio.to_thread(renderer.parse_response, token_ids, prompt_ids=prompt_ids, tools=tools)

        message: dict[str, Any] = {"role": "assistant", "content": parsed.content}
        if parsed.reasoning_content is not None:
            message["reasoning_content"] = parsed.reasoning_content
        tool_calls = [
            {
                "id": call.id or f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments if isinstance(call.arguments, str) else orjson.dumps(call.arguments or {}).decode(),
                },
            }
            for call in parsed.tool_calls
            if call.status == ToolCallParseStatus.OK and call.name
        ]
        if tool_calls:
            message["tool_calls"] = tool_calls
        reason = finish_reason(result)
        if tool_calls and reason == "stop":
            reason = "tool_calls"
        usage = {
            "prompt_tokens": len(prompt_ids),
            "completion_tokens": len(token_ids),
            "total_tokens": len(prompt_ids) + len(token_ids),
        }
        completion = {
            "id": f"chatcmpl-cortex-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", self.model_name),
            "choices": [{"index": 0, "message": message, "finish_reason": reason}],
            "usage": usage,
        }
        if not body.get("stream"):
            return web.Response(body=orjson.dumps(completion), content_type="application/json")

        # One delta carries the whole message; streaming clients only need the protocol.
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        delta = dict(message)
        if tool_calls:
            delta["tool_calls"] = [{"index": index, **call} for index, call in enumerate(tool_calls)]
        chunk = {key: completion[key] for key in ("id", "created", "model")} | {"object": "chat.completion.chunk"}
        for choice, extra in (
            ({"index": 0, "delta": delta, "finish_reason": None}, {}),
            ({"index": 0, "delta": {}, "finish_reason": reason}, {"usage": usage}),
        ):
            await response.write(b"data: " + orjson.dumps(chunk | {"choices": [choice]} | extra) + b"\n\n")
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    def renderer_for(self, chat_template_kwargs: dict[str, Any] | None) -> Renderer:
        """The run's renderer, built once per distinct ``chat_template_kwargs``."""
        key = orjson.dumps(chat_template_kwargs or {}, option=orjson.OPT_SORT_KEYS)
        if key not in self.renderers:
            self.renderers[key] = create_renderer(
                self.tokenizer, self.config.renderer, chat_template_kwargs=chat_template_kwargs or None
            )
        return self.renderers[key]

    async def handle_models(self, request: web.Request) -> web.Response:
        model = {
            "id": self.model_name,
            "object": "model",
            "root": self.model_name,
            "owned_by": "cortex",
            "max_model_len": self.config.max_model_len,
        }
        return web.json_response({"object": "list", "data": [model]})

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_barrier(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def handle_update_weights(self, request: web.Request) -> web.Response:
        weight_dir = (await request.json()).get("weight_dir")
        match = WEIGHT_DIR_PATTERN.match(Path(weight_dir).name) if weight_dir else None
        if match is None:
            raise web.HTTPBadRequest(text=f"weight_dir must name a broadcast step_<N> directory, got {weight_dir!r}")
        step = int(match.group(1))
        async with self.synced:
            await self.synced.wait_for(lambda: self.last_synced_step >= step)
        return web.json_response({"status": "ok", "step": step})

    def app(self) -> web.Application:
        app = web.Application(client_max_size=256 * 1024**2)
        app.add_routes(
            [
                web.get("/health", self.handle_health),
                web.get("/v1/models", self.handle_models),
                web.post("/pause", self.handle_barrier),
                web.post("/resume", self.handle_barrier),
                web.post("/update_weights", self.handle_update_weights),
                web.post("/inference/v1/generate", self.handle_generate),
                web.post("/v1/chat/completions", self.handle_chat_completions),
            ]
        )
        return app

    # ── training ─────────────────────────────────────────────────────────

    async def training_op(self, fn, *args, **kwargs) -> dict[str, Any]:
        async with self.cortex_request():
            return await asyncio.to_thread(fn, *args, **kwargs)

    async def sync_weights(self) -> dict[str, float]:
        """Push the training weights to the sampler. Caller holds the training lock.

        Only the weight-sync operation: the client's sync_weights() also resets the prefix
        cache, which first waits for every in-flight generation to finish."""
        start = time.perf_counter()
        await self.training_op(self.training.transport.call, sync_weights_request(self.training.jobs))
        return {"time/sync_weights": time.perf_counter() - start}

    async def offer(self, step: int, *, sync: bool) -> dict[str, float]:
        """Run the trainer side of the broadcast handshake for policy v{step}."""
        step_dir = get_step_path(get_broadcast_dir(self.output_dir), step)
        shutil.rmtree(step_dir, ignore_errors=True)
        step_dir.mkdir(parents=True)
        (step_dir / SENDER_READY_MARKER).touch()
        deadline = time.monotonic() + self.trainer.weight_broadcast.timeout
        while not (step_dir / RECEIVER_READY_MARKER).exists():
            if time.monotonic() > deadline:
                raise TimeoutError(f"No receiver acknowledged policy v{step} within {self.trainer.weight_broadcast.timeout}s")
            await asyncio.sleep(0.1)
        (step_dir / STARTED_MARKER).touch()
        timings = await self.sync_weights() if sync else {}
        async with self.synced:
            self.last_synced_step = step
            self.synced.notify_all()
        (step_dir / FINISHED_MARKER).touch()
        broadcast_dir = get_broadcast_dir(self.output_dir)
        for old_step in get_all_ckpt_steps(broadcast_dir):
            if old_step < step - 1:
                shutil.rmtree(get_step_path(broadcast_dir, old_step), ignore_errors=True)
        return timings

    async def receive_batch(self) -> list[MicroBatch]:
        # Poll rather than block a thread on the transport, so shutdown never waits on it.
        while not self.receiver.can_receive():
            await asyncio.sleep(0.2)
        return await asyncio.to_thread(self.receiver.receive)

    async def train(self) -> None:
        processing = grpo_processing(self.trainer.loss)
        min_rows = self.cortex.training.num_gpus // self.cortex.training.sp_size
        # Only the MoE provider's chunked LM head takes a per-token temperature kwarg.
        include_temperature = self.cortex.training.model_provider == "prime_rl"
        self.logger.info("Offering the startup policy (v0), which the sampler already serves")
        await self.offer(0, sync=False)

        step = 1
        while self.trainer.max_steps is None or step <= self.trainer.max_steps:
            step_start = time.perf_counter()
            micro_batches = await self.receive_batch()
            wait_for_batch = time.perf_counter() - step_start

            t0 = time.perf_counter()
            payload, batch_stats = await asyncio.to_thread(
                build_fwd_bwd_payload,
                micro_batches,
                adv_tau=self.trainer.loss.adv_tau,
                min_rows=min_rows,
                include_temperature=include_temperature,
            )
            payload_time = time.perf_counter() - t0

            async with self.training_lock:
                t0 = time.perf_counter()
                fwd_bwd = await self.training_op(self.training.fwd_bwd, payload, processing=processing)
                fwd_bwd_time = time.perf_counter() - t0
                del payload
                t0 = time.perf_counter()
                optim_step = await self.training_op(self.training.step)
                optim_step_time = time.perf_counter() - t0
                t0 = time.perf_counter()
                sync_timings = await self.offer(step, sync=True)
                broadcast_time = time.perf_counter() - t0

            metrics = step_metrics(fwd_bwd, optim_step)
            metrics.update(batch_stats)
            metrics.update(sync_timings)
            metrics.update(
                {
                    "time/step": time.perf_counter() - step_start,
                    "time/wait_for_batch": wait_for_batch,
                    "time/build_payload": payload_time,
                    "time/forward_backward": fwd_bwd_time,
                    "time/optim_step": optim_step_time,
                    "time/broadcast_weights": broadcast_time,
                    "step": step,
                }
            )
            await monitors.log(metrics, step=step)
            self.logger.success(
                f"Step {step} | {metrics['time/step']:.1f}s | Loss {metrics.get('loss/mean', float('nan')):.4f} | "
                f"Grad norm {metrics.get('optim/grad_norm', float('nan')):.4f} | "
                f"{int(batch_stats['batch/num_samples'])} samples, {int(batch_stats['batch/num_loss_tokens'])} loss tokens"
            )
            step += 1

    # ── keepalive ────────────────────────────────────────────────────────

    def idle_seconds(self) -> float:
        return 0.0 if self.requests_in_flight else time.monotonic() - self.last_request

    async def keepalive(self, interval: float) -> None:
        """Cortex reaps a job that sees no request on any of its sub-jobs (polling an in-flight one
        counts) for its idle timeout. Rollouts and training steps keep the job busy; when both pause,
        a one-token generation does. Repeated failures end the run instead of letting it stall."""
        failures = 0
        while True:
            await asyncio.sleep(max(1.0, interval - self.idle_seconds()))
            idle = self.idle_seconds()
            if idle < interval:
                continue
            try:
                await self.generate([0], {"max_tokens": 1, "temperature": 0.0}, routing_key=None)
            except Exception as error:
                failures += 1
                self.logger.warning(f"Cortex keepalive failed ({failures}/{KEEPALIVE_MAX_FAILURES}): {error!r}")
                if failures >= KEEPALIVE_MAX_FAILURES:
                    raise
                await asyncio.sleep(KEEPALIVE_RETRY_S)
                continue
            failures = 0
            self.logger.info(f"Cortex keepalive after {idle:.0f}s without a request")

    # ── lifecycle ────────────────────────────────────────────────────────

    async def run(self) -> None:
        client_config = self.client_config()
        self.logger.info(f"Attaching to Cortex job {self.config.job_id}")
        self.training = await asyncio.to_thread(ArcticRLClient, client_config)
        self.sampler = await asyncio.to_thread(AsyncArcticRLClient, client_config)
        self.tokenizer = await asyncio.to_thread(load_tokenizer, self.model_name)
        self.renderers: dict[bytes, Renderer] = {}
        self.renderer_for(None)
        self.receiver = setup_batch_receiver(self.output_dir, 0, 1, self.trainer.rollout_transport)
        await monitors.setup(
            producer="trainer",
            wandb=self.trainer.monitors.wandb,
            file=self.trainer.monitors.file,
            output_dir=self.output_dir,
            run_config=self.config,
        )

        runner = web.AppRunner(self.app(), access_log=None)
        await runner.setup()
        await web.TCPSite(runner, self.cortex.host, self.cortex.port).start()
        self.logger.info(f"Serving {self.model_name} from Cortex at {self.cortex.base_url}")

        background = []
        if self.cortex.keepalive is not None:
            background.append(asyncio.create_task(self.keepalive(self.cortex.keepalive), name="keepalive"))
        train = asyncio.create_task(self.train(), name="train")
        try:
            try:
                done, _ = await asyncio.wait([train, *background], return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if train not in done:
                    raise RuntimeError(f"Cortex worker task {next(iter(done)).get_name()} exited")
                self.logger.success("Training finished!")
            finally:
                for task in [train, *background]:
                    task.cancel()
                await monitors.finalize()
            # Like a native run's inference server, keep serving until the launcher stops this
            # process after the orchestrator exits: the orchestrator applies the final policy here.
            await asyncio.Event().wait()
        finally:
            await runner.cleanup()
            # Close the HTTP sessions only: the clients' shutdown() cancels the Cortex job,
            # which belongs to the launcher.
            await self.sampler.transport.aclose()
            self.training.transport.session.close()


def step_metrics(fwd_bwd: dict[str, Any], optim_step: dict[str, Any]) -> dict[str, float]:
    """Cortex step telemetry, under the native trainer's names where one exists."""
    metrics = {
        f"cortex/{name}": float(value)
        for name, value in (fwd_bwd.get("metrics") or {}).items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if isinstance(fwd_bwd.get("avg_loss"), int | float):
        metrics["loss/mean"] = float(fwd_bwd["avg_loss"])
    if isinstance(mismatch_kl := (fwd_bwd.get("metrics") or {}).get("trainable_logprob_lowvar_all/mean"), int | float):
        metrics["mismatch_kl/all/mean"] = float(mismatch_kl)
    if isinstance(optim_step.get("grad_norm"), int | float):
        metrics["optim/grad_norm"] = float(optim_step["grad_norm"])
    lr = optim_step.get("last_lr")
    if isinstance(lr, list) and lr:
        lr = lr[0]
    if isinstance(lr, int | float):
        metrics["optim/lr"] = float(lr)
    return metrics


def main() -> None:
    set_proc_title("Cortex")
    config = cli(CortexWorkerConfig)
    asyncio.run(CortexWorker(config).run())


if __name__ == "__main__":
    main()
