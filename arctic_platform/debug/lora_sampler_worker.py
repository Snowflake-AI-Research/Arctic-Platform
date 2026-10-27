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

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Explicit external-pool LoRA worker, not a full-parameter sync verifier."""

from __future__ import annotations

import functools
import os
from copy import deepcopy
from uuid import uuid4

from arctic_platform.debug.lora_sampler_evidence import _bind_sources
from arctic_platform.debug.lora_sampler_evidence import _require
from arctic_platform.debug.lora_sampler_evidence import engine_kwargs_for_lora
from arctic_platform.debug.lora_sampler_evidence import serving_lora_request
from arctic_platform.debug.rl_sampler_evidence import SamplerEvidenceWorker
from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import verify_sources


def _request(request):
    _require(
        type(request) is dict
        and set(request)
        == {
            "lora_int_id",
            "lora_name",
            "lora_config",
            "max_received_bytes",
            "admitted_sources",
        },
        "exact explicit adapter worker request required",
    )
    _require(
        type(request["lora_int_id"]) is int
        and request["lora_int_id"] > 0
        and type(request["lora_name"]) is str
        and request["lora_name"],
        "positive owned adapter identity required",
    )
    _require(
        type(request["lora_config"]) is dict
        and type(request["lora_config"].get("r")) is int
        and request["lora_config"]["r"] > 0,
        "actual positive configured adapter rank required",
    )
    _require(
        type(request["max_received_bytes"]) is int and request["max_received_bytes"] > 0,
        "finite explicit adapter receive budget required",
    )
    _require(
        type(request["admitted_sources"]) is dict and request["admitted_sources"],
        "actual installed source identities must be supplied",
    )
    return deepcopy(request)


def make_lora_replica_pool(pool_class, request):
    """Use the external pool's public worker_cls without changing its scheduler."""
    _require(enabled(), "LoRA sampler evidence must be explicitly enabled")
    request = _request(request)
    from arctic_platform.debug.lora_sampler_evidence import receive_and_install_lora

    _bind_sources(
        request["admitted_sources"],
        pool_class,
        LoRASamplerEvidenceWorker,
        SamplerEvidenceWorker,
        receive_and_install_lora,
    )
    import ray

    class OwnedLoRAWorker(LoRASamplerEvidenceWorker):
        def __init__(self):
            super().__init__(request)

    actor = ray.remote(runtime_env={"env_vars": {VERIFY_ENV: "1"}})(OwnedLoRAWorker)
    return functools.partial(pool_class, worker_cls=actor)


class LoRASamplerEvidenceWorker(SamplerEvidenceWorker):
    """Reuse lifecycle/metrics, with explicit adapter install and request selection.

    The first transfer is single-use. A second update needs a separate owned
    eviction protocol; it cannot silently replace the installed adapter.
    """

    def __init__(self, request):
        super().__init__()
        self._request = _request(request)
        self._installation = None
        self._lora_request = None
        self._sync_attempted = False

    async def initialize(self, engine_kwargs, extra_env=None):
        _require(self.llm is None, "LoRA sampler is already initialized")
        if extra_env:
            os.environ.update(extra_env)
        _require(enabled(), "LoRA opt-in must reach the actual sampler worker")
        verify_sources(self._request["admitted_sources"])
        import vllm.plugins
        from arctic_inference.envs import arctic_inference_effective_enabled
        from arctic_inference.server.metrics import RingStatLogger

        vllm.plugins.load_general_plugins()
        from vllm.v1.engine.async_llm import AsyncLLM

        if arctic_inference_effective_enabled():
            from arctic_inference.vllm.args import ArcticAsyncEngineArgs as EngineArgs
        else:
            from vllm.engine.arg_utils import AsyncEngineArgs as EngineArgs

        kwargs = engine_kwargs_for_lora(engine_kwargs, rank=self._request["lora_config"]["r"])
        if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
            kwargs.setdefault("attention_config", {"backend": "FLASH_ATTN"})
        args = EngineArgs(**kwargs)
        _require(
            type(args.tensor_parallel_size) is int and args.tensor_parallel_size == 8,
            "this original LoRA sampler requires the actual SAMPLE8 engine",
        )
        self._tp = args.tensor_parallel_size
        self._origins = _bind_sources(
            self._request["admitted_sources"],
            type(self),
            SamplerEvidenceWorker,
            EngineArgs,
            args.create_engine_config,
            AsyncLLM.from_vllm_config,
            AsyncLLM.collective_rpc,
            AsyncLLM.generate,
            AsyncLLM.abort,
            AsyncLLM.shutdown,
            serving_lora_request,
        )
        from arctic_platform.debug.lora_sync_hooks import maybe_admit_sampler_construction
        from arctic_platform.debug.lora_sync_hooks import maybe_verify_sampler_construction

        maybe_admit_sampler_construction(self, args, AsyncLLM.from_vllm_config, engine_kwargs)
        self.llm = AsyncLLM.from_vllm_config(args.create_engine_config(), stat_loggers=[RingStatLogger])
        try:
            maybe_verify_sampler_construction(self)
            verify_sources(self._origins)
        except BaseException as error:
            try:
                self.shutdown()
            except BaseException as cleanup_error:
                error.add_note("native LoRA engine shutdown also failed: " + repr(cleanup_error))
            raise
        self.state = "ready"

    async def sync_weights(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        bucket_size=256 * 1024 * 1024,
        engine_only=False,
        direct_mode=False,
        reverse=False,
    ):
        async with self._lock:
            _require(
                self.llm is not None and self.state in {"ready", "weights_ready"}, "actual awake LoRA sampler required"
            )
            verify_sources(self._origins)
            _require(
                not direct_mode and not reverse and rank_offset == 1 and world_size == 2,
                "only the retained external pairwise adapter transport is supported",
            )
            _require(not self._sync_attempted, "adapter transfer is single-use; no implicit replacement")
            if not engine_only:
                self._sync_attempted = True
            options = {**deepcopy(self._request), "bucket_size": bucket_size, "engine_only": engine_only}
            results = await self.llm.collective_rpc(
                "sync_lora_weights",
                args=(master_addr, master_port, rank_offset, world_size),
                kwargs=options,
            )
            verify_sources(self._origins)
            _require(
                type(results) is list and len(results) == self._tp,
                "actual adapter results from every selected TP rank required",
            )
            if engine_only:
                _require(
                    all(
                        type(row) is dict
                        and row.get("status") == "engine_ready"
                        and row.get("weight_format") == "lora"
                        for row in results
                    ),
                    "an adapter receiver engine failed initialization",
                )
                return {"status": "engine_ready", "weight_format": "lora", "rank_results": results}
            for row in results:
                _require(
                    type(row) is dict
                    and row.get("schema") == "explicit-external-lora-installation-v1"
                    and row.get("status") == "adapter_installed"
                    and row.get("weight_format") == "lora"
                    and row.get("lora_int_id") == self._request["lora_int_id"]
                    and row.get("lora_name") == self._request["lora_name"]
                    and row.get("lora_added") is True
                    and row.get("lora_activated") is True
                    and type(row.get("received")) is dict
                    and row["received"]
                    and row.get("adapter_storage_verified") is False
                    and row.get("adapter_application_verified") is False
                    and row.get("full_parameter_coverage") is False
                    and row.get("capacity_admitted") is False,
                    "a TP rank did not retain genuine adapter installation/nonadmission evidence",
                )
                _require(
                    row["evidence_sources"].items() <= self._origins.items(),
                    "TP rank installation sources differ from the admitted operation",
                )
                _require(
                    row["received"] == results[0]["received"]
                    and row["is_3d_lora_weight"] == results[0]["is_3d_lora_weight"],
                    "actual replicated adapter bytes or serving layout differ across TP ranks",
                )
            request = serving_lora_request(results[0], admitted_sources=self._origins)
            self._installation = deepcopy(results)
            self._lora_request = request
            return {
                "status": "done",
                "weight_format": "lora",
                "rank_results": results,
                "all_rank_adapter_installation": results,
                "adapter_storage_verified": False,
                "adapter_application_verified": False,
                "full_parameter_coverage": False,
                "capacity_admitted": False,
                "numerical": None,
            }

    async def generate(self, prompt, sampling_params):
        _require(
            not any(key.startswith("dss_") for key in sampling_params),
            "native router replay remains unsupported by this explicit adapter worker",
        )
        from vllm import SamplingParams

        async with self._lock:
            self._ready()
            _require(
                self._installation is not None and self._lora_request is not None,
                "actual installed adapter required before LoRA serving",
            )
            origins = _bind_sources(self._origins, SamplingParams, self.llm.generate, self.llm.abort)
            params = SamplingParams(**dict(sampling_params))
            request_id = str(uuid4())
            prompt_input = {"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt
            final = None
            try:
                async for output in self.llm.generate(
                    prompt_input,
                    params,
                    request_id=request_id,
                    lora_request=self._lora_request,
                ):
                    final = output
            except BaseException as error:
                try:
                    await self.llm.abort(request_id)
                except BaseException as abort_error:
                    error.add_note("native LoRA request abort also failed: " + repr(abort_error))
                raise
            verify_sources(origins)
            _require(final is not None and final.outputs, "empty native LoRA generation output")
            choice = final.outputs[0]
            result = {
                "text": choice.text,
                "token_ids": list(choice.token_ids),
                "finish_reason": choice.finish_reason,
                "prompt_len": len(final.prompt_token_ids or []),
                "generation_len": len(choice.token_ids),
                "prefix_cache_len": int(getattr(final, "num_cached_tokens", None) or 0),
            }
            for key, positions in (("prompt_logprobs", final.prompt_logprobs), ("logprobs", choice.logprobs)):
                if positions is not None:
                    result[key] = [
                        (
                            None
                            if position is None
                            else {
                                token: (
                                    {"logprob": value.logprob, "rank": value.rank}
                                    if hasattr(value, "logprob")
                                    else {"logprob": float(value), "rank": None}
                                )
                                for token, value in position.items()
                            }
                        )
                        for position in positions
                    ]
            return result

    async def close_weight_sync(self):
        try:
            return await super().close_weight_sync()
        finally:
            self._installation = None
            self._lora_request = None

    def shutdown(self):
        try:
            super().shutdown()
        finally:
            self._installation = None
            self._lora_request = None
