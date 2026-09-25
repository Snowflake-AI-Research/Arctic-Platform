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

# Originally lived at
# ``verl/workers/rollout/remote_rollout/arctic_rollout/arctic_rollout.py``
# in Snowflake-AI-Research/verl (fork of verl-project/verl, Apache-2.0).
# Moved here so the Arctic rollout replica ships alongside its adapter,
# and is registered with verl's ``RolloutReplicaRegistry`` by
# :mod:`arctic_platform.integrations.verl.register` via the
# ``VERL_USE_EXTERNAL_MODULES`` plugin hook.
"""Arctic rollout replica for a CPU-only VeRL driver.

Generate/sleep/wake go through :class:`ArcticRLClientWrapper` (on-prem HTTP/Ray
or Cortex). This module must not import vLLM: the driver does not host an
engine, and ``verl.third_party.vllm`` raises if the ``vllm`` package is absent.
"""

import argparse
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from typing import Optional

import ray
from ray.actor import ActorHandle
from transformers import AutoTokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.tokenizer import normalize_token_ids
from verl.workers.config import HFModelConfig
from verl.workers.config import RolloutConfig
from verl.workers.rollout.replica import RolloutMode
from verl.workers.rollout.replica import RolloutReplica
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import get_max_position_embeddings

from arctic_platform.integrations.verl.adapter import ArcticRLClientWrapper


@dataclass
class _CompletionOutput:
    index: int
    text: str
    token_ids: list
    finish_reason: str
    cumulative_logprob: Any = None
    logprobs: Any = None
    routed_experts: Any = None
    num_preempted: Any = None


@dataclass
class _RequestOutput:
    request_id: str
    outputs: list
    prompt: str
    prompt_token_ids: list
    prompt_logprobs: Any = None
    finished: bool = True


class ArcticLLMEngine:
    def __init__(
        self,
        replica_rank: int,
        arctic_rl_client: ArcticRLClientWrapper,
        tokenizer: AutoTokenizer,
    ):
        self.replica_rank = replica_rank
        self.arctic_rl_client = arctic_rl_client
        self.tokenizer = tokenizer

    async def generate(
        self,
        prompt: dict[str, Any],
        sampling_params: dict[str, Any],
        request_id: str,
        lora_request: Any = None,
        priority: int = 0,
    ) -> AsyncGenerator[_RequestOutput, None]:
        del lora_request, priority
        prompt_token_ids = prompt["prompt_token_ids"]
        gen_batch_output = await self.arctic_rl_client.generate(
            prompt_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        if getattr(self.arctic_rl_client, "_use_cortex", False) or getattr(
            self.arctic_rl_client, "_is_cortex_backend", lambda: False
        )():
            from arctic_platform.integrations.verl.cortex_generate import prompt_text_from_ids

            raw_prompt = prompt_text_from_ids(self.tokenizer, prompt_token_ids)
        else:
            raw_prompt = self.tokenizer.decode(prompt_token_ids)
        completed_outputs = []
        for i, output in enumerate(gen_batch_output):
            completed_outputs.append(
                _CompletionOutput(
                    index=i,
                    text=output["text"],
                    token_ids=output["token_ids"],
                    finish_reason=output["finish_reason"],
                )
            )

        yield _RequestOutput(
            request_id=request_id,
            outputs=completed_outputs,
            prompt=raw_prompt,
            prompt_token_ids=prompt_token_ids,
            finished=True,
        )

    async def wake_up(self, tags: list[str] = None):
        await self.arctic_rl_client.wake_up_inference(tags=tags)
        await self.reset_prefix_cache()

    async def sleep(self, level: int = 1):
        await self.arctic_rl_client.sleep_inference(level=level)

    async def reset_prefix_cache(self):
        await self.arctic_rl_client.reset_prefix_cache()


class ArcticLLMServer:
    """CPU-driver generate adapter. Sampling runs on Arctic workers / Cortex, not vLLM here."""

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        arctic_rl_client: ArcticRLClientWrapper,
        workers: list[ActorHandle] | None = None,
        replica_rank: int = 0,
        node_rank: int = 0,
        gpus_per_node: int = 1,
        nnodes: int = 1,
        cuda_visible_devices: str = "0",
    ):
        """
        Args:
            config (RolloutConfig): full config.
            model_config (HFModelConfig): model config.
            rollout_mode (RolloutMode): rollout mode.
            replica_rank (int): replica rank, a replica may contain multiple nodes.
            node_rank (int): node rank.
            gpus_per_node (int): number of gpus per node.
            nnodes (int): number of nodes.
            cuda_visible_devices (str): cuda visible devices.
        """
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        max_position_embeddings = get_max_position_embeddings(self.model_config.hf_config)
        if self.config.max_model_len is None:
            self.config.max_model_len = max_position_embeddings
        else:
            if self.config.max_model_len > max_position_embeddings:
                raise ValueError(
                    f"max_model_len ({self.config.max_model_len}) should be less than or equal to "
                    f"max_position_embeddings ({max_position_embeddings})"
                )

        self.rollout_mode = rollout_mode
        self.workers = workers if workers is not None else []

        self.replica_rank = replica_rank
        self.node_rank = node_rank
        self.gpus_per_node = gpus_per_node
        self.nnodes = nnodes
        # model weights version, set by ServerAdapter when update weights.
        self.global_steps = None

        if self.rollout_mode != RolloutMode.HYBRID and self.config.load_format == "dummy":
            # logger.warning(f"rollout mode is {self.rollout_mode}, load_format is dummy, set to auto")
            self.config.load_format = "auto"

        self._master_address = None
        self._master_port = None
        self._dp_rpc_port = None
        self._dp_master_port = None

        # Use the tokenizer already loaded on `model_config` (per upstream
        # `HFModelConfig.__post_init__`) instead of re-loading from the HF
        # hub here; avoids a hardcoded model name and an extra download.
        self.engine = ArcticLLMEngine(replica_rank, arctic_rl_client, self.model_config.tokenizer)

    def get_master_address(self):
        pass

    def get_server_address(self):
        pass

    @property
    def lora_as_adapter(self) -> bool:
        return False

    async def collective_rpc(self, **kwargs):
        pass

    async def abort_all_requests(self, reset_prefix_cache: bool = True) -> dict[str, Any]:
        del reset_prefix_cache
        return {"aborted": True}

    async def resume_generation(self):
        return None

    async def start_profile(self, **kwargs):
        del kwargs
        return None

    async def stop_profile(self):
        return None

    async def launch_server(
        self,
        master_address: str = None,
        master_port: int = None,
        dp_rpc_port: int = None,
    ):
        pass

    async def run_server(self, args: argparse.Namespace):
        pass

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        prompt_ids = normalize_token_ids(prompt_ids)

        # Calculate the maximum possible new tokens based on available context space
        # This serves as a safety upper bound
        max_possible_tokens = self.config.max_model_len - len(prompt_ids)
        if max_possible_tokens < 0:
            raise ValueError(
                f"Prompt length ({len(prompt_ids)}) exceeds the model's maximum context length "
                f"({self.config.max_model_len})."
            )

        # Determine max_tokens from sampling_params or use configured response_length as default
        if "max_tokens" in sampling_params:
            max_tokens = sampling_params.pop("max_tokens")
        elif "max_new_tokens" in sampling_params:
            # support sglang-style 'max_new_tokens' param
            max_tokens = sampling_params.pop("max_new_tokens")
        else:
            # Default to a calculation that considers configured lengths
            max_tokens = self.config.response_length + self.config.prompt_length - len(prompt_ids)

        # Clamp max_tokens to the valid range [0, max_possible_tokens]
        max_tokens = max(0, min(max_tokens, max_possible_tokens))

        assert (
            max_tokens <= max_possible_tokens
        ), f"max_tokens {max_tokens} exceeds available context space {max_possible_tokens}"
        sampling_params["logprobs"] = 0 if sampling_params.pop("logprobs", False) else None
        sampling_params.setdefault("repetition_penalty", self.config.get("repetition_penalty", 1.0))
        # sampling_params = SamplingParams(max_tokens=max_tokens, **sampling_params)
        sampling_params["max_tokens"] = max_tokens
        multi_modal_data = {}
        if image_data is not None:
            multi_modal_data["image"] = image_data
        if video_data is not None:
            multi_modal_data["video"] = video_data
        # import pdb; pdb.set_trace()
        prompt = {"prompt_token_ids": prompt_ids, "multi_modal_data": multi_modal_data}
        generator = self.engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
            priority=priority,
        )

        # print(f"arctic_async_server: {generator=}, {type(generator)=}")

        # Get final response
        final_res: Optional[_RequestOutput] = None
        async for output in generator:
            final_res = output
        assert final_res is not None

        # print(f"[ArcticLLMServer] generate OUTPUT: {final_res=}")

        token_ids = final_res.outputs[0].token_ids
        log_probs = None
        if sampling_params["logprobs"] is not None:
            log_probs = [logprobs[token_ids[i]].logprob for i, logprobs in enumerate(final_res.outputs[0].logprobs)]

        routed_experts = None
        if self.config.enable_rollout_routing_replay:
            routed_experts = final_res.outputs[0].routed_experts

        # Determine stop reason from finish_reason
        finish_reason = final_res.outputs[0].finish_reason
        if finish_reason == "abort":
            stop_reason = "aborted"
        elif finish_reason in ("stop", "length"):
            stop_reason = "completed"
        else:
            stop_reason = finish_reason  # for more stop reason in the future

        num_preempted = None

        if hasattr(final_res.outputs[0], "num_preempted"):
            num_preempted = final_res.outputs[0].num_preempted

        return TokenOutput(
            token_ids=token_ids,
            log_probs=log_probs,
            routed_experts=routed_experts,
            stop_reason=stop_reason,
            num_preempted=num_preempted,
            extra_info={"global_steps": self.global_steps},
        )

    async def wake_up(self):
        print("[ArcticLLMServer] Waking up")
        if self.node_rank != 0 or not self.config.free_cache_engine:
            return

        if self.rollout_mode == RolloutMode.HYBRID:
            # In hybrid mode, rollout is woken up inside `update_weights`.
            raise ValueError(f"wake_up not support rollout_mode {self.rollout_mode}")
        elif self.rollout_mode == RolloutMode.COLOCATED:
            # Directly wake the inference engine without a weight sync.
            await self.engine.wake_up(tags=["kv_cache", "weights"])
            await self.engine.reset_prefix_cache()
        elif self.rollout_mode == RolloutMode.STANDALONE:
            print("[ArcticLLMServer] skip wake_up in standalone mode")

    async def sleep(self):
        if self.node_rank != 0 or not self.config.free_cache_engine:
            print(
                f"[ArcticLLMServer] skip sleep {self.rollout_mode}: "
                f"node_rank={self.node_rank} free_cache_engine={self.config.free_cache_engine}"
            )
            return

        print(f"[ArcticLLMServer] Sleeping {self.rollout_mode}")
        if self.rollout_mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            await self.engine.sleep(level=1)
        elif self.rollout_mode == RolloutMode.STANDALONE:
            print("[ArcticLLMServer] skip sleep in standalone mode")

    async def clear_kv_cache(self):
        if self.node_rank == 0:
            await self.engine.reset_prefix_cache()


class ArcticReplica(RolloutReplica):
    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 1,
        is_reward_model: bool = False,
        **kwargs,
    ):
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = ray.remote(ArcticLLMServer)
        # The trainer-side `RemoteBackendTrainer` puts `main_config` and
        # the backend handle into `wg_kwargs`, which the AgentLoopManager
        # forwards here. `from_config(handle=...)` re-attaches to the
        # same driver-side backend instead of building a second one.
        self.arctic_rl_client = ArcticRLClientWrapper.from_config(
            kwargs["main_config"], handle=kwargs["backend_handle"]
        )

    def rollout_worker_use_gpu(self) -> bool:
        return False

    async def launch_servers(self):
        server = self.server_class.options(
            # override ray's default max_concurrency of 1000
            max_concurrency=4096,
        ).remote(
            replica_rank=self.replica_rank,
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            arctic_rl_client=self.arctic_rl_client,
        )
        self.servers.append(server)
        self._server_handle = server

    # NOTE: wake_up()/sleep() intentionally NOT overridden -- base RolloutReplica fans them out to each ArcticLLMServer.wake_up/sleep (drives colocate KV-cache sleep/wake); no-op overrides kept vLLM's KV reservation resident during training -> OOM.

    async def abort_request(self, request_id: str) -> dict[str, Any]:
        return {"aborted": True, "request_id": 0}
