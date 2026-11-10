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

import os

import torch
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.worker.worker_base import WorkerBase

import arctic_platform.inference.envs as envs
from arctic_platform.inference.patching import ArcticPatch
from arctic_platform.inference.utils import require_supported_vllm_version
from arctic_platform.inference.vllm.args import EngineArgsPatch, AsyncEngineArgsPatch
from arctic_platform.inference.vllm.config import (ParallelConfigPatch,
                                          SpeculativeConfigPatch,
                                          VllmConfigPatch)
from arctic_platform.inference.vllm.fp32_lm_head import (
    apply_fp32_lm_head_patches, set_fp32_lm_head_enabled)
from arctic_platform.inference.vllm.stats import (SpecDecodingStatsPatch,
                                         SpecDecodingLoggingPatch)
from arctic_platform.inference.vllm.structured_output import XgrammarBackendPatch
from arctic_platform.inference.vllm.attention import apply_forest_cascade_patches
from arctic_platform.inference.vllm.router_replay import (
    patch_scheduler_check_stop as _patch_scheduler_check_stop,
)
from arctic_platform.inference.vllm.ulysses import apply_shift_parallel_patches


logger = init_logger(__name__)
require_supported_vllm_version("ArcticInference vLLM patches")


def _stop_token_sequence_matched(request) -> bool:
    sampling_params = request.sampling_params
    if sampling_params is None:
        return False
    extra_args = getattr(sampling_params, "extra_args", None) or {}
    stop_sequences = extra_args.get("dss_stop_token_sequences") or []
    if not stop_sequences:
        return False

    output_ids = list(request.output_token_ids)
    for entry in stop_sequences:
        if isinstance(entry, dict):
            token_ids = entry.get("token_ids", entry.get("ids"))
            stop_reason = entry.get("text")
        else:
            token_ids = entry
            stop_reason = None
        if not isinstance(token_ids, list) or not token_ids:
            continue
        token_ids = [int(token_id) for token_id in token_ids]
        n_tokens = len(token_ids)
        if len(output_ids) >= n_tokens and output_ids[-n_tokens:] == token_ids:
            from vllm.v1.request import RequestStatus

            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = (
                stop_reason if stop_reason is not None else "dss_stop_token_sequence"
            )
            return True
    return False


def _patch_scheduler_check_stop() -> None:
    from vllm.v1.core.sched import scheduler as scheduler_mod
    from vllm.v1.core.sched import utils as sched_utils
    from vllm.v1.request import RequestStatus

    def check_stop_with_token_sequences(request, max_model_len: int) -> bool:
        assert not request.pooling_params

        sampling_params = request.sampling_params
        assert sampling_params is not None

        if request.num_output_tokens < sampling_params.min_tokens:
            return False

        last_token_id = request.output_token_ids[-1]
        if last_token_id == sampling_params.eos_token_id:
            request.status = RequestStatus.FINISHED_STOPPED
            return True

        if last_token_id in (sampling_params.stop_token_ids or ()):
            request.status = RequestStatus.FINISHED_STOPPED
            request.stop_reason = last_token_id
            return True

        if _stop_token_sequence_matched(request):
            return True

        if (
            request.num_tokens >= max_model_len
            or request.num_output_tokens >= request.max_tokens
        ):
            request.status = RequestStatus.FINISHED_LENGTH_CAPPED
            return True

        repetition_detection = sampling_params.repetition_detection
        if repetition_detection is not None and (
            sched_utils.check_sequence_repetition(
                request.output_token_ids,
                repetition_detection,
            )
        ):
            request.status = RequestStatus.FINISHED_REPETITION
            request.stop_reason = "repetition_detected"
            return True

        return False

    sched_utils.check_stop = check_stop_with_token_sequences
    scheduler_mod.check_stop = check_stop_with_token_sequences


class AsyncSchedulerPatch(ArcticPatch[AsyncScheduler]):

    _orig_update_after_schedule = AsyncScheduler._update_after_schedule
    _orig_update_from_output = AsyncScheduler.update_from_output

    def _update_after_schedule(self, scheduler_output):
        self._orig_update_after_schedule(scheduler_output)
        spec = self.vllm_config.speculative_config
        if spec is None or spec.custom_method not in (
            "arctic", "mlp_speculator", "arctic_suffix"
        ):
            return
        limit = spec.disable_by_batch_size
        decode_count = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue
            decode_count += 1
            if limit and decode_count > limit:
                request.spec_token_ids = []
            elif hasattr(request, "_arctic_draft_length"):
                request.spec_token_ids = request.spec_token_ids[:request._arctic_draft_length]

    def update_from_output(self, scheduler_output, model_runner_output):
        output = self._orig_update_from_output(scheduler_output, model_runner_output)
        for req_id, length in getattr(model_runner_output, "_arctic_draft_lengths", {}).items():
            request = self.requests.get(req_id)
            if request is not None:
                request._arctic_draft_length = length
        return output


class WorkerBasePatch(ArcticPatch[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from arctic_platform.inference.vllm.model_runner import GPUModelRunnerPatch

        for patch in (GPUModelRunnerPatch, WorkerPatch):
            if patch not in getattr(patch._arctic_patch_target, "_arctic_patches", {}).values():
                patch.apply_patch()

        return self._orig_init(*args, **kwargs)


class WorkerPatch(ArcticPatch[Worker]):
    """Preserve draft state and reload discarded main parameters after level-2 sleep.

    The native sleep backend retains buffers and parameters outside the main
    model's discardable set. Draft snapshots also support callers that restore
    parameters from their own pinned buffer: `_skip_main_reload_on_wake` skips
    the main disk reload; `_skip_drafter_param_snapshot` retains only draft buffers.
    """

    _orig_sleep = Worker.sleep
    _orig_wake_up = Worker.wake_up

    @staticmethod
    def _save_module_state(module, skip_params: bool = False) -> dict[str, torch.Tensor]:
        state: dict[str, torch.Tensor] = {}
        if not skip_params:
            for name, param in module.named_parameters():
                state[f"param.{name}"] = param.data.cpu().clone()
        for name, buf in module.named_buffers():
            state[f"buffer.{name}"] = buf.cpu().clone()
        return state

    @staticmethod
    def _restore_module_state(
        module, state: dict[str, torch.Tensor]
    ) -> None:
        for name, param in module.named_parameters():
            key = f"param.{name}"
            if key in state:
                param.data.copy_(state[key])
        for name, buf in module.named_buffers():
            key = f"buffer.{name}"
            if key in state:
                buf.data.copy_(state[key])

    def sleep(self, level: int = 1) -> None:
        from vllm.distributed.kv_transfer import has_kv_transfer_group, get_kv_transfer_group

        torch.accelerator.synchronize()
        discardable = self._discardable_parameters(self.model_runner.model) if level == 2 else set()
        if level == 2:
            draft = self.model_runner.get_draft_model()
            self._sleep_saved_drafter_state = (
                self._save_module_state(draft, skip_params=getattr(self, "_skip_drafter_param_snapshot", False))
                if draft is not None else {}
            )
            self._sleep_level = 2
        if has_kv_transfer_group():
            get_kv_transfer_group().unregister_kv_caches()
        cache = self.model_runner.extensible_kv_cache
        if cache is not None:
            cache.release_physical()
        self.sleep_mode_backend.suspend(discardable)
        if self.vllm_config.model_config.enable_nccl_comm_suspend:
            from vllm.distributed.parallel_state import suspend_device_comms
            suspend_device_comms()
        torch.accelerator.synchronize()

    def wake_up(self, tags: list[str] | None = None) -> None:
        self._orig_wake_up(tags=tags)

        if getattr(self, "_sleep_level", 0) == 2:
            # Skip the upstream disk reload of the main model when the
            # caller (e.g. semi-persistence) restores main params itself
            # from a host-side pinned buffer right after wake_up.  Default
            # path (flag absent / False) preserves the original behavior.
            if not getattr(self, "_skip_main_reload_on_wake", False):
                from arctic_platform.inference.vllm.model_runner import GPUModelRunnerPatch
                GPUModelRunnerPatch._orig_reload_weights(self.model_runner)

            saved_drafter = getattr(self, "_sleep_saved_drafter_state", {})
            if saved_drafter:
                draft = self.model_runner.get_draft_model()
                if draft is not None:
                    self._restore_module_state(draft, saved_drafter)
                self._sleep_saved_drafter_state = {}

            self._sleep_level = 0


def apply_arctic_patches():

    from transformers import AutoConfig
    from arctic_platform.inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)
    from arctic_platform.inference.vllm.spec_dec.config import MLPSpeculatorConfig
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    AutoConfig.register("mlp_speculator", MLPSpeculatorConfig, exist_ok=True)
    _CONFIG_REGISTRY["mlp_speculator"] = MLPSpeculatorConfig

    from vllm import ModelRegistry

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model(
        "LlamaSwiftKVForCausalLM",
        "arctic_platform.inference.vllm.swiftkv:LlamaSwiftKVForCausalLM")

    # Register ArcticSpeculator models to vLLM.
    from arctic_platform.inference.vllm.spec_dec.arctic_speculator import (
        ArcticMLPSpeculator, ArcticLSTMSpeculator)
    ModelRegistry.register_model("ArcticMLPSpeculatorPreTrainedModel",
                                 ArcticMLPSpeculator)
    ModelRegistry.register_model("ArcticLSTMSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)
    # This name is currently used in corvo
    ModelRegistry.register_model("MLPVariantSpeculatorPreTrainedModel",
                                 ArcticLSTMSpeculator)

    from arctic_platform.inference.vllm.spec_dec.v2_config import register_arctic_speculators
    register_arctic_speculators()

    WorkerBasePatch.apply_patch()

    # Async scheduler patches for spec decode (disable_by_batch_size
    # interaction + dynamic draft width allocation).
    AsyncSchedulerPatch.apply_patch()

    # Patches to vLLM arguments and configuration objects.
    EngineArgsPatch.apply_patch()
    AsyncEngineArgsPatch.apply_patch()
    ParallelConfigPatch.apply_patch()
    SpeculativeConfigPatch.apply_patch()
    SpecDecodingStatsPatch.apply_patch()
    SpecDecodingLoggingPatch.apply_patch()
    VllmConfigPatch.apply_patch()
    XgrammarBackendPatch.apply_patch()

    # Forest Cascade Attention backend (always registered; runtime-gated
    # by --forest-cascade-attn-configs).
    apply_forest_cascade_patches()

    # Main optimization patches.
    apply_shift_parallel_patches()
    _patch_scheduler_check_stop()

    # FP32 LM head: run the lm_head matmul in fp32 (weights stay in
    # their native dtype, on-the-fly upcast). The patch is always
    # installed but is a no-op unless ARCTIC_FP32_LM_HEAD=1 (or the
    # --fp32-lm-head CLI flag) is set before model construction.
    if envs.ARCTIC_FP32_LM_HEAD:
        set_fp32_lm_head_enabled(True)
    apply_fp32_lm_head_patches()

    # kvcached prefix-cache patches (only when kvcached autopatch is active).
    if os.environ.get("KVCACHED_AUTOPATCH", "").lower() in ("1", "true"):
        from arctic_platform.inference.vllm.kvcached.patches import (
            apply_kvcached_prefix_cache_patches,
        )
        apply_kvcached_prefix_cache_patches()
