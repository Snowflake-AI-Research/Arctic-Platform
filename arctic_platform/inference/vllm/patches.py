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
from arctic_platform.inference.vllm.args import AsyncEngineArgsPatch
from arctic_platform.inference.vllm.args import EngineArgsPatch
from arctic_platform.inference.vllm.attention import apply_forest_cascade_patches
from arctic_platform.inference.vllm.config import MLPSpeculatorConfigPatch
from arctic_platform.inference.vllm.config import ParallelConfigPatch
from arctic_platform.inference.vllm.config import SpeculativeConfigPatch
from arctic_platform.inference.vllm.config import VllmConfigPatch
from arctic_platform.inference.vllm.fp32_lm_head import apply_fp32_lm_head_patches
from arctic_platform.inference.vllm.fp32_lm_head import set_fp32_lm_head_enabled
from arctic_platform.inference.vllm.router_replay import patch_scheduler_check_stop as _patch_scheduler_check_stop
from arctic_platform.inference.vllm.stats import SpecDecodingLoggingPatch
from arctic_platform.inference.vllm.stats import SpecDecodingStatsPatch
from arctic_platform.inference.vllm.structured_output import XgrammarBackendPatch
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
            request.stop_reason = stop_reason if stop_reason is not None else "dss_stop_token_sequence"
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

        if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
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
    """Patch AsyncScheduler to:
    1. Respect ``disable_by_batch_size`` when allocating spec token
       placeholders (the worker only drafts for the first N requests).
    2. Use the previous step's actual draft length for dynamic placeholder
       allocation, avoiding wasted verification compute when the real draft
       width (e.g. Arctic n_predict=3) is much smaller than
       num_speculative_tokens (e.g. 12).
    3. Store ``_scheduled_spec_count`` so that the post-fix in
       ``update_from_output`` can compensate for worker-side trimming.
    """

    _orig_update_after_schedule = AsyncScheduler._update_after_schedule

    def _update_after_schedule(self, scheduler_output):
        # Call Scheduler._update_after_schedule (the grandparent),
        # skipping the base AsyncScheduler override which we are replacing.
        Scheduler._update_after_schedule(self, scheduler_output)

        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Respect disable_by_batch_size: only add spec token placeholders
        # for the first N decode requests (matching the worker's draft_limit
        # in propose_draft_token_ids).
        spec_config = getattr(self.vllm_config, "speculative_config", None)
        disable_bs = spec_config.disable_by_batch_size if spec_config else None
        decode_with_spec_count = 0
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )

            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            # Store the originally-scheduled spec count so that
            # update_from_output can compensate for worker-side trimming.
            request._scheduled_spec_count = cur_num_spec_tokens

            request.num_output_placeholders += 1 + cur_num_spec_tokens

            decode_with_spec_count += 1
            if disable_bs and decode_with_spec_count > disable_bs:
                request.spec_token_ids = []
                continue

            # Use previous step's actual draft length to size
            # placeholders.  When suffix had a good match (actual
            # > n_predict), allocate the full width so the next
            # step can verify all suffix tokens.  When suffix
            # didn't match (actual = n_predict from arctic),
            # allocate only that many to avoid wasting attention
            # compute on zero-padded positions.
            # Cold start: allocate full width (generous).
            prev_actual = getattr(request, "_prev_actual_draft_len", None)
            if prev_actual is not None:
                num_placeholders = min(max(prev_actual, 1), self.num_spec_tokens)
            else:
                num_placeholders = self.num_spec_tokens

            request.spec_token_ids = [-1] * num_placeholders

    def update_from_output(self, scheduler_output, model_runner_output):
        """Wrap Scheduler.update_from_output to store actual draft counts.

        We infer the drafter's real capability from the acceptance
        results so that the next ``_update_after_schedule`` can size
        placeholders correctly.  This works even when the worker runs
        in a separate process (where scheduler_output._actual_draft_lens
        set by the worker doesn't survive serialisation back to the
        scheduler).

        Strategy:
        1. **Primary path**: read ``_actual_draft_lens`` from the
           ``model_runner_output`` object (attached by the model runner
           to the ``ModelRunnerOutput`` dataclass, which reliably
           survives the async pipeline).
        2. **Legacy path**: read from ``scheduler_output._actual_draft_lens``
           (works in same-process non-async mode).
        3. **Fallback**: infer from acceptance results with exponential
           growth — when all drafted tokens are accepted, double the
           allocation so suffix decoding reaches full capacity in
           O(log n) steps instead of O(n).

        In v0.18, ``update_from_output`` returns ``dict[int, EngineCoreOutputs]``.
        """
        sampled_token_ids = model_runner_output.sampled_token_ids
        req_id_to_index = model_runner_output.req_id_to_index

        result = Scheduler.update_from_output(self, scheduler_output, model_runner_output)

        # Primary path: read from model_runner_output (most reliable
        # for async scheduling — the ModelRunnerOutput object is
        # returned by get_output() and guaranteed to survive).
        actual_lens = getattr(model_runner_output, "_actual_draft_lens", None)

        # Legacy path: read from scheduler_output (works for non-async
        # or same-process setups where the attribute is preserved).
        if not actual_lens:
            actual_lens = getattr(scheduler_output, "_actual_draft_lens", None)

        if actual_lens:
            for req_id, actual_len in actual_lens.items():
                request = self.requests.get(req_id)
                if request is not None:
                    request._prev_actual_draft_len = actual_len
            return result

        # Fallback: infer from acceptance results (multi-process case).
        # Uses exponential growth when all drafted tokens are accepted
        # (indicating the drafter / suffix cache can handle more), so
        # the allocation converges to num_spec_tokens in O(log n)
        # steps:
        #   step 0: 1 position  → accept 1/1 → prev = 2
        #   step 1: 2 positions → accept 2/2 → prev = 4
        #   step 2: 4 positions → accept 4/4 → prev = 8
        #   ...
        # When not all are accepted (normal drafter), linear growth:
        #   step 0: 1 position  → accept 1 → prev = 2
        #   step 1: 2 positions → accept 2 → prev = 3
        #   step 2: 3 positions → steady state (n_predict = 3)
        if not sampled_token_ids:
            return result

        for req_id in scheduler_output.num_scheduled_tokens:
            scheduled_spec = scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            if not scheduled_spec:
                continue

            req_index = req_id_to_index.get(req_id)
            if req_index is None:
                continue

            request = self.requests.get(req_id)
            if request is None:
                continue

            generated = sampled_token_ids[req_index]
            num_accepted = (len(generated) - 1) if generated else 0
            num_draft = len(scheduled_spec)

            prev = getattr(request, "_prev_actual_draft_len", None)

            if num_accepted > 0:
                if num_accepted >= num_draft and num_draft > 0:
                    # All drafted tokens accepted — the drafter (or
                    # suffix cache) could produce more if given room.
                    # Double the allocation for exponential ramp-up.
                    new_val = min(num_draft * 2, self.num_spec_tokens)
                else:
                    # Partial acceptance: grow linearly.
                    new_val = min(num_accepted + 1, self.num_spec_tokens)
                request._prev_actual_draft_len = max(prev or 0, new_val)
            elif prev is None:
                # First step for this request, zero acceptance.
                # Seed with 1 so _update_after_schedule doesn't
                # fall back to the full num_spec_tokens next time.
                request._prev_actual_draft_len = 1

        return result


class WorkerBasePatch(ArcticPatch[WorkerBase]):

    _orig_init = WorkerBase.__init__

    def __init__(self, *args, **kwargs):
        # Some patches like the GPUModelRunner will import CUDA libraries when
        # they are initialized, which will cause process forking to fail. For
        # these patches, we need to delay the initialization until after the
        # process has been forked (i.e., in the WorkerBase initializer).
        from arctic_platform.inference.vllm.model_runner import GPUModelRunnerPatch

        GPUModelRunnerPatch.apply_patch()
        WorkerPatch.apply_patch()

        return self._orig_init(*args, **kwargs)


class WorkerPatch(ArcticPatch[Worker]):
    """Fix weights offloading for sleep mode and add drafter support.

    The upstream ``load_model`` context-manager chaining bug was fixed in
    v0.18 (see https://github.com/vllm-project/vllm/pull/32947); we no
    longer patch ``load_model`` here.

    sleep/wake_up: Level-2 sleep discards all GPU weights as designed.
    On wake-up the main model is reloaded from disk via the upstream
    ``reload_weights()`` path.  The drafter (speculative model) is *not*
    covered by ``reload_weights()`` and ``drafter.load_model()`` would
    allocate a new model outside the CuMemAllocator pool, so we save the
    drafter state to CPU during sleep and restore it on wake-up instead.
    The drafter is small so the CPU cost is negligible.

    Note: FP8 quantization replaces ``ModelWeightParameter`` with plain
    ``Parameter`` during ``process_weights_after_loading``, which strips
    the TP-sharding methods that ``reload_weights()`` depends on.
    Level-2 sleep therefore only works for non-FP8 models at present.

    Opt-out flags (read from the worker instance via ``getattr``; default
    False -> original behavior).  Intended for callers that restore main
    params and/or drafter params themselves from a host-side pinned
    buffer, e.g. ``arctic_platform.inference.semi_persistence``:

    - ``_skip_main_reload_on_wake``: skip the ``_orig_reload_weights``
      disk-load of the main model inside ``wake_up`` after a level-2
      sleep.
    - ``_skip_drafter_param_snapshot``: skip the ``cpu().clone()`` of
      drafter ``named_parameters()`` inside ``sleep``; drafter
      ``named_buffers()`` are always snapshotted (sub-MB, may carry
      runtime-mutating state).
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
    def _restore_module_state(module, state: dict[str, torch.Tensor]) -> None:
        for name, param in module.named_parameters():
            key = f"param.{name}"
            if key in state:
                param.data.copy_(state[key])
        for name, buf in module.named_buffers():
            key = f"buffer.{name}"
            if key in state:
                buf.data.copy_(state[key])

    def sleep(self, level: int = 1) -> None:
        # Inlined replacement for upstream ``Worker.sleep`` (see
        # ``vllm/v1/worker/gpu_worker.py``).  We drop the
        # GPU-global ``get_memory_info`` bracket and the
        # ``assert freed_bytes >= 0`` regression check: that read is
        # GPU-global, so when peer processes on the same GPU allocate
        # concurrently with our sleep call, ``freed_bytes`` goes
        # negative and the assertion kills the child engine.  The
        # surrounding worker-level ``logger.info`` is dropped along
        # with it because the sleep backend already emits its own
        # peer-safe report from allocator-owned state.

        if level == 2:
            drafter = getattr(self.model_runner, "drafter", None)
            if drafter is not None and getattr(drafter, "model", None) is not None:
                # Skip the drafter parameter snapshot when the caller
                # (e.g. semi-persistence) restores drafter params itself
                # from a host-side pinned buffer.  Drafter buffers are
                # always saved (sub-MB, may carry runtime state).
                self._sleep_saved_drafter_state = self._save_module_state(
                    drafter.model,
                    skip_params=getattr(self, "_skip_drafter_param_snapshot", False),
                )
            else:
                self._sleep_saved_drafter_state = {}
            self._sleep_level = 2

            model = self.model_runner.model
            self._sleep_saved_buffers = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}

        # vLLM 0.30 routes sleep through a pluggable backend instead of
        # calling CuMemAllocator directly. Preserve that abstraction while
        # omitting only the process-global free-memory assertion above.
        self.sleep_mode_backend.suspend(level)
        if self.vllm_config.model_config.enable_nccl_comm_suspend:
            from vllm.distributed.parallel_state import suspend_device_comms

            suspend_device_comms()

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
                drafter = getattr(self.model_runner, "drafter", None)
                if drafter is not None and getattr(drafter, "model", None) is not None:
                    self._restore_module_state(drafter.model, saved_drafter)
                self._sleep_saved_drafter_state = {}

            self._sleep_level = 0


def apply_arctic_patches():

    from transformers import AutoConfig

    from arctic_platform.inference.common.swiftkv import LlamaSwiftKVConfig

    # Register SwiftKV model configurations to transformers.
    AutoConfig.register("llama_swiftkv", LlamaSwiftKVConfig)

    from vllm import ModelRegistry

    # Register SwiftKV model definitions to vLLM.
    ModelRegistry.register_model(
        "LlamaSwiftKVForCausalLM", "arctic_platform.inference.vllm.swiftkv:LlamaSwiftKVForCausalLM"
    )

    # Register ArcticSpeculator models to vLLM.
    from arctic_platform.inference.vllm.spec_dec.arctic_speculator import ArcticLSTMSpeculator
    from arctic_platform.inference.vllm.spec_dec.arctic_speculator import ArcticMLPSpeculator

    ModelRegistry.register_model("ArcticMLPSpeculatorPreTrainedModel", ArcticMLPSpeculator)
    ModelRegistry.register_model("ArcticLSTMSpeculatorPreTrainedModel", ArcticLSTMSpeculator)
    # This name is currently used in corvo
    ModelRegistry.register_model("MLPVariantSpeculatorPreTrainedModel", ArcticLSTMSpeculator)

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
    MLPSpeculatorConfigPatch.apply_patch()

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
        from arctic_platform.inference.vllm.kvcached.patches import apply_kvcached_prefix_cache_patches

        apply_kvcached_prefix_cache_patches()
