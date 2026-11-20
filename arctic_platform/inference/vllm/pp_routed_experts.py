# Copyright 2026 Snowflake Inc.
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

"""Return routed experts (router replay) with pipeline parallelism, Model Runner V2.

vLLM 0.33 refuses ``--enable-return-routed-experts`` with PP > 1. Each stage's
``RoutedExpertsCapturer`` fills only the columns of the layers that stage runs,
and only the executor output rank (TP rank 0 of the last stage) snapshots,
publishes and returns the rows, so the earlier stages' layers would come back
as zeros. The non-last stages' TP rank 0 also builds a data plane whose
``begin_step`` asserts once a request is scheduled a second time, because no
step output ever advances its emit cursor there.

With this patch the routes travel with PP's intermediate tensors:

* a non-last stage adds its capture rows (every layer, in the capturer's
  output dtype, as bytes) to the intermediate tensors it sends, under
  ``ROUTES_KEY``;
* a non-first stage copies the rows it receives into its own capture buffer
  before its forward, whose MoE layers then overwrite their own columns.

The last stage therefore holds every layer's routes when it snapshots them, so
its store, the scheduler's connector and the output processor see the same
per-request, all-layer rows as with PP1. The rows go rank to rank rather than
split across TP and all-gathered: on a one-token batch with sequence-parallel
MoE only TP rank 0 holds the real rows (see ``RoutedExpertsCapturer.
_restore_local_topk_ids``), and TP rank 0 is the rank that publishes. The data
plane is kept only on the last stage, and the PP refusal is lifted.

Every hook is a no-op unless ``pipeline_parallel_size > 1``, so PP1 runs exactly
as before. ``ARCTIC_PP_ROUTED_EXPERTS=0`` disables the patch, and with it
vLLM's refusal is back.

Remove once the pinned vLLM returns routed experts with PP itself.
"""

from __future__ import annotations

import os
from typing import Any

import torch

ROUTES_KEY = "arctic_routed_experts"
_PATCH_ATTR = "_arctic_pp_routed_experts"
_APPLIED = False


def pack_routes(capturer: Any, num_tokens: int) -> torch.Tensor:
    """The first ``num_tokens`` capture rows, in the output dtype, viewed as bytes.

    NCCL has no 16-bit unsigned type, so ``uint16`` rows travel as byte pairs.
    """
    rows = capturer.device_buffer[:num_tokens].to(capturer.output_dtype)
    return rows.view(torch.uint8)


def unpack_routes(capturer: Any, routes: torch.Tensor) -> None:
    """Write rows that ``pack_routes`` produced on the previous stage into the capture buffer."""
    rows = routes.view(capturer.output_dtype)
    if rows.shape[1:] != capturer.device_buffer.shape[1:]:
        raise ValueError(
            "pipeline-parallel routed-experts rows do not match the capture buffer: "
            f"received {tuple(rows.shape)}, buffer {tuple(capturer.device_buffer.shape)}"
        )
    capturer.device_buffer[: rows.shape[0]].copy_(rows)


def _pp_capturer(runner: Any) -> Any:
    """The runner's capturer when it captures routes under PP > 1, else None."""
    if not getattr(runner, "use_pp", False):
        return None
    connector = getattr(runner, "aux_output_connector", None)
    if connector is None:
        return None
    return connector._capturer


def patch_execute_model(runner_cls: Any) -> None:
    original = runner_cls.execute_model
    if getattr(original, _PATCH_ATTR, False):
        return

    from vllm.sequence import IntermediateTensors

    def execute_model(
        self,
        scheduler_output,
        intermediate_tensors=None,
        dummy_run: bool = False,
        **kwargs,
    ):
        capturer = None if dummy_run else _pp_capturer(self)
        if capturer is not None and intermediate_tensors is not None:
            # Waits for the receive, as the runner's own copy would.
            routes = intermediate_tensors.tensors.pop(ROUTES_KEY, None)
            if routes is None:
                raise RuntimeError(
                    "the previous pipeline stage sent no routed-experts rows; "
                    "every stage must run with ARCTIC_PP_ROUTED_EXPERTS=1"
                )
            unpack_routes(capturer, routes)
        output = original(
            self,
            scheduler_output,
            intermediate_tensors,
            dummy_run=dummy_run,
            **kwargs,
        )
        if capturer is not None and isinstance(output, IntermediateTensors):
            num_tokens = self.execute_model_state.input_batch.num_tokens_after_padding
            output = IntermediateTensors(
                output.tensors | {ROUTES_KEY: pack_routes(capturer, num_tokens)}
            )
        return output

    setattr(execute_model, _PATCH_ATTR, True)
    execute_model._orig = original
    runner_cls.execute_model = execute_model


def patch_should_use_all_gather(group_cls: Any) -> None:
    """Send ``ROUTES_KEY`` whole from each TP rank instead of split and all-gathered."""
    original = group_cls._should_use_all_gather
    if getattr(original, _PATCH_ATTR, False):
        return

    def _should_use_all_gather(self, key, *args, **kwargs):
        if key == ROUTES_KEY:
            return False
        return original(self, key, *args, **kwargs)

    setattr(_should_use_all_gather, _PATCH_ATTR, True)
    _should_use_all_gather._orig = original
    group_cls._should_use_all_gather = _should_use_all_gather


def patch_worker_connector(connector_cls: Any, is_last_pp_rank) -> None:
    """Keep the auxiliary-output data plane only on the last PP stage."""
    original = connector_cls.__init__
    if getattr(original, _PATCH_ATTR, False):
        return

    def __init__(self, *, vllm_config, **kwargs):
        original(self, vllm_config=vllm_config, **kwargs)
        if (
            vllm_config.parallel_config.pipeline_parallel_size > 1
            and self._store is not None
            and not is_last_pp_rank()
        ):
            store, self._store, self._buffer = self._store, None, None
            store.close()

    setattr(__init__, _PATCH_ATTR, True)
    __init__._orig = original
    connector_cls.__init__ = __init__


def patch_verify_aux_output(vllm_config_cls: Any) -> None:
    """Run vLLM's auxiliary-output checks, all but the PP one."""
    original = vllm_config_cls._verify_aux_output_compatibility
    if getattr(original, _PATCH_ATTR, False):
        return

    def _verify_aux_output_compatibility(self) -> None:
        parallel_config = self.parallel_config
        pp_size = parallel_config.pipeline_parallel_size
        if pp_size <= 1 or not self.aux_output_config.enabled:
            return original(self)
        parallel_config.pipeline_parallel_size = 1
        try:
            original(self)
        finally:
            parallel_config.pipeline_parallel_size = pp_size

    setattr(_verify_aux_output_compatibility, _PATCH_ATTR, True)
    _verify_aux_output_compatibility._orig = original
    vllm_config_cls._verify_aux_output_compatibility = _verify_aux_output_compatibility


def _is_last_pp_rank() -> bool:
    from vllm.distributed.parallel_state import get_pp_group

    return get_pp_group().is_last_rank


def ensure_pp_routed_experts() -> None:
    """Patch vLLM so routed experts are returned, all layers, with PP > 1."""
    global _APPLIED
    if _APPLIED or os.getenv("ARCTIC_PP_ROUTED_EXPERTS", "1") != "1":
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("Routed experts with pipeline parallelism")

    from vllm.config import VllmConfig
    from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    patch_verify_aux_output(VllmConfig)
    patch_worker_connector(AuxOutputWorkerConnector, _is_last_pp_rank)
    patch_should_use_all_gather(GroupCoordinator)
    patch_execute_model(GPUModelRunner)
    _APPLIED = True
