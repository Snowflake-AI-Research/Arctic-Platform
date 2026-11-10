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

import copy
from contextlib import contextmanager
from typing import Optional

import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.config import ModelConfig, ParallelConfig
from vllm.distributed.parallel_state import (init_model_parallel_group,
                                             get_world_group,
)
from vllm.model_executor.layers.attention import Attention
from vllm.v1.executor.multiproc_executor import MultiprocExecutor, WorkerProc


from arctic_platform.inference.patching import ArcticPatch

def apply_shift_parallel_patches():
    UlyssesModelConfig.apply_patch()
    UlyssesParallelState.apply_patch()
    UlyssesWorkerProc.apply_patch()
    UlyssesMultiprocExecutor.apply_patch()
    UlyssesAttention.apply_patch()


class UlyssesModelConfig(ArcticPatch[ModelConfig]):

    _orig_get_num_kv_heads = ModelConfig.get_num_kv_heads
    _orig_get_num_attention_heads = ModelConfig.get_num_attention_heads
    _orig_get_layers_start_end_indices = ModelConfig.get_layers_start_end_indices

    def get_num_kv_heads(self: ModelConfig,
                         parallel_config: ParallelConfig) -> int:
        num_kv_heads = self._orig_get_num_kv_heads(parallel_config)
        sp_size = parallel_config.ulysses_sequence_parallel_size
        return max(1, num_kv_heads // sp_size)

    def get_num_attention_heads(self: ModelConfig,
                                parallel_config: ParallelConfig) -> int:
        num_heads = self._orig_get_num_attention_heads(parallel_config)
        sp_size = parallel_config.ulysses_sequence_parallel_size
        return max(1, num_heads // sp_size)

    def get_layers_start_end_indices(
            self, parallel_config: "ParallelConfig") -> tuple[int, int]:
        config = copy.copy(parallel_config)
        config.rank //= config.ulysses_sequence_parallel_size
        return self._orig_get_layers_start_end_indices(config)


class UlyssesParallelState(ArcticPatch[parallel_state]):
    _orig_initialize_model_parallel = staticmethod(parallel_state.initialize_model_parallel)
    _orig_graph_capture = staticmethod(parallel_state.graph_capture)


    _SP = None
    _SP_TP = None

    def initialize_model_parallel(
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        prefill_context_model_parallel_size: int = 1,
        decode_context_model_parallel_size: Optional[int] = 1,
        backend: Optional[str] = None,
    ) -> None:

        from vllm.config import get_current_vllm_config

        config = get_current_vllm_config()
        sp_size = config.parallel_config.ulysses_sequence_parallel_size
        world_size = torch.distributed.get_world_size()
        backend = backend or torch.distributed.get_backend(get_world_group().device_group)
        ranks = torch.arange(world_size).reshape(
            -1, config.parallel_config.data_parallel_size,
            pipeline_model_parallel_size, prefill_context_model_parallel_size,
            sp_size, tensor_model_parallel_size,
        )
        if sp_size > 1 and config.parallel_config.enable_elastic_ep:
            raise ValueError("Ulysses sequence parallelism does not support elastic EP")

        # Treat SP as independent replicas for native TP/PP/DP/EP group creation.
        # Translate its logical ExternalDP x SP x DP x PP x PCP x TP rank order
        # into our physical ExternalDP x DP x PP x PCP x SP x TP order.
        rank_map = ranks.permute(0, 4, 1, 2, 3, 5).reshape(-1).tolist()
        original_init_group = parallel_state.init_model_parallel_group

        def init_mapped_group(group_ranks, *args, **kwargs):
            mapped = [[rank_map[rank] for rank in group] for group in group_ranks]
            return original_init_group(mapped, *args, **kwargs)

        parallel_state.init_model_parallel_group = init_mapped_group
        try:
            parallel_state._orig_initialize_model_parallel(
                tensor_model_parallel_size, pipeline_model_parallel_size,
                prefill_context_model_parallel_size,
                decode_context_model_parallel_size, backend,
            )
        finally:
            parallel_state.init_model_parallel_group = original_init_group

        sp_ranks = ranks.transpose(4, 5).reshape(-1, sp_size).tolist()
        parallel_state._SP = init_model_parallel_group(
            sp_ranks, get_world_group().local_rank, backend, group_name="sp")
        if sp_size > 1:
            # Interleaved rank order matches head ownership after Ulysses all-to-all.
            shift_ranks = ranks.transpose(4, 5).reshape(-1, sp_size * tensor_model_parallel_size).tolist()
            parallel_state._SP_TP = init_model_parallel_group(
                shift_ranks, get_world_group().local_rank, backend, group_name="sp_tp")

    @contextmanager
    def graph_capture(
        device: torch.device,
        stream: torch.cuda.Stream | None = None,
    ):
        """
        `graph_capture` is a context manager which should surround the code that
        is capturing the CUDA graph. Its main purpose is to ensure that the
        some operations will be run after the graph is captured, before the graph
        is replayed. It returns a `GraphCaptureContext` object which contains the
        necessary data for the graph capture. Currently, it only contains the
        stream that the graph capture is running on. This stream is set to the
        current CUDA stream when the context manager is entered and reset to the
        default stream when the context manager is exited. This is to ensure that
        the graph capture is running on a separate stream from the default stream,
        in order to explicitly distinguish the kernels to capture
        from other kernels possibly launched on background in the default stream.
        """
        from vllm.distributed.parallel_state import GraphCaptureContext
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )
        if parallel_state._SP_TP is None:
            with parallel_state._orig_graph_capture(device, stream) as context:
                yield context
            return
        context = GraphCaptureContext(stream or torch.cuda.Stream(device=device))

        def _get_ca(group):
            if (group is not None
                    and group.device_communicator is not None
                    and isinstance(group.device_communicator, CudaCommunicator)):
                return group.device_communicator.ca_comm
            return None

        tp_ca = _get_ca(parallel_state._TP)
        sp_tp_ca = _get_ca(parallel_state._SP_TP)

        saved_tp = tp_ca.disabled if tp_ca is not None else None
        saved_sp_tp = sp_tp_ca.disabled if sp_tp_ca is not None else None
        if tp_ca is not None:
            tp_ca.disabled = True
        if sp_tp_ca is not None:
            sp_tp_ca.disabled = True

        try:
            with parallel_state._TP.graph_capture(context), \
                 parallel_state._PP.graph_capture(context), \
                 parallel_state._SP_TP.graph_capture(context):
                yield context
        finally:
            if tp_ca is not None and saved_tp is not None:
                tp_ca.disabled = saved_tp
            if sp_tp_ca is not None and saved_sp_tp is not None:
                sp_tp_ca.disabled = saved_sp_tp


class UlyssesWorkerProc(ArcticPatch[WorkerProc]):
    _orig_shutdown = WorkerProc.shutdown

    @staticmethod
    def _destroy_ulysses_parallel():
        sp = getattr(parallel_state, '_SP', None)
        if sp is not None:
            sp.destroy()
        parallel_state._SP = None
        sp_tp = getattr(parallel_state, '_SP_TP', None)
        if sp_tp is not None:
            sp_tp.destroy()
        parallel_state._SP_TP = None

    def shutdown(self):
        try:
            self._orig_shutdown()
        finally:
            UlyssesWorkerProc._destroy_ulysses_parallel()


class UlyssesMultiprocExecutor(ArcticPatch[MultiprocExecutor]):

    def _get_parallel_sizes(self) -> tuple[int, int, int]:
        self.world_size = self.parallel_config.world_size
        assert self.world_size % self.parallel_config.nnodes_within_dp == 0, (
            f"global world_size ({self.parallel_config.world_size}) must be "
            f"divisible by nnodes_within_dp "
            f"({self.parallel_config.nnodes_within_dp}). "
        )
        self.local_world_size = self.parallel_config.local_world_size
        tp_size = self.parallel_config.tensor_parallel_size
        pp_size = self.parallel_config.pipeline_parallel_size
        pcp_size = self.parallel_config.prefill_context_parallel_size
        sp_size = self.parallel_config.ulysses_sequence_parallel_size

        assert self.world_size == tp_size * pp_size * pcp_size * sp_size, (
            f"world_size ({self.world_size}) must be equal to the "
            f"tensor_parallel_size ({tp_size}) x pipeline"
            f"_parallel_size ({pp_size}) x prefill_context"
            f"_parallel_size ({pcp_size}) x ulysses_sequence_parallel"
            f"_size ({sp_size})."
        )
        return tp_size * sp_size, pp_size, pcp_size


class UlyssesAttention(ArcticPatch[Attention]):

    _orig_init = Attention.__init__
    _orig_forward = Attention.forward

    def __init__(self, num_heads, *args, **kwargs):
        from .model_runner import is_shift_parallel_mode
        self.sp_size = parallel_state._SP.world_size
        self.sp_device_group = parallel_state._SP.device_group
        if not is_shift_parallel_mode():
            num_heads //= self.sp_size
            num_kv_heads = kwargs["num_kv_heads"]
            self.is_kv_replicated = True if num_kv_heads < self.sp_size else False
            if self.is_kv_replicated:
                self.replication_factor = self.sp_size // num_kv_heads
                num_kv_heads = 1
            else:
                num_kv_heads //= self.sp_size
            kwargs["num_kv_heads"] = num_kv_heads
        return self._orig_init(num_heads, *args, **kwargs)

    def forward(self, query, key, value, **kwargs):
        from .model_runner import is_shift_parallel_mode
        if self.sp_size == 1 or is_shift_parallel_mode():
            return self._orig_forward(query, key, value, **kwargs)

        # prepare
        q = query.view(-1, self.sp_size, self.num_heads * self.head_size)
        if self.is_kv_replicated:
            k = key.view(-1, self.sp_size // self.replication_factor, self.head_size).repeat_interleave(self.replication_factor, dim=1)
            v = value.view(-1, self.sp_size // self.replication_factor, self.head_size).repeat_interleave(self.replication_factor, dim=1)
        else:
            k = key.view(-1, self.sp_size, self.num_kv_heads * self.head_size)
            v = value.view(-1, self.sp_size, self.num_kv_heads * self.head_size)

        # pack
        qkv = torch.cat((q, k, v), dim=-1).transpose(0, 1).reshape(
            -1, (self.num_heads + 2 * self.num_kv_heads) * self.head_size)
        
        # Ulysses all-to-all 1/2
        qkv_ = torch.empty_like(qkv)
        torch.distributed.all_to_all_single(qkv_, qkv, group=self.sp_device_group)

        # unpack
        q_, k_, v_ = qkv_.split([
            self.num_heads * self.head_size, 
            self.num_kv_heads * self.head_size, 
            self.num_kv_heads * self.head_size
        ], dim=-1)

        # original attention
        c_ = self._orig_forward(q_, k_, v_, **kwargs)

        # Ulysses all-to-all 2/2
        c = torch.empty_like(c_)
        torch.distributed.all_to_all_single(c, c_, group=self.sp_device_group)
        output = (c.view(self.sp_size, -1, self.num_heads * self.head_size)
                  .transpose(0, 1)
                  .reshape(-1, self.num_heads * self.sp_size * self.head_size))
        
        return output
