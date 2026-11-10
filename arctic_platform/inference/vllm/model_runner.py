# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
import copy
from contextvars import ContextVar
from dataclasses import replace
from typing import Optional

import torch
import vllm.distributed.parallel_state as parallel_state
from vllm.config import CUDAGraphMode
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.utils.math_utils import round_up
from vllm.v1.worker.gpu import model_runner as runner_module
from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from arctic_platform.inference.patching import ArcticPatch

logger = init_logger(__name__)

SP_TP_MODE = None
_RUNNER_SP_SIZE = ContextVar("arctic_runner_sp_size", default=1)


@contextlib.contextmanager
def set_shift_parallel_mode(mode: Optional[bool]):
    """
    Swap the tensor-parallel group to an SP-compatible variant when 'mode' is True.
    """
    if mode is None:
        yield
        return

    global SP_TP_MODE

    if not is_shift_parallel_mode():
        assert not getattr(parallel_state, "_TP_STATE_PATCHED", False)
        parallel_state._ORIG_TP = parallel_state._TP

    old_mode = SP_TP_MODE
    old_tp_group = parallel_state.get_tp_group()
    SP_TP_MODE = mode

    parallel_state._TP = ((parallel_state._SP_TP or parallel_state._ORIG_TP)
                          if mode else parallel_state._ORIG_TP)

    try:
        yield
    finally:
        SP_TP_MODE = old_mode
        parallel_state._TP = old_tp_group


def is_shift_parallel_mode() -> bool:
    """Check if the shift parallel mode is enabled."""
    global SP_TP_MODE
    return SP_TP_MODE is True


def install_ulysses_dispatch():
    original = runner_module.dispatch_cg_and_sync_dp
    if getattr(original, "_arctic_ulysses", False):
        return

    def dispatch(manager, num_reqs, num_tokens, uniform_token_count, *args, **kwargs):
        size = getattr(manager, "_arctic_sp_size", _RUNNER_SP_SIZE.get())
        num_tokens = round_up(num_tokens, size)
        return original(manager, num_reqs, num_tokens, uniform_token_count, *args, **kwargs)

    dispatch._arctic_ulysses = True
    runner_module.dispatch_cg_and_sync_dp = dispatch


class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):
    _orig_init = GPUModelRunner.__init__
    _orig_load_model = GPUModelRunner.load_model
    _orig_initialize_kv_cache = GPUModelRunner.initialize_kv_cache
    _orig_prepare_inputs = GPUModelRunner.prepare_inputs
    _orig_sample = GPUModelRunner.sample
    _orig_execute_model = GPUModelRunner.execute_model
    _orig_sample_tokens = GPUModelRunner.sample_tokens
    _orig_reload_weights = GPUModelRunner.reload_weights
    _orig_profile_run = GPUModelRunner.profile_run
    _orig_capture_model = GPUModelRunner.capture_model
    _orig_needs_cudagraph_capture = GPUModelRunner.needs_cudagraph_capture

    def __init__(self, vllm_config, device):
        install_ulysses_dispatch()
        self._orig_init(vllm_config, device)
        self.use_ulysses = getattr(self.parallel_config, "ulysses_sequence_parallel_size", 1) > 1
        self.shift_model = None
        self.shift_cudagraph_manager = None
        self._step_uses_shift = False

    def load_model(self, load_dummy_weights=False, *, model=None, **kwargs):
        shift_config = (copy.deepcopy(self.vllm_config)
                        if self.parallel_config.enable_shift_parallel else None)
        self._orig_load_model(load_dummy_weights, model=model, **kwargs)
        if self.use_ulysses:
            self.monkeypatch_forward()
        if shift_config is not None:
            shift_config.parallel_config.tensor_parallel_size *= self.parallel_config.ulysses_sequence_parallel_size
            shift_config.parallel_config.ulysses_sequence_parallel_size = 1
            shift_config.compilation_config.pass_config.fuse_allreduce_rms = False
            shift_config.compilation_config.static_forward_context = {}
            before = torch.cuda.memory_allocated(self.device)
            with set_shift_parallel_mode(True):
                self.shift_model = get_model(vllm_config=shift_config)
            self.model_memory_usage += torch.cuda.memory_allocated(self.device) - before
            self.shift_vllm_config = shift_config
            self.shift_forward_context = shift_config.compilation_config.static_forward_context
            self._shift_logits_perm = self._compute_shift_logits_perm()
            permutation = self._shift_logits_perm
            if permutation is not None:
                processor = self.shift_model.logits_processor
                gather_logits = processor._gather_logits

                def ordered_gather(logits):
                    gathered = gather_logits(logits)
                    if gathered is None:
                        return None
                    # Restore shard order while the padded vocabulary is intact.
                    chunks = gathered.chunk(len(permutation), dim=-1)
                    return torch.cat([chunks[i] for i in permutation], dim=-1)

                processor._gather_logits = ordered_gather
            communicator = parallel_state._SP_TP.device_communicator
            if communicator is not None and getattr(communicator, "ca_comm", None) is not None:
                communicator.ca_comm.disabled = True

    def monkeypatch_forward(self):
        sp = parallel_state._SP
        model_forward = self.model.forward
        size, rank = sp.world_size, sp.rank_in_group

        @torch._dynamo.disable
        def ulysses_forward(*args, **kwargs):
            key = "inputs_embeds" if kwargs.get("inputs_embeds") is not None else "input_ids"
            inputs = kwargs[key]
            tokens = inputs.shape[0]
            if tokens % size:
                raise ValueError(f"Ulysses requires {size}-aligned token padding, got {tokens}")
            count = tokens // size
            start = rank * count
            kwargs[key] = inputs[start:start + count]
            kwargs["positions"] = kwargs["positions"][..., start:start + count]
            intermediate = kwargs.get("intermediate_tensors")
            if intermediate is not None:
                kwargs["intermediate_tensors"] = intermediate[start:start + count]
            with set_shift_parallel_mode(False):
                output = model_forward(*args, **kwargs)

            def gather(tensor):
                if tensor.shape[0] == tokens:
                    return tensor
                gathered = tensor.new_empty((tokens, *tensor.shape[1:]))
                torch.distributed.all_gather_into_tensor(gathered, tensor.contiguous(), group=sp.device_group)
                return gathered

            if isinstance(output, torch.Tensor):
                return gather(output)
            if isinstance(output, tuple):
                hidden, aux = output
                return gather(hidden), [gather(t) for t in aux]
            from vllm.sequence import IntermediateTensors
            return IntermediateTensors({key: gather(t) for key, t in output.tensors.items()})

        self.model.forward = ulysses_forward

    def initialize_kv_cache(self, kv_cache_config, **kwargs):
        from .swiftkv.model_state import SwiftKVModelState
        if isinstance(self.model_state, SwiftKVModelState):
            for config in (self.compilation_config, self.vllm_config.compilation_config):
                if config.cudagraph_mode == CUDAGraphMode.FULL:
                    config.cudagraph_mode = CUDAGraphMode.FULL_DECODE_ONLY
        self._orig_initialize_kv_cache(kv_cache_config, **kwargs)
        if self.use_ulysses:
            self.cudagraph_manager._arctic_sp_size = self.parallel_config.ulysses_sequence_parallel_size
            ulysses_dispatch = self.cudagraph_manager.dispatch
            size = self.parallel_config.ulysses_sequence_parallel_size

            def aligned_dispatch(num_reqs, num_tokens, *args, **kwargs):
                desc = ulysses_dispatch(num_reqs, num_tokens, *args, **kwargs)
                if desc.num_tokens % size:
                    return replace(desc, cg_mode=CUDAGraphMode.NONE, num_tokens=round_up(num_tokens, size), num_reqs=num_reqs)
                return desc

            self.cudagraph_manager.dispatch = aligned_dispatch
            for mode, descriptors in self.cudagraph_manager._capture_descs.items():
                self.cudagraph_manager._capture_descs[mode] = [
                    desc for desc in descriptors if desc.num_tokens % self.parallel_config.ulysses_sequence_parallel_size == 0
                ]
        if self.shift_model is None:
            return
        base_context = self.compilation_config.static_forward_context
        for name, layer in self.shift_forward_context.items():
            base_layer = base_context.get(name)
            if base_layer is not None and hasattr(base_layer, "kv_cache"):
                layer.kv_cache = base_layer.kv_cache
        self.shift_cudagraph_manager = ModelCudaGraphManager(
            self.shift_vllm_config, self.device, self.cudagraph_manager.cudagraph_mode,
            decode_query_lens=self.vllm_config.get_decode_cudagraph_query_lens(self.decode_query_len),
            lora_capture_cases=self.lora_capture_cases,
        )

    @contextlib.contextmanager
    def _use_shift_model(self, enabled):
        if not enabled:
            yield
            return
        model, manager = self.model, self.cudagraph_manager
        state_model = self.model_state.model
        context = self.compilation_config.static_forward_context
        self.model = self.shift_model
        self.model_state.model = self.shift_model
        self.cudagraph_manager = self.shift_cudagraph_manager
        self.compilation_config.static_forward_context = self.shift_forward_context
        try:
            with set_shift_parallel_mode(True):
                yield
        finally:
            self.model, self.cudagraph_manager = model, manager
            self.model_state.model = state_model
            self.compilation_config.static_forward_context = context

    def prepare_inputs(self, *args, **kwargs):
        batch = self._orig_prepare_inputs(*args, **kwargs)
        if self.speculative_config is not None and self.speculative_config.custom_method in (
            "arctic", "mlp_speculator", "arctic_suffix"
        ):
            # -1 marks absent drafts for verification, but embedding lookup needs
            # valid ids. Keep the mask with this batch, including async steps.
            batch._arctic_invalid_draft_inputs = batch.input_ids < 0
            batch.input_ids.clamp_(min=0)
        return batch

    def sample(self, hidden_states, input_batch, grammar_output):
        invalid = getattr(input_batch, "_arctic_invalid_draft_inputs", None)
        if invalid is not None:
            input_batch = replace(input_batch, input_ids=input_batch.input_ids.masked_fill(invalid, -1))
        return self._orig_sample(hidden_states, input_batch, grammar_output)

    def execute_model(self, scheduler_output, *args, **kwargs):
        self._step_uses_shift = (
            self.shift_model is not None
            and scheduler_output.total_num_scheduled_tokens <= self.parallel_config.shift_parallel_threshold
        )
        with self._use_shift_model(self._step_uses_shift):
            size = 1 if is_shift_parallel_mode() else self.parallel_config.ulysses_sequence_parallel_size
            token = _RUNNER_SP_SIZE.set(size)
            try:
                return self._orig_execute_model(scheduler_output, *args, **kwargs)
            finally:
                _RUNNER_SP_SIZE.reset(token)

    def sample_tokens(self, grammar_output):
        with self._use_shift_model(self._step_uses_shift):
            output = self._orig_sample_tokens(grammar_output)
        lengths = getattr(self.speculator, "last_draft_lengths", None)
        if output is not None and lengths is not None:
            model_output = getattr(output, "model_runner_output", output)
            model_output._arctic_draft_lengths = dict(lengths)
        return output

    def profile_run(self, *args, **kwargs):
        self._orig_profile_run(*args, **kwargs)
        if self.shift_model is not None:
            with self._use_shift_model(True):
                self._dummy_run(self.max_num_tokens, is_profile=True, skip_attn=not hasattr(self, "block_tables"))

    def needs_cudagraph_capture(self):
        return self._orig_needs_cudagraph_capture() or (
            self.shift_cudagraph_manager is not None and self.shift_cudagraph_manager.needs_capture()
        )

    def capture_model(self, **kwargs):
        memory = self._orig_capture_model(**kwargs)
        if self.shift_cudagraph_manager is not None:
            with self._use_shift_model(True), self._shift_graph_capture_context():
                memory += self._orig_capture_model(**kwargs)
        return memory

    def _compute_shift_logits_perm(self):
        """Compute permutation to fix logits after all-gather for the shift model.

        NCCL all_gather_into_tensor outputs chunks in sorted global-rank order,
        but vLLM's GroupCoordinator assigns rank_in_group (and thus weight shards)
        based on the unsorted rank list.  When _SP_TP has non-trivial ordering
        (e.g. [0,2,1,3]), the vocabulary shards in the gathered logits are
        scrambled.  This method returns the permutation that restores correct
        vocabulary order, or None if no reordering is needed.
        """
        sp_tp = parallel_state._SP_TP
        if sp_tp is None:
            return None
        ranks = sp_tp.ranks
        sorted_ranks = sorted(ranks)
        if ranks == sorted_ranks:
            return None
        nccl_to_shard = [ranks.index(r) for r in sorted_ranks]
        perm = sorted(range(len(nccl_to_shard)), key=lambda i: nccl_to_shard[i])
        return perm

    @contextlib.contextmanager
    def _shift_graph_capture_context(self):
        """Disable custom all-reduce on _TP and _SP_TP during shift model graph
        capture so collectives fall back to pynccl (NCCL), which handles graph
        capture natively.  Custom all-reduce communicators were not set up through
        the normal vLLM graph_capture() path, so using them inside a CUDA graph
        would crash."""
        from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator,
        )

        def _get_ca_comm(group):
            if (group is not None
                    and group.device_communicator is not None
                    and isinstance(group.device_communicator, CudaCommunicator)):
                return group.device_communicator.ca_comm
            return None

        tp_ca = _get_ca_comm(parallel_state._TP)
        sp_tp_ca = _get_ca_comm(parallel_state._SP_TP)

        saved_tp = tp_ca.disabled if tp_ca is not None else None
        saved_sp_tp = sp_tp_ca.disabled if sp_tp_ca is not None else None
        if tp_ca is not None:
            tp_ca.disabled = True
        if sp_tp_ca is not None:
            sp_tp_ca.disabled = True
        try:
            yield
        finally:
            if tp_ca is not None and saved_tp is not None:
                tp_ca.disabled = saved_tp
            if sp_tp_ca is not None and saved_sp_tp is not None:
                sp_tp_ca.disabled = saved_sp_tp
