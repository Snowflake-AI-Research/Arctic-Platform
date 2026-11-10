# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

import copy
from dataclasses import replace

import numpy as np
import torch
from vllm.config import CUDAGraphMode, set_current_vllm_config
from vllm.v1.attention.backends.flashinfer import FlashInferMetadata
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_states.default import DefaultModelState


class SwiftKVModelState(DefaultModelState):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._requests = set()
        self._prompt_logprob_requests = set()
        self._token_indices = torch.arange(self.max_num_tokens, device=self.device)
        self._decode_attn_groups = None

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self._requests.add(new_req_data.req_id)
        params = new_req_data.sampling_params
        if params is not None and params.prompt_logprobs is not None:
            self._prompt_logprob_requests.add(new_req_data.req_id)

    def remove_request(self, req_id):
        super().remove_request(req_id)
        self._requests.discard(req_id)
        self._prompt_logprob_requests.discard(req_id)

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                     attn_groups, kv_cache_config, for_capture=False, ubatch_idx=0,
                     model_specific_attn_metadata=None):
        metadata = super().prepare_attn(
            input_batch, cudagraph_mode, block_tables, slot_mappings,
            attn_groups, kv_cache_config, for_capture, ubatch_idx,
            model_specific_attn_metadata,
        )
        indices = input_batch.logits_indices
        if cudagraph_mode == CUDAGraphMode.FULL or not self._requests.intersection(input_batch.req_ids):
            # Uniform decode graphs verify every row, including padded slots.
            indices = self._token_indices[:input_batch.num_tokens_after_padding]
        elif input_batch.has_prefill and self._prompt_logprob_requests.intersection(input_batch.req_ids):
            indices = self._token_indices[:input_batch.num_tokens]
        for layer_metadata in metadata.values():
            layer_metadata.swiftkv_logits_indices = indices
        if (indices.numel() < input_batch.num_tokens
                and any(isinstance(meta, FlashInferMetadata) for meta in metadata.values())):
            self._prepare_flashinfer_decode(
                input_batch, indices, block_tables, slot_mappings,
                attn_groups, kv_cache_config, metadata, ubatch_idx,
            )
        return metadata

    def _prepare_flashinfer_decode(self, batch, indices, block_tables, slot_mappings,
                                  attn_groups, kv_cache_config, metadata, ubatch_idx):
        from ..model_runner import set_shift_parallel_mode

        if self._decode_attn_groups is None:
            model = self.model.model
            late_names = {layer.self_attn.attn.layer_name
                for layer in model.layers[model.config.num_key_value_layers:]}
            config = copy.copy(self.vllm_config)
            config.parallel_config = copy.copy(config.parallel_config)
            config.parallel_config.tensor_parallel_size *= config.parallel_config.ulysses_sequence_parallel_size
            config.parallel_config.ulysses_sequence_parallel_size = 1
            self._decode_attn_groups = []
            # Separate builders keep late-layer plans from overwriting prefill plans.
            with set_current_vllm_config(config), set_shift_parallel_mode(True):
                for groups in attn_groups:
                    decode_groups = []
                    for group in groups:
                        names = [name for name in group.layer_names if name in late_names]
                        if not names:
                            continue
                        late = replace(group, layer_names=names, metadata_builders=[])
                        late.create_metadata_builders(
                            config, self.device,
                            kernel_block_size=group.get_metadata_builder(0).kv_cache_spec.block_size,
                            num_metadata_builders=len(group.metadata_builders),
                        )
                        decode_groups.append(late)
                    self._decode_attn_groups.append(decode_groups)

        counts = np.diff(batch.cu_num_logits_np[:batch.num_reqs + 1])
        # Native FlashInfer builders expect decode requests before prefills.
        req_order = np.argsort(counts, kind="stable")
        req_order = req_order[counts[req_order] > 0]
        token_chunks = torch.split(indices, counts.tolist())
        reordered_indices = torch.cat([token_chunks[i] for i in req_order])
        request_indices = torch.as_tensor(req_order, device=self.device)
        starts_cpu = torch.from_numpy(np.concatenate(([0], np.cumsum(counts[req_order]))).astype(np.int32))
        starts_gpu = starts_cpu.to(self.device)
        cpu_seq_lens = batch.seq_lens_cpu_upper_bound[req_order]
        decode_metadata = build_attn_metadata(
            attn_groups=self._decode_attn_groups,
            num_reqs=len(req_order), num_tokens=indices.numel(),
            query_start_loc_gpu=starts_gpu, query_start_loc_cpu=starts_cpu,
            max_query_len=int(counts.max()),
            seq_lens=batch.seq_lens[request_indices],
            max_seq_len=int(cpu_seq_lens.max()),
            block_tables=tuple(table[request_indices] for table in block_tables),
            slot_mappings=slot_mappings[:, reordered_indices],
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=cpu_seq_lens,
            positions=batch.positions[reordered_indices],
            is_prefilling=torch.from_numpy(batch.is_prefilling_np[req_order]),
            ubatch_idx=ubatch_idx,
        )
        for meta in metadata.values():
            meta.swiftkv_decode_metadata = decode_metadata
            meta.swiftkv_decode_indices = reordered_indices
            meta.swiftkv_inverse_sort_indices = torch.argsort(reordered_indices)
