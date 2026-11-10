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

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator, DraftModelSpeculator

from arctic_platform.inference.envs import ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK
from arctic_platform.inference.suffix_decoding.cache import SuffixDecodingCache

logger = init_logger(__name__)


class SuffixProposer(BaseSpeculator):
    def __init__(self, vllm_config, device, req_states):
        self.vllm_config = vllm_config
        self.speculative_config = vllm_config.speculative_config
        self.device = device
        self.req_states = req_states
        self.num_speculative_steps = self.speculative_config.num_speculative_tokens
        self.max_model_len = vllm_config.model_config.max_model_len
        self._new_cache()

    def _new_cache(self):
        config = self.speculative_config
        self.cache = SuffixDecodingCache(
            max_tree_depth=config.suffix_cache_max_depth,
            max_cached_requests=config.suffix_cache_max_requests,
        )
        self._lengths = {}
        self.last_draft_lengths = {}

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        pass

    def reset_state(self):
        self._new_cache()

    def update_max_model_len(self, max_model_len):
        self.max_model_len = max_model_len

    def on_requests_finished(self, finished_req_ids):
        for req_id in sorted(finished_req_ids):
            if req_id in self.cache.active_requests:
                self.cache.stop_request(req_id)
            self._lengths.pop(req_id, None)
            self.last_draft_lengths.pop(req_id, None)

    def propose(self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens,
                temperature, seeds, dp_sync_state=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False,
                num_speculative_tokens=None):
        width = self.num_speculative_steps
        result = torch.full((input_batch.num_reqs, width), -1,
                            dtype=torch.int64, device=self.device)
        self.last_draft_lengths = dict.fromkeys(input_batch.req_ids, 0)
        if dummy_run:
            return result
        config = self.speculative_config
        limit = config.disable_by_batch_size or input_batch.num_reqs
        lengths = self.req_states.total_len.gpu[input_batch.idx_mapping].cpu().tolist()
        sampled = num_sampled.cpu().tolist()
        k = width if num_speculative_tokens is None else num_speculative_tokens
        for row, req_id in enumerate(input_batch.req_ids):
            idx = self.req_states.req_id_to_index[req_id]
            total = lengths[row]
            prompt_len = int(self.req_states.prompt_len.np[idx])
            if req_id not in self.cache.active_requests:
                prompt = self.req_states.all_token_ids.gpu[idx, :prompt_len].cpu().tolist()
                self.cache.start_request(req_id, prompt)
                self._lengths[req_id] = prompt_len
            previous = self._lengths[req_id]
            if total > previous:
                response = self.req_states.all_token_ids.gpu[idx, previous:total].cpu().tolist()
                self.cache.add_active_response(req_id, response)
                self._lengths[req_id] = total
            if row >= limit or sampled[row] == 0 or k == 0:
                continue
            max_spec = min(k, self.max_model_len - total - 1)
            if max_spec <= 0:
                continue
            start = max(0, total - config.suffix_cache_max_depth)
            pattern = self.req_states.all_token_ids.gpu[idx, start:total].cpu().tolist()
            draft = self.cache.speculate(
                req_id, pattern, max_spec_tokens=max_spec,
                max_spec_factor=config.suffix_max_spec_factor,
                max_spec_offset=config.suffix_max_spec_offset,
                min_token_prob=config.suffix_min_token_prob,
            )
            tokens = draft.token_ids[:max_spec]
            self.last_draft_lengths[req_id] = len(tokens)
            if tokens:
                result[row, :len(tokens)] = torch.tensor(tokens, device=self.device)
        return result


class ArcticProposer(DraftModelSpeculator):
    def __init__(self, vllm_config, device, req_states):
        super().__init__(vllm_config, device)
        self.req_states = req_states
        self.suffix = (SuffixProposer(vllm_config, device, req_states)
                       if self.speculative_config.enable_suffix_decoding else None)

    def load_draft_model(self, target_model, target_attn_layer_names):
        draft_config_model_config = self.speculative_config.draft_model_config
        spec_model_archs = draft_config_model_config.hf_config.architectures
        if not isinstance(spec_model_archs, list):
            logger.error(
                f"Draft model architectures {spec_model_archs} is not a list. "
            )
            raise TypeError()
        if len(spec_model_archs) != 1:
            logger.error(
                f"Draft model architectures {spec_model_archs} does not have exactly one architecture. "
            )
            raise ValueError()
        if spec_model_archs[0] not in [
                "ArcticMLPSpeculatorPreTrainedModel",
                "ArcticLSTMSpeculatorPreTrainedModel",
                "MLPVariantSpeculatorPreTrainedModel",
        ]:
            logger.error(
                f"Draft model architecture {spec_model_archs} is not supported by Arctic Speculator. "
            )
            raise ValueError()

        if not ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK:
            base_model_arch = self.vllm_config.model_config.architectures[0]
            if not hasattr(draft_config_model_config.hf_config, "base_model_archs"):
                logger.error(
                    "Draft model config does not have base_model_archs attribute. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
                assert False
            base_model_archs_in_spec_config = draft_config_model_config.hf_config.base_model_archs
            if base_model_arch not in base_model_archs_in_spec_config:
                logger.error(
                    f"Draft model trained with base model architectures {base_model_archs_in_spec_config} "
                    f"does not match the base model architecture {base_model_arch} in the vLLM config. "
                    "Set ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK=1 to skip this assertion."
                )
                assert False

        draft_config = copy.copy(self.vllm_config)
        draft_config.model_config = draft_config_model_config
        draft_config.parallel_config = self.speculative_config.draft_parallel_config
        draft_config.speculative_config = None
        draft_config.load_config = self.speculative_config.draft_load_config or self.vllm_config.load_config
        draft_config.quant_config = VllmConfig._get_quantization_config(
            draft_config_model_config, draft_config.load_config)
        return get_model(vllm_config=draft_config)

    def init_cudagraph_manager(self, cudagraph_mode):
        pass

    def capture(self):
        # The neural speculators capture their proposal graphs by batch size.
        pass

    def set_attn(self, *args, **kwargs):
        pass

    def prepare_propose(self, *args, **kwargs):
        pass

    def on_requests_finished(self, finished_req_ids):
        if self.suffix is not None:
            self.suffix.on_requests_finished(finished_req_ids)

    def reset_state(self):
        if self.suffix is not None:
            self.suffix.reset_state()

    def update_max_model_len(self, max_model_len):
        super().update_max_model_len(max_model_len)
        if self.suffix is not None:
            self.suffix.update_max_model_len(max_model_len)

    def propose(self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens,
                temperature, seeds, dp_sync_state=None, dummy_run=False,
                skip_attn_for_dummy_run=False, mm_inputs=None, is_profile=False,
                num_speculative_tokens=None):
        width = self.num_speculative_steps
        batch_size = input_batch.num_reqs
        result = torch.full((batch_size, width), -1,
                            dtype=torch.int64, device=self.device)
        k = width if num_speculative_tokens is None else num_speculative_tokens
        neural_width = min(width, self.model.n_predict)
        limit = min(self.speculative_config.disable_by_batch_size or batch_size, batch_size)
        self.last_draft_lengths = {
            req_id: min(k, neural_width) if row < limit else 0
            for row, req_id in enumerate(input_batch.req_ids)
        }
        if k and batch_size:
            # The state is already advanced by v2 postprocess_sampled. The hidden
            # row that predicted the last accepted token is before the rejected tail.
            hidden_indices = input_batch.query_start_loc[1:limit + 1].long() - 1 - num_rejected[:limit]
            hidden_states = last_hidden_states[hidden_indices]
            context_tokens = last_sampled[input_batch.idx_mapping[:limit]].reshape(-1)
            drafts = self.model.generate_proposals(
                input_ids=context_tokens,
                previous_hidden_states=hidden_states,
                num_predict_tokens=neural_width,
            )
            result[:limit, :neural_width] = drafts
            result.masked_fill_((num_sampled == 0).unsqueeze(1), -1)
        if self.suffix is not None:
            suffix = self.suffix.propose(
                input_batch, attn_metadata, slot_mappings, last_hidden_states,
                aux_hidden_states, num_sampled, num_rejected, last_sampled,
                next_prefill_tokens, temperature, seeds, dp_sync_state,
                dummy_run, skip_attn_for_dummy_run, mm_inputs, is_profile, k,
            )
            suffix_lengths = (suffix >= 0).sum(dim=1)
            # Preserve the old hybrid choice: suffix replaces the entire neural
            # chain only when it offers a longer continuation.
            use_suffix = suffix_lengths > neural_width
            result = torch.where(use_suffix.unsqueeze(1), suffix, result)
            for req_id, length in self.suffix.last_draft_lengths.items():
                self.last_draft_lengths[req_id] = max(self.last_draft_lengths[req_id], length)
        result[:, k:] = -1
        positions = self.req_states.total_len.gpu[input_batch.idx_mapping].unsqueeze(1)
        slots = torch.arange(width, device=self.device).unsqueeze(0)
        result.masked_fill_(positions + slots >= self.max_model_len - 1, -1)
        return result
