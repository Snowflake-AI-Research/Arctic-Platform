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

"""Reject drafts whose structured-output grammar rows were not validated.

Async scheduling can leave placeholder drafts after a pause. The scheduler
records which drafts the grammar checked; v2 sampling masks the unchecked
input ids without changing the batch used by the model or speculator.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

_APPLIED = False


def attach_spec_token_ids(grammar_output: Any, scheduler_output: Any) -> Any:
    """Record the draft ids ``grammar_bitmask`` just filled rows for.

    The copy is the list as of bitmask construction. A later in-place edit of
    the scheduler output must not change which drafts sampling treats as
    validated.
    """
    if grammar_output is None:
        return None
    spec_tokens = scheduler_output.scheduled_spec_decode_tokens
    recorded = {
        req_id: list(spec_tokens[req_id])
        for req_id in grammar_output.structured_output_request_ids
        if req_id in spec_tokens
    }
    object.__setattr__(grammar_output, "spec_token_ids", recorded)
    return grammar_output


def mask_unvalidated_drafts_v2(grammar_output: Any, input_batch: Any) -> Any:
    """Model Runner V2: the batch to verify, with -1 from each first unchecked draft.

    V2 verifies ``input_ids[logits_indices]``: request ``i`` owns logits rows
    ``cu_num_logits_np[i]`` up to the next entry, and its draft ``k`` sits at
    ``logits_indices[cu_num_logits_np[i] + 1 + k]`` (adaptive verification off,
    the default). Only the copy handed to sampling changes; the drafter keeps
    the real ids.
    """
    rows = []
    for i, req_id in enumerate(input_batch.req_ids):
        num_drafts = int(input_batch.num_draft_tokens_per_req[i])
        scheduled = list((getattr(grammar_output, "spec_token_ids", None) or {}).get(req_id, ()))[:num_drafts]
        if -1 in scheduled:
            first = int(input_batch.cu_num_logits_np[i]) + 1
            rows.extend(range(first + scheduled.index(-1), first + num_drafts))
    if not rows:
        return input_batch
    input_ids = input_batch.input_ids.clone()
    positions = input_batch.logits_indices[torch.tensor(rows, device=input_ids.device)]
    input_ids[positions] = -1
    return replace(input_batch, input_ids=input_ids)


def ensure_spec_decode_grammar_fix() -> None:
    """Patch vLLM so sampling cannot accept a draft the grammar did not check."""
    global _APPLIED
    if _APPLIED:
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("speculative decoding grammar mask")

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    if not getattr(Scheduler.get_grammar_bitmask, "_arctic_spec_grammar", False):
        original_get_grammar_bitmask = Scheduler.get_grammar_bitmask

        def get_grammar_bitmask(self, scheduler_output):
            grammar_output = original_get_grammar_bitmask(self, scheduler_output)
            return attach_spec_token_ids(grammar_output, scheduler_output)

        get_grammar_bitmask._arctic_spec_grammar = True
        get_grammar_bitmask._orig = original_get_grammar_bitmask
        Scheduler.get_grammar_bitmask = get_grammar_bitmask

    if not getattr(GPUModelRunner.sample, "_arctic_spec_grammar", False):
        original_sample_v2 = GPUModelRunner.sample

        def sample_v2(self, hidden_states, input_batch, grammar_output):
            if grammar_output is not None and input_batch.num_draft_tokens:
                input_batch = mask_unvalidated_drafts_v2(grammar_output, input_batch)
            return original_sample_v2(self, hidden_states, input_batch, grammar_output)

        sample_v2._arctic_spec_grammar = True
        sample_v2._orig = original_sample_v2
        GPUModelRunner.sample = sample_v2

    _APPLIED = True
