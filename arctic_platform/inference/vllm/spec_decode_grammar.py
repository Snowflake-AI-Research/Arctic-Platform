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

"""Reject speculative drafts whose grammar rows were not built for them.

vLLM 0.30.0 fills a structured-output bitmask row from the live grammar only
until it sees a placeholder draft id (-1). Later rows, including the bonus
row, are left unconstrained. Under async scheduling the rejection sampler does
not verify that placeholder list. It verifies the drafter's own token ids,
which can be real tokens in those unconstrained rows.

That happens on the first decode step after a weight sync resumes generation
with ``pause_mode="keep"``: the pause drains the in-flight step, so the
scheduler's spec-token list is still the -1 placeholders while the runner
still holds the drafts proposed before the pause.

The scheduler records the ids each request's rows were filled for. Sampling
then rejects every draft from the first -1, so the step emits one token from
the row that was actually constrained. Drafts that were validated before the
bitmask was built are left alone.

Apply this whether or not ``ARCTIC_INFERENCE_ENABLED`` is set. DFlash on DSS
runs on the vanilla vLLM path. Remove the patch once the pinned vLLM rejects
these drafts itself.

See https://github.com/Snowflake-AI-Research/DSS-Issue-Tracker/blob/main/issues/2026-10-02-dflash-speculative-decoding-samples-tokens-outside-the-struc-c850e84c7f7d42918b0a98471d69fad8/ISSUE.md
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import torch

_APPLIED = False
_NO_OVERRIDE = object()
_OVERRIDE_ATTR = "_arctic_grammar_sample_metadata"


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


def reject_unvalidated_drafts(
    grammar_output: Any,
    req_ids: list[str],
    metadata: Any,
) -> Any:
    """Reject each draft from the first one whose grammar row was not filled.

    ``-1`` is the rejection sampler's placeholder: a draft id of -1 is always
    rejected, and the recovered token is drawn from that position's target
    row. The row at the first -1 was filled before padding was noticed, so it
    still carries the grammar. Rows after it do not.
    """
    scheduled_by_req = getattr(grammar_output, "spec_token_ids", None) or {}
    draft_token_ids = metadata.draft_token_ids
    reject = np.zeros(draft_token_ids.shape[0], dtype=bool)
    start = 0
    for req_id, num_drafts in zip(req_ids, metadata.num_draft_tokens):
        num_drafts = int(num_drafts)
        scheduled = list(scheduled_by_req.get(req_id, ()))[:num_drafts]
        if -1 in scheduled:
            first_invalid = scheduled.index(-1)
            reject[start + first_invalid : start + num_drafts] = True
        start += num_drafts
    if not reject.any():
        return metadata
    mask = torch.from_numpy(reject).to(draft_token_ids.device, non_blocking=True)
    return replace(
        metadata,
        draft_token_ids=draft_token_ids.masked_fill(mask, -1),
    )


def stage_sample_metadata(runner: Any, grammar_output: Any) -> None:
    """Choose the spec-decode metadata ``_sample`` should verify.

    The original ``sample_tokens`` also passes this metadata into draft
    proposal. Only verification should see the rejected ids, so the override
    is consumed by the ``_sample`` wrapper rather than written back onto
    ``execute_model_state``.
    """
    override: Any = _NO_OVERRIDE
    state = getattr(runner, "execute_model_state", None)
    if grammar_output is not None and state is not None and state.spec_decode_metadata is not None:
        override = reject_unvalidated_drafts(
            grammar_output,
            runner.input_batch.req_ids,
            state.spec_decode_metadata,
        )
    setattr(runner, _OVERRIDE_ATTR, override)


def metadata_for_sample(runner: Any, spec_decode_metadata: Any) -> Any:
    override = getattr(runner, _OVERRIDE_ATTR, _NO_OVERRIDE)
    if override is _NO_OVERRIDE:
        return spec_decode_metadata
    return override


def clear_sample_metadata(runner: Any) -> None:
    setattr(runner, _OVERRIDE_ATTR, _NO_OVERRIDE)


def ensure_spec_decode_grammar_fix() -> None:
    """Patch vLLM so sampling cannot accept a draft the grammar did not check."""
    global _APPLIED
    if _APPLIED:
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("speculative decoding grammar mask")

    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if not getattr(Scheduler.get_grammar_bitmask, "_arctic_spec_grammar", False):
        original_get_grammar_bitmask = Scheduler.get_grammar_bitmask

        def get_grammar_bitmask(self, scheduler_output):
            grammar_output = original_get_grammar_bitmask(self, scheduler_output)
            return attach_spec_token_ids(grammar_output, scheduler_output)

        get_grammar_bitmask._arctic_spec_grammar = True
        get_grammar_bitmask._orig = original_get_grammar_bitmask
        Scheduler.get_grammar_bitmask = get_grammar_bitmask

    if not getattr(GPUModelRunner.sample_tokens, "_arctic_spec_grammar", False):
        original_sample_tokens = GPUModelRunner.sample_tokens

        def sample_tokens(self, grammar_output):
            stage_sample_metadata(self, grammar_output)
            try:
                return original_sample_tokens(self, grammar_output)
            finally:
                clear_sample_metadata(self)

        sample_tokens._arctic_spec_grammar = True
        sample_tokens._orig = original_sample_tokens
        GPUModelRunner.sample_tokens = sample_tokens

    if not getattr(GPUModelRunner._sample, "_arctic_spec_grammar", False):
        original_sample = GPUModelRunner._sample

        def _sample(self, logits, spec_decode_metadata):
            return original_sample(
                self,
                logits,
                metadata_for_sample(self, spec_decode_metadata),
            )

        _sample._arctic_spec_grammar = True
        _sample._orig = original_sample
        GPUModelRunner._sample = _sample

    _APPLIED = True
