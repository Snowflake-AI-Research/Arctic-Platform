# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Warn when returned routed experts differ from the routes a request computed.

vLLM returns a request's routed experts from its auxiliary-output store for
every keyed block below the step's end, including blocks the request recomputed
because its KV prefix hit stopped short of them, e.g. when EAGLE-style
speculative decoding drops the newest prefix block. A store entry that is still
referenced keeps its first payload, so those positions can return another
request's routes. This logs, once per request, how many recomputed positions
came back with routes other than the ones the request computed. Outputs are
unchanged.
"""

from __future__ import annotations

import logging

from arctic_platform.inference.patching import ArcticPatch

logger = logging.getLogger(__name__)

_APPLIED = False


def ensure_routed_experts_recompute_warning() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("routed-experts recompute warning")

    from vllm.distributed.aux_output_connector.worker import AuxOutputWorkerConnector

    original_commit_output = AuxOutputWorkerConnector._commit_output

    class RoutedExpertsRecomputeWarningPatch(ArcticPatch[AuxOutputWorkerConnector]):
        def _commit_output(self, pending):
            computed = {}
            for request_id, token_start, start, end, rejected in zip(
                pending.request_ids,
                pending.token_starts,
                pending.query_start_loc[:-1],
                pending.query_start_loc[1:],
                pending.num_rejected,
            ):
                state = self._requests[request_id]
                # Where this step's rows start, as the original resolves it.
                capture_start = int(token_start) if state.capture_cursor is None else state.capture_cursor
                if state.emit_cursor < capture_start:
                    computed[request_id] = (capture_start, pending.routed_experts[start : end - int(rejected)])
            outputs = original_commit_output(self, pending)
            block_size = self._buffer.block_size
            for request_id, (capture_start, rows) in computed.items():
                output = outputs.get(request_id)
                if output is None:
                    continue
                state = self._requests[request_id]
                stored_end = min(state.capture_cursor // block_size, len(state.aux_output_keys)) * block_size
                if stored_end <= capture_start:
                    continue
                returned = output.rows[capture_start - output.token_start : stored_end - output.token_start]
                differ = (returned != rows[: stored_end - capture_start]).reshape(len(returned), -1).any(axis=1)
                if differ.any():
                    logger.warning(
                        "routed experts: request %s computed positions [%d, %d) (block size %d) but %d of them "
                        "returned stored routes that differ from its own",
                        request_id,
                        capture_start,
                        stored_end,
                        block_size,
                        int(differ.sum()),
                    )
            return outputs

    RoutedExpertsRecomputeWarningPatch.apply_patch()
    _APPLIED = True
