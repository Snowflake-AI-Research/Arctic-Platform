"""Minimal vLLM compatibility hooks for router replay.

These patches are intentionally independent from the full ArcticInference
vLLM patch stack so weight sync and router replay can run with
``ARCTIC_INFERENCE_ENABLED=0``.
"""

from __future__ import annotations

import os

from arctic_platform.inference.utils import require_supported_vllm_version


def _check_supported_vllm_version() -> None:
    if os.getenv("ARCTIC_INFERENCE_SKIP_VERSION_CHECK", "0") != "1":
        require_supported_vllm_version("Router replay")


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


def patch_scheduler_check_stop() -> None:
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

    check_stop_with_token_sequences._arctic_router_replay_patch = True
    sched_utils.check_stop = check_stop_with_token_sequences
    scheduler_mod.check_stop = check_stop_with_token_sequences


def ensure_router_replay_vllm_patches() -> None:
    _check_supported_vllm_version()

    from vllm.v1.core.sched import scheduler as scheduler_mod

    if getattr(scheduler_mod.check_stop, "_arctic_router_replay_patch", False):
        return
    patch_scheduler_check_stop()
