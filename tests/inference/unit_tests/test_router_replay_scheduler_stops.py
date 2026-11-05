from types import SimpleNamespace

from vllm.v1.request import RequestStatus

from arctic_platform.inference.vllm import patches


class _SamplingParams:
    def __init__(
        self,
        *,
        min_tokens=0,
        eos_token_id=99,
        stop_token_ids=None,
        extra_args=None,
        repetition_detection=None,
    ):
        self.min_tokens = min_tokens
        self.eos_token_id = eos_token_id
        self.stop_token_ids = stop_token_ids
        self.extra_args = extra_args or {}
        self.repetition_detection = repetition_detection


class _Request:
    pooling_params = None
    max_tokens = 128

    def __init__(self, output_token_ids, sampling_params, num_tokens=None):
        self.output_token_ids = list(output_token_ids)
        self.num_output_tokens = len(output_token_ids)
        self.num_tokens = num_tokens if num_tokens is not None else self.num_output_tokens
        self.sampling_params = sampling_params
        self.status = None
        self.stop_reason = None


def _check_stop(request, max_model_len=1024):
    patches._patch_scheduler_check_stop()
    from vllm.v1.core.sched import utils as sched_utils

    return sched_utils.check_stop(request, max_model_len)


def test_scheduler_visible_stop_token_sequence_matches_token_ids():
    request = _Request(
        [10, 510, 1773, 29],
        _SamplingParams(
            extra_args={
                "dss_stop_token_sequences": [
                    {"token_ids": [510, 1773, 29], "include_in_output": False}
                ]
            }
        ),
    )

    assert _check_stop(request) is True
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == "dss_stop_token_sequence"


def test_scheduler_visible_stop_token_sequence_accepts_legacy_ids():
    request = _Request(
        [10, 510, 8944, 29],
        _SamplingParams(
            extra_args={
                "dss_stop_token_sequences": [
                    {"text": "</answer>", "ids": [510, 8944, 29]}
                ]
            }
        ),
    )

    assert _check_stop(request) is True
    assert request.status == RequestStatus.FINISHED_STOPPED
    assert request.stop_reason == "</answer>"


def test_scheduler_visible_stop_token_sequence_waits_for_full_suffix():
    request = _Request(
        [10, 510, 1773],
        _SamplingParams(
            extra_args={
                "dss_stop_token_sequences": [
                    {"token_ids": [510, 1773, 29], "include_in_output": False}
                ]
            }
        ),
    )

    assert _check_stop(request) is False
    assert request.status is None


def test_scheduler_visible_stop_token_sequence_respects_min_tokens():
    request = _Request(
        [510, 1773, 29],
        _SamplingParams(
            min_tokens=4,
            extra_args={
                "dss_stop_token_sequences": [
                    {"token_ids": [510, 1773, 29], "include_in_output": False}
                ]
            },
        ),
    )

    assert _check_stop(request) is False
    assert request.status is None


def test_scheduler_visible_stop_preserves_eos_and_stop_token_ids():
    eos_request = _Request([1, 99], _SamplingParams(eos_token_id=99))
    token_request = _Request([1, 77], _SamplingParams(stop_token_ids=[77]))

    assert _check_stop(eos_request) is True
    assert eos_request.status == RequestStatus.FINISHED_STOPPED
    assert _check_stop(token_request) is True
    assert token_request.status == RequestStatus.FINISHED_STOPPED
    assert token_request.stop_reason == 77
