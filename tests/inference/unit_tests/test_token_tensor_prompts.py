"""Token-id prompts given as 1-D integer tensors: same generation input as ``list[int]``."""
from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.inference.server.replica_pool import ReplicaPool
from arctic_platform.inference.server.scheduler import _compute_prefix_hash
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

WorkerClass = InferenceWorker.__ray_metadata__.modified_class

START, END = 10, 11


def _new_worker(parser=None):
    worker = WorkerClass.__new__(WorkerClass)
    worker._active_lora_int_id = None
    worker._active_lora_name = None
    worker._active_lora_is_3d = False
    worker._grammar_stop_token_ids = ()
    worker._stream_cleanup_failed = False
    worker._engine_streams = {}
    worker.state = WorkerLifecycleState.READY
    worker._router_replay_tx = None
    worker._replica_label = None
    worker._reasoning_parser = parser
    worker._return_reasoning_content = False
    worker.llm = SimpleNamespace(get_tokenizer=lambda: object())
    return worker


def _think_parser(start=START, end=END):
    return SimpleNamespace(
        start_token_id=start,
        end_token_id=end,
        is_reasoning_end=lambda ids: end in ids,
    )


PROMPTS = [
    [1, 2, 3],
    [1, START, 2],
    [1, START, 2, END, 3],
    [1, END, 2, START, 3],
    [START, 1, 2],
    [1, 2, START],
    [END, 1, 2],
    [1, 2, END],
    [START],
    [7],
]


@pytest.mark.parametrize("parser", [None, _think_parser(), SimpleNamespace(is_reasoning_end=lambda ids: False)])
@pytest.mark.parametrize("enable_thinking", [None, False, True])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("prompt", PROMPTS)
def test_tensor_prefill_matches_list(prompt, dtype, enable_thinking, parser):
    worker = _new_worker(parser)
    expected = worker._prefill_think_for_generation(list(prompt), enable_thinking=enable_thinking)
    got = worker._prefill_think_for_generation(torch.tensor(prompt, dtype=dtype), enable_thinking=enable_thinking)
    assert got == expected
    assert all(type(t) is int for t in got[0])


@pytest.mark.parametrize("start,end", [(START, END), (START, START)])
def test_vectorized_reasoning_open_matches_loop(start, end):
    worker = _new_worker(_think_parser(start, end))
    rng = random.Random(0)
    for _ in range(500):
        prompt = [rng.choice([1, 2, start, end]) for _ in range(rng.randint(1, 12))]
        assert worker._is_reasoning_open(torch.tensor(prompt, dtype=torch.int32)) == worker._is_reasoning_open(prompt)


@pytest.mark.parametrize("enable_thinking", [None, True])
def test_generate_sends_vllm_the_same_input_for_tensor_and_list(enable_thinking):
    captured = []

    async def fake_generate_once(prompt_input, _params, _request_id, *, reasoning_ended, **_kwargs):
        captured.append((prompt_input, reasoning_ended))
        choice = SimpleNamespace(text="x", token_ids=[1], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[1], num_cached_tokens=0, prompt_logprobs=None)

    worker = _new_worker(_think_parser())
    worker._generate_once = fake_generate_once
    params = {"max_tokens": 1} if enable_thinking is None else {"max_tokens": 1, "enable_thinking": True}
    prompt = [5, 6, START, 7, END, 8]
    asyncio.run(worker.generate(prompt, dict(params)))
    asyncio.run(worker.generate(torch.tensor(prompt, dtype=torch.int32), dict(params)))
    assert captured[0] == captured[1]
    assert type(captured[1][0]["prompt_token_ids"]) is list


@pytest.mark.parametrize(
    "prompt",
    [
        (1, 2, 3),
        torch.tensor([1, 2, 3]).numpy(),
        torch.tensor([[1, 2], [3, 4]]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([True, False]),
        torch.tensor([], dtype=torch.int32),
        {"prompt_token_ids": [1, 2]},
    ],
)
def test_generate_rejects_inputs_it_would_read_as_text(prompt):
    async def fail_generate_once(*_args, **_kwargs):
        raise AssertionError("must not reach vLLM")

    worker = _new_worker()
    worker._generate_once = fail_generate_once
    with pytest.raises((TypeError, ValueError)):
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))


class _RecordingScheduler:
    def __init__(self):
        self.prompts = []

    def submit(self, prompt, sampling_params, routing_key=None, strict=False):
        self.prompts.append(prompt)
        return object()


def _new_pool():
    pool = ReplicaPool.__new__(ReplicaPool)
    pool._scheduler = _RecordingScheduler()
    pool._sleeping = False
    return pool


def _assert_compact_tensor(prompt, expected):
    assert torch.is_tensor(prompt)
    assert prompt.tolist() == expected
    assert prompt.is_contiguous()
    assert prompt.untyped_storage().nbytes() == prompt.numel() * prompt.element_size()


def test_pool_submits_a_1d_tensor_as_one_prompt():
    pool = _new_pool()
    prompt = torch.tensor([1, 2, 3], dtype=torch.int32)
    assert len(pool.submit_generate_futures(prompt, {})) == 1
    assert pool._scheduler.prompts[0] is prompt


def test_pool_splits_a_2d_tensor_into_compact_rows():
    pool = _new_pool()
    batch = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    assert len(pool.submit_generate_futures(batch, {})) == 2
    _assert_compact_tensor(pool._scheduler.prompts[0], [1, 2, 3])
    _assert_compact_tensor(pool._scheduler.prompts[1], [4, 5, 6])


def test_pool_compacts_tensor_views_in_a_batch_and_keeps_lists():
    pool = _new_pool()
    buffer = torch.arange(10, dtype=torch.int32)
    pool.submit_generate_futures([buffer[2:5], [7, 8], buffer[::3]], {})
    _assert_compact_tensor(pool._scheduler.prompts[0], [2, 3, 4])
    assert pool._scheduler.prompts[1] == [7, 8]
    _assert_compact_tensor(pool._scheduler.prompts[2], [0, 3, 6, 9])


def test_pool_keeps_list_prompts_unchanged():
    pool = _new_pool()
    pool.submit_generate_futures([1, 2, 3], {})
    pool.submit_generate_futures([[1, 2], "text"], {})
    assert pool._scheduler.prompts == [[1, 2, 3], [1, 2], "text"]


@pytest.mark.parametrize(
    "prompts",
    [
        torch.zeros((1, 2, 3), dtype=torch.int32),
        torch.tensor([1.0, 2.0]),
        [torch.tensor([1.0])],
        [torch.tensor([], dtype=torch.int64)],
    ],
)
def test_pool_rejects_malformed_tensors_before_submitting(prompts):
    pool = _new_pool()
    with pytest.raises((TypeError, ValueError)):
        pool.submit_generate_futures(prompts, {})
    assert pool._scheduler.prompts == []


def test_prefix_hash_of_a_tensor_ignores_integer_width_and_layout():
    tokens = [3, 1, 4, 1, 5, 9, 2, 6]
    as_int32 = torch.tensor(tokens, dtype=torch.int32)
    expected = hash(as_int32.numpy().tobytes())
    assert _compute_prefix_hash(as_int32) == expected
    assert _compute_prefix_hash(torch.tensor(tokens, dtype=torch.int64)) == expected
    assert _compute_prefix_hash(torch.tensor(tokens + tokens, dtype=torch.int64)[::2]) == hash(
        torch.tensor((tokens + tokens)[::2], dtype=torch.int32).numpy().tobytes()
    )
    assert _compute_prefix_hash(torch.tensor(tokens[::-1], dtype=torch.int32)) != expected
    assert _compute_prefix_hash(tokens) == hash(tuple(tokens))
