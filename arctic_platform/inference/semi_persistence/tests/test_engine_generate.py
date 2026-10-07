"""Unit tests for ``_SemiPEngine``'s generate path in ``server/semip_engine.py``.

``_SemiPEngine`` stands in for vLLM's ``AsyncLLM`` when a job restores a
semi-p image. Two properties of ``AsyncLLM`` that callers rely on are checked
here against a fake ``Instance``:

* concurrent generates reach the engine together, so the child can batch
  them, instead of one at a time;
* every response carries the real prompt and completion token ids.

The fake fires completions from its own thread, as the demuxer does.

Run from the package directory::

    python -m pytest tests/test_engine_generate.py -v
"""
from __future__ import annotations

import ast
import asyncio
import os
import sys
import threading
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import _PKG, se  # noqa: E402


def _load_apply_result():
    """``Instance._apply_result``, without importing torch and pynvml."""
    path = os.path.join(_PKG, "instance.py")
    with open(path) as handle:
        tree = ast.parse(handle.read(), path)
    cls = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "Instance")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "_apply_result")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"),
         namespace)
    return namespace["_apply_result"]


_apply_result = _load_apply_result()


class _FakeInstance:
    """The slice of ``Instance`` that ``_SemiPEngine`` drives."""

    def __init__(self):
        self.listeners = []
        self.prompts = {}            # req_id -> prompts, in submit order
        self.generate_results = {}
        self.last_req_id = None
        self.cmds = []
        self._latched = None
        self._next = 0

    def add_cmd_listener(self, cmd, callback):
        assert cmd == "generate"
        self.listeners.append(callback)

    def generate(self, prompts, sampling_params, reasoning_ended=None):
        self.reasoning_ended = reasoning_ended
        rid = f"inst0-{self._next}"
        self._next += 1
        self.last_req_id = rid
        self.prompts[rid] = prompts

    def finish(self, rid, prompt_ids=None, completion_ids=None, error=None,
               info=None):
        if error is None:
            self.generate_results[rid] = {
                "outputs": [[f"text-{rid}"]],
                "prompt_token_ids": [prompt_ids],
                "completion_token_ids": [[completion_ids]],
                "finish_reasons": ["stop"],
            }
        elif self._latched is None:
            self._latched = error
        info = {"req_id": rid} if info is None else info
        for callback in self.listeners:
            callback("generate", 0.1, error, info)

    def wait(self):
        err, self._latched = self._latched, None
        if err is not None:
            raise RuntimeError(f"command 'generate' failed: {err}")

    def sleep(self):
        self.cmds.append("sleep")

    def teardown(self):
        self.cmds.append("teardown")


def _engine(inst):
    return se._SemiPEngine(inst, tokenizer_path=None, model=None)


async def _one(engine, prompt, reasoning_ended=None):
    async for out in engine.generate(prompt, {"max_tokens": 4},
                                     reasoning_ended=reasoning_ended):
        return out


def _from_thread(fn, *args, **kwargs):
    thread = threading.Thread(target=fn, args=args, kwargs=kwargs)
    thread.start()
    thread.join()


async def _until(predicate, timeout=2.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        assert loop.time() < deadline, "condition never became true"
        await asyncio.sleep(0.01)


def test_concurrent_generates_are_in_flight_together():
    async def main():
        inst = _FakeInstance()
        engine = _engine(inst)
        a = asyncio.create_task(_one(engine, "a"))
        b = asyncio.create_task(_one(engine, "b"))
        await _until(lambda: len(inst.prompts) == 2)
        by_prompt = {p[0]: rid for rid, p in inst.prompts.items()}
        # Out of order, to show each caller gets its own result.
        _from_thread(inst.finish, by_prompt["b"], [3], [30, 31])
        _from_thread(inst.finish, by_prompt["a"], [1, 2], [10])
        out_a, out_b = await asyncio.gather(a, b)
        assert out_a.prompt_token_ids == [1, 2]
        assert out_a.outputs[0].token_ids == [10]
        assert out_b.prompt_token_ids == [3]
        assert out_b.outputs[0].token_ids == [30, 31]
        assert out_b.outputs[0].text == f"text-{by_prompt['b']}"
        assert inst.generate_results == {}

    asyncio.run(main())


def test_child_result_survives_instance_into_response():
    """What the child sends at completion reaches the response unchanged."""
    child_info = {
        "req_id": "inst0-0",
        "outputs": [["Paris"]],
        "prompt_token_ids": [[1, 2, 3]],
        "completion_token_ids": [[[271, 57590, 248044]]],
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "prompt_logprobs": [[None, {2: "lp2"}, {3: "lp3"}]],
        "completion_logprobs": [[[{271: "a"}, {57590: "b"}, {248044: "c"}]]],
        "num_cached_tokens": 2,
        "finish_reasons": ["length"],
    }
    inst = types.SimpleNamespace(last_info={}, generate_results={})
    _apply_result(inst, "generate", child_info)
    out = _engine(_FakeInstance())._to_request_output(
        inst.generate_results["inst0-0"])
    assert out.prompt_token_ids == [1, 2, 3]
    assert out.outputs[0].token_ids == [271, 57590, 248044]
    assert out.outputs[0].text == "Paris"
    assert out.outputs[0].finish_reason == "length"
    assert out.num_cached_tokens == 2
    assert out.outputs[0].logprobs == [{271: "a"}, {57590: "b"}, {248044: "c"}]
    assert out.prompt_logprobs == [None, {2: "lp2"}, {3: "lp3"}]


def test_reasoning_ended_reaches_the_instance():
    async def main():
        inst = _FakeInstance()
        task = asyncio.create_task(_one(_engine(inst), "a", reasoning_ended=True))
        await _until(lambda: inst.prompts)
        assert inst.reasoning_ended is True
        _from_thread(inst.finish, inst.last_req_id, [1], [2])
        await task

    asyncio.run(main())


class _FakeSamplingParams:
    """vLLM's SamplingParams as _sampling_params_to_dict sees it."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    @classmethod
    def from_optional(cls, n=1, temperature=None, max_tokens=16,
                      logprobs=None, structured_outputs=None, extra_args=None,
                      logit_bias=None, output_kind=None, stream_interval=None,
                      skip_clone=None):
        raise AssertionError("only its signature is read")


def test_sampling_params_forward_every_caller_field():
    guided = object()
    params = _FakeSamplingParams(
        n=1, temperature=0.0, max_tokens=None, logprobs=1,
        structured_outputs=guided,
        extra_args={"dss_stop_token_sequences": [[1, 2]]},
        logit_bias={5: -1.0}, output_kind="DELTA", stream_interval=4,
        skip_clone=True, _all_stop_token_ids={7})
    sp = se._sampling_params_to_dict(params)
    assert sp == {
        "n": 1, "temperature": 0.0, "max_tokens": None, "logprobs": 1,
        "structured_outputs": guided,
        "extra_args": {"dss_stop_token_sequences": [[1, 2]]},
        "logit_bias": {5: -1.0},
    }


def test_logprobs_reach_the_response():
    sample = [{10: "lp10"}, {11: "lp11"}]
    prompt = [None, {2: "lp2"}]
    res = {"outputs": [["x"]], "prompt_token_ids": [[1, 2]],
           "completion_token_ids": [[[10, 11]]],
           "completion_logprobs": [[sample]], "prompt_logprobs": [prompt]}
    out = _engine(_FakeInstance())._to_request_output(
        res, {"logprobs": 1, "prompt_logprobs": 1})
    assert out.outputs[0].logprobs == sample
    assert out.prompt_logprobs == prompt


def test_requested_logprobs_that_never_arrive_raise():
    res = {"outputs": [["x"]], "prompt_token_ids": [[1]],
           "completion_token_ids": [[[10]]]}
    engine = _engine(_FakeInstance())
    assert engine._to_request_output(res, {"logprobs": None}).outputs[0].logprobs is None
    for name in ("logprobs", "prompt_logprobs"):
        try:
            engine._to_request_output(res, {name: 1})
        except RuntimeError as exc:
            assert name in str(exc)
        else:
            raise AssertionError(f"expected RuntimeError for {name}")


def test_missing_token_ids_raise_instead_of_zeros():
    async def main():
        inst = _FakeInstance()
        engine = _engine(inst)
        task = asyncio.create_task(_one(engine, "a"))
        await _until(lambda: inst.prompts)
        rid = inst.last_req_id

        def finish_without_ids():
            inst.generate_results[rid] = {
                "outputs": [["hi"]], "prompt_tokens": 2,
                "completion_tokens": 1}
            for callback in inst.listeners:
                callback("generate", 0.1, None, {"req_id": rid})

        _from_thread(finish_without_ids)
        try:
            await task
        except RuntimeError as exc:
            assert "no token ids" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")

    asyncio.run(main())


def test_failed_generate_fails_alone_and_not_the_next_sleep():
    async def main():
        inst = _FakeInstance()
        engine = _engine(inst)
        bad = asyncio.create_task(_one(engine, "bad"))
        good = asyncio.create_task(_one(engine, "good"))
        await _until(lambda: len(inst.prompts) == 2)
        by_prompt = {p[0]: rid for rid, p in inst.prompts.items()}
        _from_thread(inst.finish, by_prompt["bad"], error="boom")
        _from_thread(inst.finish, by_prompt["good"], [1], [2])
        try:
            await bad
        except RuntimeError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
        assert (await good).outputs[0].token_ids == [2]
        assert await engine.collective_rpc("sleep") == [None]
        assert inst.cmds == ["sleep"]

    asyncio.run(main())


def test_sleep_waits_for_inflight_and_holds_new_generates():
    async def main():
        inst = _FakeInstance()
        engine = _engine(inst)
        first = asyncio.create_task(_one(engine, "first"))
        await _until(lambda: len(inst.prompts) == 1)
        sleep = asyncio.create_task(engine.collective_rpc("sleep"))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(_one(engine, "second"))
        await asyncio.sleep(0.05)
        assert inst.cmds == []
        assert len(inst.prompts) == 1

        _from_thread(inst.finish, "inst0-0", [1], [2])
        await sleep
        assert inst.cmds == ["sleep"]
        await first
        await _until(lambda: len(inst.prompts) == 2)
        _from_thread(inst.finish, "inst0-1", [3], [4])
        assert (await second).outputs[0].token_ids == [4]

    asyncio.run(main())


def test_dead_child_fails_every_waiter():
    async def main():
        inst = _FakeInstance()
        engine = _engine(inst)
        tasks = [asyncio.create_task(_one(engine, p)) for p in "abc"]
        await _until(lambda: len(inst.prompts) == 3)
        _from_thread(inst.finish, None,
                     error="child process died during generate", info={})
        for task in tasks:
            try:
                await task
            except RuntimeError as exc:
                assert "child process died" in str(exc)
            else:
                raise AssertionError("expected RuntimeError")
        assert await engine.collective_rpc("sleep") == [None]

    asyncio.run(main())


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
