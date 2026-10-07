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

    def generate(self, prompts, sampling_params):
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


async def _one(engine, prompt):
    async for out in engine.generate(prompt, {"max_tokens": 4}):
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
