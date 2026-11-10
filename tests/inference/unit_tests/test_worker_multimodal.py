"""Media prompts: the worker hands vLLM a ``TokensPrompt`` and returns the expanded ids and media hashes."""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import random
import struct
import sys
import types
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest


class _StubVllm(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Any ``vllm.*`` import gets a module whose missing attributes are plain classes."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "vllm" or fullname.startswith("vllm."):
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda name: (
            _stub_attr(spec.name, name) if not name.startswith("__") else _raise(name)
        )
        return module

    def exec_module(self, module):
        pass


def _raise(name):
    raise AttributeError(name)


def _stub_attr(module_name, name):
    value = type(name, (), {"__init__": lambda self, *a, **k: None})
    setattr(sys.modules[module_name], name, value)
    return value


class _FakeHasher:
    """vLLM 0.32.0's signature: the algorithm is positional-only and required."""

    calls = []

    @staticmethod
    def hash_kwargs(algorithm, /, **kwargs):
        _FakeHasher.calls.append((algorithm, kwargs))
        return f"{algorithm}:" + repr(sorted(kwargs.items()))


def _is_vllm(name):
    return name == "vllm" or name.startswith("vllm.")


@contextlib.contextmanager
def _vllm_stand_in():
    """Swap every ``vllm`` / ``vllm.*`` module for the stand-in, then put back exactly what was there.

    The streaming tests leave their own partial ``vllm`` in ``sys.modules``; it is hidden while
    the stand-in is in place and restored after, whichever file runs first.
    """
    saved = {name: module for name, module in sys.modules.items() if _is_vllm(name)}
    for name in saved:
        del sys.modules[name]
    finder = _StubVllm()
    sys.meta_path.insert(0, finder)
    try:
        vllm = importlib.import_module("vllm")
        # The real ones: TokensPrompt is a TypedDict, so a dict; SamplingParams keeps its kwargs.
        vllm.SamplingParams = lambda **kwargs: SimpleNamespace(all_stop_token_ids=set(), **kwargs)
        importlib.import_module("vllm.inputs").TokensPrompt = dict
        importlib.import_module("vllm.multimodal.hasher").MultiModalHasher = _FakeHasher
        yield vllm
    finally:
        sys.meta_path.remove(finder)
        for name in [name for name in sys.modules if _is_vllm(name)]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _stand_in_vllm():
    _FakeHasher.calls.clear()
    with _vllm_stand_in():
        yield


with _vllm_stand_in():
    import arctic_platform.inference

    # The package __init__ applies vLLM patches; import its modules without it.
    if "arctic_platform.inference.server" not in sys.modules:
        server_package = types.ModuleType("arctic_platform.inference.server")
        server_package.__path__ = [str(Path(arctic_platform.inference.__file__).parent / "server")]
        sys.modules["arctic_platform.inference.server"] = server_package
        arctic_platform.inference.server = server_package

    from arctic_platform.inference.server import replica_pool, streaming, worker as worker_module
    from arctic_platform.inference.server.media_prompts import MediaDecodeError
    from arctic_platform.inference.server.replica_pool import ReplicaPool
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

WorkerClass = InferenceWorker.__ray_metadata__.modified_class

START, END = 10, 11
IMAGE_PAD = 151655
IMAGE = object()
EXPANDED = [1, 2, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 3]


def _model_config(
    algorithm="blake3", media_io_kwargs=None, merge=dict, mm_processor_kwargs=None, mm_processor_cache_gb=4
):
    """A ModelConfig with a MultiModalConfig; ``merge`` stands in for ``merge_mm_processor_kwargs``."""
    mm_config = SimpleNamespace(
        mm_hasher_algorithm=algorithm,
        media_io_kwargs=media_io_kwargs or {},
        mm_processor_kwargs=mm_processor_kwargs,
        merge_mm_processor_kwargs=merge,
        mm_processor_cache_gb=mm_processor_cache_gb,
    )
    return SimpleNamespace(model="qwen-vl", max_model_len=4096, multimodal_config=mm_config)


def _renderer(enable_prefix_caching=True):
    """vLLM's renderer keeps the engine's ``VllmConfig`` as ``config`` (vllm/renderers/base.py lines 106-110)."""
    cache_config = SimpleNamespace(enable_prefix_caching=enable_prefix_caching)
    return SimpleNamespace(config=SimpleNamespace(cache_config=cache_config))


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
    worker._chat_only_reasoner = False
    worker.llm = SimpleNamespace(
        get_tokenizer=lambda: object(),
        model_config=_model_config(),
        renderer=_renderer(),
    )
    return worker


def _think_parser():
    return SimpleNamespace(start_token_id=START, end_token_id=END, is_reasoning_end=lambda ids: END in ids)


class _FakeEngine:
    """Records what vLLM gets and expands the one image placeholder to four tokens."""

    def __init__(self):
        self.calls = []

    def get_tokenizer(self):
        return object()

    model_config = _model_config()
    renderer = _renderer()

    async def generate(self, prompt, params, **kwargs):
        self.calls.append((prompt, params, kwargs))
        ids = prompt["prompt_token_ids"] if isinstance(prompt, dict) else [0]
        if isinstance(prompt, dict) and "multi_modal_data" in prompt:
            ids = [t for i in ids for t in ([IMAGE_PAD] * 4 if i == IMAGE_PAD else [i])]
        choice = SimpleNamespace(index=0, text="ok", token_ids=[7], finish_reason="stop", logprobs=None)
        yield SimpleNamespace(outputs=[choice], prompt_token_ids=ids, num_cached_tokens=0, prompt_logprobs=None)


def _generate(worker, prompt, params=None, model_config=None, enable_prefix_caching=True):
    worker.llm = _FakeEngine()
    if model_config is not None:
        worker.llm.model_config = model_config
    worker.llm.renderer = _renderer(enable_prefix_caching)
    result = asyncio.run(worker.generate(prompt, dict(params or {"max_tokens": 1})))
    return result, worker.llm.calls


def _media_prompt(**extra):
    return {"prompt_token_ids": [1, 2, IMAGE_PAD, 3], "multi_modal_data": {"image": IMAGE}, **extra}


def test_stand_in_hides_and_restores_another_tests_vllm():
    # What tests/inference/streaming leaves behind: a bare ``vllm`` package with no SamplingParams.
    outer = {name: module for name, module in sys.modules.items() if _is_vllm(name)}
    planted = {name: types.ModuleType(name) for name in ("vllm", "vllm.exceptions")}
    meta_path = list(sys.meta_path)
    for name in outer:
        del sys.modules[name]
    sys.modules.update(planted)
    try:
        with _vllm_stand_in() as vllm:
            assert sys.modules["vllm"] is vllm and vllm is not planted["vllm"]
            assert "vllm.exceptions" not in sys.modules
            assert importlib.import_module("vllm.multimodal.hasher").MultiModalHasher is _FakeHasher
        assert {name: module for name, module in sys.modules.items() if _is_vllm(name)} == planted
        assert sys.meta_path == meta_path
    finally:
        for name in planted:
            sys.modules.pop(name, None)
        sys.modules.update(outer)


def test_worker_declares_media_support():
    assert worker_module.SUPPORTS_MEDIA is True
    assert WorkerClass.supports_media is True


def test_media_prompt_reaches_vllm_as_tokens_prompt():
    kwargs = {"max_pixels": 1024}
    prompt = _media_prompt(mm_processor_kwargs=kwargs, multi_modal_uuids={"image": ["img-0"]})
    result, calls = _generate(_new_worker(), prompt)
    (sent, _params, _kwargs), = calls
    assert sent == prompt
    assert sent["multi_modal_data"]["image"] is IMAGE
    assert sent["mm_processor_kwargs"] is kwargs
    assert result["prompt_token_ids"] == EXPANDED
    assert result["prompt_len"] == len(EXPANDED)
    hasher = _FakeHasher
    assert result["mm_hashes"] == {
        "image": [hasher.hash_kwargs("blake3", model_id="qwen-vl", image="img-0", mm_processor_kwargs=kwargs)]
    }
    assert result["token_ids"] == [7]


def test_media_prompt_drops_none_keys():
    _result, calls = _generate(_new_worker(), _media_prompt(mm_processor_kwargs=None, multi_modal_uuids=None))
    assert set(calls[0][0]) == {"prompt_token_ids", "multi_modal_data"}


def _hash_calls(prompt, model_config, enable_prefix_caching=True):
    calls = _FakeHasher.calls
    calls.clear()
    result, _calls = _generate(
        _new_worker(), prompt, model_config=model_config, enable_prefix_caching=enable_prefix_caching
    )
    return result["mm_hashes"], list(calls)


def _direct_hash_calls(prompt, model_config):
    """``_media_hashes`` itself, for prompt keys ``generate`` refuses (media prompts carry no ``media_io_kwargs``)."""
    calls = _FakeHasher.calls
    calls.clear()
    hashes = worker_module._media_hashes(prompt, model_config, _renderer().config.cache_config, "req-1")
    return hashes, list(calls)


def test_media_hashes_use_engine_algorithm_positionally():
    images = [object(), object()]
    prompt = _media_prompt(multi_modal_data={"image": images}, multi_modal_uuids={"image": ["a", None]})
    hashes, calls = _hash_calls(prompt, _model_config(algorithm="sha512"))
    # No hash factors: the caller's uuid is the hash as is, the other item is hashed.
    assert calls == [("sha512", {"model_id": "qwen-vl", "image": images[1]})]
    assert hashes == {"image": ["a", "sha512:" + repr([("image", images[1]), ("model_id", "qwen-vl")])]}


def test_media_hash_factors_are_nested_and_scoped_per_modality():
    image, clip = object(), object()
    prompt = _media_prompt(
        multi_modal_data={"image": image, "video": clip},
        mm_processor_kwargs={"fps": 2, "images_kwargs": {"max_pixels": 64}, "videos_kwargs": {"max_frames": 4}},
        media_io_kwargs={"video": {"num_frames": 8}},
    )
    hashes, calls = _direct_hash_calls(prompt, _model_config())
    assert calls == [
        (
            "blake3",
            {
                "model_id": "qwen-vl",
                "image": image,
                "mm_processor_kwargs": {"fps": 2, "images_kwargs": {"max_pixels": 64}},
            },
        ),
        (
            "blake3",
            {
                "model_id": "qwen-vl",
                "video": clip,
                "media_io_kwargs": {"video": {"num_frames": 8}},
                "mm_processor_kwargs": {"fps": 2, "videos_kwargs": {"max_frames": 4}},
            },
        ),
    ]
    assert set(hashes) == {"image", "video"}


def test_media_uuid_is_hashed_in_place_of_item_with_factors():
    image = object()
    kwargs = {"do_normalize": False}
    prompt = _media_prompt(
        multi_modal_data={"image": [image]}, multi_modal_uuids={"image": ["img-0"]}, mm_processor_kwargs=kwargs
    )
    hashes, calls = _hash_calls(prompt, _model_config())
    assert calls == [("blake3", {"model_id": "qwen-vl", "image": "img-0", "mm_processor_kwargs": kwargs})]
    assert hashes["image"] != ["img-0"]


# A deployment that sets both kwargs at engine level. vLLM 0.32's default mm_device_do_normalize makes
# its merged processor kwargs carry do_normalize / do_rescale even when the prompt has none.
def _engine_kwargs_config(**kwargs):
    engine_kwargs = {"max_pixels": 256}
    merge = lambda request: {**engine_kwargs, **request, "do_normalize": False, "do_rescale": False}  # noqa: E731
    return _model_config(
        mm_processor_kwargs=engine_kwargs,
        merge=merge,
        media_io_kwargs={"image": {"image_mode": "L"}, "video": {"num_frames": 8}},
        **kwargs,
    )


def _vllm_mm_hashes(model_id, mm_data, mm_uuids, mm_processor_kwargs, media_io_kwargs):
    """``ProcessorInputs.get_mm_hashes`` (vllm/multimodal/processing/inputs.py, v0.32.0), step by step."""
    hasher, scoped = _FakeHasher, {"image": "images_kwargs", "video": "videos_kwargs", "audio": "audio_kwargs"}
    mm_hashes = {}
    for modality, items in mm_data.items():  # line 45
        # Lines 49-62: this modality's media_io_kwargs, and the processor kwargs minus other modalities' scopes.
        modality_media_io = media_io_kwargs.get(modality, {})
        modality_media_io = {modality: modality_media_io} if modality_media_io else {}
        processor = {k: v for k, v in mm_processor_kwargs.items() if k not in scoped.values() or k == scoped[modality]}
        # Lines 70-75: only the non-empty factors.
        factors = {k: v for k, v in (("media_io_kwargs", modality_media_io), ("mm_processor_kwargs", processor)) if v}
        uuids = mm_uuids.get(modality, [None] * len(items))  # lines 77-81
        mm_hashes[modality] = [
            # Lines 91-102: hash the uuid in the item's place, or the item; lines 103-105: else the uuid as is.
            hasher.hash_kwargs("blake3", model_id=model_id, **{modality: item if uuid is None else uuid}, **factors)
            if uuid is None or factors
            else uuid
            for item, uuid in zip(items, uuids)
        ]
    return mm_hashes


def test_engine_kwargs_are_not_hash_factors():
    image, clip = object(), object()
    prompt = _media_prompt(multi_modal_data={"image": [image], "video": [clip]}, multi_modal_uuids={"image": ["img-0"]})
    hashes, _calls = _hash_calls(prompt, _engine_kwargs_config())
    # vllm/renderers/base.py (v0.32.0) lines 950-955 take both kwargs from the prompt, which has none, and
    # lines 915-921 put them on ProcessorInputs as {}: the engine's are not merged in.
    expected = _vllm_mm_hashes("qwen-vl", {"image": [image], "video": [clip]}, {"image": ["img-0"]}, {}, {})
    assert hashes == expected
    assert hashes["image"] == ["img-0"]


def test_engine_kwargs_are_not_merged_into_request_kwargs():
    image, clip = object(), object()
    request_kwargs = {"fps": 2, "images_kwargs": {"max_pixels": 64}}
    request_media_io = {"video": {"num_frames": 4}}
    prompt = _media_prompt(
        multi_modal_data={"image": [image], "video": [clip]},
        multi_modal_uuids={"image": ["img-0"]},
        mm_processor_kwargs=request_kwargs,
        media_io_kwargs=request_media_io,
    )
    hashes, calls = _direct_hash_calls(prompt, _engine_kwargs_config())
    # Lines 950-955 and 915-921 again: ProcessorInputs gets the prompt's kwargs exactly.
    expected = _vllm_mm_hashes(
        "qwen-vl", {"image": [image], "video": [clip]}, {"image": ["img-0"]}, request_kwargs, request_media_io
    )
    assert hashes == expected
    assert calls == [
        ("blake3", {"model_id": "qwen-vl", "image": "img-0", "mm_processor_kwargs": request_kwargs}),
        (
            "blake3",
            {
                "model_id": "qwen-vl",
                "video": clip,
                "media_io_kwargs": request_media_io,
                "mm_processor_kwargs": {"fps": 2},
            },
        ),
    ]


def test_generate_hashes_request_kwargs_without_engine_kwargs():
    image = object()
    request_kwargs = {"images_kwargs": {"max_pixels": 64}}
    prompt = _media_prompt(
        multi_modal_data={"image": [image]}, multi_modal_uuids={"image": ["img-0"]}, mm_processor_kwargs=request_kwargs
    )
    hashes, _calls = _hash_calls(prompt, _engine_kwargs_config())
    # Through generate: lines 950-955 and 915-921 hand ProcessorInputs the prompt's kwargs, not the engine merge.
    assert hashes == _vllm_mm_hashes("qwen-vl", {"image": [image]}, {"image": ["img-0"]}, request_kwargs, {})


@pytest.mark.parametrize("request_kwargs", [None, {"fps": 2}])
def test_media_hashes_take_per_request_uuids_when_both_caches_are_off(monkeypatch, request_kwargs):
    monkeypatch.setattr(worker_module, "uuid4", lambda: "req-1")
    images = [object(), object()]
    prompt = _media_prompt(
        multi_modal_data={"image": images},
        multi_modal_uuids={"image": [None, "img-1"]},
        mm_processor_kwargs=request_kwargs,
    )
    hashes, _calls = _hash_calls(prompt, _engine_kwargs_config(mm_processor_cache_gb=0), enable_prefix_caching=False)
    # vllm/renderers/base.py lines 874-889: with mm_processor_cache_gb == 0 and prefix caching off, an item without
    # a uuid gets f"{mm_req_id}-{modality}-{i}"; a given uuid is kept. vLLM's mm_req_id is a renderer counter
    # (line 906) the worker cannot see, so the worker's request id stands in for it.
    uuids = {"image": ["req-1-image-0", "img-1"]}
    assert hashes == _vllm_mm_hashes("qwen-vl", {"image": images}, uuids, request_kwargs or {}, {})
    if request_kwargs is None:
        assert hashes["image"] == ["req-1-image-0", "img-1"]


@pytest.mark.parametrize("cache_gb, prefix_caching", [(0, True), (4, False)])
def test_media_hashes_hash_content_when_either_cache_is_on(monkeypatch, cache_gb, prefix_caching):
    monkeypatch.setattr(worker_module, "uuid4", lambda: "req-1")
    image = object()
    prompt = _media_prompt(multi_modal_data={"image": [image]})
    hashes, _calls = _hash_calls(
        prompt, _engine_kwargs_config(mm_processor_cache_gb=cache_gb), enable_prefix_caching=prefix_caching
    )
    assert hashes == _vllm_mm_hashes("qwen-vl", {"image": [image]}, {}, {}, {})


def test_media_prompt_needs_multimodal_model():
    async def fail_generate_once(*_args, **_kwargs):
        raise AssertionError("must not reach vLLM")

    worker = _new_worker()
    worker._generate_once = fail_generate_once
    worker.llm.model_config.multimodal_config = None
    with pytest.raises(ValueError, match="multi_modal_data"):
        asyncio.run(worker.generate(_media_prompt(), {"max_tokens": 1}))


def test_media_uuid_count_must_match_items():
    async def fail_generate_once(*_args, **_kwargs):
        raise AssertionError("must not reach vLLM")

    worker = _new_worker()
    worker._generate_once = fail_generate_once
    prompt = _media_prompt(multi_modal_uuids={"image": ["a", "b"]})
    with pytest.raises(ValueError, match="multi_modal_uuids"):
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))


@pytest.mark.parametrize(
    "prompt",
    [
        {"prompt_token_ids": [1, 2], "multi_modal_data": {}},
        {"prompt_token_ids": [], "multi_modal_data": {"image": IMAGE}},
        {"prompt_token_ids": [1, "2"], "multi_modal_data": {"image": IMAGE}},
        {"prompt": "text", "multi_modal_data": {"image": IMAGE}},
        {"prompt_token_ids": [1], "multi_modal_data": {"image": IMAGE}, "mm_processor_kwargs": [1]},
        {"prompt_token_ids": [1], "multi_modal_data": {"image": IMAGE}, "extra": 1},
    ],
)
def test_bad_media_prompts_never_reach_vllm(prompt):
    async def fail_generate_once(*_args, **_kwargs):
        raise AssertionError("must not reach vLLM")

    worker = _new_worker()
    worker._generate_once = fail_generate_once
    with pytest.raises(ValueError):
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))


@pytest.mark.parametrize("enable_thinking", [None, False, True])
@pytest.mark.parametrize(
    "token_ids",
    [[1, IMAGE_PAD, 2], [1, START, IMAGE_PAD, 2], [START, 1, END, IMAGE_PAD], [IMAGE_PAD]],
)
def test_media_prefill_think_matches_token_list(token_ids, enable_thinking):
    worker = _new_worker(_think_parser())
    expected_ids, expected_fallback, expected_ended = worker._prefill_think_for_generation(
        list(token_ids), enable_thinking=enable_thinking
    )
    prompt = {"prompt_token_ids": list(token_ids), "multi_modal_data": {"image": IMAGE}}
    got, fallback, ended = worker._prefill_think_for_generation(prompt, enable_thinking=enable_thinking)
    assert got == {**prompt, "prompt_token_ids": expected_ids}
    assert got["multi_modal_data"]["image"] is IMAGE
    assert (fallback, ended) == (expected_fallback, expected_ended)
    assert prompt["prompt_token_ids"] == token_ids


def test_media_prompt_with_thinking_gets_think_token_not_text():
    prompt = _media_prompt()
    _result, calls = _generate(_new_worker(_think_parser()), prompt, {"max_tokens": 1, "enable_thinking": True})
    sent = calls[0][0]
    assert sent["prompt_token_ids"] == [1, 2, IMAGE_PAD, 3, START]
    assert calls[0][2]["reasoning_ended"] is False


@pytest.mark.parametrize("prompt", ["hello", [1, 2, 3]])
def test_text_only_results_are_unchanged(prompt):
    result, calls = _generate(_new_worker(), prompt)
    assert "prompt_token_ids" not in result and "mm_hashes" not in result
    assert calls[0][0] == ({"prompt_token_ids": prompt} if isinstance(prompt, list) else prompt)


class _RecordingScheduler:
    def __init__(self):
        self.submitted = []

    def submit(self, prompt, params, routing_key=None, strict=False):
        self.submitted.append((prompt, params, routing_key))
        return object()


def _pool():
    pool = ReplicaPool.__new__(ReplicaPool)
    pool._scheduler = _RecordingScheduler()
    pool._sleeping = False
    return pool


def test_pool_takes_one_media_prompt_as_one_request():
    pool = _pool()
    prompt = _media_prompt()
    assert len(pool.submit_generate_futures(prompt, {"max_tokens": 1})) == 1
    (sent, params, key), = pool._scheduler.submitted
    assert sent is prompt and params == {"max_tokens": 1}
    assert key == replica_pool._media_routing_key(prompt, None)


def test_pool_takes_media_prompts_as_batch_members():
    pool = _pool()
    prompts = [_media_prompt(), [1, 2, 3], _media_prompt(prompt_token_ids=[4, IMAGE_PAD])]
    assert len(pool.submit_generate_futures(prompts, routing_key=[None, None, "turns-1"])) == 3
    sent = pool._scheduler.submitted
    assert [s[0] for s in sent] == prompts
    assert sent[1][2] is None and sent[2][2] == "turns-1"


def test_media_routing_key_follows_token_ids():
    a = replica_pool._media_routing_key(_media_prompt(), None)
    assert a == replica_pool._media_routing_key(_media_prompt(multi_modal_data={"image": object()}), None)
    assert a != replica_pool._media_routing_key(_media_prompt(prompt_token_ids=[9, IMAGE_PAD]), None)
    assert replica_pool._media_routing_key(_media_prompt(), "key") == "key"
    assert replica_pool._media_routing_key([1, 2], None) is None


def test_stream_validation_passes_media_prompt_through():
    prompt = _media_prompt(multi_modal_uuids=None)
    checked, _params = streaming.validate_request(prompt, {"max_tokens": 4})
    assert checked == {"prompt_token_ids": [1, 2, IMAGE_PAD, 3], "multi_modal_data": {"image": IMAGE}}
    assert checked["multi_modal_data"]["image"] is IMAGE
    with pytest.raises(ValueError):
        streaming.validate_request({"prompt_token_ids": [1]}, {"max_tokens": 4})
    assert streaming.validate_request([1, 2], {"max_tokens": 4})[0] == [1, 2]


_NOT_MEDIA_DICTS = [{"prompt_token_ids": [1, 2]}, {"prompt": "text"}, {}]


@pytest.mark.parametrize("prompt", _NOT_MEDIA_DICTS)
def test_dict_without_multimodal_data_is_not_a_media_prompt(prompt):
    from arctic_platform.inference.server import media_prompts

    assert not media_prompts.is_media_prompt(prompt)
    assert media_prompts.is_media_prompt({**prompt, "multi_modal_data": {}})


@pytest.mark.parametrize("prompt", _NOT_MEDIA_DICTS)
def test_worker_refuses_dict_without_multimodal_data_as_before(prompt):
    async def fail_generate_once(*_args, **_kwargs):
        raise AssertionError("must not reach vLLM")

    worker = _new_worker()
    worker._generate_once = fail_generate_once
    with pytest.raises(TypeError, match="^Expected a text prompt, a token-id list, .* got dict$") as error:
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))
    assert "multi_modal_data" not in str(error.value)


@pytest.mark.parametrize("prompt", _NOT_MEDIA_DICTS)
def test_stream_refuses_dict_without_multimodal_data_as_before(prompt):
    with pytest.raises(ValueError, match="^Expected one prepared text prompt, token-ID list, ") as error:
        streaming.validate_request(prompt, {"max_tokens": 4})
    assert "multi_modal_data" not in str(error.value)
    from dataclasses import asdict

    worker = _new_worker()
    with pytest.raises(ValueError, match="^Expected one prepared text prompt, token-ID list, "):
        worker.start_stream("not-media", prompt, {"max_tokens": 1}, 20, asdict(streaming.StreamLimits()))
    assert worker._engine_streams == {}


@pytest.mark.parametrize("prompt", _NOT_MEDIA_DICTS)
def test_think_prefill_does_not_read_token_ids_of_dict_without_multimodal_data(prompt):
    worker = _new_worker(_think_parser())
    assert worker._prefill_think_for_generation(prompt, enable_thinking=None) == (prompt, [], None)


@pytest.mark.parametrize("prompt", _NOT_MEDIA_DICTS)
def test_pool_hands_dict_without_multimodal_data_to_the_worker_whole(prompt):
    pool = _pool()
    assert len(pool.submit_generate_futures(prompt, {"max_tokens": 1})) == 1
    assert len(pool.submit_generate_futures([prompt, [1, 2]], routing_key=[None, "k"])) == 2
    (one, _, one_key), (member, _, member_key), (_, _, list_key) = pool._scheduler.submitted
    assert one is prompt and member is prompt
    # No media routing key: only a media prompt is keyed by its token ids.
    assert one_key is None and member_key is None and list_key == "k"
    assert replica_pool._media_routing_key(prompt, None) is None
    with pytest.raises(TypeError, match="^Expected a text prompt, a token-id list, .* got dict$"):
        asyncio.run(_new_worker().generate(one, {"max_tokens": 1}))


def test_media_prompt_takes_token_tensor_ids():
    import torch

    prompt = _media_prompt(prompt_token_ids=torch.tensor([1, 2, IMAGE_PAD, 3], dtype=torch.int32))
    result, calls = _generate(_new_worker(), prompt)
    sent = calls[0][0]["prompt_token_ids"]
    assert sent == [1, 2, IMAGE_PAD, 3] and all(type(t) is int for t in sent)
    assert result["prompt_token_ids"] == EXPANDED
    assert replica_pool._media_routing_key(prompt, None) == replica_pool._media_routing_key(_media_prompt(), None)
    with pytest.raises((TypeError, ValueError)):
        _generate(_new_worker(), _media_prompt(prompt_token_ids=torch.tensor([1.0, 2.0])))


class _StreamEngine(_FakeEngine):
    async def abort(self, request_id):
        pass

    def get_num_unfinished_requests(self):
        return 0


def test_stream_hands_media_prompt_to_vllm():
    from dataclasses import asdict

    async def scenario():
        worker = _new_worker()
        worker.llm = _StreamEngine()
        worker._stream_sampling_params = lambda params: params
        prompt = _media_prompt(mm_processor_kwargs={"max_pixels": 1024})
        assert worker.start_stream("media", prompt, {"max_tokens": 1}, 20, asdict(streaming.StreamLimits()))
        events = []
        reader = worker.stream_events("media")
        async for batch in reader:
            events.extend(batch)
            if batch[-1]["type"] in ("completed", "terminal_error"):
                break
            worker.acknowledge_stream("media", batch[-1]["sequence"])
        await reader.aclose()
        (sent, _params, kwargs), = worker.llm.calls
        assert sent == prompt
        assert sent["multi_modal_data"]["image"] is IMAGE
        assert kwargs["request_id"] == "media"
        return events

    events = asyncio.run(scenario())
    assert [e["type"] for e in events] == ["delta", "choice_finished", "usage", "completed"]
    assert events[2]["prompt_tokens"] == len(EXPANDED)


# --- raw encoded bytes: decoded in the worker with vLLM's loaders ---


def _png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


# Real minimal headers: PNG's signature and a 1x1 IHDR chunk, a JPEG SOI and JFIF APP0
# segment, and an MP4 ``ftyp`` box. A tag after the header tells test items apart.
PNG = b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
JPEG = b"\xff\xd8\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"


def _ftyp(major, *compatible):
    body = major + struct.pack(">I", 0x200) + b"".join(compatible)
    return struct.pack(">I", 8 + len(body)) + b"ftyp" + body


MP4 = _ftyp(b"isom", b"isom", b"iso2", b"avc1", b"mp41")


def _image(tag):
    return PNG + tag


def _video(tag):
    return MP4 + tag


_MEDIA = {"image": _image, "video": _video}


class _Decoded:
    """Stands in for ``MediaWithBytes``: the decoded media with the original bytes."""

    def __init__(self, kind, media, original_bytes):
        self.kind, self.media, self.original_bytes = kind, media, original_bytes


class _Cv2Error(Exception):
    """``cv2.error``: OpenCV's exception class has this lowercase name."""


_Cv2Error.__name__ = _Cv2Error.__qualname__ = "error"


class InvalidDataError(ValueError):
    """``av.error.InvalidDataError``, a ``ValueError`` (PyAV's ``FFmpegError`` base aside)."""


def _decoder_failures():
    """What vLLM's loaders let through, by modality: real Pillow classes, OpenCV's and PyAV's by name."""
    from PIL import Image, UnidentifiedImageError

    return {
        "image": [
            UnidentifiedImageError,
            Image.DecompressionBombError,
            OSError,  # Pillow's "image file is truncated"
            SyntaxError,  # Pillow's "broken PNG file"
            ValueError,
        ],
        "video": [_Cv2Error, InvalidDataError, OSError, ValueError],
    }


_DECODE_FAILURES = [(modality, cls) for modality, classes in _decoder_failures().items() for cls in classes]


def _failing_bytes(modality, cls):
    """Bytes the stand-in loaders fail on with *cls*; the decoder's message echoes them."""
    return _MEDIA[modality](b"secret-input-%d" % _DECODE_FAILURES.index((modality, cls)))


class _Built(list):
    """The loaders built, in order; ``loaded`` is each ``load_bytes`` call's (modality, bytes)."""

    def __init__(self):
        super().__init__()
        self.loaded = []


_VIDEO_LIBRARIES = ("cv2", "torchcodec", "PyNvVideoCodec")


def _video_libraries(monkeypatch, *present, backend=None):
    """Make ``find_spec`` find only the *present* video libraries; set or clear the env var backend."""
    real = importlib.util.find_spec
    looked_up = []

    def find_spec(name, package=None):
        if name in _VIDEO_LIBRARIES:
            looked_up.append(name)
            return importlib.machinery.ModuleSpec(name, None) if name in present else None
        return real(name, package)

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)
    if backend is None:
        monkeypatch.delenv("VLLM_VIDEO_LOADER_BACKEND", raising=False)
    else:
        monkeypatch.setenv("VLLM_VIDEO_LOADER_BACKEND", backend)
    return looked_up


@pytest.fixture
def fake_loaders(monkeypatch):
    # The default video backend's library is there, whether or not OpenCV is installed.
    _video_libraries(monkeypatch, "cv2")
    built = _Built()
    failures = {_failing_bytes(modality, cls): cls for modality, cls in _DECODE_FAILURES}
    loaded = built.loaded

    def load(kind, data):
        loaded.append((kind, data))
        if data in failures:
            raise failures[data](f"cannot decode {data!r}")
        return _Decoded(kind, f"pil:{data!r}" if kind == "image" else "frames", data)

    class ImageMediaIO:
        def __init__(self, image_mode="RGB", **kwargs):
            self.kwargs = {"image_mode": image_mode, **kwargs}
            built.append(("image", self.kwargs))

        @classmethod
        def merge_kwargs(cls, default, runtime):
            return {**(default or {}), **(runtime or {})}

        def load_bytes(self, data):
            return load("image", data)

    class VideoMediaIO:
        def __init__(self, image_io, num_frames=32, **kwargs):
            built.append(("video", {"num_frames": num_frames, **kwargs}))

        @classmethod
        def merge_kwargs(cls, default, runtime):
            return {**(default or {}), **(runtime or {})}

        def load_bytes(self, data):
            return load("video", data)

    for name, cls in (("image", ImageMediaIO), ("video", VideoMediaIO)):
        module = types.ModuleType(f"vllm.multimodal.media.{name}")
        setattr(module, f"{name.capitalize()}MediaIO", cls)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    return built


def test_media_prompts_module_declares_support():
    from arctic_platform.inference.server import media_prompts

    assert media_prompts.SUPPORTS_MEDIA is True
    assert streaming.is_media_prompt is media_prompts.is_media_prompt
    assert worker_module.SUPPORTS_MEDIA is media_prompts.SUPPORTS_MEDIA


def test_image_bytes_are_decoded_and_keep_their_bytes(fake_loaders):
    prompt = _media_prompt(multi_modal_data={"image": [_image(b"one"), JPEG], "video": [_video(b"vid")]})
    result, calls = _generate(_new_worker(), prompt)
    (sent, _params, _kwargs), = calls
    images = sent["multi_modal_data"]["image"]
    assert [(i.kind, i.original_bytes) for i in images] == [("image", _image(b"one")), ("image", JPEG)]
    assert sent["multi_modal_data"]["video"][0].original_bytes == _video(b"vid")
    # The hasher saw the decoded items, not the bytes.
    hashed = _FakeHasher_calls()
    assert all(isinstance(kwargs.get("image", kwargs.get("video")), _Decoded) for _alg, kwargs in hashed)
    assert prompt["multi_modal_data"]["image"] == [_image(b"one"), JPEG]
    assert result["mm_hashes"].keys() == {"image", "video"}


def _FakeHasher_calls():
    return _FakeHasher.calls[-3:]


def test_decode_uses_engine_media_io_kwargs(fake_loaders):
    config = _model_config(media_io_kwargs={"image": {"image_mode": "L"}, "video": {"num_frames": 8}})
    _generate(_new_worker(), _media_prompt(multi_modal_data={"image": [PNG], "video": [MP4]}), model_config=config)
    assert ("image", {"image_mode": "L"}) in fake_loaders
    assert ("video", {"num_frames": 8}) in fake_loaders


def test_already_decoded_items_pass_through(fake_loaders):
    _result, calls = _generate(_new_worker(), _media_prompt())
    assert calls[0][0]["multi_modal_data"]["image"] is IMAGE
    assert fake_loaders == []


def _assert_decode_error(error, modality, index, cls):
    assert type(error) is MediaDecodeError and isinstance(error, ValueError)
    assert str(error) == f"could not decode {modality} {index} ({cls.__name__})"
    assert error.args == (str(error),)
    assert "secret" not in repr(error)
    # Not chained: neither the cause nor the context holds the decoder's exception or its message.
    assert error.__cause__ is None and error.__context__ is None and error.__suppress_context__


def test_decoder_failure_classes_have_their_real_names():
    names = {modality: [cls.__name__ for cls in classes] for modality, classes in _decoder_failures().items()}
    assert names == {
        "image": ["UnidentifiedImageError", "DecompressionBombError", "OSError", "SyntaxError", "ValueError"],
        "video": ["error", "InvalidDataError", "OSError", "ValueError"],
    }


@pytest.mark.parametrize("modality, cls", _DECODE_FAILURES, ids=lambda v: getattr(v, "__name__", v))
def test_generate_raises_media_decode_error_for_each_decoder_failure(fake_loaders, modality, cls):
    worker = _new_worker()
    worker.llm = _FakeEngine()
    good = _MEDIA[modality]
    items = [good(b"good-0"), _failing_bytes(modality, cls), good(b"good-2")]
    prompt = _media_prompt(multi_modal_data={modality: items})
    with pytest.raises(MediaDecodeError) as caught:
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))
    _assert_decode_error(caught.value, modality, 1, cls)
    # The item after the bad one is never decoded, and vLLM gets nothing.
    assert fake_loaders.loaded == [(modality, good(b"good-0")), (modality, _failing_bytes(modality, cls))]
    assert worker.llm.calls == []


def test_a_bad_item_not_in_a_list_is_index_zero(fake_loaders):
    prompt = _media_prompt(multi_modal_data={"image": [_image(b"good-0")], "video": _failing_bytes("video", _Cv2Error)})
    with pytest.raises(MediaDecodeError) as caught:
        _generate(_new_worker(), prompt)
    _assert_decode_error(caught.value, "video", 0, _Cv2Error)


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_bad_item_index_counts_within_its_modality(fake_loaders, wrap):
    from PIL import Image

    bad = _failing_bytes("image", Image.DecompressionBombError)
    prompt = _media_prompt(
        multi_modal_data={"video": [wrap(_video(b"clip"))], "image": [IMAGE, wrap(_image(b"good-1")), wrap(bad)]}
    )
    with pytest.raises(MediaDecodeError) as caught:
        _generate(_new_worker(), prompt)
    _assert_decode_error(caught.value, "image", 2, Image.DecompressionBombError)
    assert fake_loaders.loaded == [("video", _video(b"clip")), ("image", _image(b"good-1")), ("image", bad)]


def test_media_decode_error_is_exported():
    from arctic_platform.inference.server import media_prompts

    assert "MediaDecodeError" in media_prompts.__all__
    assert issubclass(media_prompts.MediaDecodeError, ValueError)


def test_video_without_a_decoder_is_refused(monkeypatch):
    # OpenCV is there but vLLM's video loader cannot be built.
    _video_libraries(monkeypatch, "cv2")

    def broken(name):
        module = types.ModuleType(f"vllm.multimodal.media.{name}")

        def missing(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'cv2'")

        setattr(module, f"{name.capitalize()}MediaIO", type(
            f"{name}Io", (), {"merge_kwargs": classmethod(lambda cls, d, r: {}), "__init__": missing}
        ))
        monkeypatch.setitem(sys.modules, module.__name__, module)

    broken("image")
    broken("video")
    worker = _new_worker()
    worker.llm = _FakeEngine()
    with pytest.raises(ValueError, match="Cannot decode video") as caught:
        asyncio.run(worker.generate(_media_prompt(multi_modal_data={"video": [MP4]}), {"max_tokens": 1}))
    # A missing loader is not a bad item: it keeps its type and message.
    assert type(caught.value) is ValueError
    assert str(caught.value) == "Cannot decode video: this worker has no vLLM video decoder (No module named 'cv2')"
    assert worker.llm.calls == []


@pytest.mark.parametrize(
    "present, decodable", [(("cv2",), True), ((), False), (("torchcodec", "PyNvVideoCodec"), False)]
)
def test_default_video_backend_is_opencv(monkeypatch, present, decodable):
    from arctic_platform.inference.server import media_prompts

    looked_up = _video_libraries(monkeypatch, *present)
    expected = {"image", "video"} if decodable else {"image"}
    assert media_prompts.decodable_modalities({}) == expected
    assert media_prompts.decodable_modalities() == expected
    assert media_prompts.decodable_modalities({"video": {"num_frames": 8}}) == expected
    assert set(looked_up) == {"cv2"}


_CODECS = [("opencv", "cv2"), ("torchcodec", "torchcodec"), ("pynvvideocodec", "PyNvVideoCodec")]


@pytest.mark.parametrize("loader", [None, "opencv", "qwen3_vl", "qwen2_vl", "opencv_dynamic", "no_such_loader"])
@pytest.mark.parametrize("codec, library", _CODECS)
def test_named_codec_needs_its_library(monkeypatch, loader, codec, library):
    from arctic_platform.inference.server import media_prompts

    kwargs = {"video": {"backend": codec} | ({} if loader is None else {"video_backend": loader})}
    others = [name for name in _VIDEO_LIBRARIES if name != library]
    _video_libraries(monkeypatch, library)
    assert media_prompts.decodable_modalities(kwargs) == {"image", "video"}
    # The other codecs' libraries do not stand in for this one's.
    _video_libraries(monkeypatch, *others)
    assert media_prompts.decodable_modalities(kwargs) == {"image"}


@pytest.mark.parametrize("loader", ["opencv", "qwen3_vl", "glm46v", "molmo2", "no_such_loader"])
def test_a_loader_without_a_codec_decodes_with_opencv(monkeypatch, loader):
    from arctic_platform.inference.server import media_prompts

    looked_up = _video_libraries(monkeypatch, "cv2")
    assert media_prompts.decodable_modalities({"video": {"video_backend": loader}}) == {"image", "video"}
    _video_libraries(monkeypatch, "torchcodec", "PyNvVideoCodec")
    assert media_prompts.decodable_modalities({"video": {"video_backend": loader}}) == {"image"}
    assert set(looked_up) == {"cv2"}


def test_qwen3_vl_loader_with_the_pynvvideocodec_codec_needs_pynvvideocodec(monkeypatch):
    from arctic_platform.inference.server import media_prompts

    kwargs = {"video": {"video_backend": "qwen3_vl", "backend": "pynvvideocodec"}}
    _video_libraries(monkeypatch, "PyNvVideoCodec")
    assert media_prompts.decodable_modalities(kwargs) == {"image", "video"}
    _video_libraries(monkeypatch, "cv2", "torchcodec")
    assert media_prompts.decodable_modalities(kwargs) == {"image"}


@pytest.mark.parametrize("codec", [None, "opencv", "torchcodec", "deepstream", "no_such_codec"])
def test_pynvvideocodec_loader_always_needs_pynvvideocodec(monkeypatch, codec):
    from arctic_platform.inference.server import media_prompts

    kwargs = {"video": {"video_backend": "pynvvideocodec"} | ({} if codec is None else {"backend": codec})}
    looked_up = _video_libraries(monkeypatch, "PyNvVideoCodec")
    assert media_prompts.decodable_modalities(kwargs) == {"image", "video"}
    _video_libraries(monkeypatch, "cv2", "torchcodec")
    assert media_prompts.decodable_modalities(kwargs) == {"image"}
    assert set(looked_up) == {"PyNvVideoCodec"}


@pytest.mark.parametrize("backend", [["opencv"], 1])
def test_a_codec_that_is_not_a_name_is_not_decodable(monkeypatch, backend):
    from arctic_platform.inference.server import media_prompts

    looked_up = _video_libraries(monkeypatch, *_VIDEO_LIBRARIES)
    assert media_prompts.decodable_modalities({"video": {"backend": backend}}) == {"image"}
    assert looked_up == []


@pytest.mark.parametrize("codec", ["deepstream", "ffmpeg", "OpenCV", ""])
@pytest.mark.parametrize("loader", [None, "opencv", "qwen3_vl"])
def test_deepstream_and_unknown_codecs_are_not_decodable(monkeypatch, codec, loader):
    from arctic_platform.inference.server import media_prompts

    kwargs = {"video": {"backend": codec} | ({} if loader is None else {"video_backend": loader})}
    looked_up = _video_libraries(monkeypatch, *_VIDEO_LIBRARIES)
    assert media_prompts.decodable_modalities(kwargs) == {"image"}
    assert looked_up == []


@pytest.mark.parametrize("loader", ["opencv", "qwen3_vl", "opencv_dynamic", "no_such_loader"])
@pytest.mark.parametrize("codec, library", [(None, "cv2"), *_CODECS])
def test_env_var_names_the_loader(monkeypatch, loader, codec, library):
    from arctic_platform.inference.server import media_prompts

    kwargs = {"video": {"num_frames": 8} | ({} if codec is None else {"backend": codec})}
    _video_libraries(monkeypatch, library, backend=loader)
    assert media_prompts.decodable_modalities(kwargs) == {"image", "video"}
    _video_libraries(monkeypatch, *(name for name in _VIDEO_LIBRARIES if name != library), backend=loader)
    assert media_prompts.decodable_modalities(kwargs) == {"image"}


@pytest.mark.parametrize("codec", [None, "opencv", "torchcodec"])
def test_env_var_pynvvideocodec_loader_needs_pynvvideocodec(monkeypatch, codec):
    from arctic_platform.inference.server import media_prompts

    kwargs = {} if codec is None else {"video": {"backend": codec}}
    _video_libraries(monkeypatch, "PyNvVideoCodec", backend="pynvvideocodec")
    assert media_prompts.decodable_modalities(kwargs) == {"image", "video"}
    _video_libraries(monkeypatch, "cv2", "torchcodec", backend="pynvvideocodec")
    assert media_prompts.decodable_modalities(kwargs) == {"image"}


def test_the_entry_loader_wins_over_the_env_var(monkeypatch):
    from arctic_platform.inference.server import media_prompts

    _video_libraries(monkeypatch, "cv2", backend="pynvvideocodec")
    assert media_prompts.decodable_modalities({"video": {"video_backend": "qwen3_vl"}}) == {"image", "video"}
    _video_libraries(monkeypatch, "PyNvVideoCodec", backend="qwen3_vl")
    assert media_prompts.decodable_modalities({"video": {"video_backend": "pynvvideocodec"}}) == {"image", "video"}


def test_decodable_modalities_does_not_import_the_library(monkeypatch):
    from arctic_platform.inference.server import media_prompts

    _video_libraries(monkeypatch, "cv2")
    monkeypatch.delitem(sys.modules, "cv2", raising=False)
    assert media_prompts.decodable_modalities({}) == {"image", "video"}
    assert "cv2" not in sys.modules


def test_decodable_modalities_is_exported():
    from arctic_platform.inference.server import media_prompts

    assert "decodable_modalities" in media_prompts.__all__
    assert "SUPPORTS_MEDIA" in media_prompts.__all__ and media_prompts.SUPPORTS_MEDIA is True
    assert "decodable_modalities" in media_prompts.__doc__


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize(
    "media_io_kwargs, present",
    [
        ({}, ("torchcodec", "PyNvVideoCodec")),  # no cv2 for the default opencv codec
        ({"video": {"video_backend": "qwen3_vl"}}, ("torchcodec", "PyNvVideoCodec")),
        ({"video": {"video_backend": "pynvvideocodec", "backend": "opencv"}}, ("cv2", "torchcodec")),
        ({"video": {"backend": "deepstream"}}, _VIDEO_LIBRARIES),
    ],
)
def test_video_is_refused_before_any_loader_when_not_decodable(
    fake_loaders, monkeypatch, wrap, single, media_io_kwargs, present
):
    _video_libraries(monkeypatch, *present)
    worker = _new_worker()
    worker.llm = _FakeEngine()
    worker.llm.model_config = _model_config(media_io_kwargs=media_io_kwargs)
    video = wrap(_video(b"secret-input"))
    prompt = _media_prompt(multi_modal_data={"image": [_image(b"good-0")], "video": video if single else [video]})
    with pytest.raises(MediaDecodeError) as caught:
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))
    error = caught.value
    assert type(error) is MediaDecodeError and error.args == ("video decoding is not available",)
    assert error.__cause__ is None and error.__context__ is None
    # No loader was built, not even the image one, and vLLM gets nothing.
    assert fake_loaders == [] and fake_loaders.loaded == []
    assert worker.llm.calls == []


def test_video_refusal_uses_the_engine_media_io_kwargs(fake_loaders, monkeypatch):
    from arctic_platform.inference.server import media_prompts

    _video_libraries(monkeypatch, "cv2")
    kwargs = {"video": {"video_backend": "pynvvideocodec"}}
    with pytest.raises(MediaDecodeError, match="^video decoding is not available$"):
        media_prompts.decode_media(_media_prompt(multi_modal_data={"video": [MP4]}), kwargs)
    assert fake_loaders == []


def test_images_and_decoded_video_are_unaffected_when_video_is_not_decodable(fake_loaders, monkeypatch):
    _video_libraries(monkeypatch)
    frames = object()
    prompt = _media_prompt(multi_modal_data={"image": [_image(b"one"), JPEG], "video": [frames]})
    _result, calls = _generate(_new_worker(), prompt)
    sent = calls[0][0]["multi_modal_data"]
    assert [i.original_bytes for i in sent["image"]] == [_image(b"one"), JPEG]
    assert sent["video"] == [frames]
    assert [kind for kind, _kwargs in fake_loaders] == ["image"]


def test_unknown_modality_bytes_are_refused(fake_loaders):
    with pytest.raises(ValueError, match="^Cannot decode raw audio bytes; send audio items already decoded$") as caught:
        _generate(_new_worker(), _media_prompt(multi_modal_data={"audio": [b"a"]}))
    assert type(caught.value) is ValueError


@pytest.mark.parametrize(
    "modality, data",
    [
        ("image", PNG),
        ("image", JPEG),
        ("image", b"\xff\xd8\xff\xdb" + bytes(8)),  # a JPEG with no JFIF segment
        ("video", MP4),
        ("video", _ftyp(b"mp42", b"mp42", b"isom")),
        ("video", _ftyp(b"avc1", b"avc1")),
        ("video", _ftyp(b"iso5", b"iso5", b"dash")),
    ],
    ids=["png", "jpeg", "jpeg-dqt", "mp4-isom", "mp4-mp42", "mp4-avc1", "mp4-iso5"],
)
def test_allowed_formats_reach_the_loader(fake_loaders, modality, data):
    _result, calls = _generate(_new_worker(), _media_prompt(multi_modal_data={modality: [data]}))
    assert fake_loaders.loaded == [(modality, data)]
    assert calls[0][0]["multi_modal_data"][modality][0].original_bytes == data


_TAG = b"secret-input"
_REFUSED = [
    ("image", b"GIF89a\x01\x00\x01\x00\x00\x00\x00" + _TAG),
    ("image", b"GIF87a\x01\x00\x01\x00\x00\x00\x00" + _TAG),
    ("image", b"II*\x00\x08\x00\x00\x00" + _TAG),  # little-endian TIFF
    ("image", b"MM\x00*\x00\x00\x00\x08" + _TAG),  # big-endian TIFF
    ("image", b"BM" + struct.pack("<IHHI", 58, 0, 0, 54) + _TAG),
    ("image", b"RIFF" + struct.pack("<I", 26) + b"WEBPVP8 " + _TAG),
    ("image", random.Random(0).randbytes(64) + _TAG),
    ("image", b""),
    ("image", PNG[:7]),  # a truncated PNG signature
    ("image", MP4 + _TAG),
    ("video", PNG + _TAG),
    ("video", JPEG + _TAG),
    ("video", b"RIFF" + struct.pack("<I", 26) + b"AVI LIST" + _TAG),
    ("video", _ftyp(b"qt  ", b"qt  ") + _TAG),  # QuickTime, not MP4
    ("video", bytes(4) + MP4 + _TAG),  # ftyp not the first box
    ("video", b"\x1aE\xdf\xa3" + _TAG),  # Matroska / WebM
    ("video", random.Random(1).randbytes(64) + _TAG),
    ("video", b""),
]


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
@pytest.mark.parametrize(
    "modality, data",
    _REFUSED,
    ids=[
        "gif89a", "gif87a", "tiff-le", "tiff-be", "bmp", "webp", "random-image", "empty-image", "short-png",
        "mp4-as-image", "png-as-video", "jpeg-as-video", "avi", "quicktime", "late-ftyp", "matroska",
        "random-video", "empty-video",
    ],
)
def test_unsupported_formats_are_refused_before_any_loader(fake_loaders, modality, data, wrap):
    worker = _new_worker()
    worker.llm = _FakeEngine()
    # The refused item comes second: nothing, not even the good item before it, is decoded.
    prompt = _media_prompt(multi_modal_data={modality: [_MEDIA[modality](b"good-0"), wrap(data)]})
    with pytest.raises(MediaDecodeError) as caught:
        asyncio.run(worker.generate(prompt, {"max_tokens": 1}))
    error = caught.value
    assert str(error) == f"could not decode {modality} 1 (unsupported format)"
    assert error.args == (str(error),)
    assert "secret" not in repr(error)
    assert error.__cause__ is None and error.__context__ is None
    assert fake_loaders == [] and fake_loaders.loaded == []
    assert worker.llm.calls == []


def test_unsupported_format_index_counts_within_its_modality(fake_loaders):
    prompt = _media_prompt(multi_modal_data={"image": [IMAGE, PNG], "video": [MP4, MP4, b"GIF89a"]})
    with pytest.raises(MediaDecodeError, match=r"^could not decode video 2 \(unsupported format\)$"):
        _generate(_new_worker(), prompt)
    assert fake_loaders.loaded == []


def test_unsupported_single_item_is_index_zero(fake_loaders):
    with pytest.raises(MediaDecodeError, match=r"^could not decode image 0 \(unsupported format\)$"):
        _generate(_new_worker(), _media_prompt(multi_modal_data={"image": b"BM"}))
    assert fake_loaders == []


def test_media_formats_allowlist():
    from arctic_platform.inference.server import media_prompts

    assert "MEDIA_FORMATS" in media_prompts.__all__
    assert media_prompts.MEDIA_FORMATS.keys() == {"image", "video"}
    assert media_prompts.MEDIA_FORMATS["image"] == ((0, PNG[:8]), (0, b"\xff\xd8\xff"))
    assert {magic[:4] for offset, magic in media_prompts.MEDIA_FORMATS["video"] if offset == 4} == {b"ftyp"}


@pytest.mark.parametrize(
    "prompt, message",
    [
        ({"prompt_token_ids": [], "multi_modal_data": {"image": [b"secret-input-0"]}}, "prompt_token_ids"),
        ({"prompt_token_ids": [1], "multi_modal_data": {"image": [b"secret-input-0"]}, "extra": 1}, "Unknown"),
    ],
)
def test_bad_prompt_shape_with_bad_bytes_keeps_its_error(fake_loaders, prompt, message):
    with pytest.raises(ValueError, match=message) as caught:
        _generate(_new_worker(), prompt)
    assert type(caught.value) is ValueError
    assert fake_loaders.loaded == []


@pytest.mark.parametrize("modality, cls", _DECODE_FAILURES, ids=lambda v: getattr(v, "__name__", v))
def test_stream_raises_media_decode_error_for_each_decoder_failure(fake_loaders, monkeypatch, modality, cls):
    from dataclasses import asdict

    raised = []

    def recording_decode(*args, **kwargs):
        try:
            return decode_media(*args, **kwargs)
        except Exception as exc:
            raised.append(exc)
            raise

    decode_media = streaming.decode_media
    monkeypatch.setattr(streaming, "decode_media", recording_decode)

    async def scenario():
        worker = _new_worker()
        worker.llm = _StreamEngine()
        worker._stream_sampling_params = lambda params: params
        good = _MEDIA[modality]
        prompt = _media_prompt(
            multi_modal_data={modality: [good(b"good-0"), _failing_bytes(modality, cls), good(b"good-2")]}
        )
        worker.start_stream("media", prompt, {"max_tokens": 1}, 20, asdict(streaming.StreamLimits()))
        events = []
        reader = worker.stream_events("media")
        async for batch in reader:
            events.extend(batch)
            if batch[-1]["type"] in ("completed", "terminal_error"):
                break
            worker.acknowledge_stream("media", batch[-1]["sequence"])
        await reader.aclose()
        return worker.llm.calls, events

    calls, events = asyncio.run(scenario())
    (exc,) = raised
    _assert_decode_error(exc, modality, 1, cls)
    good = _MEDIA[modality]
    assert fake_loaders.loaded == [(modality, good(b"good-0")), (modality, _failing_bytes(modality, cls))]
    assert calls == []
    assert events[-1]["type"] == "terminal_error"
    assert "secret" not in repr(events)


def test_stream_decodes_media_bytes(fake_loaders):
    from dataclasses import asdict

    async def scenario():
        worker = _new_worker()
        worker.llm = _StreamEngine()
        worker._stream_sampling_params = lambda params: params
        prompt = _media_prompt(multi_modal_data={"image": [JPEG]})
        worker.start_stream("media", prompt, {"max_tokens": 1}, 20, asdict(streaming.StreamLimits()))
        reader = worker.stream_events("media")
        async for batch in reader:
            if batch[-1]["type"] in ("completed", "terminal_error"):
                break
            worker.acknowledge_stream("media", batch[-1]["sequence"])
        await reader.aclose()
        (sent, _params, _kwargs), = worker.llm.calls
        return sent

    sent = asyncio.run(scenario())
    assert sent["multi_modal_data"]["image"][0].original_bytes == JPEG


def test_stream_refuses_unsupported_format_before_any_loader(fake_loaders, monkeypatch):
    from dataclasses import asdict

    raised = []

    def recording_decode(*args, **kwargs):
        try:
            return decode_media(*args, **kwargs)
        except Exception as exc:
            raised.append(exc)
            raise

    decode_media = streaming.decode_media
    monkeypatch.setattr(streaming, "decode_media", recording_decode)

    async def scenario():
        worker = _new_worker()
        worker.llm = _StreamEngine()
        worker._stream_sampling_params = lambda params: params
        prompt = _media_prompt(multi_modal_data={"image": [PNG, b"GIF89a" + _TAG]})
        worker.start_stream("media", prompt, {"max_tokens": 1}, 20, asdict(streaming.StreamLimits()))
        events = []
        reader = worker.stream_events("media")
        async for batch in reader:
            events.extend(batch)
            if batch[-1]["type"] in ("completed", "terminal_error"):
                break
            worker.acknowledge_stream("media", batch[-1]["sequence"])
        await reader.aclose()
        return worker.llm.calls, events

    calls, events = asyncio.run(scenario())
    (exc,) = raised
    assert type(exc) is MediaDecodeError and str(exc) == "could not decode image 1 (unsupported format)"
    assert calls == []
    assert fake_loaders == [] and fake_loaders.loaded == []
    assert events[-1]["type"] == "terminal_error"
    assert "secret" not in repr(events)
