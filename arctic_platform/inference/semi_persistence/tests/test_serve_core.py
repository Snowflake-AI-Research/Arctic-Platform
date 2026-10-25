"""The child's ``serve_core`` command, without a GPU, vLLM or ZMQ.

After a restore, ``serve_core`` turns the child's in-process ``EngineCore`` into
a ZMQ-served ``EngineCoreProc`` so a real ``AsyncLLM`` in the server process can
drive it. Two module-level helpers carry the logic, ``_semip_serve_core`` and
``_semip_serving_tick``; they are extracted from ``vllm_child.py`` through the
AST, as ``test_communicator_checkpoint_hooks.py`` does, and run against fakes
of the vLLM classes they import. The closure edits -- the command branch, the
main loop's serving branch and ``_drain_engine`` -- are checked by AST.

The real socket handshake needs vLLM and a GPU pod; nothing here covers it.

Run from the package directory::

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_serve_core.py -v
"""
from __future__ import annotations

import ast
import enum
import os
import queue
import sys
import threading
import time
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_CHILD = os.path.join(_PKG, "vllm_child.py")

_HELPERS = ("_semip_core_summary", "_semip_serve_core", "_semip_serving_tick")
_CONSTS = ("_SERVING_REFUSED_CMDS", "_SERVING_IDLE_WAIT_S",
           "_SERVING_READY_TIMEOUT_S")


def _tree():
    with open(_CHILD) as f:
        return ast.parse(f.read())


def _load_helpers():
    tree = _tree()
    nodes = [n for n in tree.body
             if isinstance(n, ast.FunctionDef) and n.name in _HELPERS]
    missing = set(_HELPERS) - {n.name for n in nodes}
    assert not missing, f"missing helper(s) in vllm_child.py: {sorted(missing)}"
    consts = [n for n in tree.body
              if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id in _CONSTS
                      for t in n.targets)]
    found = {t.id for n in consts for t in n.targets if isinstance(t, ast.Name)}
    assert found == set(_CONSTS), (
        f"missing constant(s) in vllm_child.py: {sorted(set(_CONSTS) - found)}")
    ns = {}
    exec(compile(ast.Module(body=consts + nodes, type_ignores=[]), _CHILD,
                 "exec"), ns)
    return ns


@pytest.fixture
def child():
    return _load_helpers()


# --------------------------------------------------------------------------
# Fakes of the vLLM classes the helpers import.
# --------------------------------------------------------------------------

class _RequestType(enum.Enum):
    ADD = b"\x00"
    EXECUTOR_FAILED = b"\x06"


class _ShutdownState(enum.IntEnum):
    RUNNING = 0
    REQUESTED = 1


class _FakeEngineCoreProc:
    """Stands in for ``EngineCoreProc``: socket threads that just record."""

    fail_input_thread = False

    def process_input_sockets(self, addresses, coord, identity, ready):
        self.socket_calls.append(("input", list(addresses), coord, identity))
        if not self.fail_input_thread:
            ready.set()

    def process_output_sockets(self, paths, coord, engine_index):
        self.socket_calls.append(("output", list(paths), coord, engine_index))


@pytest.fixture
def fake_vllm(monkeypatch):
    engine = types.ModuleType("vllm.v1.engine")
    engine.EngineCoreRequestType = _RequestType
    core = types.ModuleType("vllm.v1.engine.core")
    core.EngineCoreProc = _FakeEngineCoreProc
    core.EngineShutdownState = _ShutdownState
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.v1", types.ModuleType("vllm.v1")),
                      ("vllm.v1.engine", engine),
                      ("vllm.v1.engine.core", core)):
        monkeypatch.setitem(sys.modules, name, mod)
    _FakeEngineCoreProc.fail_input_thread = False
    return core


def _vllm_config(**over):
    ns = types.SimpleNamespace
    cfg = ns(
        model_config=ns(model="/models/qwen", tokenizer="/models/qwen",
                        dtype="torch.bfloat16", max_model_len=32768),
        parallel_config=ns(tensor_parallel_size=1,
                           worker_extension_cls="arl.WeightSync"),
        scheduler_config=ns(max_num_seqs=256, max_num_batched_tokens=8192),
        cache_config=ns(block_size=16, num_gpu_blocks=4096,
                        enable_prefix_caching=True),
        structured_outputs_config=ns(reasoning_parser="qwen3", backend="auto"),
        speculative_config=None,
        lora_config=None,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


class _FakeExecutor:
    def __init__(self):
        self.failure_callback = None

    def register_failure_callback(self, cb):
        self.failure_callback = cb


class _FakeEngineCore:
    """The child's in-process core before ``serve_core``."""

    def __init__(self):
        self.vllm_config = _vllm_config()
        self.model_executor = _FakeExecutor()
        self.socket_calls = []


# --------------------------------------------------------------------------
# _semip_core_summary
# --------------------------------------------------------------------------

def test_summary_is_plain_values(child):
    s = child["_semip_core_summary"](_vllm_config())
    assert s == {
        "model": "/models/qwen",
        "tokenizer": "/models/qwen",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "worker_extension_cls": "arl.WeightSync",
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "reasoning_parser": "qwen3",
        "structured_outputs_backend": "auto",
        "speculative_method": "",
        "num_speculative_tokens": 0,
        "lora": False,
        "max_model_len": 32768,
        "block_size": 16,
        "num_gpu_blocks": 4096,
        "enable_prefix_caching": True,
    }


def test_summary_reads_speculative_and_lora(child):
    ns = types.SimpleNamespace
    s = child["_semip_core_summary"](_vllm_config(
        speculative_config=ns(method="mtp", num_speculative_tokens=3),
        lora_config=ns(max_loras=1)))
    assert (s["speculative_method"], s["num_speculative_tokens"]) == ("mtp", 3)
    assert s["lora"] is True


# --------------------------------------------------------------------------
# _semip_serve_core
# --------------------------------------------------------------------------

def test_serve_core_swaps_class_and_sets_proc_state(child, fake_vllm):
    core = _FakeEngineCore()
    summary = child["_semip_serve_core"](core, "ipc:///tmp/in", "ipc:///tmp/out")

    assert type(core) is _FakeEngineCoreProc
    assert isinstance(core.input_queue, queue.Queue)
    assert isinstance(core.output_queue, queue.Queue)
    assert core.engine_index == 0
    assert core.engines_running is False
    assert core.shutdown_state is _ShutdownState.RUNNING
    assert core.tensor_ipc_receiver is None
    assert core.has_coordinator is False
    assert core.frontend_stats_publish_address is None
    assert core.publish_dp_lb_stats is False
    assert core.last_counts == (0, 0)
    assert core.process_input_queue_block is False
    assert core.enable_fault_tolerance is False
    assert summary == child["_semip_core_summary"](core.vllm_config)

    core.output_thread.join(timeout=5)
    assert sorted(core.socket_calls) == [
        ("input", ["ipc:///tmp/in"], None, b"\x00\x00"),
        ("output", ["ipc:///tmp/out"], None, 0),
    ]


def test_executor_failure_reaches_the_input_queue(child, fake_vllm):
    core = _FakeEngineCore()
    child["_semip_serve_core"](core, "ipc:///tmp/in", "ipc:///tmp/out")
    core.model_executor.failure_callback()
    assert core.input_queue.get_nowait() == (_RequestType.EXECUTOR_FAILED, b"")


def test_serve_core_fails_when_input_thread_dies(child, fake_vllm):
    child["_SERVING_READY_TIMEOUT_S"] = 0.2
    _FakeEngineCoreProc.fail_input_thread = True
    with pytest.raises(RuntimeError, match="input thread died"):
        child["_semip_serve_core"](_FakeEngineCore(), "ipc://a", "ipc://b")


# --------------------------------------------------------------------------
# _semip_serving_tick
# --------------------------------------------------------------------------

class _TickCore:
    def __init__(self, work=0):
        self.work = work
        self.input_queue = queue.Queue()
        self.calls = []

    def has_work(self):
        return self.work > 0

    def _process_input_queue(self):
        self.calls.append("input")

    def _process_engine_step(self):
        self.calls.append("step")
        self.work -= 1

    def _handle_client_request(self, kind, payload):
        self.calls.append(("request", kind, payload))


class _Pipe:
    def __init__(self, pending=False):
        self.pending = pending
        self.polls = []

    def poll(self, timeout=0):
        self.polls.append(timeout)
        if not self.pending and timeout:
            time.sleep(timeout)
        return self.pending


def test_tick_never_steps_while_asleep(child):
    core, pipe = _TickCore(work=3), _Pipe()
    assert child["_semip_serving_tick"](core, pipe, True, 0.01) is False
    assert core.calls == ["input"]
    assert pipe.polls == [0.01]


def test_tick_steps_once_and_polls_the_pipe_when_busy(child):
    core, pipe = _TickCore(work=3), _Pipe(pending=True)
    assert child["_semip_serving_tick"](core, pipe, False, 0.01) is True
    assert core.calls == ["input", "step"]
    assert pipe.polls == [0]


def test_idle_tick_answers_a_waiting_pipe_command_at_once(child):
    core, pipe = _TickCore(), _Pipe(pending=True)
    assert child["_semip_serving_tick"](core, pipe, False, 5.0) is True
    assert core.calls == ["input"]


def test_idle_tick_wait_is_bounded(child):
    core, pipe = _TickCore(), _Pipe()
    t0 = time.monotonic()
    assert child["_semip_serving_tick"](core, pipe, False, 0.02) is False
    assert time.monotonic() - t0 < 1.0
    assert core.calls == ["input"]


def test_idle_tick_wakes_for_a_request(child):
    core, pipe = _TickCore(), _Pipe()
    threading.Timer(0.01, core.input_queue.put, args=(("add", b"r1"),)).start()
    assert child["_semip_serving_tick"](core, pipe, False, 5.0) is False
    assert core.calls == ["input", ("request", "add", b"r1")]


def test_default_idle_wait_is_a_few_milliseconds(child):
    assert 0 < child["_SERVING_IDLE_WAIT_S"] <= 0.01


# --------------------------------------------------------------------------
# The closure edits, by AST.
# --------------------------------------------------------------------------

def _child_loop():
    return next(n for n in _tree().body
                if isinstance(n, ast.FunctionDef) and n.name == "vllm_child_loop")


def _nested(fn, name):
    return next(n for n in ast.walk(fn)
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _names(node):
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def _attrs(node):
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


def test_refused_commands(child):
    assert child["_SERVING_REFUSED_CMDS"] == {
        "generate", "pause", "resume", "prepare_criu_dump", "serve_core"}


def _main_loop():
    """The ``try: while True:`` at the end of ``vllm_child_loop``."""
    return next(n for n in reversed(_child_loop().body)
                if isinstance(n, ast.Try)
                and any(isinstance(m, ast.While) for m in n.body))


def test_main_loop_ticks_the_core_and_refuses_before_dispatch():
    body = next(n for n in _main_loop().body if isinstance(n, ast.While))
    assert "_semip_serving_tick" in _names(body)
    tick = next(n for n in ast.walk(body)
                if isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "_semip_serving_tick")
    assert [a.id for a in tick.args] == ["_served_core", "pipe_conn", "_dormant"]

    src = ast.get_source_segment(open(_CHILD).read(), body)
    refuse = src.index("_SERVING_REFUSED_CMDS")
    assert refuse < src.index('if cmd == "generate":')
    assert refuse < src.index("_handle_command(cmd, kwargs)")


def test_fatal_reporter_tells_the_front_end():
    handler = _main_loop().handlers[0]
    assert "_send_engine_dead" in _attrs(handler)


def test_drain_steps_the_served_core():
    drain = _nested(_child_loop(), "_drain_engine")
    assert "_served_core" in _names(drain)
    assert "_process_engine_step" in _attrs(drain)


def test_serve_core_branch():
    handle = _nested(_child_loop(), "_handle_command")
    nonlocals = {name for n in ast.walk(handle) if isinstance(n, ast.Nonlocal)
                 for name in n.names}
    assert "_served_core" in nonlocals
    branch = next(n for n in ast.walk(handle)
                  if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and isinstance(n.test.comparators[0], ast.Constant)
                  and n.test.comparators[0].value == "serve_core")
    assert "_semip_serve_core" in _names(branch)


def _command_branch(name):
    handle = _nested(_child_loop(), "_handle_command")
    branch = next(n for n in ast.walk(handle)
                  if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                  and isinstance(n.test.comparators[0], ast.Constant)
                  and n.test.comparators[0].value == name)
    return ast.get_source_segment(open(_CHILD).read(), branch)


def test_sleep_keeps_requests_and_resumes_the_scheduler_when_it_fails():
    """An "abort" pause resets caches the routed-experts connector refuses to drop."""
    src = _command_branch("sleep")
    assert 'llm.sleep(level=2, mode="keep")' in src
    on_failure = src[src.index("except BaseException:"):src.index("torch.cuda.synchronize")]
    assert 'llm.wake_up(tags=["scheduling"])' in on_failure
    assert on_failure.rstrip().endswith("raise")


def test_a_serving_sleep_stages_lora_and_restore_weights_puts_it_back():
    sleep = _command_branch("sleep")
    stage = sleep.index("_semip_stage_lora")
    assert "_served_core is not None" in sleep[:stage]
    assert stage < sleep.index("llm.sleep(")
    assert "_semip_restore_lora" in _command_branch("restore_weights")


def test_child_applies_every_required_patch_before_building_the_engine():
    """The child builds its own ``LLM``, so it has to apply what the cold path
    applies. Naming patches one by one let it fall behind ``main``'s list."""
    with open(_CHILD) as handle:
        src = handle.read()
    apply = src.find("apply_required_vllm_patches()")
    build = src.find("llm = LLM(**vllm_config)")
    assert apply != -1, "the child must call apply_required_vllm_patches()"
    assert build != -1
    assert apply < build, "the patches must be in place before LLM() runs"
