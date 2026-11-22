"""The ``SEMIP_DUMP_WARMUP`` switch and the dump's GPU-memory readings.

The pre-dump generation exists so a broken engine fails before an image is
spent on it. Whether what it leaves on the GPU makes the checkpoint -- and so
the image and its cuda_restore -- larger is an A/B question, so the switch
skips it at both dump sites while the default keeps today's call sequence
exactly. Every rank's GPU memory is logged at init, after the generation (or
where it would have run), after sleep and immediately before the checkpoint.

``instance`` and ``vllm_child`` import NVML and torch at module scope, so their
pieces are lifted out of the source; ``semip_engine`` is loaded with the
bootstrap in ``test_image_cache_key``.

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_dump_warmup.py -v
"""
from __future__ import annotations

import ast
import logging
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import se  # noqa: E402

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INSTANCE = os.path.join(_PKG, "instance.py")
_CHILD = os.path.join(_PKG, "vllm_child.py")

# The dump's command sequence before the switch existed, gpu_mem aside.
_DEFAULT_SEQUENCE = ["init", "generate", "attach", "stage", "save_weights",
                     "release_host_weights", "detach", "sleep", "wait",
                     "cuda_checkpoint", "criu_dump", "wait", "teardown"]
_PHASES = ["init", "post-warmup", "post-sleep", "pre-checkpoint"]


def _lift(path, names, cls=None, namespace=None):
    with open(path) as handle:
        tree = ast.parse(handle.read(), filename=path)
    scope = tree
    if cls is not None:
        scope = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.ClassDef) and n.name == cls)
    body = [n for n in scope.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {f.name for f in body} == set(names)
    ns = dict(namespace or {})
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), ns)
    return ns


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, "", "1", "on", "true", "yes", "ON"])
def test_the_warmup_is_on_unless_turned_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(se._DUMP_WARMUP_ENV, raising=False)
    else:
        monkeypatch.setenv(se._DUMP_WARMUP_ENV, value)
    assert se._dump_warmup_enabled() is True


@pytest.mark.parametrize("value", ["0", "false", "off", "FALSE", " Off "])
def test_zero_false_and_off_turn_it_off(monkeypatch, value):
    monkeypatch.setenv(se._DUMP_WARMUP_ENV, value)
    assert se._dump_warmup_enabled() is False


# --------------------------------------------------------------------------
# The single-pod dump
# --------------------------------------------------------------------------

class _FakeInstance:
    """Records every call a dump makes, with its arguments."""

    def __init__(self, calls):
        self.calls = calls
        self.meta_extra = None
        self.last_info = {}

    def criu_dump(self, filename=None, meta_extra=None):
        self.meta_extra = meta_extra
        self.calls.append(("criu_dump", (), {}))
        return self

    def __getattr__(self, name):
        def _call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            return self
        return _call


def _single_dump(monkeypatch, warmup):
    if warmup is None:
        monkeypatch.delenv(se._DUMP_WARMUP_ENV, raising=False)
    else:
        monkeypatch.setenv(se._DUMP_WARMUP_ENV, warmup)
    calls, made = [], []

    def fake_instance(vllm_config, model_dir):
        made.append(_FakeInstance(calls))
        return made[-1]

    monkeypatch.setattr(se, "Instance", fake_instance)
    monkeypatch.setattr(se, "_raise_pid_floor", lambda: 100_001)
    monkeypatch.setattr(se, "_record_env_files", lambda model_dir: None)
    monkeypatch.setattr(se, "_log_image_stats", lambda model_dir: None)
    monkeypatch.setattr(se.time, "sleep", lambda seconds: None)
    se._dump({}, "/tmp/model_dir", [0], image_ref="img@sha256:abc",
             driver_version="580.159.03")
    return calls, made[0].meta_extra


def _phases(calls):
    out = [args[0] for name, args, _ in calls if name == "gpu_mem"]
    out += [kw["gpu_mem_phase"] for name, _, kw in calls
            if name == "cuda_checkpoint" and kw.get("gpu_mem_phase")]
    return out


@pytest.mark.parametrize("warmup", [None, "1"])
def test_the_default_keeps_the_dump_sequence_and_its_meta(monkeypatch, caplog,
                                                          warmup):
    caplog.set_level(logging.INFO, logger=se.logger.name)
    calls, meta = _single_dump(monkeypatch, warmup)
    assert [c[0] for c in calls if c[0] != "gpu_mem"] == _DEFAULT_SEQUENCE
    (_, args, kwargs), = [c for c in calls if c[0] == "generate"]
    assert args == ([se._DUMP_PROMPT], se._DUMP_SAMPLING) and kwargs == {}
    assert "dump_warmup" not in meta
    assert "pre-dump warm-up generate kept" in caplog.text


@pytest.mark.parametrize("warmup", ["0", "false", "off"])
def test_off_skips_the_generate_and_marks_the_image(monkeypatch, caplog,
                                                    warmup):
    caplog.set_level(logging.INFO, logger=se.logger.name)
    calls, meta = _single_dump(monkeypatch, warmup)
    expected = [c for c in _DEFAULT_SEQUENCE if c != "generate"]
    assert [c[0] for c in calls if c[0] != "gpu_mem"] == expected
    assert meta["dump_warmup"] is False
    assert (f"semi_p: pre-dump warm-up generate skipped "
            f"(SEMIP_DUMP_WARMUP={warmup})") in caplog.text


@pytest.mark.parametrize("warmup", [None, "0"])
def test_memory_is_read_at_every_phase_in_order(monkeypatch, warmup):
    calls, _ = _single_dump(monkeypatch, warmup)
    assert _phases(calls) == _PHASES
    names = [c[0] for c in calls]
    first, second, third = [i for i, n in enumerate(names) if n == "gpu_mem"]
    assert names[first - 1] == "init"
    assert names[second + 1] == "attach"
    assert names[second - 1] == ("generate" if warmup is None else "gpu_mem")
    assert names[third - 2:third + 2] == ["sleep", "wait", "gpu_mem",
                                          "cuda_checkpoint"]


# --------------------------------------------------------------------------
# The multi-node dump, whose leader runs the same switch
# --------------------------------------------------------------------------

class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _Agent:
    def __init__(self, calls):
        self.config_digest = _Remote(se._config_hash)
        self.init = _Remote(lambda *a: {"pid_floor": 7, "unprivileged":
                                        se._unprivileged_mode()})
        self.cuda_checkpoint = _Remote(lambda: calls.append(("agent", (), {})))
        self.criu_dump = _Remote(lambda meta: calls.append(
            ("agent_criu_dump", (meta,), {})))
        self.teardown = _Remote(lambda: None)


def _multinode_dump(monkeypatch, tmp_path, warmup):
    monkeypatch.setenv(se._DUMP_WARMUP_ENV, warmup)
    calls, made = [], []
    ray = types.ModuleType("ray")
    ray.get = lambda x, timeout=None: x
    monkeypatch.setitem(sys.modules, "ray", ray)
    semip = sys.modules["arctic_platform.inference.semi_persistence"]

    def fake_instance(vllm_config, model_dir, multinode=None):
        made.append(_FakeInstance(calls))
        return made[-1]

    monkeypatch.setattr(semip, "Instance", fake_instance, raising=False)
    monkeypatch.setattr(semip, "MultiNode", lambda **kw: kw, raising=False)
    monkeypatch.setattr(se, "_leader_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(se, "_raise_pid_floor", lambda: 100_001)
    monkeypatch.setattr(se, "_record_env_files", lambda model_dir: None)
    monkeypatch.setattr(se, "_log_image_stats", lambda model_dir: None)
    monkeypatch.setattr(se.time, "sleep", lambda seconds: None)
    key = tmp_path / "key"
    paths = types.SimpleNamespace(nnodes=2, key_dir=str(key),
                                  model_dir=str(key / "node0"))
    se._dump_multinode({"model": "m"}, paths, list(range(8)), [_Agent(calls)],
                       master_port=29500, ifname="eth0", image_ref="img",
                       driver_version="580")
    return calls, made[0].meta_extra


def test_the_multinode_leader_keeps_its_generate_by_default(monkeypatch,
                                                            tmp_path):
    calls, meta = _multinode_dump(monkeypatch, tmp_path, "1")
    assert sum(1 for c in calls if c[0] == "generate") == 1
    assert _phases(calls) == _PHASES
    agent_meta = next(c[1][0] for c in calls if c[0] == "agent_criu_dump")
    assert "dump_warmup" not in meta and "dump_warmup" not in agent_meta


def test_the_multinode_leader_skips_it_and_every_node_records_that(
        monkeypatch, tmp_path):
    calls, meta = _multinode_dump(monkeypatch, tmp_path, "0")
    assert not [c for c in calls if c[0] == "generate"]
    assert _phases(calls) == _PHASES
    agent_meta = next(c[1][0] for c in calls if c[0] == "agent_criu_dump")
    assert meta["dump_warmup"] is False and agent_meta["dump_warmup"] is False


def test_no_dump_site_calls_generate_past_the_switch():
    """Both sites go through ``_dump_warmup``; a third caller would bypass it."""
    with open(se.__file__) as handle:
        tree = ast.parse(handle.read())
    callers = sorted(
        fn.name for fn in ast.walk(tree) if isinstance(fn, ast.FunctionDef)
        for sub in ast.walk(fn)
        if isinstance(sub, ast.Call)
        and getattr(sub.func, "attr", None) == "generate"
        and any(getattr(a, "id", None) == "_DUMP_PROMPT"
                for e in sub.args for a in ast.walk(e)))
    assert callers == ["_dump_warmup"]


# --------------------------------------------------------------------------
# Instance: the pre-checkpoint reading follows the NCCL teardown
# --------------------------------------------------------------------------

_INST = _lift(_INSTANCE, ["cuda_checkpoint", "gpu_mem"], cls="Instance")


def _instance(n_gpus, nnodes=1, node_rank=0):
    sent = []
    inst = types.SimpleNamespace(n_gpus=n_gpus, nnodes=nnodes,
                                 node_rank=node_rank, _log=lambda *a: None)
    inst._send = lambda cmd, **kw: sent.append((cmd, kw)) or inst
    inst.gpu_mem = lambda phase: _INST["gpu_mem"](inst, phase)
    return inst, sent


@pytest.mark.parametrize("n_gpus,nnodes,teardown", [
    (8, 1, ["drop_graphs_if_nccl", "destroy_nccl"]),
    (8, 2, ["drop_graphs", "destroy_nccl"]),
    (1, 1, []),
], ids=["tp8", "multinode_leader", "tp1"])
def test_the_reading_sits_between_the_teardown_and_the_checkpoint(
        n_gpus, nnodes, teardown):
    inst, sent = _instance(n_gpus, nnodes)
    _INST["cuda_checkpoint"](inst, gpu_mem_phase="pre-checkpoint")
    assert [c for c, _ in sent] == teardown + ["gpu_mem", "cuda_checkpoint"]
    assert sent[-2] == ("gpu_mem", {"phase": "pre-checkpoint"})


def test_without_a_phase_the_checkpoint_sends_what_it_always_did():
    inst, sent = _instance(8)
    _INST["cuda_checkpoint"](inst)
    assert [c for c, _ in sent] == ["drop_graphs_if_nccl", "destroy_nccl",
                                    "cuda_checkpoint"]


# --------------------------------------------------------------------------
# The child's per-rank reading
# --------------------------------------------------------------------------

def test_the_rank_reading_only_queries():
    asked = []

    def query(name, value):
        return lambda device=None: asked.append((name, device)) or value

    cuda = types.SimpleNamespace(
        mem_get_info=query("mem_get_info", (30 << 30, 80 << 30)),
        memory_reserved=query("memory_reserved", 6 << 30),
        memory_allocated=query("memory_allocated", 5 << 30))
    ns = _lift(_CHILD, ["_semip_gpu_mem"],
               namespace={"torch": types.SimpleNamespace(cuda=cuda)})
    worker = types.SimpleNamespace(device="cuda:3", rank=3)
    assert ns["_semip_gpu_mem"](worker) == {
        "rank": 3, "used": 50 << 30, "total": 80 << 30,
        "reserved": 6 << 30, "allocated": 5 << 30}
    assert {d for _, d in asked} == {"cuda:3"}


def test_the_child_logs_one_grepable_line_per_rank():
    with open(_CHILD) as handle:
        src = handle.read()
    assert '"semi_p: gpu-mem %s rank=%s used=%.3f "' in src
    assert '"reserved=%.3f allocated=%.3f total=%.3f GiB"' in src
    assert 'elif cmd == "gpu_mem":' in src


# --------------------------------------------------------------------------
# The image a dump wrote
# --------------------------------------------------------------------------

F = se._ZERO_SCAN_FRAME_BYTES


def _image(tmp_path):
    image = tmp_path / se._IMAGE_DIR
    image.mkdir()
    frames = [bytes(F), b"\x01" + bytes(F - 1), bytes(F), bytes(F // 2)]
    (image / "pages-1.img").write_bytes(b"".join(frames))
    (image / "pages-2.img").write_bytes(b"")
    (image / "pages-3.img").write_bytes(bytes(F - 1) + b"\x02")
    (image / "core-1.img").write_bytes(b"x" * 100)
    (image / "meta.json").write_text("{}")
    return image


def test_zero_frames_are_counted_as_the_publisher_frames_them(tmp_path):
    image = _image(tmp_path)
    assert se._count_zero_frames(str(image / "pages-1.img")) == 3
    assert se._count_zero_frames(str(image / "pages-3.img")) == 0
    assert se._count_zero_frames(str(image / "pages-2.img")) == 0


def test_frames_past_the_first_read_are_counted(tmp_path):
    """The scan reads 8 frames at a time; frames 8 and 9 come from the second."""
    path = tmp_path / "pages-9.img"
    path.write_bytes(bytes(9 * F) + b"\x03" + bytes(F - 1) + bytes(7))
    assert se._count_zero_frames(str(path)) == 10


@pytest.mark.parametrize("scan", ["", "1"])
def test_the_image_is_sized_and_scanned_only_on_request(tmp_path, monkeypatch,
                                                        caplog, scan):
    _image(tmp_path)
    monkeypatch.setenv(se._DUMP_ZERO_SCAN_ENV, scan)
    caplog.set_level(logging.INFO, logger=se.logger.name)
    se._log_image_stats(str(tmp_path))
    pages = 3.5 * F + F
    assert (f"pages={pages / 2**30:.3f} GiB in 2 pages file(s), "
            f"zero_frames={'3 (' if scan else 'not scanned'}") in caplog.text
    assert "in 5 file(s)" in caplog.text


def test_a_missing_image_is_reported_empty_not_raised(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=se.logger.name)
    se._log_image_stats(str(tmp_path / "missing"))
    assert "raw=0.000 GiB in 0 file(s)" in caplog.text
