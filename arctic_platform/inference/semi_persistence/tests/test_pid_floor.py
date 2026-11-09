"""Unit tests for the dump-side PID floor in ``server/semip_engine.py``.

The floor is what makes an image restorable in a pod other than the one that
dumped it: a fresh device-manager container hands out the same low pids every
time, so an image that recorded them collides with the restoring pod's own Ray
worker -- a thread of the very actor driving the restore, which therefore
cannot be asked to exit. See ``semip_engine._PID_FLOOR_ENV``.

Two halves are tested differently. The decision logic is pure and runs against
temporary files standing in for the two sysctls, which is why
``_NS_LAST_PID_PATH``, ``_PID_MAX_PATH`` and ``_write_sysctl`` are module
attributes rather than literals. The burn loop is source text executed in a
fresh interpreter, where the only failure mode worth testing is the one a
reviewer cannot see -- that the text runs at all, and that its iteration bound
holds -- so those two run it for real against a target a few pids away.

Nothing here forks 100k times, needs ``/proc``, or needs Linux.

Run from the package directory::

    python -m pytest tests/test_pid_floor.py -v

Or directly::

    python tests/test_pid_floor.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types

# The bootstrap that loads semip_engine.py without importing the
# arctic_inference packages lives there; see that module's docstring.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_image_cache_key import se  # noqa: E402


class _Sysctls:
    """Temporary files standing in for ns_last_pid and pid_max."""

    def __init__(self, counter=1234, pid_max=4194304):
        self.dir = tempfile.mkdtemp(prefix="pidfloor-")
        self.counter_path = os.path.join(self.dir, "ns_last_pid")
        self.pid_max_path = os.path.join(self.dir, "pid_max")
        self.write_counter(counter)
        with open(self.pid_max_path, "w") as handle:
            handle.write(str(pid_max))
        self._saved = (se._NS_LAST_PID_PATH, se._PID_MAX_PATH,
                       se._write_sysctl)
        se._NS_LAST_PID_PATH = self.counter_path
        se._PID_MAX_PATH = self.pid_max_path
        # Read-only by default, which is what the containers this runs in see.
        se._write_sysctl = lambda path, value: False

    def write_counter(self, value):
        with open(self.counter_path, "w") as handle:
            handle.write(str(value))

    def read_counter(self):
        with open(self.counter_path) as handle:
            return int(handle.read().strip())

    def close(self):
        (se._NS_LAST_PID_PATH, se._PID_MAX_PATH,
         se._write_sysctl) = self._saved
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)


class _Burner:
    """Stands in for ``subprocess.run`` on the burner, recording its argv.

    Replaces the module reference ``se.subprocess`` rather than patching
    ``subprocess.run`` itself: that attribute belongs to the stdlib module
    object every importer shares, so patching it would also stand in front of
    the real burns this file runs further down.
    """

    def __init__(self, sysctls, reaches=None, returncode=0, raises=None):
        self.sysctls = sysctls
        self.reaches = reaches          # None -> leave the counter alone
        self.returncode = returncode
        self.raises = raises
        self.calls = []
        self._saved = se.subprocess
        se.subprocess = types.SimpleNamespace(
            run=self, TimeoutExpired=subprocess.TimeoutExpired)

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises is not None:
            raise self.raises
        if self.reaches is not None:
            self.sysctls.write_counter(self.reaches)
        return types.SimpleNamespace(returncode=self.returncode,
                                     stdout=b"", stderr=b"burner said no")

    def close(self):
        se.subprocess = self._saved


def _set_env(**values):
    """Set or clear environment variables, returning their previous state."""
    previous = {name: os.environ.get(name) for name in values}
    for name, value in values.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    return previous


def _restore_env(previous):
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


# --------------------------------------------------------------------------
# _env_int
# --------------------------------------------------------------------------


def test_env_int_takes_the_override():
    previous = _set_env(SEMIP_PID_FLOOR="12345")
    try:
        assert se._env_int(se._PID_FLOOR_ENV, 100_000) == 12345
    finally:
        _restore_env(previous)


def test_env_int_ignores_a_malformed_override():
    """A typo in a tuning knob must not fail a job -- it warns and defaults."""
    previous = _set_env(SEMIP_PID_FLOOR="two hundred thousand")
    try:
        assert se._env_int(se._PID_FLOOR_ENV, 100_000) == 100_000
    finally:
        _restore_env(previous)


# --------------------------------------------------------------------------
# _write_sysctl
# --------------------------------------------------------------------------


def test_write_sysctl_reports_success_and_failure():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "counter")
        assert se._write_sysctl(path, 4321) is True
        with open(path) as handle:
            assert handle.read() == "4321"
        assert se._write_sysctl(os.path.join(tmp, "nope", "counter"), 1) is False


# --------------------------------------------------------------------------
# _raise_pid_floor: the paths that decline
# --------------------------------------------------------------------------


def test_floor_is_disabled_by_zero():
    sysctls, previous = _Sysctls(), _set_env(SEMIP_PID_FLOOR="0")
    burner = _Burner(sysctls)
    try:
        assert se._raise_pid_floor() is None
        assert burner.calls == []
        assert sysctls.read_counter() == 1234
    finally:
        burner.close()
        _restore_env(previous)
        sysctls.close()


def test_floor_is_refused_when_pid_max_leaves_no_room():
    """The legacy pid_max of 32768 has no safe floor, so none is half-applied.

    A floor close to pid_max is not a floor: the counter wraps back under it
    and the next pod's low ids are in the image's range again.
    """
    sysctls = _Sysctls(pid_max=32768)
    previous = _set_env(SEMIP_PID_FLOOR=None)
    burner = _Burner(sysctls)
    try:
        assert se._raise_pid_floor() is None
        assert burner.calls == []
    finally:
        burner.close()
        _restore_env(previous)
        sysctls.close()


def test_floor_is_refused_when_the_sysctls_are_unreadable():
    """No /proc at all -- a laptop, or a kernel that hides it. Never raises."""
    saved = (se._NS_LAST_PID_PATH, se._PID_MAX_PATH)
    se._NS_LAST_PID_PATH = "/nonexistent/ns_last_pid"
    se._PID_MAX_PATH = "/nonexistent/pid_max"
    previous = _set_env(SEMIP_PID_FLOOR=None)
    try:
        assert se._raise_pid_floor() is None
    finally:
        _restore_env(previous)
        (se._NS_LAST_PID_PATH, se._PID_MAX_PATH) = saved


def test_floor_is_a_no_op_when_the_counter_is_already_past_it():
    sysctls = _Sysctls(counter=150_000)
    previous = _set_env(SEMIP_PID_FLOOR="100000")
    burner = _Burner(sysctls)
    try:
        assert se._raise_pid_floor() == 150_000
        assert burner.calls == []
    finally:
        burner.close()
        _restore_env(previous)
        sysctls.close()


# --------------------------------------------------------------------------
# _raise_pid_floor: the paths that place the floor
# --------------------------------------------------------------------------


def test_floor_takes_the_write_when_the_namespace_allows_one():
    """Where ns_last_pid is writable the burn is skipped entirely."""
    sysctls = _Sysctls()
    previous = _set_env(SEMIP_PID_FLOOR="100000")
    burner = _Burner(sysctls)
    se._write_sysctl = lambda path, value: (sysctls.write_counter(value), True)[1]
    try:
        assert se._raise_pid_floor() == 100_000
        assert burner.calls == [], "the write path must not also burn"
    finally:
        burner.close()
        _restore_env(previous)
        sysctls.close()


def test_floor_burns_when_the_counter_cannot_be_written():
    sysctls = _Sysctls(counter=2000)
    previous = _set_env(SEMIP_PID_FLOOR="100000", SEMIP_PID_BURN_WORKERS="4")
    burner = _Burner(sysctls, reaches=100_001)
    try:
        assert se._raise_pid_floor() == 100_001
        assert len(burner.calls) == 1
        argv, kwargs = burner.calls[0]
        assert argv[0] == sys.executable
        assert argv[1] == "-c" and argv[2] == se._PID_BURN_SOURCE
        assert argv[3:] == ["100000", "4", str(100_000 - 2000
                                               + se._PID_BURN_SLACK)]
        assert kwargs["timeout"] == se._PID_BURN_TIMEOUT_S
    finally:
        burner.close()
        _restore_env(previous)
        sysctls.close()


def test_floor_does_not_claim_a_burn_that_fell_short():
    """A burn that ran but did not arrive is a miss, not a floor."""
    sysctls = _Sysctls(counter=2000)
    previous = _set_env(SEMIP_PID_FLOOR="100000")
    burner = _Burner(sysctls, reaches=40_000)
    try:
        assert se._raise_pid_floor() is None
    finally:
        burner.close()
        _restore_env(previous)
        sysctls.close()


def test_floor_survives_a_burner_that_fails_or_hangs():
    sysctls = _Sysctls(counter=2000)
    previous = _set_env(SEMIP_PID_FLOOR="100000")
    try:
        # Constructed one at a time: each installs itself on creation, so a
        # pair built up front would nest, and the outer one would be restored
        # as if it were the original.
        for failure in ({"returncode": 1},
                        {"raises": subprocess.TimeoutExpired("py", 1)}):
            burner = _Burner(sysctls, **failure)
            try:
                assert se._raise_pid_floor() is None
            finally:
                burner.close()
    finally:
        _restore_env(previous)
        sysctls.close()


# --------------------------------------------------------------------------
# The burn loop itself, run for real
# --------------------------------------------------------------------------


def test_burn_source_advances_the_counter():
    """The inlined source is executed by a fresh interpreter, so it must run.

    A syntax error or a stray name in it would otherwise surface only on a
    cluster, one image build later.
    """
    target = os.getpid() + 40
    result = subprocess.run(
        [sys.executable, "-c", se._PID_BURN_SOURCE, str(target), "2", "5000"],
        capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode()
    assert int(result.stdout.split()[-1]) >= target, result.stdout


def test_burn_source_stops_at_its_limit():
    """The bound that keeps an orphaned burner from forking forever.

    ``subprocess.run``'s timeout kills the burner it spawned, not the burners
    that one forked, so the limit is what actually terminates them.
    """
    result = subprocess.run(
        [sys.executable, "-c", se._PID_BURN_SOURCE, "1000000000", "2", "20"],
        capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode()
    assert int(result.stdout.split()[-1]) < 1000000000


# --------------------------------------------------------------------------
# _dump wiring
# --------------------------------------------------------------------------


class _FakeInstance:
    """Records the call sequence a dump makes, and the meta it would write."""

    def __init__(self, log):
        self.log = log
        self.meta_extra = None
        log.append("Instance()")

    def criu_dump(self, filename=None, meta_extra=None):
        self.meta_extra = meta_extra
        self.log.append("criu_dump")

    def __getattr__(self, name):
        return lambda *a, **k: self.log.append(name)


def test_dump_places_the_floor_before_the_instance_and_records_it():
    """The ordering is the whole mechanism: after ``Instance`` it is too late.

    The ids being placed are the ones the child tree will record, so the burn
    has to happen before anything in that tree exists.
    """
    log = []
    saved = (se.Instance, se._raise_pid_floor, se._record_env_files, se.time)
    instances = []

    def fake_instance(vllm_config, model_dir):
        inst = _FakeInstance(log)
        instances.append(inst)
        return inst

    se.Instance = fake_instance
    se._raise_pid_floor = lambda: log.append("floor") or 100_001
    se._record_env_files = lambda model_dir: None
    se.time = types.SimpleNamespace(perf_counter=saved[3].perf_counter,
                                    sleep=lambda seconds: None)
    try:
        se._dump({}, "/tmp/model_dir", [0], image_ref="img@sha256:abc",
                 driver_version="580.159.03")
        assert log[0] == "floor", log
        assert log[1] == "Instance()", log
        assert instances[0].meta_extra == {
            "image_ref": "img@sha256:abc",
            "driver_version": "580.159.03",
            "pid_floor": 100_001,
            "unprivileged": se._unprivileged_mode(),
        }
    finally:
        (se.Instance, se._raise_pid_floor, se._record_env_files,
         se.time) = saved


def test_dump_records_a_null_floor_when_none_was_placed():
    """An image that could not be floored says so, rather than staying silent."""
    log = []
    saved = (se.Instance, se._raise_pid_floor, se._record_env_files, se.time)
    instances = []

    def fake_instance(vllm_config, model_dir):
        inst = _FakeInstance(log)
        instances.append(inst)
        return inst

    se.Instance = fake_instance
    se._raise_pid_floor = lambda: None
    se._record_env_files = lambda model_dir: None
    se.time = types.SimpleNamespace(perf_counter=saved[3].perf_counter,
                                    sleep=lambda seconds: None)
    try:
        se._dump({}, "/tmp/model_dir", [0], image_ref="img@sha256:abc",
                 driver_version="580.159.03")
        assert instances[0].meta_extra["pid_floor"] is None
    finally:
        (se.Instance, se._raise_pid_floor, se._record_env_files,
         se.time) = saved


if __name__ == "__main__":
    tests = [
        test_env_int_takes_the_override,
        test_env_int_ignores_a_malformed_override,
        test_write_sysctl_reports_success_and_failure,
        test_floor_is_disabled_by_zero,
        test_floor_is_refused_when_pid_max_leaves_no_room,
        test_floor_is_refused_when_the_sysctls_are_unreadable,
        test_floor_is_a_no_op_when_the_counter_is_already_past_it,
        test_floor_takes_the_write_when_the_namespace_allows_one,
        test_floor_burns_when_the_counter_cannot_be_written,
        test_floor_does_not_claim_a_burn_that_fell_short,
        test_floor_survives_a_burner_that_fails_or_hangs,
        test_burn_source_advances_the_counter,
        test_burn_source_stops_at_its_limit,
        test_dump_places_the_floor_before_the_instance_and_records_it,
        test_dump_records_a_null_floor_when_none_was_placed,
    ]
    failures = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
    sys.exit(1 if failures else 0)
