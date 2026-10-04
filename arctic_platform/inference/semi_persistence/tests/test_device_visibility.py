"""Unit tests for the device-visibility check and dump-path pre-creation.

A TP>1 image restores only into a pod that can reopen every device node its
captured state refers to, and on the scheduled path a pod is given only its
allocated slice of ``/dev``. So an image dumped on GPUs 2,3 is unrestorable in
a pod holding 0,1 -- twenty restores, no exception, at TP=2 and TP=4 alike.

Left unchecked that mismatch surfaces from inside ``ncclCommInitRank`` as
``NCCL error: unhandled system error``, which reads like a network fault and is
not one. These tests pin the predicate, the TP=1 exemption, the escape hatch,
and the one case it must stay quiet about: images published before
``device_nodes`` existed, which have no recorded set and must not be stranded.

Two claims the first cut of this made are now measured and false, so they are
not asserted anywhere here. The check does **not** run before the copy -- the
backstop lives inside ``_restore``, which is reached only after
``_materialize_from_source`` returns -- and the copy it was said to save is
``image/`` plus ``compilation/`` at ~6.7 GB, not 80 GB, because ``weights/`` are
read in place off the mirror. Declining in the miss path is what actually skips
it, and what turns a refusal into a cold start rather than a failed job.

Run from the package directory:

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_device_visibility.py -v
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                      # .../semi_persistence
_INFERENCE = os.path.dirname(_PKG)                 # .../arctic_platform/inference
_ENGINE = os.path.join(_INFERENCE, "server", "semip_engine.py")


def _load_semip_engine():
    """Import semip_engine.py without executing the arctic_platform packages."""
    pkg = types.ModuleType("arctic_platform.inference")
    pkg.__path__ = []
    semip = types.ModuleType("arctic_platform.inference.semi_persistence")
    semip.Instance = object  # only referenced at call time, never here
    sys.modules.setdefault("arctic_platform.inference", pkg)
    sys.modules.setdefault("arctic_platform.inference.semi_persistence", semip)

    spec = importlib.util.spec_from_file_location("semip_engine_device_test",
                                                  _ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


se = _load_semip_engine()


class _Devices:
    """Stand in for this pod's /dev by pointing the scan at a temp directory."""

    def __init__(self, names):
        self.names = names

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        for name in self.names:
            open(os.path.join(self._dir.name, name), "w").close()
        self._saved = se._DEVICE_GLOBS
        se._DEVICE_GLOBS = (os.path.join(self._dir.name, "nvidia[0-9]*"),
                            os.path.join(self._dir.name, "uverbs[0-9]*"))
        return self

    def __exit__(self, *exc):
        se._DEVICE_GLOBS = self._saved
        self._dir.cleanup()
        return False


def _img(nodes, tp=2):
    """An image meta recording ``nodes``, dumped at tensor-parallel degree ``tp``.

    The TP is not incidental. The constraint is specific to the multi-rank case,
    so every fixture has to say which side of that line it is on; a meta with no
    ``vllm_config`` reads as TP=1 and is exempt by design.
    """
    return {"device_nodes": list(nodes),
            "vllm_config": {"tensor_parallel_size": tp}}


# --------------------------------------------------------------------------
# _image_tp -- which side of the TP>1 line an image is on
# --------------------------------------------------------------------------


def test_image_tp_reads_the_baked_config():
    assert se._image_tp(_img([], tp=4)) == 4


def test_image_tp_defaults_to_one_when_unstated():
    """vLLM's own default, and the degree that imposes no device constraint."""
    assert se._image_tp({}) == 1
    assert se._image_tp({"vllm_config": {}}) == 1


def test_image_tp_survives_a_float_or_a_junk_value():
    """meta.json round-trips through JSON, where 2 can come back as 2.0."""
    assert se._image_tp({"vllm_config": {"tensor_parallel_size": 2.0}}) == 2
    assert se._image_tp({"vllm_config": {"tensor_parallel_size": "?"}}) == 1
    assert se._image_tp({"vllm_config": {"tensor_parallel_size": None}}) == 1


# --------------------------------------------------------------------------
# _missing_device_nodes -- the predicate both the miss path and the backstop use
# --------------------------------------------------------------------------


def test_nothing_missing_when_slots_match():
    with _Devices(["nvidia2", "nvidia3", "uverbs4", "uverbs5"]):
        assert se._missing_device_nodes(
            _img(["nvidia2", "nvidia3", "uverbs4"])) == []


def test_missing_names_every_absent_node():
    with _Devices(["nvidia0", "nvidia1"]):
        assert se._missing_device_nodes(
            _img(["nvidia2", "nvidia3"])) == ["nvidia2", "nvidia3"]


def test_tp1_is_exempt_even_on_a_total_mismatch():
    """A TP=1 image has no communicator to rebuild and has always travelled.

    This is the regression the first cut of the check shipped: it read TP>1 in
    its docstring but ran unconditionally, so a TP=1 image dumped on GPU 4 was
    refused in a pod holding GPU 6 -- with a message claiming NCCL would fail,
    when at TP=1 there is no NCCL involved at all.
    """
    with _Devices(["nvidia6", "nvidia7"]):
        assert se._missing_device_nodes(
            _img(["nvidia4", "nvidia5"], tp=1)) == []


# --------------------------------------------------------------------------
# _check_device_visibility -- the backstop that still raises
# --------------------------------------------------------------------------


def test_matching_slots_pass():
    with _Devices(["nvidia2", "nvidia3", "uverbs4", "uverbs5"]):
        se._check_device_visibility(
            _img(["nvidia2", "nvidia3", "uverbs4"]), "/cache/key")


def test_superset_passes():
    """A pod holding more than the image needs is fine -- presence is the rule."""
    with _Devices(["nvidia0", "nvidia1", "nvidia2", "nvidia3"]):
        se._check_device_visibility(_img(["nvidia2", "nvidia3"]), "/cache/key")


def test_mismatched_slots_raise_and_name_the_devices():
    """The message has to carry what is missing; that is its whole point."""
    with _Devices(["nvidia0", "nvidia1"]):
        with pytest.raises(RuntimeError) as excinfo:
            se._check_device_visibility(_img(["nvidia2", "nvidia3"]),
                                        "/cache/key")
    msg = str(excinfo.value)
    assert "nvidia2" in msg and "nvidia3" in msg
    assert "/cache/key" in msg
    # It must also say what this pod *does* have, or the reader cannot tell a
    # slot mismatch from an empty /dev.
    assert "nvidia0" in msg


def test_partial_overlap_still_raises():
    """One missing device is enough; NCCL does not care that the rest resolved."""
    with _Devices(["nvidia2"]):
        with pytest.raises(RuntimeError) as excinfo:
            se._check_device_visibility(_img(["nvidia2", "nvidia3"]),
                                        "/cache/key")
    assert "nvidia3" in str(excinfo.value)


def test_tp1_mismatch_does_not_raise():
    """The gate has to hold at the backstop too, not only in the predicate."""
    with _Devices(["nvidia6"]):
        se._check_device_visibility(_img(["nvidia4"], tp=1), "/cache/key")


def test_legacy_image_without_device_nodes_is_left_alone():
    """Absence is not evidence -- refusing these would strand every published key."""
    with _Devices([]):
        se._check_device_visibility(
            {"gpus": [2, 3], "vllm_config": {"tensor_parallel_size": 2}},
            "/cache/key")
        se._check_device_visibility(_img([]), "/cache/key")


def test_escape_hatch_downgrades_to_a_warning():
    """The override exists for the day this is wrong about a real placement.

    It is also how a mismatch is reached on purpose: driving a known-bad
    placement all the way into ncclCommInitRank is what finally produced the
    NCCL output behind 'unhandled system error'.
    """
    with _Devices(["nvidia0"]):
        os.environ[se._DEVICE_CHECK_ENV] = "0"
        try:
            se._check_device_visibility(_img(["nvidia2"]), "/cache/key")
        finally:
            os.environ.pop(se._DEVICE_CHECK_ENV, None)


def test_escape_hatch_only_accepts_falsey_spellings():
    """A stray value must not silently disable the guard."""
    with _Devices(["nvidia0"]):
        os.environ[se._DEVICE_CHECK_ENV] = "1"
        try:
            with pytest.raises(RuntimeError):
                se._check_device_visibility(_img(["nvidia2"]), "/cache/key")
        finally:
            os.environ.pop(se._DEVICE_CHECK_ENV, None)


def test_device_check_disabled_reads_only_falsey_spellings():
    for value, disabled in (("0", True), ("false", True), ("no", True),
                            (" 0 ", True), ("1", False), ("true", False),
                            ("", False)):
        os.environ[se._DEVICE_CHECK_ENV] = value
        try:
            assert se._device_check_disabled() is disabled, value
        finally:
            os.environ.pop(se._DEVICE_CHECK_ENV, None)
    assert se._device_check_disabled() is False


# --------------------------------------------------------------------------
# a quiescence failure is not retried
# --------------------------------------------------------------------------

# The message as the Ray actor actually produces it, wrapped by the Instance.
_QUIESCE_MSG = (
    "command 'cuda_restore' failed: RuntimeError: CUDA restore skipped: 2 of 3 "
    "pid(s) never quiesced within 30.0s of the CRIU restore and are blocked in "
    "the kernel, so the ranks would come up with no GPUs and reinit_nccl would "
    "hang. States: 100137=D (disk sleep), 100136=D (disk sleep)")


def test_a_quiescence_failure_is_not_retried():
    """022eb9d retried this; the retry could never have worked.

    It fired in 4 of 4 quiescence failures on hardware and rescued none. Now
    understood: the ranks spin because the restore host's CLOCK_MONOTONIC is
    behind the dump host's, and a retry lands on the same host with the same
    clock, so it re-asks a question whose answer cannot change. The spin itself
    is handled where it occurs, by worker._is_userspace_spin; what survives to
    here is a tree genuinely blocked in the kernel, which must fail at once.
    """
    calls = []

    def always_stuck(model_dir, engine_kwargs, gpus, **kwargs):
        calls.append(1)
        raise RuntimeError(_QUIESCE_MSG)

    saved_restore, saved_sleep = se._restore, se.time.sleep
    se._restore, se.time.sleep = always_stuck, lambda _s: None
    try:
        with pytest.raises(RuntimeError) as excinfo:
            se._restore_with_port_retry({}, {}, [0], after="test")
    finally:
        se._restore, se.time.sleep = saved_restore, saved_sleep
    assert "never quiesced" in str(excinfo.value)
    assert len(calls) == 1, calls        # no retry
