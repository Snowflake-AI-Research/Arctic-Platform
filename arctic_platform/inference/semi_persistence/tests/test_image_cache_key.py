"""Unit tests for the derived image-cache key and the published-image source.

Both halves of ``server/semip_engine.py``'s cache logic are pure functions of
their inputs and the filesystem, so all of this runs on a laptop in
milliseconds: no cluster, no GPU, no CRIU, no vLLM. What it covers that an
end-to-end job cannot is *which* guard fired -- the source path declines in
seven distinguishable ways, all of which look identical from outside (a cold
start).

Run from the package directory:

    cd arctic_platform/inference/semi_persistence
    python -m pytest tests/test_image_cache_key.py -v

Or directly:

    python tests/test_image_cache_key.py

``semip_engine`` lives under ``arctic_platform/inference/server``, whose package
``__init__`` pulls in ray and torch via ``replica_pool`` and ``weight_sync``.
Importing it the ordinary way would make these tests neither fast nor
hermetic, and the key derivation itself needs none of that -- so the module is
loaded straight from its file with a stub standing in for the one name it
imports at module scope. If ``semip_engine`` ever grows a second module-scope
import, this bootstrap is what will notice.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import types

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

    spec = importlib.util.spec_from_file_location("semip_engine_under_test",
                                                  _ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


se = _load_semip_engine()


BASE_CONFIG = {
    "enable_chunked_prefill": True,
    "gpu_memory_utilization": 0.8,
    "max_model_len": 128,
    "model": "/mnt/neutrino/base-models/Qwen/Qwen3.6-35B-A3B",
    "tensor_parallel_size": 1,
}


# --------------------------------------------------------------------------
# _config_hash
# --------------------------------------------------------------------------


def test_config_hash_is_order_independent():
    """Key ordering is a dict artefact, not a config difference."""
    shuffled = dict(reversed(list(BASE_CONFIG.items())))
    assert se._config_hash(BASE_CONFIG) == se._config_hash(shuffled)


def test_config_hash_is_stable_across_calls():
    assert se._config_hash(BASE_CONFIG) == se._config_hash(dict(BASE_CONFIG))


def test_config_hash_covers_gpu_memory_utilization():
    """The knob vLLM's own compute_hash deliberately ignores.

    ``CacheConfig.compute_hash`` lists ``gpu_memory_utilization`` in its
    ignored factors, correctly -- it does not change the compiled graph. It
    does change the KV cache allocation, and therefore the process memory a
    CRIU image is a snapshot of, so it has to move this key.
    """
    other = dict(BASE_CONFIG, gpu_memory_utilization=0.7)
    assert se._config_hash(BASE_CONFIG) != se._config_hash(other)


def test_config_hash_covers_tensor_parallel_size():
    other = dict(BASE_CONFIG, tensor_parallel_size=2)
    assert se._config_hash(BASE_CONFIG) != se._config_hash(other)


def test_config_hash_does_not_normalize_model_path():
    """Two roots holding the same model tail are not assumed equivalent.

    Collapsing these to ``Qwen/Qwen3.6-35B-A3B`` would alias a stale mirror
    onto a live one. A changed root costing a miss is the harmless direction.
    """
    other = dict(BASE_CONFIG, model="/data-fast/mirror/Qwen/Qwen3.6-35B-A3B")
    assert se._config_hash(BASE_CONFIG) != se._config_hash(other)


def test_config_hash_length():
    assert len(se._config_hash(BASE_CONFIG)) == se._HASH_LEN


# --------------------------------------------------------------------------
# _config_hash / _device_binding: the device allocation in the key
#
# A TP>1 image is restorable only where every device node it captured can be
# reopened, so the allocation is part of its identity. These pin the three
# properties that makes the key depend on -- it separates allocations, it
# ignores their order, and it leaves TP=1 alone -- plus the cross-copy
# agreement the key now rests on.
# --------------------------------------------------------------------------


TP2_CONFIG = dict(BASE_CONFIG, tensor_parallel_size=2)


class _Devices:
    """Point the device scan at a temp directory standing in for ``/dev``."""

    def __init__(self, names):
        self.names = names

    def __enter__(self):
        self._dir = tempfile.TemporaryDirectory()
        for name in self.names:
            open(os.path.join(self._dir.name, name), "w").close()
        self._saved = se._DEVICE_GLOBS
        se._DEVICE_GLOBS = (os.path.join(self._dir.name, "*"),)
        return self

    def __exit__(self, *exc):
        se._DEVICE_GLOBS = self._saved
        self._dir.cleanup()
        return False


def test_config_hash_separates_two_device_allocations():
    """The whole point: one config on two slots is two images, two keys.

    Before this, both dumped into one directory and the second overwrote the
    first, so a config had exactly one restorable placement no matter how many
    times it was dumped.
    """
    a = se._config_hash(TP2_CONFIG, ["nvidia4", "nvidia5"])
    b = se._config_hash(TP2_CONFIG, ["nvidia6", "nvidia7"])
    assert a != b


def test_config_hash_ignores_device_order():
    """The binding is a set. ``_gpu_migration_permutation`` handles the rest.

    Measured on a live TP=4 restore: an image dumped on ``[3, 1, 2, 0]`` came
    back on ``[3, 2, 0, 1]``. Hashing the order would have split that image
    into keys no restore needs.
    """
    forward = se._config_hash(TP2_CONFIG, ["nvidia4", "nvidia5"])
    reversed_ = se._config_hash(TP2_CONFIG, ["nvidia5", "nvidia4"])
    assert forward == reversed_


def test_config_hash_without_devices_is_the_bare_config():
    """``None`` must mean "unbound", not "bound to nothing".

    An empty list is a real (if degenerate) allocation and has to hash
    differently, or a pod whose ``/dev`` scan came back empty would silently
    collide with every TP=1 image of the same config.
    """
    assert se._config_hash(TP2_CONFIG, None) == se._config_hash(TP2_CONFIG)
    assert se._config_hash(TP2_CONFIG, []) != se._config_hash(TP2_CONFIG)


def test_device_binding_is_none_at_tp1():
    """TP=1 restores onto any slot, so binding it would only fragment it."""
    with _Devices(["nvidia3", "uverbs6", "uverbs7"]):
        assert se._device_binding(BASE_CONFIG) is None


def test_device_binding_reads_the_pod_allocation_at_tp2():
    with _Devices(["nvidia4", "nvidia5", "uverbs8", "uverbs9"]):
        assert se._device_binding(TP2_CONFIG) == [
            "nvidia4", "nvidia5", "uverbs8", "uverbs9"]


def test_device_binding_threshold_matches_the_restore_check():
    """The key and the veto must bind the devices at the same TP.

    A key that bound them where ``_missing_device_nodes`` did not would publish
    images nothing ever looks up; the reverse is the placement lottery this
    replaced. Asserted against the check's own behaviour rather than against
    the constant, so redefining one without the other fails here.
    """
    with _Devices(["nvidia4", "nvidia5"]):
        for tp in (1, 2, 4):
            meta = {"vllm_config": {"tensor_parallel_size": tp},
                    "device_nodes": ["nvidia6", "nvidia7"]}
            key_binds = se._device_binding(
                dict(BASE_CONFIG, tensor_parallel_size=tp)) is not None
            check_binds = bool(se._missing_device_nodes(meta))
            assert key_binds == check_binds, tp


def test_tp1_binds_the_key_but_not_the_check_with_several_replicas():
    """The one place the two part ways, on purpose.

    Several TP=1 replicas dump under <key>/replica<K>/ where a lone one dumps
    flat, so the pod's allocation has to keep a 1-GPU and an 8-GPU pod off one
    key. Restorability is unchanged -- a TP=1 image still lands on any slot --
    so the device check keeps exempting it.
    """
    with _Devices(["nvidia0", "nvidia1", "uverbs0", "uverbs1"]):
        assert se._device_binding(BASE_CONFIG, replica_count=1) is None
        assert se._device_binding(BASE_CONFIG, replica_count=2) == [
            "nvidia0", "nvidia1", "uverbs0", "uverbs1"]
    meta = {"vllm_config": {"tensor_parallel_size": 1},
            "device_nodes": ["nvidia6", "nvidia7"]}
    assert se._missing_device_nodes(meta) == []


# --------------------------------------------------------------------------
# replica slots
# --------------------------------------------------------------------------


def _with_slot(count, slot, body):
    saved = {k: os.environ.get(k)
             for k in ("SEMIP_NUM_REPLICAS", "SEMIP_REPLICA_ID")}
    for key, value in (("SEMIP_NUM_REPLICAS", count),
                       ("SEMIP_REPLICA_ID", slot)):
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        return body()
    finally:
        for key, was in saved.items():
            if was is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = was


def test_replica_slot_defaults_to_a_lone_replica():
    assert _with_slot(None, None, se._replica_slot) == (0, 1)
    # A slot without a count above 1 changes nothing.
    assert _with_slot(1, 5, se._replica_slot) == (0, 1)


def test_replica_slot_reads_both_variables():
    assert _with_slot(8, 3, se._replica_slot) == (3, 8)


def test_replica_slot_refuses_a_missing_or_foreign_slot():
    """Guessing would put two replicas of one pod on one image."""
    for slot in (None, "", "x", -1, 4):
        _expect_raises(lambda: _with_slot(4, slot, se._replica_slot),
                       RuntimeError)


def test_resolve_model_dir_nests_a_replica_under_the_shared_key():
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ["SEMIP_IMAGE_CACHE"] = "/data-fast/image-cache_neutrino"
    try:
        with _Devices(["nvidia0", "nvidia1"]):
            lone = _with_slot(None, None,
                              lambda: se._resolve_model_dir(TP2_CONFIG))
            many = _with_slot(4, 2, lambda: se._resolve_model_dir(TP2_CONFIG))
        assert os.path.basename(lone.model_dir) == lone.key_prefix
        assert many.model_dir == os.path.join(
            "/data-fast/image-cache_neutrino", many.key_prefix, "replica2")
        assert (many.replica_slot, many.replica_count) == (2, 4)
        # At TP>1 the device set already separates pod shapes, so the replica
        # count adds nothing to the key itself.
        assert many.key_prefix == lone.key_prefix
    finally:
        se._env_hash = original
        os.environ.pop("SEMIP_IMAGE_CACHE", None)


def test_replica_source_picks_this_slot_inside_the_skeleton():
    def body(tmp):
        skel = os.path.join(tmp, "skel")
        os.makedirs(os.path.join(skel, "replica0"))
        os.makedirs(os.path.join(skel, "replica1"))
        many = se._ImagePaths("/c/k/replica1", "k", None, None, "img", "580",
                              1, 2)
        lone = se._ImagePaths("/c/k", "k", None, None, "img", "580")
        assert se._replica_source(skel, many) == os.path.join(skel, "replica1")
        assert se._replica_source(skel, lone) == skel
        assert se._replica_source(None, many) is None
    _in_tmp(body)


def test_replica_source_misses_a_flat_skeleton_under_a_replica_key():
    """Every replica reads the same answer, so the pod cold-starts together."""
    def body(tmp):
        skel = os.path.join(tmp, "skel")
        os.makedirs(os.path.join(skel, "image"))
        many = se._ImagePaths("/c/k/replica0", "k", None, None, "img", "580",
                              0, 2)
        assert se._replica_source(skel, many) is None
    _in_tmp(body)


def test_strict_materialize_raises_for_a_missing_replica():
    """All-or-none: one replica short of its directory fails the job."""
    def body(tmp):
        _expect_raises(
            lambda: se._materialize_from_source(
                os.path.join(tmp, "skel", "replica3"), "/c/k/replica3",
                strict=True),
            RuntimeError)
        # The same absence is an ordinary miss for a lone replica.
        assert se._materialize_from_source(
            os.path.join(tmp, "skel"), "/c/k") is False
    _in_tmp(body)


def test_unprivileged_mismatch_names_both_sides():
    saved = os.environ.get("SEMIP_UNPRIVILEGED")
    os.environ["SEMIP_UNPRIVILEGED"] = "1"
    try:
        assert se._unprivileged_mismatch({"unprivileged": True}) is None
        assert se._unprivileged_mismatch({}) is None    # predates the field
        assert "SEMIP_UNPRIVILEGED=0" in se._unprivileged_mismatch(
            {"unprivileged": False})
    finally:
        if saved is None:
            os.environ.pop("SEMIP_UNPRIVILEGED", None)
        else:
            os.environ["SEMIP_UNPRIVILEGED"] = saved


def test_resolve_model_dir_separates_allocations_of_one_config():
    """End to end: the same TP=2 job on two slots resolves to two directories."""
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ["SEMIP_IMAGE_CACHE"] = "/data-fast/image-cache_neutrino"
    try:
        with _Devices(["nvidia0", "nvidia1", "uverbs0", "uverbs1"]):
            first = se._resolve_model_dir(TP2_CONFIG).model_dir
        with _Devices(["nvidia2", "nvidia3", "uverbs2", "uverbs3"]):
            second = se._resolve_model_dir(TP2_CONFIG).model_dir
        assert first != second, first
        # Still two 12-hex hashes joined by an underscore: criu validates that
        # exact string and semip_publish._DERIVED_KEY_RE matches it.
        assert re.fullmatch(r"[0-9a-f]{12}_eeeeeeeeeeee",
                            os.path.basename(first)), first
    finally:
        se._env_hash = original
        os.environ.pop("SEMIP_IMAGE_CACHE", None)


def test_resolve_model_dir_is_stable_for_one_allocation():
    """Two jobs on the same devices must land on the same directory."""
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ["SEMIP_IMAGE_CACHE"] = "/data-fast/image-cache_neutrino"
    try:
        with _Devices(["nvidia6", "nvidia7", "uverbs12", "uverbs13"]):
            first = se._resolve_model_dir(TP2_CONFIG).model_dir
        with _Devices(["nvidia6", "nvidia7", "uverbs12", "uverbs13"]):
            second = se._resolve_model_dir(TP2_CONFIG).model_dir
        assert first == second, (first, second)
    finally:
        se._env_hash = original
        os.environ.pop("SEMIP_IMAGE_CACHE", None)


def test_the_two_visible_device_nodes_copies_agree():
    """``worker``'s copy records what the engine's copy now keys the cache on.

    They always had to agree or the check would refuse a restore that would
    have worked. Now the key is derived from the engine's copy, so drift makes
    every TP>1 job compute a key no publish ever wrote: a permanent silent
    miss instead of a loud refusal. Same hazard as ``model_slug``, pinned the
    same way.
    """
    worker_src = os.path.join(_PKG, "worker.py")
    with open(worker_src) as fh:
        worker_text = fh.read()

    # The globs decide which nodes are in the set at all, so equality of the
    # tuples is most of the agreement.
    match = re.search(r"^_DEVICE_GLOBS = (\(.*?\))$", worker_text, re.M)
    assert match, "worker.py no longer defines _DEVICE_GLOBS"
    assert ast.literal_eval(match.group(1)) == se._DEVICE_GLOBS

    # And both must return basenames, sorted, deduplicated -- the engine's
    # output is hashed, so an unsorted or duplicated list is a different key.
    body = worker_text.split("def _visible_device_nodes(")[1]
    body = body.split("\ndef ")[0]
    for token in ("os.path.basename", "sorted(", "set()"):
        assert token in body, token


# --------------------------------------------------------------------------
# _driver_version
# --------------------------------------------------------------------------


def test_driver_version_parses_nvrm_line(monkeypatch=None):
    line = ("NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  "
            "580.159.03  Release Build  (root@host)  Thu Jun 25 02:57:11 2026\n"
            "GCC version:  gcc version 11.5.0")
    original = se._read_text
    se._read_text = lambda path: line
    try:
        assert se._driver_version() == "580.159.03"
    finally:
        se._read_text = original


def test_driver_version_raises_on_unparseable():
    original = se._read_text
    se._read_text = lambda path: "something else entirely"
    try:
        _expect_raises(se._driver_version, RuntimeError)
    finally:
        se._read_text = original


# --------------------------------------------------------------------------
# _resolve_model_dir
# --------------------------------------------------------------------------


def test_resolve_model_dir_composes_both_hashes():
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ["SEMIP_IMAGE_CACHE"] = "/data-fast/image-cache_neutrino"
    try:
        paths = se._resolve_model_dir(BASE_CONFIG)
        expected = os.path.join(
            "/data-fast/image-cache_neutrino",
            f"{se._config_hash(BASE_CONFIG)}_eeeeeeeeeeee")
        assert paths.model_dir == expected, paths.model_dir
        assert paths.image_ref == "img@sha256:abc"
        assert paths.driver_version == "580.159.03"
    finally:
        se._env_hash = original
        os.environ.pop("SEMIP_IMAGE_CACHE", None)


def test_resolve_model_dir_strips_trailing_slash():
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ["SEMIP_IMAGE_CACHE"] = "/data-fast/image-cache_neutrino/"
    try:
        model_dir = se._resolve_model_dir(BASE_CONFIG).model_dir
        assert "//" not in model_dir, model_dir
    finally:
        se._env_hash = original
        os.environ.pop("SEMIP_IMAGE_CACHE", None)


def test_resolve_model_dir_defaults_the_cache_root():
    """Unset means the operator's /data-fast mount, not an error."""
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ.pop("SEMIP_IMAGE_CACHE", None)
    try:
        model_dir = se._resolve_model_dir(BASE_CONFIG).model_dir
        assert os.path.dirname(model_dir) == se._DEFAULT_IMAGE_CACHE, model_dir
    finally:
        se._env_hash = original


def test_resolve_model_dir_separates_configs_under_one_env():
    """Same environment, different config -> different directory, not a clash."""
    original = se._env_hash
    se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.159.03")
    os.environ["SEMIP_IMAGE_CACHE"] = "/cache"
    try:
        a = se._resolve_model_dir(BASE_CONFIG).model_dir
        b = se._resolve_model_dir(
            dict(BASE_CONFIG, gpu_memory_utilization=0.7)).model_dir
        assert a != b
    finally:
        se._env_hash = original
        os.environ.pop("SEMIP_IMAGE_CACHE", None)


# --------------------------------------------------------------------------
# _pod_image_ref
# --------------------------------------------------------------------------


def _fake_pod_body(statuses):
    return {"status": {"containerStatuses": statuses}}


class _FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _with_fake_pod(body, fn):
    import urllib.request as urlreq
    original_open = urlreq.urlopen
    original_read = se._read_text
    original_ssl = se.ssl.create_default_context
    se._read_text = lambda path: "stub"
    se.ssl.create_default_context = lambda **kwargs: None
    urlreq.urlopen = lambda *a, **k: _FakeResponse(body)
    os.environ["HOSTNAME"] = "dz-test-device-manager-abc"
    try:
        return fn()
    finally:
        urlreq.urlopen = original_open
        se._read_text = original_read
        se.ssl.create_default_context = original_ssl


def test_pod_image_ref_matches_the_named_container():
    body = _fake_pod_body([
        {"name": "istio-proxy", "imageID": "proxy@sha256:1111"},
        {"name": "device-manager", "imageID": "dss@sha256:2222"},
    ])
    got = _with_fake_pod(body, se._pod_image_ref)
    assert got == "dss@sha256:2222", got


def test_pod_image_ref_raises_rather_than_guessing():
    """A sidecar's digest would be a wrong key with no symptom."""
    body = _fake_pod_body([{"name": "istio-proxy", "imageID": "p@sha256:1111"}])
    _with_fake_pod(
        body, lambda: _expect_raises(se._pod_image_ref, RuntimeError))


def test_pod_image_ref_rejects_a_digestless_image_id():
    body = _fake_pod_body([{"name": "device-manager", "imageID": "dss:latest"}])
    _with_fake_pod(
        body, lambda: _expect_raises(se._pod_image_ref, RuntimeError))


# --------------------------------------------------------------------------
# The published-image source
# --------------------------------------------------------------------------

# The first known-good pair, from the dump in handoff 6 section 1. Any change
# to either hash function must be checked against it.
KEY = "77d95928d2ac_1fad1e633806"

# The weight hash a published skeleton names. Only its shape matters here: 12
# hex, appended to the derived key, which is how a restore learns which shard
# directory to read without a pointer file.
WEIGHT_HASH = "beefbeefbeef"

# The published path component BASE_CONFIG's model resolves to: the name
# tail, not the absolute path dss hands the engine. Written out rather than
# derived from BASE_CONFIG, so these tests assert the intended value instead of
# re-implementing _model_slug and passing even if it were wrong.
SLUG = "Qwen3.6-35B-A3B"


def _in_tmp(body):
    """Run ``body(tmpdir)`` against a fresh temporary directory."""
    tmp = tempfile.mkdtemp(prefix="semip-image-source-")
    try:
        return body(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write(path, text="x", mode=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    if mode is not None:
        os.chmod(path, mode)
    return path


def _fake_published(tmp, *, model_dir, verified=True, weights=True, uid=None,
                    env_files=None, tp=1, verified_at=None, marker_text=None,
                    weight_hash=WEIGHT_HASH, key=KEY,
                    weights_verified=True, weights_verified_at=None):
    """Build a plausible published skeleton the way the DaemonSet leaves one.

    Shaped like the real tree rather than the old flat one: the skeleton holds
    only the two path-bound directories and names its weights in its own name,
    while the shards sit in a sibling ``weight/<wt12>/`` that other skeletons of
    the same model may share.
    """
    slug = SLUG
    root = os.path.join(tmp, "mirror", slug)
    source_dir = os.path.join(root, se._SKELETON_DIR,
                              f"{key}_{weight_hash}")
    meta = {
        "model_dir": model_dir,
        "uid": os.getuid() if uid is None else uid,
        "gpus": [0],
        "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000000"],
        "vllm_config": dict(BASE_CONFIG),
        "image_ref": "dss@sha256:abc",
        "driver_version": "580.159.03",
    }
    if env_files is not None:
        meta["env_files"] = env_files
    _write(os.path.join(source_dir, "image", "meta.json"), json.dumps(meta))
    _write(os.path.join(source_dir, "image", "files.img"), "not really criu")
    _write(os.path.join(source_dir, "compilation", "triton", "kernel.json"),
           "{}")
    if weights:
        tail = ("weights_meta.json" if tp == 1
                else os.path.join("rank0", "weights_meta.json"))
        weights_dir = os.path.join(root, se._WEIGHT_DIR, weight_hash)
        _write(os.path.join(weights_dir, tail), "{}")
        if weights_verified:
            # The weight directory carries its own marker, stamped by its own
            # pass: the daemon verifies it independently of the skeleton, so the
            # fixture has to as well or nothing here resembles a real node.
            _write(os.path.join(weights_dir, se._VERIFIED_MARKER), json.dumps({
                "manifest_digest": "e" * 64,
                "verified_at": (time.time() if weights_verified_at is None
                                else weights_verified_at),
            }))
    if verified:
        # Shaped like the daemon's own marker, and written last, as it writes
        # it -- so by default nothing under the directory is newer than the
        # stamp and the freshness check passes.
        body = marker_text if marker_text is not None else json.dumps({
            "manifest_digest": "d" * 64,
            "verified_at": time.time() if verified_at is None else verified_at,
        })
        _write(os.path.join(source_dir, se._VERIFIED_MARKER), body)
    return source_dir


def _fresh_model_dir(tmp):
    model_dir = os.path.join(tmp, "cache", KEY)
    os.makedirs(model_dir)
    return model_dir


def _is_miss(model_dir):
    """What restore_and_wrap's hit predicate would say about ``model_dir``."""
    return not os.path.isfile(os.path.join(model_dir, "image", "meta.json"))


def _weight_root(source_dir):
    """The ``weight/`` sibling of a published skeleton."""
    return os.path.join(
        os.path.dirname(os.path.dirname(source_dir)), se._WEIGHT_DIR)


def _weight_hash_of(source_dir):
    """The hash a skeleton's name carries, the way the engine parses it."""
    name = os.path.basename(source_dir)
    return name.rsplit("_", 1)[1] if name.count("_") >= 2 else None


def _materialize(source_dir, model_dir):
    """``_materialize_from_source`` with the roots the published layout implies.

    The weights are no longer inside the directory being copied, so every call
    has to supply where they live -- which is the whole point of the split.
    """
    return se._materialize_from_source(
        source_dir, model_dir, _weight_root(source_dir),
        _weight_hash_of(source_dir))


def test_materialize_without_a_source_is_a_miss():
    """No SEMIP_IMAGE_SOURCE is a supported deployment, not an error."""
    assert se._materialize_from_source(None, "/cache/key") is False


def test_materialize_tolerates_a_source_that_was_never_published():
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        missing = os.path.join(tmp, "mirror", KEY)
        assert _materialize(missing, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_refuses_an_unverified_directory():
    """The marker is the whole safety argument: without it, a sync may be mid-flight."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir, verified=False)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_refuses_a_directory_being_resynced():
    """A stale marker over fresh bytes is the one case presence cannot catch.

    The daemon writes the marker after the payload, so a re-sync landing new
    bytes under the *previous* pass's marker reads as verified while it is half
    overwritten. This is the only decline on the accept side of the gate: miss
    it and the job fails inside CRIU rather than cold-starting.
    """
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir,
                                 verified_at=time.time() - 3600)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_accepts_a_settled_directory():
    """Nothing newer than the stamp: the sync finished before it was written."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir)
        assert _materialize(source, model_dir) is True
    _in_tmp(body)


def test_materialize_refuses_unverified_weights():
    """A verified skeleton vouches for the skeleton, and for nothing else.

    The daemon discovers and verifies the two directories independently, and a
    skeleton has been observed stamped while its own weights were still
    downloading. ``weights_meta.json`` is an ordinary payload file in an
    unordered s5cmd batch -- and the worst one to gate on, since it sorts after
    ``shard_*`` and is thousands of times smaller -- so its presence is no
    evidence the shards arrived.
    """
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir,
                                 weights_verified=False)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_refuses_weights_being_resynced():
    """Stale marker over fresh shards, the weight-side twin of the skeleton case."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir,
                                 weights_verified_at=time.time() - 3600)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_restore_weights_dir_refuses_unverified_weights():
    """The other caller gates on the same answer, so it declines the same way.

    This is the one that matters most: it is reached on a *hit*, where nothing
    else is going to look at the shards before load_weights reads them at their
    recorded offsets.
    """
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir,
                                 weights_verified=False)
        assert se._weights_dir_for_restore(
            model_dir, _weight_root(source), WEIGHT_HASH) is None
    _in_tmp(body)


def test_materialize_falls_back_to_presence_for_an_old_marker():
    """An older daemon writes no timestamp; keep the weaker gate, not none."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir, marker_text="")
        assert se._verified_at(os.path.join(source, se._VERIFIED_MARKER)) is None
        assert _materialize(source, model_dir) is True
    _in_tmp(body)


def test_materialize_refuses_an_image_dumped_under_another_model_dir():
    """image/ and compilation/ bake the path; criu_restore rejects any other."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir="/somewhere/else/" + KEY)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_refuses_an_image_dumped_by_another_uid():
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir,
                                 uid=os.getuid() + 1)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_refuses_a_publish_that_withheld_the_weights():
    """We never copy weights, so a weightless publish is unrestorable."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir, weights=False)
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_refuses_on_a_recorded_size_mismatch():
    """env_files finally earns its keep: refuse before spending the copy."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        lib = _write(os.path.join(tmp, "usr", "libcuda.so.580"), "12345")
        source = _fake_published(tmp, model_dir=model_dir,
                                 env_files=[[lib, 999999, None]])
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
    _in_tmp(body)


def test_materialize_accepts_a_matching_environment():
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        lib = _write(os.path.join(tmp, "usr", "libcuda.so.580"), "12345")
        source = _fake_published(tmp, model_dir=model_dir,
                                 env_files=[[lib, 5, None]])
        assert _materialize(source, model_dir) is True
    _in_tmp(body)


def test_materialize_copies_the_bound_directories_and_not_the_weights():
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir)
        assert _materialize(source, model_dir) is True
        assert os.path.isfile(os.path.join(model_dir, "image", "meta.json"))
        assert os.path.isfile(
            os.path.join(model_dir, "compilation", "triton", "kernel.json"))
        # 66 GB stays on the mirror and is read in place.
        assert not os.path.exists(os.path.join(model_dir, "weight"))
        # And the staging directory does not outlive a success.
        assert not os.path.exists(os.path.join(model_dir, se._INCOMING_NAME))
    _in_tmp(body)


def test_materialize_preserves_file_modes():
    """CRIU re-validates the recorded mode of every path it re-maps."""
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir)
        _write(os.path.join(source, "compilation", "private.so"), "x",
               mode=0o640)
        assert _materialize(source, model_dir) is True
        copied = os.path.join(model_dir, "compilation", "private.so")
        assert stat.S_IMODE(os.stat(copied).st_mode) == 0o640
    _in_tmp(body)


def test_materialize_repairs_a_mode_the_publish_flattened():
    """The real cross-node failure: S3 has no mode, so 0755 comes back 0644.

    Preserving the mirror's mode is not enough when the mirror's mode is
    already wrong; the image's recorded mode is the only surviving original.
    """
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir)
        # As the DaemonSet leaves it: executable bit gone.
        _write(os.path.join(source, "compilation", "sampling.so"), "kernel",
               mode=0o644)
        dumped = os.path.join(model_dir, "compilation", "sampling.so")
        meta_path = os.path.join(source, "image", "meta.json")
        with open(meta_path) as handle:
            meta = json.load(handle)
        meta["env_files"] = [[dumped, 6, None, 0o100755]]
        _write(meta_path, json.dumps(meta))

        assert _materialize(source, model_dir) is True
        assert stat.S_IMODE(os.stat(dumped).st_mode) == 0o755
        # The mirror is shared and readOnly to us; only our copy was touched.
        mirrored = os.path.join(source, "compilation", "sampling.so")
        assert stat.S_IMODE(os.stat(mirrored).st_mode) == 0o644
    _in_tmp(body)


def test_materialize_flips_the_image_last():
    """An interrupted materialize must read as a miss, not as a broken hit.

    A plain file where ``image/`` has to go makes the final ``os.replace``
    fail with ENOTDIR, which stands in for any interruption -- a killed pod, a
    full disk. ``compilation/`` is already flipped by then, which is what makes
    this the ordering test as well.
    """
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir)
        _write(os.path.join(model_dir, "image"), "in the way")
        assert _materialize(source, model_dir) is False
        assert _is_miss(model_dir)
        assert os.path.isdir(os.path.join(model_dir, "compilation"))
        assert not os.path.exists(os.path.join(model_dir, se._INCOMING_NAME))
    _in_tmp(body)


# --------------------------------------------------------------------------
# _check_env_files
# --------------------------------------------------------------------------


def test_meta_json_writes_one_mapping_per_line():
    """Five lines per mapping at indent=2 is ~2000 lines of paths; this is 1."""
    meta = {
        "model_dir": "/data-fast/image-cache/aaa_bbb",
        "pid_floor": 100001,
        "env_files": [["/usr/lib/libc.so.6", 2125328, "abc123"],
                      ["/usr/lib/libm.so.6", 940000, None]],
    }
    text = se._meta_json_text(meta)
    assert text == '\n'.join([
        '{',
        '  "model_dir": "/data-fast/image-cache/aaa_bbb",',
        '  "pid_floor": 100001,',
        '  "env_files": [',
        '    ["/usr/lib/libc.so.6", 2125328, "abc123"],',
        '    ["/usr/lib/libm.so.6", 940000, null]',
        '  ]',
        '}',
    ]), text
    assert json.loads(text) == meta, "must round-trip unchanged"
    # The point of it: five lines per mapping becomes one.
    assert len(text.splitlines()) < len(
        json.dumps(meta, indent=2).splitlines())


def test_meta_json_is_an_ordinary_dump_without_mappings():
    for meta in ({"model_dir": "/d", "env_files": []}, {"model_dir": "/d"}):
        assert se._meta_json_text(meta) == json.dumps(meta, indent=2)


def test_meta_json_is_not_confused_by_a_path_that_looks_like_structure():
    """The splice never matches on dumped text, so contents cannot mislead it."""
    meta = {"model_dir": '/d"}', "env_files": [['/x"}', 1, None]]}
    assert json.loads(se._meta_json_text(meta)) == meta


def test_check_env_files_reports_a_size_mismatch():
    def body(tmp):
        target = _write(os.path.join(tmp, "usr", "lib.so"), "12345")
        meta = {"env_files": [[target, 999, None]]}
        fatal = se._check_env_files(meta, os.path.join(tmp, "cache", KEY),
                                    under_model_dir=False)
        assert len(fatal) == 1 and target in fatal[0], fatal
    _in_tmp(body)


def test_check_env_files_treats_a_missing_mapping_as_advisory():
    """Ghosted and recreated files explain an absence; a wrong size does not."""
    meta = {"env_files": [["/nonexistent/ghost.so", 10, None]]}
    assert se._check_env_files(meta, "/cache/" + KEY,
                               under_model_dir=False) == []


def test_check_env_files_scopes_on_the_model_dir():
    """The environment is checkable before the copy, the copy only after it."""
    def body(tmp):
        model_dir = os.path.join(tmp, "cache", KEY)
        inside = _write(os.path.join(model_dir, "compilation", "k.so"), "abc")
        outside = _write(os.path.join(tmp, "usr", "lib.so"), "abc")
        meta = {"env_files": [[inside, 1, None], [outside, 2, None]]}

        env_half = se._check_env_files(meta, model_dir, under_model_dir=False)
        copied_half = se._check_env_files(meta, model_dir, under_model_dir=True)
        assert len(env_half) == 1 and outside in env_half[0], env_half
        assert len(copied_half) == 1 and inside in copied_half[0], copied_half
    _in_tmp(body)


def test_check_env_files_accepts_an_image_that_recorded_none():
    assert se._check_env_files({}, "/cache/" + KEY,
                               under_model_dir=False) == []


# --------------------------------------------------------------------------
# _apply_recorded_modes
# --------------------------------------------------------------------------


def test_apply_recorded_modes_never_touches_a_path_outside_the_scope():
    """The safety property: env_files also lists /usr, which is not ours.

    A publish cannot flatten the environment's own libraries -- they never went
    through S3 -- so there is nothing here to repair and every reason not to
    try.
    """
    def body(tmp):
        scope = os.path.join(tmp, "cache", KEY, "compilation")
        outside = _write(os.path.join(tmp, "usr", "libcuda.so.580"), "12345",
                         mode=0o644)
        meta = {"env_files": [[outside, 5, None, 0o100755]]}
        assert se._apply_recorded_modes(meta, scope) == 0
        assert stat.S_IMODE(os.stat(outside).st_mode) == 0o644
    _in_tmp(body)


def test_apply_recorded_modes_skips_an_image_dumped_before_modes():
    """Three-element rows predate the mode column and must not crash."""
    def body(tmp):
        scope = os.path.join(tmp, "compilation")
        target = _write(os.path.join(scope, "kernel.so"), "x", mode=0o644)
        meta = {"env_files": [[target, 1, "abc123"]]}
        assert se._apply_recorded_modes(meta, scope) == 0
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o644
    _in_tmp(body)


def test_apply_recorded_modes_is_a_no_op_when_the_mode_already_matches():
    def body(tmp):
        scope = os.path.join(tmp, "compilation")
        target = _write(os.path.join(scope, "kernel.so"), "x", mode=0o755)
        meta = {"env_files": [[target, 1, None, 0o100755]]}
        assert se._apply_recorded_modes(meta, scope) == 0
    _in_tmp(body)


def test_apply_recorded_modes_does_not_follow_a_symlink():
    """chmod follows symlinks, and a link's target can sit outside the scope."""
    def body(tmp):
        scope = os.path.join(tmp, "compilation")
        outside = _write(os.path.join(tmp, "usr", "libcuda.so.580"), "12345",
                         mode=0o644)
        link = os.path.join(scope, "libcuda.so")
        os.makedirs(scope, exist_ok=True)
        os.symlink(outside, link)
        meta = {"env_files": [[link, 5, None, 0o100755]]}
        assert se._apply_recorded_modes(meta, scope) == 0
        assert stat.S_IMODE(os.stat(outside).st_mode) == 0o644
    _in_tmp(body)


def test_apply_recorded_modes_survives_a_path_it_cannot_chmod():
    """Best-effort: a failure costs a cold start, never an exception."""
    def body(tmp):
        scope = os.path.join(tmp, "compilation")
        good = _write(os.path.join(scope, "kernel.so"), "x", mode=0o644)
        absent = os.path.join(scope, "vanished.so")
        meta = {"env_files": [[absent, 1, None, 0o100755],
                              [good, 1, None, 0o100755]]}
        # The survivor is still repaired; the casualty is logged, not raised.
        assert se._apply_recorded_modes(meta, scope) == 1
        assert stat.S_IMODE(os.stat(good).st_mode) == 0o755
    _in_tmp(body)


def test_apply_recorded_modes_ignores_the_file_type_bits():
    """Recorded modes are CRIU's st_mode; only the permission bits are ours."""
    def body(tmp):
        scope = os.path.join(tmp, "compilation")
        target = _write(os.path.join(scope, "kernel.so"), "x", mode=0o644)
        meta = {"env_files": [[target, 1, None, 0o100755]]}
        assert se._apply_recorded_modes(meta, scope) == 1
        assert stat.S_IMODE(os.stat(target).st_mode) == 0o755
    _in_tmp(body)


# --------------------------------------------------------------------------
# Where the weights come from
# --------------------------------------------------------------------------


def test_has_weights_accepts_both_layouts():
    def body(tmp):
        tp1 = os.path.join(tmp, "tp1")
        _write(os.path.join(tp1, "weight", "weights_meta.json"), "{}")
        tp2 = os.path.join(tmp, "tp2")
        _write(os.path.join(tp2, "weight", "rank0", "weights_meta.json"), "{}")
        bare = os.path.join(tmp, "bare")
        os.makedirs(os.path.join(bare, "weight"))
        assert se._has_weights(tp1)
        assert se._has_weights(tp2)
        assert not se._has_weights(bare)
    _in_tmp(body)


def test_weights_dir_prefers_the_local_copy():
    """A dump of our own, or a hit: Instance's default is already right."""
    def body(tmp):
        model_dir = os.path.join(tmp, "cache", KEY)
        _write(os.path.join(model_dir, "weight", "weights_meta.json"), "{}")
        source = _fake_published(tmp, model_dir=model_dir)
        assert se._weights_dir_for_restore(
            model_dir, _weight_root(source), WEIGHT_HASH) is None
    _in_tmp(body)


def test_weights_dir_reads_the_hash_the_skeleton_names():
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir)
        assert (se._weights_dir_for_restore(
            model_dir, _weight_root(source), WEIGHT_HASH)
            == os.path.join(_weight_root(source), WEIGHT_HASH))
    _in_tmp(body)


def test_weights_dir_is_none_for_a_hash_that_was_unpublished():
    """A dangling reference is a miss, not a failure.

    This is the hazard the split introduces: weights are shared, so retiring a
    hash can orphan a skeleton that still names it. The restore has to
    cold-start rather than fail, and say so.
    """
    def body(tmp):
        model_dir = _fresh_model_dir(tmp)
        source = _fake_published(tmp, model_dir=model_dir, weights=False)
        assert se._weights_dir_for_restore(
            model_dir, _weight_root(source), WEIGHT_HASH) is None
    _in_tmp(body)


def test_weights_dir_is_none_when_nothing_has_them():
    """Let load_weights raise naming the local path it expected."""
    def body(tmp):
        assert se._weights_dir_for_restore(
            _fresh_model_dir(tmp), None, None) is None
    _in_tmp(body)


# --------------------------------------------------------------------------
# _resolve_published_skeleton: the weight hash is discovered, never derived
# --------------------------------------------------------------------------


def _paths_for(tmp, *, slug=None):
    """An ``_ImagePaths`` pointing at ``tmp``'s fake mirror."""
    slug = SLUG if slug is None else slug
    root = os.path.join(tmp, "mirror", slug)
    return se._ImagePaths(
        model_dir=os.path.join(tmp, "cache", KEY),
        key_prefix=KEY,
        skeleton_root=os.path.join(root, se._SKELETON_DIR),
        weight_root=os.path.join(root, se._WEIGHT_DIR),
        image_ref="img@sha256:abc",
        driver_version="580.1",
    )


def test_resolve_skeleton_finds_the_one_match_and_its_hash():
    def body(tmp):
        model_dir = os.path.join(tmp, "cache", KEY)
        source = _fake_published(tmp, model_dir=model_dir)
        found, wt = se._resolve_published_skeleton(_paths_for(tmp))
        assert found == source
        assert wt == WEIGHT_HASH
    _in_tmp(body)


def test_resolve_skeleton_is_a_miss_with_nothing_published():
    def body(tmp):
        found, wt = se._resolve_published_skeleton(_paths_for(tmp))
        assert (found, wt) == (None, None)
    _in_tmp(body)


def test_resolve_skeleton_refuses_to_choose_between_two_weight_versions():
    """Two versions coexisting is the rollback; picking one is not ours to do.

    A tie-break here would silently change which weights a job serves, so the
    refusal cold-starts instead -- the same shape as every other decline on
    this path.
    """
    def body(tmp):
        model_dir = os.path.join(tmp, "cache", KEY)
        _fake_published(tmp, model_dir=model_dir, weight_hash="a" * 12)
        _fake_published(tmp, model_dir=model_dir, weight_hash="b" * 12)
        found, wt = se._resolve_published_skeleton(_paths_for(tmp))
        assert (found, wt) == (None, None)
    _in_tmp(body)


def test_resolve_skeleton_ignores_another_configs_skeletons():
    """The prefix is the whole match: a different cfg or env is a different key."""
    def body(tmp):
        model_dir = os.path.join(tmp, "cache", KEY)
        other = "1234567890ab_ba0987654321"
        _fake_published(tmp, model_dir=model_dir, key=other)
        found, wt = se._resolve_published_skeleton(_paths_for(tmp))
        assert (found, wt) == (None, None)
    _in_tmp(body)


def test_resolve_skeleton_skips_a_presplit_directory():
    """A flat pre-split key carries no hash, so it cannot be resolved."""
    def body(tmp):
        paths = _paths_for(tmp)
        os.makedirs(os.path.join(paths.skeleton_root, KEY))
        found, wt = se._resolve_published_skeleton(paths)
        assert (found, wt) == (None, None)
    _in_tmp(body)


def test_resolve_skeleton_without_a_mirror_is_a_miss():
    paths = se._ImagePaths("/cache/" + KEY, KEY, None, None, "img", "580.1")
    assert se._resolve_published_skeleton(paths) == (None, None)


# --------------------------------------------------------------------------
# _resolve_model_dir, source half
# --------------------------------------------------------------------------


def _with_env(**values):
    """Set env vars for one call (``None`` unsets one), restoring them afterwards."""
    def decorate(fn):
        def wrapper():
            original = {k: os.environ.get(k) for k in values}
            for k, v in values.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            stub = se._env_hash
            se._env_hash = lambda: ("eeeeeeeeeeee", "img@sha256:abc", "580.1")
            try:
                return fn()
            finally:
                se._env_hash = stub
                for key, was in original.items():
                    if was is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = was
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorate


@_with_env(SEMIP_IMAGE_CACHE="/data-fast/image-cache_neutrino",
           SEMIP_IMAGE_SOURCE="/mnt/neutrino/base-models/image-cache")
def test_resolve_model_dir_composes_the_published_roots():
    """The local key stays flat; the mirror groups by model and splits the two.

    The local name cannot carry the weight hash -- model_dir is fixed before the
    weights exist -- so the two layouts differ by design, and this is the seam
    that lets one weight directory serve several backend images.
    """
    paths = se._resolve_model_dir(BASE_CONFIG)
    slug = SLUG
    base = "/mnt/neutrino/base-models/image-cache"
    assert os.path.basename(paths.model_dir) == paths.key_prefix
    # Spelled out rather than taken from the constants: this is the wire
    # contract the bucket and the node DaemonSet already hold, so it has to
    # fail when the constants move, not follow them.
    assert paths.skeleton_root == os.path.join(base, slug, "skeleton")
    assert paths.weight_root == os.path.join(base, slug, "weight")


@_with_env(SEMIP_IMAGE_CACHE="/data-fast/image-cache_neutrino",
           SEMIP_IMAGE_SOURCE="")
def test_resolve_model_dir_has_no_source_when_set_empty():
    """An empty source is the A/B switch: cold-start on a miss."""
    paths = se._resolve_model_dir(BASE_CONFIG)
    assert paths.skeleton_root is None
    assert paths.weight_root is None


@_with_env(SEMIP_IMAGE_CACHE=None, SEMIP_IMAGE_SOURCE=None)
def test_resolve_model_dir_defaults_both_roots_when_unset():
    """A job that says only semi_p: true gets the operator's two mounts."""
    paths = se._resolve_model_dir(BASE_CONFIG)
    assert os.path.dirname(paths.model_dir) == "/data-fast/image-cache_neutrino"
    assert paths.skeleton_root == os.path.join(
        "/mnt/neutrino/base-models/image-cache", SLUG, "skeleton")


@_with_env(SEMIP_IMAGE_CACHE="/elsewhere/cache",
           SEMIP_IMAGE_SOURCE="/elsewhere/mirror")
def test_resolve_model_dir_env_still_overrides_the_defaults():
    paths = se._resolve_model_dir(BASE_CONFIG)
    assert os.path.dirname(paths.model_dir) == "/elsewhere/cache"
    assert paths.skeleton_root.startswith("/elsewhere/mirror/")


@_with_env(SEMIP_IMAGE_CACHE="/", SEMIP_IMAGE_SOURCE=None)
def test_resolve_model_dir_refuses_a_root_that_names_no_directory():
    _expect_raises(lambda: se._resolve_model_dir(BASE_CONFIG), ValueError)


@_with_env(SEMIP_IMAGE_CACHE="/data-fast/image-cache_neutrino",
           SEMIP_IMAGE_SOURCE="/data-fast/image-cache_neutrino/")
def test_resolve_model_dir_drops_a_source_equal_to_the_cache():
    """A directory cannot be materialized from itself."""
    paths = se._resolve_model_dir(BASE_CONFIG)
    assert paths.skeleton_root is None
    assert paths.weight_root is None


@_with_env(SEMIP_IMAGE_CACHE="/data-fast/image-cache_neutrino",
           SEMIP_IMAGE_SOURCE="/mnt/neutrino/base-models/image-cache")
def test_resolve_model_dir_has_no_source_without_a_model():
    """No model means no address in the published tree, so treat it as a miss."""
    paths = se._resolve_model_dir(
        {k: v for k, v in BASE_CONFIG.items() if k != "model"})
    assert paths.skeleton_root is None
    assert paths.weight_root is None


# --------------------------------------------------------------------------


def _expect_raises(fn, exc_type):
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # noqa: BLE001 - report the wrong type clearly
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}")
    raise AssertionError(f"expected {exc_type.__name__}, nothing raised")


if __name__ == "__main__":
    tests = [
        test_config_hash_is_order_independent,
        test_config_hash_is_stable_across_calls,
        test_config_hash_covers_gpu_memory_utilization,
        test_config_hash_covers_tensor_parallel_size,
        test_config_hash_does_not_normalize_model_path,
        test_config_hash_length,
        test_config_hash_separates_two_device_allocations,
        test_config_hash_ignores_device_order,
        test_config_hash_without_devices_is_the_bare_config,
        test_device_binding_is_none_at_tp1,
        test_device_binding_reads_the_pod_allocation_at_tp2,
        test_device_binding_threshold_matches_the_restore_check,
        test_resolve_model_dir_separates_allocations_of_one_config,
        test_resolve_model_dir_is_stable_for_one_allocation,
        test_the_two_visible_device_nodes_copies_agree,
        test_driver_version_parses_nvrm_line,
        test_driver_version_raises_on_unparseable,
        test_resolve_model_dir_composes_both_hashes,
        test_resolve_model_dir_strips_trailing_slash,
        test_resolve_model_dir_requires_the_cache_root,
        test_resolve_model_dir_separates_configs_under_one_env,
        test_pod_image_ref_matches_the_named_container,
        test_pod_image_ref_raises_rather_than_guessing,
        test_pod_image_ref_rejects_a_digestless_image_id,
        test_materialize_without_a_source_is_a_miss,
        test_materialize_tolerates_a_source_that_was_never_published,
        test_materialize_refuses_an_unverified_directory,
        test_materialize_refuses_a_directory_being_resynced,
        test_materialize_accepts_a_settled_directory,
        test_materialize_falls_back_to_presence_for_an_old_marker,
        test_materialize_refuses_an_image_dumped_under_another_model_dir,
        test_materialize_refuses_an_image_dumped_by_another_uid,
        test_materialize_refuses_a_publish_that_withheld_the_weights,
        test_materialize_refuses_on_a_recorded_size_mismatch,
        test_materialize_accepts_a_matching_environment,
        test_materialize_copies_the_bound_directories_and_not_the_weights,
        test_materialize_preserves_file_modes,
        test_materialize_flips_the_image_last,
        test_meta_json_writes_one_mapping_per_line,
        test_meta_json_is_an_ordinary_dump_without_mappings,
        test_meta_json_is_not_confused_by_a_path_that_looks_like_structure,
        test_check_env_files_reports_a_size_mismatch,
        test_check_env_files_treats_a_missing_mapping_as_advisory,
        test_check_env_files_scopes_on_the_model_dir,
        test_check_env_files_accepts_an_image_that_recorded_none,
        test_has_weights_accepts_both_layouts,
        test_weights_dir_prefers_the_local_copy,
        test_weights_dir_falls_back_to_the_mirror,
        test_weights_dir_is_none_when_nothing_has_them,
        test_resolve_model_dir_composes_the_source_from_the_same_key,
        test_resolve_model_dir_has_no_source_when_unset,
        test_resolve_model_dir_drops_a_source_equal_to_the_cache,
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
