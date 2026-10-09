"""Unit tests for the published skeleton/weight split in ``semip_publish.py``.

The property the whole split rests on is that **a new backend image re-uploads
no weights**. That is what the end-to-end test here asserts, by publishing the
same weights twice under two different env hashes and checking that the second
run issues no weights upload and lands on the same ``weight/<wt12>/``.

It is worth testing rather than assuming because it was measured, not designed:
16 of 16 cross-env pairs in the published cache hold byte-identical weights. The
hash is what turns that observation into something the tooling can rely on per
dump, so the hash's own properties -- order independence, path sensitivity --
are tested too.

Two failure modes get their own tests because both are silent in production.
A sentinel-vs-prefix confusion would skip an interrupted upload and publish a
skeleton over half a weight directory. And a manifest whose paths are not
relative to the directory that holds it fails *verification* on every node
rather than failing the publish, so the directory simply never becomes usable.

No AWS, no network, no pod: ``_aws`` and ``_published`` are the seams, and the
module is plain stdlib so it imports directly.

Run from the package directory::

    python -m pytest tests/test_publish_layout.py -v

Or directly::

    python tests/test_publish_layout.py
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
_PUBLISH = os.path.join(_PKG, "scripts", "semip_publish.py")


def _load_publish():
    """Import ``semip_publish`` by path, without importing the package."""
    spec = importlib.util.spec_from_file_location("semip_publish", _PUBLISH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["semip_publish"] = mod
    spec.loader.exec_module(mod)
    return mod


sp = _load_publish()

CFG = "77d95928d2ac"
ENV_A = "1fad1e633806"
ENV_B = "84490f235789"
SLUG = "Qwen3.8-27B"


def _in_tmp(body):
    tmp = tempfile.mkdtemp(prefix="semip-publish-")
    try:
        return body(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _write(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        handle.write(text)
    return path


def _model_dir(tmp, env_hash, *, weights_text="shard-bytes", tp=1,
               model=SLUG):
    """A dumped directory, shaped the way the local cache holds one."""
    key = f"{CFG}_{env_hash}"
    model_dir = os.path.join(tmp, "cache", key)
    meta = {
        "model_dir": model_dir,
        "uid": os.getuid(),
        "vllm_config": {"model": model, "tensor_parallel_size": tp},
        "image_ref": "dss@sha256:abc",
        "driver_version": "580.159.03",
    }
    _write(os.path.join(model_dir, "image", "meta.json"), json.dumps(meta))
    _write(os.path.join(model_dir, "image", "files.img"), "criu-ish")
    _write(os.path.join(model_dir, "compilation", "triton", "k.json"), "{}")
    if tp == 1:
        _write(os.path.join(model_dir, sp.WEIGHT_DIR, "weights_meta.json"), "{}")
        _write(os.path.join(model_dir, sp.WEIGHT_DIR, "shard_0000.bin"),
               weights_text)
    else:
        for rank in range(tp):
            base = os.path.join(model_dir, sp.WEIGHT_DIR, f"rank{rank}")
            _write(os.path.join(base, "weights_meta.json"), "{}")
            _write(os.path.join(base, "shard_0000.bin"),
                   f"{weights_text}-r{rank}")
    return model_dir


class _Bucket:
    """Records uploads and answers the sentinel check from what it recorded."""

    def __init__(self):
        self.copies: list[tuple[str, ...]] = []
        self.removed: list[str] = []
        self.sentinels: set[str] = set()
        self.manifests: dict[str, dict] = {}
        self.staged: dict[str, dict[str, dict]] = {}
        self.cleared: list[str] = []

    def stage_put(self, bucket, key, name, obj, scratch):
        self.staged.setdefault(key, {})[name] = json.loads(json.dumps(obj))

    def stage_list(self, bucket, key):
        return dict(self.staged.get(key, {}))

    def stage_clear(self, bucket, key):
        self.cleared.append(key)
        self.staged.pop(key, None)

    def node_count(self, bucket: str, key: str):
        manifest = self.manifests.get(f"s3://{bucket}/{key}")
        return None if manifest is None else manifest.get("nodes", 1)

    def aws(self, *args: str) -> None:
        self.copies.append(args)
        if args[:2] == ("s3", "cp") and args[-1].endswith(
                sp.MANIFEST_FILENAME):
            # Mirrors the real invariant: a directory only counts as published
            # once its sentinel lands, which is written last.
            dest = args[-1].rsplit("/", 1)[0]
            self.sentinels.add(dest)
            with open(args[-2]) as handle:
                self.manifests[dest] = json.load(handle)
        if args[:2] == ("s3", "rm"):
            self.removed.append(args[-1])

    def published(self, bucket: str, key: str) -> bool:
        return f"s3://{bucket}/{key}" in self.sentinels

    def replica_count(self, bucket: str, key: str):
        manifest = self.manifests.get(f"s3://{bucket}/{key}")
        return None if manifest is None else manifest.get("replicas", 1)

    def uploaded_dirs(self) -> list[str]:
        return [a[-1] for a in self.copies
                if a[:2] == ("s3", "cp") and "--recursive" in a]


def _with_bucket(body):
    """Run *body(bucket)* with the AWS seams replaced."""
    bucket = _Bucket()
    originals = (sp._aws, sp._published, sp._skeleton_referrers, sp.image_digest,
                 sp._published_replica_count, sp._published_node_count,
                 sp._stage_put, sp._stage_list, sp._stage_clear,
                 sp._RENDEZVOUS_POLL_S)
    sp._aws = bucket.aws
    sp._published = bucket.published
    sp._skeleton_referrers = lambda *a, **k: []
    sp.image_digest = lambda: ("abc", "dss@sha256:abc")
    sp._published_replica_count = bucket.replica_count
    sp._published_node_count = bucket.node_count
    sp._stage_put = bucket.stage_put
    sp._stage_list = bucket.stage_list
    sp._stage_clear = bucket.stage_clear
    sp._RENDEZVOUS_POLL_S = 0.01
    try:
        return body(bucket)
    finally:
        (sp._aws, sp._published, sp._skeleton_referrers,
         sp.image_digest, sp._published_replica_count,
         sp._published_node_count, sp._stage_put, sp._stage_list,
         sp._stage_clear, sp._RENDEZVOUS_POLL_S) = originals


# --------------------------------------------------------------------------
# weights_hash
# --------------------------------------------------------------------------

def _rows(*pairs):
    return [{"path": p, "size_bytes": 1, "sha256": d} for p, d in pairs]


def test_weights_hash_ignores_the_order_it_is_given():
    """Filesystem walk order must not change a content hash.

    Two nodes hashing the same bytes have to agree, or the dedup never fires.
    """
    a = _rows(("weight/shard_0000.bin", "aa"), ("weight/shard_0001.bin", "bb"))
    assert sp.weights_hash(a) == sp.weights_hash(list(reversed(a)))


def test_weights_hash_tracks_the_bytes():
    a = _rows(("weight/shard_0000.bin", "aa"))
    b = _rows(("weight/shard_0000.bin", "ab"))
    assert sp.weights_hash(a) != sp.weights_hash(b)


def test_weights_hash_tracks_the_layout_not_just_the_bytes():
    """TP1's flat shards and TP>1's rank dirs must never share a hash.

    The paths differ before the bytes do, and the per-rank byte totals differ
    too, so a cross-TP mispairing cannot resolve to one weight directory.
    """
    flat = _rows(("weight/shard_0000.bin", "aa"))
    ranked = _rows(("weight/rank0/shard_0000.bin", "aa"))
    assert sp.weights_hash(flat) != sp.weights_hash(ranked)


def test_weights_hash_ignores_the_skeleton():
    """Only weights may move the hash, or a rebuilt skeleton would fork it."""
    base = _rows(("weight/shard_0000.bin", "aa"))
    with_skel = base + _rows(("image/files.img", "zz"),
                             ("compilation/k.json", "yy"))
    assert sp.weights_hash(base) == sp.weights_hash(with_skel)


def test_weights_hash_does_not_depend_on_the_directory_name():
    """The weight directory can be renamed without forking every hash.

    The prefix is the same constant on every row of every dump, so it
    discriminates nothing -- but while it was hashed, renaming the directory
    changed the hash of byte-identical weights. Unifying the local ``weights``
    with the published ``weight`` paid that fork once (one re-dump and one
    re-upload per model, 760 GB for GLM-5.3); this is what stops the next rename
    from paying it again.

    Also what makes the hash recomputable from a published weight manifest,
    whose paths are already relative to the directory holding it.
    """
    rows = _rows((f"{sp.WEIGHT_DIR}/rank0/shard_0000.bin", "aa"),
                 (f"{sp.WEIGHT_DIR}/rank1/shard_0000.bin", "bb"))
    before = sp.weights_hash(rows)
    original = sp.WEIGHT_DIR
    try:
        sp.WEIGHT_DIR = "shards"
        renamed = _rows(("shards/rank0/shard_0000.bin", "aa"),
                        ("shards/rank1/shard_0000.bin", "bb"))
        assert sp.weights_hash(renamed) == before
    finally:
        sp.WEIGHT_DIR = original


def test_weights_hash_is_none_without_weights():
    assert sp.weights_hash(_rows(("image/files.img", "zz"))) is None


def test_weights_hash_is_a_path_component_length():
    assert len(sp.weights_hash(_rows(("weight/s.bin", "aa")))) == sp.HASH_LEN


# --------------------------------------------------------------------------
# manifest scoping
# --------------------------------------------------------------------------

def test_weight_manifest_paths_are_relative_to_the_weight_dir():
    """The daemon verifies <dir>/<path>, and the shards sit at the dir's top."""
    files = _rows(("weight/shard_0000.bin", "aa"),
                  ("weight/rank0/weights_meta.json", "bb"),
                  ("image/files.img", "zz"))
    m = sp._scoped_manifest(files, (sp.WEIGHT_DIR,), strip=sp.WEIGHT_DIR)
    assert sorted(f["path"] for f in m["files"]) == [
        "rank0/weights_meta.json", "shard_0000.bin"]


def test_thread_count_cannot_change_the_manifest_or_the_hash():
    """Parallel hashing must be bit-identical to serial, at any width.

    The weight hash is a path component and a dedup key, so if it moved with the
    thread count two nodes would disagree about whether they hold the same
    weights -- and the manifest rows would stop being diffable between runs.
    """
    def body(tmp):
        md = _model_dir(tmp, ENV_A, tp=2)
        serial = sp.build_manifest(md_path := sp.Path(md),
                                   list(sp.SKELETON_DIRS) + [sp.WEIGHT_DIR],
                                   workers=1)
        for n in (2, 8, 32):
            par = sp.build_manifest(md_path,
                                    list(sp.SKELETON_DIRS) + [sp.WEIGHT_DIR],
                                    workers=n)
            assert par == serial, f"manifest differs at {n} threads"
            assert sp.weights_hash(par["files"]) == sp.weights_hash(
                serial["files"]), f"weight hash differs at {n} threads"
    _in_tmp(body)


def test_skeleton_manifest_keeps_its_two_directories_and_drops_weights():
    files = _rows(("image/files.img", "zz"),
                  ("compilation/k.json", "yy"),
                  ("weight/shard_0000.bin", "aa"))
    m = sp._scoped_manifest(files, sp.SKELETON_DIRS)
    assert sorted(f["path"] for f in m["files"]) == [
        "compilation/k.json", "image/files.img"]


# --------------------------------------------------------------------------
# derived_key
# --------------------------------------------------------------------------

def test_derived_key_accepts_the_flat_local_name():
    def body(tmp):
        md = _model_dir(tmp, ENV_A)
        meta = json.loads(
            open(os.path.join(md, "image", "meta.json")).read())
        assert sp.derived_key(sp.Path(md), meta) == f"{CFG}_{ENV_A}"
    _in_tmp(body)


def test_derived_key_refuses_a_renamed_directory():
    """A key that disagrees with the dump would publish where nothing resolves.

    Only the *name* is checked here, deliberately: a different parent is legal
    (publishing may run against a copy), and the full-path comparison that
    ``image/`` and ``compilation/`` actually depend on is
    ``_materialize_from_source``'s, on every consuming node.
    """
    def body(tmp):
        md = _model_dir(tmp, ENV_A)
        meta = json.loads(
            open(os.path.join(md, "image", "meta.json")).read())
        meta["model_dir"] = f"/somewhere/else/{CFG}_{ENV_B}"
        _expect_exit(lambda: sp.derived_key(sp.Path(md), meta))
    _in_tmp(body)


def test_derived_key_refuses_an_undrived_name():
    def body(tmp):
        md = os.path.join(tmp, "qwen_35b")
        os.makedirs(md)
        _expect_exit(lambda: sp.derived_key(sp.Path(md), {}))
    _in_tmp(body)


# --------------------------------------------------------------------------
# model_slug
# --------------------------------------------------------------------------

def test_model_slug_reads_the_dumped_config():
    assert sp.model_slug({"vllm_config": {"model": "/" + SLUG}}) == SLUG


def test_model_slug_keeps_only_the_name():
    """dss resolves model to an absolute path; publishing it whole would nest
    the mirror's own mount inside itself and fork the weight dir if it moved.

    The org goes too: one level reads better, and nothing load-bearing rides on
    it, since identity is in the hashes rather than the path.
    """
    assert sp.model_slug({"vllm_config": {
        "model": "/mnt/neutrino/base-models/Qwen/Qwen3.6-35B-A3B"}}
    ) == "Qwen3.6-35B-A3B"


def test_model_slug_refuses_a_config_without_a_model():
    _expect_exit(lambda: sp.model_slug({"vllm_config": {}}))


def test_model_slug_drops_traversal_segments():
    assert sp.model_slug({"vllm_config": {"model": "a/../b/c"}}) == "c"


# The table both implementations are checked against. Includes the real value
# observed in a live job, which is the case that motivated the tail.
_SLUG_CASES = [
    "/mnt/neutrino/base-models/Qwen/Qwen3.6-35B-A3B",
    "/mnt/neutrino/base-models/Qwen/Qwen3.8-27B",
    "Qwen/Qwen3.8-27B",
    "/Qwen/Qwen3.8-27B/",
    "bare-model-name",
    "a/b/c/d/e",
    "a/../b/c",
    "",
    None,
]


def test_engine_and_publish_agree_on_every_slug():
    """Publish writes where the engine reads, so the two must never diverge.

    A disagreement is not a crash: the engine resolves a directory nothing was
    published to, reports a miss, and cold-starts forever with nothing above
    WARNING. The two are duplicated because this script is stdlib-only, so this
    table is what stops drift merging unnoticed.
    """
    engine = _load_engine_slug()
    if engine is None:
        return  # vLLM absent: nothing to compare against on this machine
    for raw in _SLUG_CASES:
        mine = engine(raw)
        try:
            theirs = sp.model_slug({"vllm_config": {"model": raw}})
        except SystemExit:
            theirs = ""          # publish refuses where the engine returns ""
        assert mine == theirs, (
            f"slug mismatch for {raw!r}: engine {mine!r} != publish {theirs!r}")


def _load_engine_slug():
    """``semip_engine._model_slug``, lifted by AST so vLLM is not imported."""
    import ast
    path = os.path.join(os.path.dirname(_PKG), "server", "semip_engine.py")
    if not os.path.isfile(path):
        return None
    tree = ast.parse(open(path).read())
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_model_slug":
            exec(compile(ast.Module([node], []), path, "exec"), ns)
            return ns["_model_slug"]
    return None


# --------------------------------------------------------------------------
# Node-label derivation, pinned against the live cluster
# --------------------------------------------------------------------------

# Read off the 8 cache nodes on 2026-09-25, right after the first publish under
# the split. Each right-hand side is a label key the DaemonSet actually wrote and
# that was observed at value ``ready``. The last two are over 63 characters
# before truncation, so they exercise the ``[:55] + "-" + sha256[:7]`` branch --
# which is the half most likely to drift, and the half that silently breaks the
# readiness check if it does.
#
# The left-hand sides are the paths as published **that day**, before the org
# level was dropped and the two subdirs were pluralized. They are deliberately
# not updated to the current layout: what is pinned here is the sanitizer's
# behaviour on inputs the cluster was seen to answer for, and rewriting the
# inputs would turn an observation into a guess. The layout the paths describe
# is irrelevant to the transform, which only ever sees a string.
_LIVE_LABELS = [
    ("image-cache/Qwen/Qwen3.8-27B/weight/7fd0c0cb3d49",
     "image-cache.Qwen.Qwen3.8-27B.weight.7fd0c0cb3d49"),
    ("image-cache/Qwen/Qwen3.6-35B-A3B/weight/ac012bb36e7a",
     "image-cache.Qwen.Qwen3.6-35B-A3B.weight.ac012bb36e7a"),
    ("image-cache/Qwen/Qwen3.8-27B/skeleton/"
     "552cc22540cd_1585beecd614_7fd0c0cb3d49",
     "image-cache.Qwen.Qwen3.8-27B.skeleton.552cc22540cd_1585-5533f5b"),
    ("image-cache/Qwen/Qwen3.6-35B-A3B/skeleton/"
     "77d95928d2ac_1585beecd614_ac012bb36e7a",
     "image-cache.Qwen.Qwen3.6-35B-A3B.skeleton.77d95928d2ac-7678595"),
]


def test_label_derivation_matches_the_live_cluster():
    """The readiness check is only as good as this replication.

    ``sync_node_cache._sanitize_label_segment`` lives in another repo and its own
    docstring requires consumers to reproduce it exactly. If it drifts, this
    script computes a label nothing carries and reports 0 nodes ready forever --
    a permanent false negative, not an error.
    """
    for rel, expected in _LIVE_LABELS:
        got = sp.sanitize_label_segment(rel)
        assert got == expected, f"{rel}\n  got      {got}\n  expected {expected}"
        assert len(got) <= 63


def test_label_derivation_handles_the_degenerate_cases():
    assert sp.sanitize_label_segment("a/b").startswith("a.b")
    # All-punctuation collapses to nothing, so it must still emit a legal key.
    weird = sp.sanitize_label_segment("///")
    assert weird.startswith("m.") and len(weird) <= 63
    # Leading and trailing non-alphanumerics are trimmed, not collapsed inward.
    assert sp.sanitize_label_segment("/x/") == "x"


# --------------------------------------------------------------------------
# The property the split exists for
# --------------------------------------------------------------------------

def test_a_second_backend_image_uploads_no_weights():
    """The whole point: same weights, new env hash, no shard re-upload.

    Today this costs a full weights copy per backend image on every one of the
    8 nodes -- about 560 GB for a 70 GB model, and ~5.7 TB for GLM-5.3 at TP8.
    """
    def body(tmp):
        def run(bucket):
            first = _model_dir(tmp, ENV_A)
            key_a = sp.publish(sp.Path(first), "bkt", "image-cache")
            weight_uploads_a = [d for d in bucket.uploaded_dirs()
                                if f"/{sp.WEIGHT_DIR}/" in d]
            assert len(weight_uploads_a) == 1, "first publish must send weights"

            before = len(bucket.copies)
            second = _model_dir(tmp, ENV_B)
            key_b = sp.publish(sp.Path(second), "bkt", "image-cache")
            new = bucket.copies[before:]
            weight_uploads_b = [
                a[-1] for a in new
                if a[:2] == ("s3", "cp") and f"/{sp.WEIGHT_DIR}/" in a[-1]]
            assert not weight_uploads_b, (
                f"second publish re-uploaded weights: {weight_uploads_b}")

            # Same weights, so the same weight directory -- and two skeletons,
            # because the backend image differs and its compiled artifacts do too.
            assert key_a.rsplit("_", 1)[1] == key_b.rsplit("_", 1)[1]
            assert key_a.startswith(f"{CFG}_{ENV_A}_")
            assert key_b.startswith(f"{CFG}_{ENV_B}_")
            skels = [d for d in bucket.uploaded_dirs()
                     if f"/{sp.SKELETON_DIR}/" in d]
            assert len({d.rsplit("/", 2)[0] for d in skels}) == 1, skels
        _with_bucket(run)
    _in_tmp(body)


def test_different_weights_get_their_own_directory():
    def body(tmp):
        def run(bucket):
            a = _model_dir(tmp, ENV_A, weights_text="one")
            b = _model_dir(tmp, ENV_B, weights_text="two")
            key_a = sp.publish(sp.Path(a), "bkt", "image-cache")
            key_b = sp.publish(sp.Path(b), "bkt", "image-cache")
            assert key_a.rsplit("_", 1)[1] != key_b.rsplit("_", 1)[1]
            weights = {d for d in bucket.uploaded_dirs()
                       if f"/{sp.WEIGHT_DIR}/" in d}
            assert len(weights) == 2, weights
        _with_bucket(run)
    _in_tmp(body)


def test_weights_land_before_the_skeleton_that_names_them():
    """Order is the invariant: a skeleton resolvable before its weights exist
    would make every node materialize an image it cannot load."""
    def body(tmp):
        def run(bucket):
            sp.publish(sp.Path(_model_dir(tmp, ENV_A)), "bkt", "image-cache")
            dirs = bucket.uploaded_dirs()
            first_weight = next(i for i, d in enumerate(dirs)
                                if f"/{sp.WEIGHT_DIR}/" in d)
            first_skel = next(i for i, d in enumerate(dirs)
                              if f"/{sp.SKELETON_DIR}/" in d)
            assert first_weight < first_skel, dirs
        _with_bucket(run)
    _in_tmp(body)


def test_a_partial_weight_upload_is_not_skipped():
    """A prefix without a sentinel is an interrupted upload, not a hit.

    ``publish`` writes the payload first and the sentinel last precisely so this
    case is distinguishable, and the check has to test the sentinel to benefit.
    """
    def body(tmp):
        def run(bucket):
            sp.publish(sp.Path(_model_dir(tmp, ENV_A)), "bkt", "image-cache")
            # Drop the sentinel, keeping the payload: the shape an interrupted
            # upload leaves behind.
            bucket.sentinels = {s for s in bucket.sentinels
                                if f"/{sp.WEIGHT_DIR}/" not in s}
            before = len(bucket.copies)
            sp.publish(sp.Path(_model_dir(tmp, ENV_B)), "bkt", "image-cache")
            resent = [a[-1] for a in bucket.copies[before:]
                      if f"/{sp.WEIGHT_DIR}/" in a[-1]]
            assert resent, "an unsentinelled weight dir must be re-uploaded"
        _with_bucket(run)
    _in_tmp(body)


def test_a_dry_run_reports_the_real_published_state():
    """A dry run is how --skeleton-only is judged safe, so it must probe S3.

    It used to hard-code the weights as absent, which warned that a published
    directory was missing -- the unsafe direction to be wrong in.
    """
    def body(tmp):
        def run(bucket):
            sp.publish(sp.Path(_model_dir(tmp, ENV_A)), "bkt", "image-cache")
            before = len(bucket.copies)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                sp.publish(sp.Path(_model_dir(tmp, ENV_B)), "bkt",
                           "image-cache", skeleton_only=True, dry_run=True)
            log = out.getvalue()
            assert "weights already published" in log, log
            assert "NOT published" not in log, log
            assert len(bucket.copies) == before, bucket.copies[before:]
        _with_bucket(run)
    _in_tmp(body)


def test_publishing_without_weights_refuses():
    """The engine reads weights in place, so a weightless dump is unpublishable."""
    def body(tmp):
        def run(bucket):
            md = _model_dir(tmp, ENV_A)
            shutil.rmtree(os.path.join(md, sp.WEIGHT_DIR))
            _expect_exit(
                lambda: sp.publish(sp.Path(md), "bkt", "image-cache"))
        _with_bucket(run)
    _in_tmp(body)


def test_unpublishing_weights_refuses_while_a_skeleton_needs_them():
    """Orphaning is silent -- a cold start, not an error -- so refuse up front."""
    def body(tmp):
        bucket = _Bucket()
        original = (sp._aws, sp._skeleton_referrers)
        sp._aws = bucket.aws
        sp._skeleton_referrers = lambda *a, **k: [f"{CFG}_{ENV_A}_beefbeefbeef"]
        try:
            _expect_exit(lambda: sp.unpublish_weights(
                "bkt", "image-cache", SLUG, "beefbeefbeef"))
            assert not bucket.removed, "refused, so nothing may be removed"
            # --force is the escape hatch for a leftover whose skeletons are gone.
            sp.unpublish_weights("bkt", "image-cache", SLUG, "beefbeefbeef",
                                 force=True)
            assert bucket.removed
        finally:
            sp._aws, sp._skeleton_referrers = original
    _in_tmp(body)


def test_unpublishing_a_skeleton_leaves_the_weights_alone():
    def body(tmp):
        def run(bucket):
            md = _model_dir(tmp, ENV_A)
            key = sp.publish(sp.Path(md), "bkt", "image-cache")
            bucket.removed.clear()
            sp.unpublish_skeleton(sp.Path(md), "bkt", "image-cache", key=key)
            assert bucket.removed
            assert not any(f"/{sp.WEIGHT_DIR}/" in r
                           for r in bucket.removed), bucket.removed
        _with_bucket(run)
    _in_tmp(body)


# --------------------------------------------------------------------------
# Several replicas in one pod: reconstructed and deduplicated at publish time
# --------------------------------------------------------------------------

def _replica_dump(tmp, env_hash, n, *, weights=None, slots=None):
    """``<key>/replica<K>/`` for each slot, the way a multi-replica pod dumps."""
    key = f"{CFG}_{env_hash}"
    root = os.path.join(tmp, "cache", key)
    for k in (range(n) if slots is None else slots):
        rdir = os.path.join(root, f"{sp.REPLICA_DIR_PREFIX}{k}")
        meta = {
            "model_dir": rdir,
            "uid": os.getuid(),
            "vllm_config": {"model": SLUG, "tensor_parallel_size": 1},
            "image_ref": "dss@sha256:abc",
            "driver_version": "580.159.03",
        }
        _write(os.path.join(rdir, "image", "meta.json"), json.dumps(meta))
        _write(os.path.join(rdir, "image", "core-1.img"), f"tree-{k}")
        _write(os.path.join(rdir, "compilation", "triton", "k.json"), "{}")
        _write(os.path.join(rdir, sp.WEIGHT_DIR, "weights_meta.json"), "{}")
        _write(os.path.join(rdir, sp.WEIGHT_DIR, "shard_0000.bin"),
               (weights or {}).get(k, "shard-bytes"))
    return root


def test_replica_weights_hash_like_a_flat_dump():
    """The replica level is stripped with the weight dir, so one copy names all."""
    flat = _rows(("weight/shard_0000.bin", "aa"), ("weight/weights_meta.json", "bb"))
    nested = _rows(("replica3/weight/shard_0000.bin", "aa"),
                   ("replica3/weight/weights_meta.json", "bb"))
    assert sp.weights_hash(nested, under="replica3/") == sp.weights_hash(flat)


def test_a_replica_dump_publishes_every_replica_and_one_weight_copy():
    def body(tmp):
        def run(bucket):
            md = _replica_dump(tmp, ENV_A, 3)
            key = sp.publish(sp.Path(md), "bkt", "image-cache")
            dirs = bucket.uploaded_dirs()
            weights = [d for d in dirs if f"/{sp.WEIGHT_DIR}/" in d]
            assert len(weights) == 1, weights
            skel = f"s3://bkt/image-cache/{SLUG}/{sp.SKELETON_DIR}/{key}"
            assert sorted(d for d in dirs if d.startswith(skel)) == sorted(
                f"{skel}/replica{k}/{sub}" for k in range(3)
                for sub in sp.SKELETON_DIRS)
            manifest = bucket.manifests[skel]
            assert manifest["replicas"] == 3
            assert {r["path"].split("/", 1)[0] for r in manifest["files"]} == {
                "replica0", "replica1", "replica2"}
            assert not any(f"/{sp.WEIGHT_DIR}/" in r["path"]
                           for r in manifest["files"])
            wt = bucket.manifests[weights[0]]
            assert sorted(r["path"] for r in wt["files"]) == [
                "shard_0000.bin", "weights_meta.json"]
        _with_bucket(run)
    _in_tmp(body)


def test_replicas_with_different_weights_refuse():
    def body(tmp):
        def run(bucket):
            md = _replica_dump(tmp, ENV_A, 2, weights={1: "other-bytes"})
            _expect_exit(lambda: sp.publish(sp.Path(md), "bkt", "image-cache"))
            assert not bucket.copies
        _with_bucket(run)
    _in_tmp(body)


def test_a_gap_in_the_replicas_refuses():
    """A pod of this shape could only half restore, and the engine fails it."""
    def body(tmp):
        md = _replica_dump(tmp, ENV_A, 0, slots=[0, 2])
        _expect_exit(lambda: sp.load_layout(sp.Path(md)))
    _in_tmp(body)


def test_a_replica_that_has_not_dumped_refuses():
    def body(tmp):
        md = _replica_dump(tmp, ENV_A, 2)
        os.remove(os.path.join(md, "replica1", "image", "meta.json"))
        _expect_exit(lambda: sp.load_layout(sp.Path(md)))
    _in_tmp(body)


def test_a_replica_recorded_under_another_slot_refuses():
    def body(tmp):
        md = _replica_dump(tmp, ENV_A, 2)
        path = os.path.join(md, "replica1", "image", "meta.json")
        meta = json.load(open(path))
        meta["model_dir"] = os.path.join(md, "replica0")
        _write(path, json.dumps(meta))
        _expect_exit(lambda: sp.load_layout(sp.Path(md)))
    _in_tmp(body)


def test_publishing_over_a_different_replica_count_refuses():
    def body(tmp):
        def run(bucket):
            three = sp.Path(_replica_dump(tmp, ENV_A, 3))
            sp.publish(three, "bkt", "image-cache")
            shutil.rmtree(three / "replica2")
            _expect_exit(lambda: sp.publish(three, "bkt", "image-cache"))
        _with_bucket(run)
    _in_tmp(body)


def test_a_flat_dump_is_unchanged_by_the_replica_layout():
    def body(tmp):
        md = sp.Path(_model_dir(tmp, ENV_A))
        key, _, replicas = sp.load_layout(md)
        assert key == f"{CFG}_{ENV_A}" and replicas == []
    _in_tmp(body)


# --------------------------------------------------------------------------
# One engine over several pods: each pod holds only its node<k>/ node-partition
# --------------------------------------------------------------------------

DUMP_ID = "4cc6c196ea1c4c59b59df9842c2ea85e"


def _node_pod(tmp, pod, k, *, nnodes=2, tp=4, dump_id=DUMP_ID,
              weights_text="shard-bytes", ranks=None):
    """One pod's ``<key>/node<k>/`` plus the ``weight/`` ranks it saved."""
    key = f"{CFG}_{ENV_A}"
    root = os.path.join(tmp, pod, "cache", key)
    ndir = os.path.join(root, f"{sp.NODE_DIR_PREFIX}{k}")
    meta = {
        # What a restore materializes into, which is the same on every pod.
        "model_dir": os.path.join("/data-fast/image-cache_neutrino", key,
                                  f"{sp.NODE_DIR_PREFIX}{k}"),
        "uid": os.getuid(),
        "vllm_config": {"model": SLUG, "tensor_parallel_size": tp,
                        "nnodes": nnodes},
        "image_ref": "dss@sha256:abc",
        "driver_version": "580.159.03",
        "nnodes": nnodes, "node_rank": k, "dump_id": dump_id,
    }
    _write(os.path.join(ndir, "image", "meta.json"), json.dumps(meta))
    _write(os.path.join(ndir, "image", "core-1.img"), f"tree-{k}")
    _write(os.path.join(ndir, "compilation", "triton", "k.json"), "{}")
    local = tp // nnodes
    for rank in (range(k * local, (k + 1) * local) if ranks is None else ranks):
        base = os.path.join(root, sp.WEIGHT_DIR, f"rank{rank}")
        _write(os.path.join(base, "weights_meta.json"), "{}")
        _write(os.path.join(base, "shard_0000.bin"), f"{weights_text}-r{rank}")
    return root


def _publish_pods(roots, **kw):
    """Publish every pod's node-partition concurrently, as the operator does."""
    import threading
    results, errors = {}, {}

    def run(i, root):
        try:
            results[i] = sp.publish(sp.Path(root), "bkt", "image-cache", **kw)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors[i] = exc
    threads = [threading.Thread(target=run, args=(i, r))
               for i, r in enumerate(roots)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    return results, errors


def test_two_pods_publish_one_skeleton_and_one_weight_directory():
    def body(tmp):
        def run(bucket):
            roots = [_node_pod(tmp, "pod-a", 0), _node_pod(tmp, "pod-b", 1)]
            results, errors = _publish_pods(roots)
            assert not errors, errors
            assert results[0] == results[1]
            key = results[0]
            skel = f"s3://bkt/image-cache/{SLUG}/{sp.SKELETON_DIR}/{key}"
            manifest = bucket.manifests[skel]
            assert manifest["nodes"] == 2 and manifest["dump_id"] == DUMP_ID
            assert {r["path"].split("/", 1)[0] for r in manifest["files"]} == {
                "node0", "node1"}
            assert not any(f"/{sp.WEIGHT_DIR}/" in r["path"]
                           for r in manifest["files"])
            wt_dest = [d for d in bucket.manifests if f"/{sp.WEIGHT_DIR}/" in d]
            assert len(wt_dest) == 1
            ranks = {r["path"].split("/", 1)[0]
                     for r in bucket.manifests[wt_dest[0]]["files"]}
            assert ranks == {f"rank{r}" for r in range(4)}
            # Each pod uploads its own shards into the one directory.
            assert [d for d in bucket.uploaded_dirs()
                    if f"/{sp.WEIGHT_DIR}/" in d] == wt_dest * 2
            assert sorted(d for d in bucket.uploaded_dirs()
                          if d.startswith(skel)) == sorted(
                f"{skel}/node{k}/{sub}" for k in range(2)
                for sub in sp.SKELETON_DIRS)
            assert bucket.cleared and not bucket.staged
        _with_bucket(run)
    _in_tmp(body)


def test_the_sentinels_follow_every_pods_uploads():
    """Node 0 writes them, after the other pod's done marker, weights first."""
    def body(tmp):
        def run(bucket):
            roots = [_node_pod(tmp, "pod-a", 0), _node_pod(tmp, "pod-b", 1)]
            _, errors = _publish_pods(roots)
            assert not errors, errors
            dests = [a[-1] for a in bucket.copies]
            sentinels = [i for i, a in enumerate(bucket.copies)
                         if a[:2] == ("s3", "cp")
                         and a[-1].endswith(sp.MANIFEST_FILENAME)]
            uploads = [i for i, a in enumerate(bucket.copies)
                       if "--recursive" in a]
            assert len(sentinels) == 2
            assert max(uploads) < min(sentinels)
            assert f"/{sp.WEIGHT_DIR}/" in dests[sentinels[0]]
            assert f"/{sp.SKELETON_DIR}/" in dests[sentinels[1]]
        _with_bucket(run)
    _in_tmp(body)


def test_the_weight_hash_spans_every_pods_ranks():
    """Each pod alone would hash only its ranks; the published hash is the
    union's, which is what every pod must agree on."""
    def body(tmp):
        def run(bucket):
            roots = [_node_pod(tmp, "pod-a", 0), _node_pod(tmp, "pod-b", 1)]
            results, _ = _publish_pods(roots)
            wt = results[0].rsplit("_", 1)[1]
            alone = sp.weights_hash(sp.build_manifest(
                sp.Path(roots[0]), [sp.WEIGHT_DIR])["files"])
            assert wt != alone
        _with_bucket(run)
    _in_tmp(body)


def test_missing_ranks_refuse_before_anything_is_uploaded():
    def body(tmp):
        def run(bucket):
            roots = [_node_pod(tmp, "pod-a", 0),
                     _node_pod(tmp, "pod-b", 1, ranks=[2])]
            _, errors = _publish_pods(roots)
            assert set(errors) == {0, 1}
            assert all(isinstance(e, SystemExit) for e in errors.values())
            assert not bucket.uploaded_dirs() and not bucket.sentinels
        _with_bucket(run)
    _in_tmp(body)


def test_node_partitions_of_different_configs_refuse():
    def body(tmp):
        def run(bucket):
            a = _node_pod(tmp, "pod-a", 0)
            b = _node_pod(tmp, "pod-b", 1)
            path = os.path.join(b, "node1", "image", "meta.json")
            meta = json.load(open(path))
            meta["image_ref"] = "dss@sha256:other"
            _write(path, json.dumps(meta))
            _, errors = _publish_pods([a, b])
            assert errors and not bucket.sentinels
        _with_bucket(run)
    _in_tmp(body)


def test_a_lone_node_partition_times_out_rather_than_publishing():
    def body(tmp):
        def run(bucket):
            root = sp.Path(_node_pod(tmp, "pod-a", 0))
            _expect_exit(lambda: sp.publish_nodes(
                root, "bkt", "image-cache", rendezvous_timeout=0.05))
            assert not bucket.sentinels and not bucket.uploaded_dirs()
        _with_bucket(run)
    _in_tmp(body)


def test_a_node_partition_recorded_under_another_rank_refuses():
    def body(tmp):
        root = _node_pod(tmp, "pod-a", 1)
        path = os.path.join(root, "node1", "image", "meta.json")
        meta = json.load(open(path))
        meta["node_rank"] = 0
        _write(path, json.dumps(meta))
        _expect_exit(lambda: sp.load_node_layout(sp.Path(root)))
    _in_tmp(body)


def test_a_node_partition_moved_after_its_dump_refuses():
    def body(tmp):
        root = _node_pod(tmp, "pod-a", 1)
        path = os.path.join(root, "node1", "image", "meta.json")
        meta = json.load(open(path))
        meta["model_dir"] = "/data-fast/image-cache_neutrino/other_key/node1"
        _write(path, json.dumps(meta))
        _expect_exit(lambda: sp.load_node_layout(sp.Path(root)))
    _in_tmp(body)


def test_a_node_partition_still_answers_the_key_and_model():
    """unpublish_skeleton and --status need only these two."""
    def body(tmp):
        root = sp.Path(_node_pod(tmp, "pod-a", 1))
        key, meta, replicas = sp.load_layout(root)
        assert key == f"{CFG}_{ENV_A}" and replicas == []
        assert sp.model_slug(meta) == SLUG and sp._has_image(root)
    _in_tmp(body)


def _expect_exit(fn):
    try:
        fn()
    except SystemExit:
        return
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"expected SystemExit, got {type(exc).__name__}: {exc}")
    raise AssertionError("expected SystemExit, nothing raised")


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"  [ok  ] {name}")
        except AssertionError as exc:
            fails.append(f"{name}: {exc}")
            print(f"  [FAIL] {name}: {exc}")
    print()
    print("FAILURES:", fails if fails else "none")
    raise SystemExit(1 if fails else 0)
