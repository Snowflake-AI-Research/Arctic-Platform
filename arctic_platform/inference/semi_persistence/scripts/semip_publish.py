"""Publish a dumped semi-p ``model_dir`` into the cluster model bucket.

The node-local ``base-models`` tree is a mirror of a per-cluster S3 bucket,
pulled by the ``neutrino-model-cache`` DaemonSet every 300 s.  Writing a
directory into that bucket under a ``_neutrino_manifest.json`` sentinel is
therefore enough to have it appear on every GPU node, with no new
infrastructure.  See ``skills/IMAGE_CACHE.md`` for the pipeline and its
constraints.

Skeleton and weights are published **separately**, because only one of them is
bound to a path and only one of them is worth deduplicating::

    s3://<bucket>/image-cache/<name>/
      skeleton/<cfg12>_<env12>_<wt12>/    image/ compilation/
      weight/<wt12>/                      shards + weights_meta.json

``<name>`` is the last element of ``vllm_config["model"]`` -- see ``model_slug``.

The node cache mirrors that tree verbatim under
``/mnt/neutrino/base-models/image-cache/``.  Nesting is safe: the DaemonSet's
``_discover_model_dirs`` defines a model dir as the parent of any
``_neutrino_manifest.json``, "flat or nested", and ``_remove_orphans`` reaps per
sentinel via ``rglob`` -- so a skeleton and a weight directory are independently
synced and independently retired.

**Why the split.** ``image/`` and ``compilation/`` bake the absolute
``model_dir`` they were dumped with (criu records file-backed mappings by mount
and device, and ``image/`` is criu's own ``-D`` dir), so they must be
materialized to that same local path on the consuming node.  ``weight/`` bakes
nothing -- 0 of 416 mappings in the measured image live under it, because the
dump writes the shards and detaches before criu runs -- so it is read in place
off the mirror and can live anywhere.  Measured consequence: the same config
dumped under different backend images produces **byte-identical** weights (16 of
16 cross-env pairs in the published cache), so a new backend image needs a new
skeleton and no new weights at all.

**The weight hash is the binding, and it is in the skeleton's name.** Nothing
inside the skeleton references a weight hash -- no pointer file to flip, nothing
mutable in a digest-verified manifest, and no way for the two halves to disagree,
since the key was produced by the dump that made both.  Two weight versions of
one config are two skeleton directories that coexist, which is also the rollback.
``semip_engine`` resolves a restore by listing ``skeleton/`` for
``<cfg12>_<env12>_*`` and refusing on more than one match.

The hash itself is free: ``build_manifest`` already computes a SHA-256 per file
for the sentinel, so the weight hash is a SHA-256 over those ``(relpath,
sha256)`` rows.  It does cost a full read of ``weight/`` to produce them, which
is the one real expense here.

**The local layout does not change**, and cannot.  ``model_dir`` is fixed at
process startup because the compile caches live under it, long before the weights
exist, so the weight hash can never be part of the local directory name::

    /data-fast/image-cache_neutrino/<cfg12>_<env12>/   image/ compilation/ weight/

Hence ``derived_key`` checks that the published key's ``<cfg12>_<env12>`` prefix
matches, not the whole name.

``image_ref`` and ``driver_version`` are copied into the manifest from the
image's own ``meta.json`` rather than read off this pod, so they describe the
environment the image was *dumped* in even when it is published later from
somewhere else.

Usage, from inside a device-manager pod after a dump::

    python3 semip_publish.py <model_dir> --dry-run
    python3 semip_publish.py <model_dir>            # skeleton + weights if new
    python3 semip_publish.py <model_dir> --skeleton-only

    python3 semip_publish.py <model_dir> --unpublish-skeleton
    python3 semip_publish.py --model <org/model> --unpublish-weights <wt12>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST_FILENAME = "_neutrino_manifest.json"
# The bucket is per cluster, so there is no default: a wrong one would publish
# where no node mirrors from.
DEFAULT_BUCKET_ENV = "MODEL_BUCKET"
DEFAULT_PREFIX = "image-cache"

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"

# The layout, and a *copy* of it.  ``instance.py`` creates these directories and
# ``semip_engine`` composes both trees; this script can import neither, because
# it is copied into a pod on its own and run with a bare ``python3`` in pods that
# do not all have arctic_inference installed.  So the spellings are duplicated
# here and held to the originals by ``tests/test_layout_names.py`` -- the same
# treatment ``sanitize_label_segment`` gets against the DaemonSet's copy.
#
# One name each, used for the local directory and the published one alike.
# ``WEIGHT_DIR`` in particular is both ``<model_dir>/weight/`` and
# ``<model>/weight/<wt12>/``; it was plural locally until the two were unified.
#
# Nothing here may become a bare literal again.  A published weight manifest
# once carried paths that were still scoped to the dump's own directory, so
# every node asked S3 for ``weight/<wt12>/weights/rank0/shard_0000.bin``, got a
# 404, and cold-started for a day with the shards sitting correctly beside the
# manifest.  Note that sharing a name does not make that mistake impossible:
# ``_scoped_manifest`` still re-scopes every row against the directory that will
# hold it, and has to.
SKELETON_DIR = "skeleton"
WEIGHT_DIR = "weight"
IMAGE_DIR = "image"
COMPILATION_DIR = "compilation"

# What lives in a skeleton: exactly the path-bound directories, uploaded under
# their local names.  Order is this script's upload order and means nothing --
# the restore copies the same two in the opposite order and says why.
SKELETON_DIRS = (IMAGE_DIR, COMPILATION_DIR)

# Truncation for every hash in a path, matching the cfg/env hashes the engine
# already derives.  12 hex is 48 bits, which is ample for a cache that holds
# tens of keys and keeps the label sanitizer's 63-char budget comfortable.
HASH_LEN = 12

# What a derived directory name looks like: two 12-hex hashes joined by "_".
# Matched rather than assumed, so an old-style dir (``qwen_35b``) is rejected
# with an explanation instead of being published under a key nothing resolves to.
_DERIVED_KEY_RE = re.compile(r"[0-9a-f]{12}_[0-9a-f]{12}")

# A published skeleton name: the derived key plus the weight hash it was dumped
# with.  Used to parse referrers back out of the bucket listing.
_SKELETON_KEY_RE = re.compile(r"([0-9a-f]{12}_[0-9a-f]{12})_([0-9a-f]{12})$")

# Several replicas in one pod each dump into <key>/replica<K>/ (K is the slot
# ReplicaPool gives the replica on its node); a pod holding one replica dumps
# flat into <key>/.  The replica set is reconstructed here, from the directory
# alone -- nothing in meta.json says how many replicas there were -- and
# published as one skeleton with one sentinel, so a node verifies every replica
# at once.  Pinned to ``semip_engine._REPLICA_DIR_PREFIX`` by test_layout_names.
REPLICA_DIR_PREFIX = "replica"
_REPLICA_DIR_RE = re.compile(REPLICA_DIR_PREFIX + r"(\d+)")


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_json(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# The weight hash
# ---------------------------------------------------------------------------


def weights_hash(files: list[dict], under: str = "") -> str | None:
    """Content hash of ``weight/`` from the per-file digests already computed.

    ``None`` when the directory holds no weights, which the caller reports
    rather than publishing a skeleton nothing can restore.

    Composed from ``build_manifest``'s rows rather than by re-reading the
    shards, so it costs nothing beyond the hashing the sentinel already needs.
    The path that goes in is relative to the weight directory -- ``WEIGHT_DIR``
    is stripped, exactly as ``_scoped_manifest`` strips it for the sentinel --
    and the rows are sorted, so:

    * the same bytes under the same names hash the same on any node, in any
      filesystem order -- which is what makes the cross-backend-image dedup
      work at all;
    * a TP change cannot collide with the layout it replaces, since the paths
      themselves differ (flat ``shard_NNNN.bin`` at TP1, ``rank<N>/`` above it)
      and the per-rank byte totals differ too.

    Truncated to ``HASH_LEN`` because it becomes a path component.  A collision
    would mean two different weight sets sharing a directory, so the budget is
    deliberately the same as the cfg and env hashes' rather than smaller.

    **The directory's own name is deliberately not an input.**  It used to be:
    the rows went in with their ``build_manifest`` prefix, which is the same
    constant on every row of every dump and so discriminates nothing, while
    quietly making the hash depend on a name.  Renaming the local directory then
    forked every published weight hash for byte-identical weights, which is what
    unifying the local ``weights`` with the published ``weight`` cost -- one
    re-dump and one re-upload per model, 760 GB for GLM-5.3.  Stripping the
    prefix costs that same fork once, so the two landed together and there is
    nothing further to pay: the directory can be renamed again for free.

    What the hash still tracks is the path *inside* the directory, which is what
    keeps a TP change from colliding with the layout it replaces.

    A second property falls out of stripping it.  These are the paths a
    published weight manifest already stores, so the hash is now recomputable
    from that manifest alone -- meaning a published directory can be checked
    against the ``<wt12>`` in its own name, with no local dump to compare to.
    That is the check ``_published`` wants: it gates on the sentinel existing,
    so a manifest that parses but is scoped wrong reads as complete, which is
    exactly how ``GLM-5.3/weight/a1cd5612899d`` satisfied a reuse check while no
    node could resolve a single one of its keys.

    ``under`` scopes it to one replica's ``replica<K>/weight/``; the replica
    level is stripped with the rest, so every replica of a dump hashes alike.
    """
    prefix = under + WEIGHT_DIR + "/"
    # ``build_manifest`` only descends directories, so every weight row is
    # "<WEIGHT_DIR>/..." and a bare "<WEIGHT_DIR>" row cannot occur.
    rows = sorted(
        (f["path"][len(prefix):], f["sha256"])
        for f in files
        if f["path"].startswith(prefix)
    )
    if not rows:
        return None
    h = hashlib.sha256()
    for path, digest in rows:
        # NUL-delimited so no path can impersonate a digest boundary.
        h.update(path.encode())
        h.update(b"\0")
        h.update(digest.encode())
        h.update(b"\0")
    return h.hexdigest()[:HASH_LEN]


def model_slug(meta: dict) -> str:
    """The model **name** path component, from the image's own config.

    Read out of ``meta.json`` rather than off this pod so the destination
    describes the model that was dumped, the same way ``image_ref`` does.

    Only the last element of the recorded path is used. dss resolves ``model``
    through ``resolve_model_path`` before the engine sees it, so it is absolute:
    publishing it verbatim would nest the mirror's own mount inside itself
    (``image-cache/mnt/neutrino/base-models/Qwen/...``) and would fork the shared
    weight directory if that mount ever moved, re-uploading identical bytes under
    a second path.

    Only grouping and readability depend on this -- the weight hash carries
    identity, and the skeleton key carries ``cfg12`` over the full absolute path,
    so two orgs shipping one name would share a directory but never a key or a
    weight hash. A *missing* value is still an error, because it would silently
    split one model's weights across two directories and lose the dedup.

    **``semip_engine._model_slug`` must compute exactly this**, since publish
    writes where the engine reads and a disagreement is a permanent, silent cache
    miss. Duplicated rather than shared because this script is stdlib-only and
    runs by path inside a pod; ``test_publish_layout`` cross-checks the two.
    """
    raw = (meta.get("vllm_config") or {}).get("model")
    parts = [p for p in str(raw or "").strip("/").split("/")
             if p and p not in (".", "..")]
    slug = parts[-1] if parts else ""
    if not slug:
        raise SystemExit(
            "meta.json carries no usable vllm_config.model, so there is no model "
            f"path to publish under (got {raw!r}). Re-dump this image.")
    return slug


# ---------------------------------------------------------------------------
# Key 1: the config hash
# ---------------------------------------------------------------------------


def derived_key(model_dir: Path, meta: dict) -> str:
    """The image's cache key: the name ``_resolve_model_dir`` gave the directory.

    This is the ``<cfg12>_<env12>`` prefix only.  The published skeleton appends
    ``_<wt12>``, but the *local* directory cannot carry it: ``model_dir`` is
    fixed at process startup, because the compile caches are written under it
    long before the weights that the hash is taken over exist.  So the check
    below compares the prefix rather than the whole published name.

    Two guards, because both failure modes are silent otherwise.  The name must
    look like a derived key, or an old-style directory would be published where
    nothing resolves to it.  And ``meta.json``'s recorded ``model_dir`` must
    agree with where the directory actually sits, or the image has been moved
    since the dump and is no longer restorable under the path it names.
    """
    key = model_dir.name
    if not _DERIVED_KEY_RE.fullmatch(key):
        raise SystemExit(
            f"{model_dir} is not a derived image directory: expected a name like "
            f"<cfg12>_<env12> (two 12-hex hashes), got {key!r}. Only a directory "
            f"named by semip_engine._resolve_model_dir can be published -- a key "
            f"computed here could not be the one a later restore looks up.")

    recorded = meta.get("model_dir")
    if recorded and Path(recorded).name != key:
        raise SystemExit(
            f"this image was dumped as {recorded} but now sits at {model_dir}. "
            f"image/ and compilation/ bake absolute paths, so publishing it "
            f"under {key!r} would distribute an image no node can restore.")
    return key


def replica_dirs(model_dir: Path) -> list[Path]:
    """A multi-replica dump's ``replica<K>`` directories in slot order; ``[]`` if flat."""
    found = []
    if model_dir.is_dir():
        for child in model_dir.iterdir():
            match = _REPLICA_DIR_RE.fullmatch(child.name)
            if match and child.is_dir():
                found.append((int(match.group(1)), child))
    return [path for _, path in sorted(found)]


def load_layout(model_dir: Path) -> tuple[str, dict, list[Path]]:
    """``(key, meta, replicas)`` for a dumped directory, flat or per-replica.

    ``meta`` is replica 0's for a multi-replica dump.  Every replica is checked
    before anything is hashed, because a skeleton published with a replica
    missing or foreign is one that some pod can only half restore -- and the
    engine fails that pod's job rather than mix restores with cold starts:

    * slots are contiguous from 0, so no replica is missing from the middle;
    * every replica has dumped (``image/meta.json`` exists);
    * each recorded ``model_dir`` is ``<key>/replica<K>`` for its own K, since
      the baked paths have to resolve where the engine will put them;
    * all replicas agree on the config, image, driver and uid.
    """
    replicas = replica_dirs(model_dir)
    if not replicas:
        meta = json.loads((model_dir / IMAGE_DIR / "meta.json").read_text())
        return derived_key(model_dir, meta), meta, []

    key = model_dir.name
    if not _DERIVED_KEY_RE.fullmatch(key):
        raise SystemExit(
            f"{model_dir} holds {REPLICA_DIR_PREFIX}<K>/ directories but is not "
            f"named like a derived key (<cfg12>_<env12>)")
    slots = [int(_REPLICA_DIR_RE.fullmatch(r.name).group(1)) for r in replicas]
    if slots != list(range(len(slots))):
        raise SystemExit(
            f"{model_dir} has replicas {slots}, not 0..{len(slots) - 1}; a "
            f"replica is missing, so a pod of this shape cannot fully restore")
    metas = []
    for rdir in replicas:
        meta_path = rdir / IMAGE_DIR / "meta.json"
        if not meta_path.is_file():
            raise SystemExit(
                f"{rdir} has not dumped yet ({meta_path} missing); publish "
                f"once every replica has")
        meta = json.loads(meta_path.read_text())
        recorded = meta.get("model_dir")
        if recorded and (Path(recorded).name != rdir.name
                         or Path(recorded).parent.name != key):
            raise SystemExit(
                f"{rdir} was dumped as {recorded}; image/ and compilation/ "
                f"bake that path, so it cannot be published as {key}/{rdir.name}")
        metas.append(meta)
    for rdir, meta in zip(replicas[1:], metas[1:]):
        for field in ("vllm_config", "image_ref", "driver_version", "uid"):
            if meta.get(field) != metas[0].get(field):
                raise SystemExit(
                    f"{rdir.name} disagrees with {replicas[0].name} on {field} "
                    f"({meta.get(field)!r} vs {metas[0].get(field)!r}); these "
                    f"are not replicas of one dump")
    return key, metas[0], replicas


def _has_image(model_dir: Path) -> bool:
    """Whether *model_dir* holds a dump in either layout."""
    return ((model_dir / IMAGE_DIR / "meta.json").is_file()
            or bool(replica_dirs(model_dir)))


def _payload_subdirs(replicas: list[Path]) -> list[str]:
    """Every directory a publish hashes, relative to the dumped ``model_dir``."""
    subs = list(SKELETON_DIRS) + [WEIGHT_DIR]
    if not replicas:
        return subs
    return [f"{r.name}/{sub}" for r in replicas for sub in subs]


# ---------------------------------------------------------------------------
# Key 2: the environment hash
# ---------------------------------------------------------------------------


def _k8s_get(path: str, timeout: float = 30.0) -> dict:
    """GET *path* from the in-cluster API with this pod's ServiceAccount token.

    What the token may read is narrower than it looks, and the readiness check
    below is shaped by it: **nodes are readable, DaemonSets are not** (403). So
    "verified on every node" is answered from node labels rather than from the
    DaemonSet's ``desiredNumberScheduled``, which would otherwise be the
    authoritative denominator.
    """
    token = Path(SA_DIR, "token").read_text().strip()
    ctx = ssl.create_default_context(cafile=os.path.join(SA_DIR, "ca.crt"))
    req = urllib.request.Request(
        f"https://kubernetes.default.svc{path}",
        headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return json.load(resp)


def image_digest() -> tuple[str, str]:
    """Return ``(short_hash, full_ref)`` for this container's image.

    Reads the pod's own object through the in-cluster API with the
    ServiceAccount token.  ``status.containerStatuses[].imageID`` carries the
    registry content digest, which is immutable -- unlike the tag, which moves.
    """
    namespace = Path(SA_DIR, "namespace").read_text().strip()
    pod = os.environ.get("HOSTNAME", "")
    body = _k8s_get(f"/api/v1/namespaces/{namespace}/pods/{pod}")

    container = os.environ.get("SEMIP_CONTAINER_NAME", "device-manager")
    statuses = body.get("status", {}).get("containerStatuses", [])
    match = next((c for c in statuses if c.get("name") == container), None)
    if match is None and statuses:
        match = statuses[0]
    if match is None:
        raise RuntimeError("no containerStatuses on this pod")

    image_id = match.get("imageID", "")
    if "sha256:" not in image_id:
        raise RuntimeError(f"imageID carries no digest: {image_id!r}")
    return image_id.split("sha256:", 1)[1][:12], image_id


# ---------------------------------------------------------------------------
# Is it there? Two independent questions.
#
# **In the bucket** is about the upload: the sentinel exists, so a directory is
# complete rather than half-written. It says nothing about reachability.
#
# **Verified on every node** is about distribution, and it is the one that
# decides whether a restore can actually use the thing. The DaemonSet stamps
# ``model.neutrino.snowflake.com/<sanitized>=ready`` on its node only after
# digest-verifying every file in the directory against the bucket manifest, so a
# full count simultaneously proves the upload was correct, the sync finished, and
# the nested layout was discovered. Kept separate from the bucket check because
# they fail for different reasons and at different times: measured on the first
# publish, a skeleton was ready ~40 s after upload and its 66 GB weight directory
# ~140 s, with a ~5m18s pass interval in front of both.
# ---------------------------------------------------------------------------

# Runs of characters not allowed in a label-key segment collapse to a single
# dot; segments must start and end with an alphanumeric. Replicated from the
# DaemonSet's sync_node_cache._sanitize_label_segment, whose own docstring says
# it "MUST be replicated exactly by any consumer that needs to compute the label
# for the same model name". test_publish_layout pins it against four label keys
# read off the live cluster, two of which exercise the truncation branch.
_LABEL_DISALLOWED_RUN = re.compile(r"[^A-Za-z0-9_.-]+")
_LABEL_TRIM_ENDS = re.compile(r"(^[^A-Za-z0-9]+)|([^A-Za-z0-9]+$)")
_LABEL_MAX_SEGMENT = 63
_NODE_LABEL_PREFIX = "model.neutrino.snowflake.com/"

# The pools the model-cache DaemonSet runs on, used only as the denominator for
# "every node". Needed because the DaemonSet object itself is forbidden to this
# token, so its desiredNumberScheduled cannot be read.
_CACHE_NODE_POOLS = ("neutrino-pool-h200", "neutrino-pool-b200")


def sanitize_label_segment(name: str) -> str:
    """Map a model-dir path to the label-key segment the DaemonSet would use."""
    s = _LABEL_DISALLOWED_RUN.sub(".", name)
    s = _LABEL_TRIM_ENDS.sub("", s)
    if not s:
        return "m." + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
    if len(s) > _LABEL_MAX_SEGMENT:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:7]
        s = s[:55].rstrip(".-_") + "-" + digest
    return s


def _node_count(selector: str) -> int:
    body = _k8s_get("/api/v1/nodes?labelSelector="
                    + urllib.parse.quote(selector, safe="(),=!"))
    return len(body.get("items", []))


def cluster_readiness(rel_dir: str) -> tuple[int, int]:
    """``(ready, total)`` nodes for the published directory at *rel_dir*.

    *rel_dir* is the path under the bucket root, which is what the daemon names a
    model dir -- e.g. ``image-cache/Qwen3.8-27B/weight/7fd0c0cb3d49``.

    Both counts come from one server-side selector each, so neither pulls the
    cluster's full node list (575 against the 8 that matter here).
    """
    pools = ",".join(_CACHE_NODE_POOLS)
    denom = f"workergroup in ({pools})"
    total = _node_count(denom)
    if not total:
        # Unknown topology: fall back to "nodes the daemon has ever stamped".
        # Weaker -- a freshly added node carries no labels yet and so is not
        # counted -- but better than reporting 0/0 as success.
        body = _k8s_get("/api/v1/nodes")
        managed = [n for n in body.get("items", [])
                   if any(k.startswith(_NODE_LABEL_PREFIX)
                          for k in (n["metadata"].get("labels") or {}))]
        return (sum(1 for n in managed
                    if (n["metadata"].get("labels") or {}).get(
                        _NODE_LABEL_PREFIX + sanitize_label_segment(rel_dir))
                    == "ready"), len(managed))
    label = _NODE_LABEL_PREFIX + sanitize_label_segment(rel_dir)
    return _node_count(f"{denom},{label}=ready"), total


def report_state(bucket: str, prefix: str, slug: str, skel_key: str,
                 wt_hash: str) -> bool:
    """Print both checks for one published pair. True when everything is ready."""
    rows = [(SKELETON_DIR, f"{prefix}/{slug}/{SKELETON_DIR}/{skel_key}"),
            (WEIGHT_DIR, f"{prefix}/{slug}/{WEIGHT_DIR}/{wt_hash}")]
    all_ready = True
    for what, rel in rows:
        in_bucket = _published(bucket, rel)
        try:
            ready, total = cluster_readiness(rel)
            nodes = f"{ready}/{total} nodes verified"
            ok = in_bucket and total > 0 and ready == total
        except Exception as exc:  # noqa: BLE001 - a check must not fail a publish
            nodes = f"node check unavailable ({type(exc).__name__}: {exc})"
            ok = False
        all_ready = all_ready and ok
        print(f"  {what:9} {'in bucket' if in_bucket else 'MISSING from bucket'}"
              f", {nodes}")
        print(f"  {'':9} {rel}")
    return all_ready


def wait_verified(bucket: str, prefix: str, slug: str, skel_key: str,
                  wt_hash: str, timeout: float, interval: float = 20.0) -> bool:
    """Poll both checks until everything is ready, or *timeout* elapses.

    Default timeout is generous on purpose: the daemon's pass interval alone is
    over five minutes, and a large weight directory then takes minutes more to
    download and digest-verify.
    """
    deadline = time.time() + timeout
    while True:
        print(f"-- state at {time.strftime('%H:%M:%S')} "
              f"({deadline - time.time():.0f}s budget left)")
        if report_state(bucket, prefix, slug, skel_key, wt_hash):
            print("all published directories are verified on every node")
            return True
        if time.time() + interval >= deadline:
            print(f"still not fully verified after {timeout:.0f}s. This is not "
                  f"necessarily wrong -- a large weight directory can outlast "
                  f"the budget -- but nothing will restore from it until the "
                  f"counts reach their totals.")
            return False
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Manifest + upload
# ---------------------------------------------------------------------------


def build_manifest(model_dir: Path, subdirs: list[str],
                   workers: int | None = None) -> dict:
    """Per-file sizes and SHA-256s for the sentinel, and so for the weight hash.

    This is the expensive half of a publish: every byte of every subdir is read,
    which is 58 GB for a 27B at TP1 and 760 GB for GLM-5.3 at TP8.

    Hashed across a thread pool because ``hashlib`` releases the GIL, so this
    scales with cores instead of being serialized by the interpreter -- measured
    2.2 GiB/s on one core against 9.3 GiB/s across eight, on hardware with SHA
    extensions (the H200 nodes are Xeon Platinum 8488C, 192 cores, ``sha_ni``).
    SHA-256 rather than a faster-looking BLAKE2b for the same reason: it is the
    hardware-accelerated one here, and roughly 2x BLAKE2b's software speed.

    ``ex.map`` preserves input order, so rows stay in sorted path order whatever
    order the hashes complete in. The weight hash sorts its own rows anyway, but
    a manifest whose order wandered between runs would be needlessly hard to diff.
    """
    if workers is None:
        workers = max(1, min(16, os.cpu_count() or 8))
    files = []
    total = 0
    for sub in subdirs:
        root = model_dir / sub
        if not root.is_dir():
            print(f"  - {sub}/ absent, skipping")
            continue
        paths = [p for p in sorted(root.rglob("*")) if p.is_file()]
        started = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            digests = list(pool.map(_sha256_file, paths))
        nbytes = 0
        for path, digest in zip(paths, digests):
            size = path.stat().st_size
            files.append(
                {
                    "path": str(path.relative_to(model_dir)),
                    "size_bytes": size,
                    "sha256": digest,
                }
            )
            total += size
            nbytes += size
        elapsed = max(time.time() - started, 1e-9)
        print(f"  - {sub}/ hashed: {len(paths)} files, "
              f"{nbytes / 2**30:.1f} GiB in {elapsed:.1f}s "
              f"({nbytes / 2**30 / elapsed:.2f} GiB/s, {workers} threads)")
    return {"files": files, "total_bytes": total}


def _scoped_manifest(files: list[dict], keep: tuple[str, ...],
                     strip: str | None = None) -> dict:
    """The rows for one published directory, with paths relative to *it*.

    The DaemonSet verifies each file at ``<local model dir>/<path>``, so a
    sentinel's paths have to be relative to the directory that holds it.  The
    skeleton keeps ``image/...`` and ``compilation/...`` as they are, because it
    is their parent; the weight directory *is* the local ``weight/``, holding the
    shards at its own top level, so that prefix is stripped.

    The local and published names being identical does not make the stripping
    optional, and reading it that way is the one mistake this function exists to
    prevent.  Getting it wrong does not fail a publish -- it fails the
    *verification* on every node, quietly: the directory never gets a
    ``.neutrino_verified``, and the only symptom is a 404 per missing key in a
    DaemonSet log nobody reads.
    """
    rows = []
    total = 0
    for f in files:
        first = f["path"].split("/", 1)[0]
        if first not in keep:
            continue
        row = dict(f)
        if strip:
            prefix = strip + "/"
            if not row["path"].startswith(prefix):
                continue
            row["path"] = row["path"][len(prefix):]
        rows.append(row)
        total += f["size_bytes"]
    return {"files": rows, "total_bytes": total}


def _replica_skeleton_manifest(files: list[dict], names: list[str]) -> dict:
    """Skeleton rows of a multi-replica dump: ``replica<K>/{image,compilation}/...``.

    Paths keep their replica level, because the skeleton is their common parent
    and holds one sentinel for all of them -- so a node verifies (and a pod
    restores) every replica or none.
    """
    rows = []
    total = 0
    for f in files:
        parts = f["path"].split("/", 2)
        if len(parts) >= 2 and parts[0] in names and parts[1] in SKELETON_DIRS:
            rows.append(dict(f))
            total += f["size_bytes"]
    return {"files": rows, "total_bytes": total}


def _published_replica_count(bucket: str, key: str) -> int | None:
    """Replicas in the skeleton already published at *key*; ``None`` if none or unknown."""
    if not _published(bucket, key):
        return None
    try:
        rc, out = _aws_out("s3", "cp", f"s3://{bucket}/{key}/{MANIFEST_FILENAME}",
                           "-")
        if rc != 0:
            return None
        return int(json.loads(out).get("replicas", 1))
    except (OSError, ValueError, TypeError):
        return None


def _aws(*args: str) -> None:
    result = subprocess.run(["aws", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"aws {' '.join(args)} failed: {result.stderr[:400]}")


def _aws_out(*args: str) -> tuple[int, str]:
    """``aws`` for the queries, where a non-zero exit is an answer not a failure."""
    result = subprocess.run(["aws", *args], capture_output=True, text=True)
    return result.returncode, result.stdout


def _published(bucket: str, key: str) -> bool:
    """Whether a completed directory sits at *key*.

    Tests the **sentinel**, never the prefix.  ``publish`` uploads the payload
    first and the sentinel last precisely so an interrupted upload is never
    discovered as a complete model dir by the node DaemonSet -- which means a
    prefix that exists without a sentinel is a partial upload and has to be
    re-uploaded, not skipped.
    """
    rc, _ = _aws_out("s3api", "head-object", "--bucket", bucket,
                     "--key", f"{key}/{MANIFEST_FILENAME}")
    return rc == 0


def _skeleton_referrers(bucket: str, prefix: str, slug: str,
                        wt_hash: str) -> list[str]:
    """Published skeleton keys bound to *wt_hash*.

    The binding lives in the skeleton's name, so counting referrers is a listing
    rather than a fan-out of file reads -- which is the whole reason the weight
    hash went into the key instead of a pointer file.
    """
    root = f"{prefix}/{slug}/{SKELETON_DIR}"
    rc, out = _aws_out("s3", "ls", f"s3://{bucket}/{root}/")
    if rc != 0:
        return []
    found = []
    for line in out.splitlines():
        name = line.split()[-1].rstrip("/") if line.split() else ""
        m = _SKELETON_KEY_RE.fullmatch(name)
        if m and m.group(2) == wt_hash:
            found.append(name)
    return sorted(found)


def _upload_dir(src: Path, dest: str, label: str) -> None:
    print(f"uploading {label} ...")
    _aws("s3", "cp", "--recursive", "--only-show-errors", str(src), dest)


def _write_sentinel(model_dir: Path, manifest: dict, dest: str,
                    name: str) -> None:
    """Write the sentinel last, which is what marks the directory complete."""
    local = model_dir / f".{name}.{MANIFEST_FILENAME}"
    local.write_text(json.dumps(manifest))
    try:
        _aws("s3", "cp", "--only-show-errors", str(local),
             f"{dest}/{MANIFEST_FILENAME}")
    finally:
        local.unlink(missing_ok=True)


def publish(model_dir: Path, bucket: str, prefix: str,
            skeleton_only: bool = False, dry_run: bool = False,
            force_weights: bool = False,
            hash_workers: int | None = None,
            wait_timeout: float | None = None) -> str:
    """Publish the skeleton, and the weights only if that hash is new.

    Returns the published skeleton key.

    *model_dir* is the dumped ``<root>/<cfg12>_<env12>``, whether it holds one
    flat image or ``replica<K>/`` directories; see ``load_layout``.
    """
    key, meta, replicas = load_layout(model_dir)
    slug = model_slug(meta)

    print(f"key prefix  : {key}   (the local directory's derived name)")
    print(f"model       : {slug}")
    print(f"image_ref   : {meta.get('image_ref')}")
    print(f"driver      : {meta.get('driver_version')}")

    # Advisory only. A mismatch means this pod is not the one that dumped the
    # image -- fine to publish, since the key already encodes the dumping
    # environment, but worth saying out loud.
    try:
        _, pod_image = image_digest()
        if meta.get("image_ref") and pod_image != meta["image_ref"]:
            print(f"  ! this pod runs {pod_image}, the image was dumped under "
                  f"{meta['image_ref']}; publishing under the dumped identity")
    except Exception as exc:  # noqa: BLE001 - never fail a publish over a check
        print(f"  ! could not read this pod's image ({type(exc).__name__}: {exc})")

    # Everything is hashed, including weights we may not upload: the hash is what
    # decides whether to upload them, so it cannot be skipped. This is the one
    # expensive step -- a full read of weight/ -- and it is why the dump-side
    # alternative exists as a later option.
    print("hashing files ...")
    full = build_manifest(model_dir, _payload_subdirs(replicas),
                          workers=hash_workers)
    names = [r.name for r in replicas]
    if replicas:
        # Every replica saved its own copy of the same weights; one is uploaded.
        # Unequal hashes mean the replicas are not interchangeable, which no
        # deduplication should paper over.
        by_replica = {n: weights_hash(full["files"], under=n + "/")
                      for n in names}
        empty = [n for n, h in by_replica.items() if h is None]
        if empty:
            raise SystemExit(
                f"{', '.join(empty)} under {model_dir} hold no {WEIGHT_DIR}/ "
                f"files; every replica must have dumped its weights")
        if len(set(by_replica.values())) != 1:
            raise SystemExit(
                f"the replicas under {model_dir} dumped different weights "
                f"({by_replica}); refusing to publish one copy for all of them")
        weight_src = names[0]
        wt_hash = by_replica[weight_src]
        print(f"replicas    : {len(names)}, one weight hash across all of them")
    else:
        weight_src = ""
        wt_hash = weights_hash(full["files"])
    if wt_hash is None:
        raise SystemExit(
            f"{model_dir}/{WEIGHT_DIR}/ holds no files, so there is no weight "
            f"hash and nothing a restore could read. The engine reads weights in "
            f"place off the mirror rather than copying them, so a skeleton "
            f"published without them cold-starts on every node.")

    skel_key = f"{key}_{wt_hash}"
    skel_rel = f"{prefix}/{slug}/{SKELETON_DIR}/{skel_key}"
    skel_dest = f"s3://{bucket}/{skel_rel}"
    wt_dest = f"s3://{bucket}/{prefix}/{slug}/{WEIGHT_DIR}/{wt_hash}"

    if replicas:
        skel = _replica_skeleton_manifest(full["files"], names)
        skel["replicas"] = len(names)
        wts = _scoped_manifest(full["files"], (weight_src,),
                               strip=f"{weight_src}/{WEIGHT_DIR}")
    else:
        skel = _scoped_manifest(full["files"], SKELETON_DIRS)
        wts = _scoped_manifest(full["files"], (WEIGHT_DIR,), strip=WEIGHT_DIR)

    # One key, one shape. Overwriting a skeleton of another replica count would
    # leave the old shape's files under the new sentinel, which the node cache
    # then treats as orphans -- and pods of the old shape lose their image.
    already = _published_replica_count(bucket, skel_rel)
    if already is not None and already != (len(names) or 1):
        raise SystemExit(
            f"{skel_rel} is already published with {already} replica(s); this "
            f"dump has {len(names) or 1}. Unpublish it first if it is stale.")

    for m, extra in ((skel, {"cache_key": skel_key}),
                     (wts, {"weight_hash": wt_hash})):
        m["model_dir"] = str(model_dir)
        m["model"] = slug
        m["weight_hash"] = wt_hash
        m["image_ref"] = meta.get("image_ref")
        m["driver_version"] = meta.get("driver_version")
        m["published_at"] = time.time()
        m.update(extra)

    print(f"weight hash : {wt_hash}   ({len(wts['files'])} files, "
          f"{wts['total_bytes'] / 1e9:.1f} GB)")
    print(f"skeleton    : {skel_key}   ({len(skel['files'])} files, "
          f"{skel['total_bytes'] / 1e9:.1f} GB)")

    # Probed on a dry run too. `_published` is a read-only head-object on the
    # sentinel, and whether this hash is already up is the one thing a dry run
    # exists to tell you: it decides both the size of the upload (up to 760 GB
    # for GLM) and whether --skeleton-only is safe. Short-circuiting it to False
    # made the dry run claim published weights were missing, which is the
    # unsafe direction to be wrong in.
    weights_present = _published(bucket, f"{prefix}/{slug}/"
                                f"{WEIGHT_DIR}/{wt_hash}")
    skip_weights = skeleton_only or (weights_present and not force_weights)
    if weights_present:
        refs = _skeleton_referrers(bucket, prefix, slug, wt_hash)
        print(f"  = weights already published, shared with {len(refs)} "
              f"skeleton(s): {', '.join(refs) or 'none yet'}")
        if force_weights:
            print("  ! --force-weights: re-uploading identical bytes")
    elif skeleton_only:
        print("  ! --skeleton-only and these weights are NOT published; the "
              "skeleton will cold-start on every node until they are")

    # (local dir, destination, label) for every skeleton directory, in upload
    # order; a multi-replica skeleton keeps each replica under its own name.
    skel_uploads = [
        (model_dir / name / sub, f"{skel_dest}/{name + '/' if name else ''}{sub}",
         f"{name + '/' if name else ''}{sub}/")
        for name in (names or [""]) for sub in SKELETON_DIRS
        if (model_dir / name / sub).is_dir()]

    if dry_run:
        print("DRY RUN -- nothing uploaded. Would have written:")
        for _, dest, _ in skel_uploads:
            print(f"  {dest}/")
        print(f"  {skel_dest}/{MANIFEST_FILENAME}")
        print(f"  {wt_dest}/  (subject to the published-hash check)")
        return skel_key

    # Weights first. A skeleton is what a restore resolves, so publishing it
    # before its weights exist opens a window where every node materializes an
    # image it cannot load and cold-starts instead.
    if skip_weights:
        print("skipping weights upload")
    else:
        _upload_dir(model_dir / weight_src / WEIGHT_DIR, wt_dest,
                    f"{weight_src + '/' if weight_src else ''}{WEIGHT_DIR}/")
        _write_sentinel(model_dir, wts, wt_dest, WEIGHT_DIR)

    for src, dest, label in skel_uploads:
        _upload_dir(src, dest, label)
    _write_sentinel(model_dir, skel, skel_dest, SKELETON_DIR)

    print(f"published. expect it on every node within 300 s at:\n"
          f"  /mnt/neutrino/base-models/{prefix}/{slug}/"
          f"{SKELETON_DIR}/{skel_key}/\n"
          f"  /mnt/neutrino/base-models/{prefix}/{slug}/"
          f"{WEIGHT_DIR}/{wt_hash}/")
    if wait_timeout is not None:
        wait_verified(bucket, prefix, slug, skel_key, wt_hash, wait_timeout)
    return skel_key


def unpublish_skeleton(model_dir: Path, bucket: str, prefix: str,
                       key: str | None = None) -> None:
    """Remove published skeletons for this config, and never their weights.

    Weights are shared, so removing them here would break every *other*
    skeleton bound to the same hash -- quietly, as a cold start rather than an
    error.  Use ``--unpublish-weights`` for those, which checks referrers first.
    """
    want, meta, _ = load_layout(model_dir)
    slug = model_slug(meta)
    root = f"{prefix}/{slug}/{SKELETON_DIR}"

    if key:
        targets = [key]
    else:
        rc, out = _aws_out("s3", "ls", f"s3://{bucket}/{root}/")
        targets = []
        for line in out.splitlines() if rc == 0 else []:
            name = line.split()[-1].rstrip("/") if line.split() else ""
            m = _SKELETON_KEY_RE.fullmatch(name)
            if m and m.group(1) == want:
                targets.append(name)
    if not targets:
        print("no published skeleton for this config; nothing to remove")
        return
    for name in targets:
        dest = f"s3://{bucket}/{root}/{name}"
        print(f"removing {dest} ...")
        _aws("s3", "rm", "--recursive", "--only-show-errors", dest)
    print(f"removed {len(targets)} skeleton(s); weights left in place. "
          f"Nodes drop them on their next pass (<=300 s)")


def unpublish_weights(bucket: str, prefix: str, slug: str, wt_hash: str,
                      force: bool = False) -> None:
    """Remove a weight directory, refusing while a skeleton still needs it."""
    refs = _skeleton_referrers(bucket, prefix, slug, wt_hash)
    if refs and not force:
        raise SystemExit(
            f"refusing: {len(refs)} published skeleton(s) are bound to weight "
            f"hash {wt_hash} and would be left unrestorable -- as a silent cold "
            f"start, not an error:\n  " + "\n  ".join(refs) +
            f"\nUnpublish those skeletons first, or pass --force if they are "
            f"already gone and this is a leftover.")
    if refs:
        print(f"  ! --force: orphaning {len(refs)} skeleton(s): "
              f"{', '.join(refs)}")
    dest = f"s3://{bucket}/{prefix}/{slug}/{WEIGHT_DIR}/{wt_hash}"
    print(f"removing {dest} ...")
    _aws("s3", "rm", "--recursive", "--only-show-errors", dest)
    print("removed; nodes drop it on their next pass (<=300 s)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?",
                        help="the dumped directory, e.g. "
                             "/data-fast/image-cache_neutrino/<cfg12>_<env12>. "
                             "Not needed for --unpublish-weights")
    parser.add_argument("--bucket", default=os.environ.get(DEFAULT_BUCKET_ENV),
                        help=f"the cluster's model bucket (default: "
                             f"${DEFAULT_BUCKET_ENV})")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--skeleton-only", action="store_true",
                        help="publish image/ + compilation/ and never the "
                             "weights. Only safe when this weight hash is "
                             "already published -- which is the normal case "
                             "for a new backend image, and is detected "
                             "automatically, so you rarely need this")
    parser.add_argument("--force-weights", action="store_true",
                        help="re-upload weights even when the hash is already "
                             "published (repairs a corrupted directory)")
    parser.add_argument("--unpublish-skeleton", action="store_true",
                        help="remove published skeletons for this config, "
                             "leaving the shared weights alone")
    parser.add_argument("--key",
                        help="with --unpublish-skeleton, remove exactly this "
                             "<cfg12>_<env12>_<wt12> rather than every skeleton "
                             "for the config")
    parser.add_argument("--unpublish-weights", metavar="WT12",
                        help="remove a weight directory, refusing while any "
                             "published skeleton is still bound to it")
    parser.add_argument("--model", metavar="ORG/MODEL",
                        help="model path for --unpublish-weights, when no "
                             "model_dir is given")
    parser.add_argument("--force", action="store_true",
                        help="with --unpublish-weights, proceed even though "
                             "skeletons still reference the hash")
    parser.add_argument("--status", action="store_true",
                        help="report both checks and publish nothing: whether "
                             "each directory is in the bucket, and how many "
                             "nodes have digest-verified it. Needs the weight "
                             "hash, so it re-hashes the payload unless "
                             "--weight-hash is given")
    parser.add_argument("--wait-verified", type=float, nargs="?", const=1200.0,
                        default=None, metavar="SECS",
                        help="after publishing (or with --status), poll until "
                             "every directory is verified on every node. "
                             "Default 1200s: the daemon's pass interval alone "
                             "is over 5 min, and a large weight directory then "
                             "needs minutes more to download and verify")
    parser.add_argument("--weight-hash", metavar="WT12",
                        help="with --status, the hash to look up, so no local "
                             "payload has to be re-read")
    parser.add_argument("--hash-workers", type=int, default=None,
                        metavar="N",
                        help="threads used to hash the payload (default "
                             "min(16, cpu_count)). hashlib drops the GIL, so "
                             "this scales with cores; raise it for a large "
                             "weights directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="hash and report the destinations without uploading")
    args = parser.parse_args(argv)
    if not args.bucket:
        parser.error(f"no bucket: pass --bucket or set {DEFAULT_BUCKET_ENV}")

    if args.unpublish_weights:
        slug = args.model
        if not slug and args.model_dir:
            md = Path(args.model_dir.rstrip("/"))
            if _has_image(md):
                slug = model_slug(load_layout(md)[1])
        if not slug:
            print("--unpublish-weights needs --model <org/model> (or a "
                  "model_dir whose meta.json names it)")
            return 1
        unpublish_weights(args.bucket, args.prefix, slug.strip("/"),
                          args.unpublish_weights, force=args.force)
        return 0

    if args.status:
        # Two ways in: a local model_dir (derive everything, re-hashing the
        # payload for the weight hash), or --model plus --weight-hash and --key
        # for a pure lookup that reads nothing local.
        if args.model and args.weight_hash:
            slug, wt_hash = args.model.strip("/"), args.weight_hash
            skel_key = args.key or "<unknown>"
            if not args.key:
                print("  ! no --key given, so only the weight directory can be "
                      "checked; pass --key <cfg12>_<env12>_<wt12> for both")
        elif args.model_dir:
            model_dir = Path(args.model_dir.rstrip("/"))
            key, meta, replicas = load_layout(model_dir)
            slug = model_slug(meta)
            wt_hash = args.weight_hash
            if not wt_hash:
                # One replica's weights name the hash for all of them; publish
                # refuses a dump whose replicas disagree.
                under = f"{replicas[0].name}/" if replicas else ""
                print("hashing to derive the weight hash ...")
                full = build_manifest(model_dir, [under + WEIGHT_DIR],
                                      workers=args.hash_workers)
                wt_hash = weights_hash(full["files"], under=under)
            skel_key = f"{key}_{wt_hash}"
        else:
            parser.error("--status needs a model_dir, or --model with "
                         "--weight-hash")
        print(f"model       : {slug}")
        print(f"skeleton    : {skel_key}")
        print(f"weight hash : {wt_hash}")
        if args.wait_verified is not None:
            ok = wait_verified(args.bucket, args.prefix, slug, skel_key,
                               wt_hash, args.wait_verified)
        else:
            ok = report_state(args.bucket, args.prefix, slug, skel_key, wt_hash)
        return 0 if ok else 2

    if not args.model_dir:
        parser.error("model_dir is required unless using --unpublish-weights")

    model_dir = Path(args.model_dir.rstrip("/"))
    if not _has_image(model_dir):
        print(f"no image at {model_dir}/image/meta.json nor under "
              f"{model_dir}/{REPLICA_DIR_PREFIX}<K>/ -- nothing to publish")
        return 1

    if args.unpublish_skeleton:
        unpublish_skeleton(model_dir, args.bucket, args.prefix, key=args.key)
        return 0

    publish(model_dir, args.bucket, args.prefix,
            skeleton_only=args.skeleton_only, dry_run=args.dry_run,
            force_weights=args.force_weights,
            hash_workers=args.hash_workers,
            wait_timeout=args.wait_verified)
    return 0


if __name__ == "__main__":
    sys.exit(main())
