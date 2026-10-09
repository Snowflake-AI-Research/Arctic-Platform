"""Semi-persistence engine adapter for the ArcticInference sampling worker.

`InferenceWorker.initialize` normally builds an in-process vLLM ``AsyncLLM``.
When a job sets ``semi_p: true`` we instead **restore a pre-warmed CRIU image**
via ``arctic_platform.inference.semi_persistence.Instance`` and wrap it in
``_SemiPEngine``, which presents the subset of the ``AsyncLLM`` surface the
worker touches.

The vLLM engine itself lives out-of-process inside the ``Instance``'s vLLM
child; every call here forwards over the ``Instance`` queue/pipe (blocking, so
we run it via ``asyncio.to_thread`` and serialize with a lock).

Coverage: full ``self.llm`` surface **except weight-sync**, which raises
``NotImplementedError`` (deferred to a later take; never hit by pure sampling).
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import ssl
import stat
import struct
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from types import SimpleNamespace
from typing import Any, NamedTuple

# The library's modules import their siblings by bare name (``import
# semip_logging``), so only the attribute form of the import works: the
# package's lazy ``__getattr__`` puts the package directory on ``sys.path``
# on first attribute access. ``import
# arctic_platform.inference.semi_persistence.instance`` still raises ModuleNotFoundError.
from arctic_platform.inference.semi_persistence import Instance

logger = logging.getLogger("arctic_platform.inference.server.semip")

# collective_rpc method names that are weight-sync (RL) features. Deferred.
_WEIGHT_SYNC_RPCS = frozenset(
    {"sync_weights", "sync_weights_broadcast", "sync_spec_weights", "close_weight_sync"}
)

# engine_kwargs keys that are dss/arctic plumbing rather than vLLM engine args,
# and so must not reach the Instance's vllm_config on the dump path. The child
# builds the engine with ``LLM(**vllm_config)``, and whatever we pass is baked
# into meta.json for every later restore to be compared against, so a stray key
# is permanent rather than merely wrong once.
_NON_VLLM_ENGINE_KEYS = frozenset({
    "ray_num_gpus",
    "router_replay_max_cache_bytes",
})

# Values meaning "the operator did not ask for this". A ModelConfig default
# that vLLM rejects is dropped when it holds one of these; anything else is a
# real request we cannot honour, so it raises. See _strip_kwargs_vllm_rejects.
# "{}" is here for forest_cascade_attn_configs, whose documented default means
# "all backend defaults" and which worker.py itself pops on the non-Arctic path.
_UNSET_LIKE = (None, False, 0, "", "{}")

# Root of the derived image cache. The default is a cluster fact rather than a
# guess: /data-fast is the data-nvme volume the neutrino operator mounts into
# every device-manager pod (operator/controller/resources.go), and an image is
# only findable at the path its hashes name, so every pod has to agree on it.
# The variable still overrides it, for a run outside that layout.
_IMAGE_CACHE_ENV = "SEMIP_IMAGE_CACHE"
_DEFAULT_IMAGE_CACHE = "/data-fast/image-cache_neutrino"

# Root of the read-only mirror that published images arrive on: the operator's
# model-cache mount (resources.go's modelCacheMountPath) plus the image-cache
# directory the neutrino-model-cache DaemonSet syncs into it. A pod whose mirror
# holds nothing reads that as an ordinary miss. Setting the variable to "" turns
# the mirror off -- "cold-start on a miss", the A/B switch -- which is distinct
# from leaving it unset.
_IMAGE_SOURCE_ENV = "SEMIP_IMAGE_SOURCE"
_DEFAULT_IMAGE_SOURCE = "/mnt/neutrino/base-models/image-cache"

# Read by semi_persistence/worker._unprivileged in the Instance's worker, which
# inherits this process's environment. On by default: these pods are granted only
# CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE and the image's criu carries just
# those, so the privileged path cannot work here. The value decides what the
# dump writes (the capability level every task records), so it is recorded in
# meta.json and a restore under the other value is refused.
_UNPRIVILEGED_ENV = "SEMIP_UNPRIVILEGED"

# The published tree, which is *not* shaped like the local cache:
#
#   <source>/<model>/skeleton/<cfg12>_<env12>_<wt12>/   image/ compilation/
#   <source>/<model>/weight/<wt12>/                     shards
#
# Only the local cache has to keep the flat <cfg12>_<env12> name, because criu
# validates that exact string and the compile caches were written under it. The
# mirror is free, which is what lets one weight directory serve every backend
# image that produces the same weights. See semip_publish.py.
#
# Four names, and this module owns all four: it is the only side that composes
# both trees, the publisher holds a pinned copy, and tests/test_layout_names.py
# holds every other side to these by lifting their spellings out of the source
# -- neither Instance, the vLLM child nor the publisher can import this module.
#
# Each of these was a bare literal until now, "image" fourteen times across the
# three sides that build the tree, and that is not a cosmetic debt: it is why a
# published weight manifest whose paths were scoped to the wrong directory took
# a day to diagnose. Every pod asked S3 for
# weight/<wt12>/weights/rank0/shard_0000.bin, got a 404, and cold-started while
# the shards sat correctly beside the manifest.
_SKELETON_DIR = "skeleton"
_IMAGE_DIR = "image"
_COMPILATION_DIR = "compilation"

# One name for the shards in both trees: <model_dir>/weight/ locally and
# <source>/<model>/weight/<wt12>/ published. The local dump directory was
# "weights" until the two were unified, and the singular is now the only
# spelling -- test_layout_names refuses a bare "weights" anywhere on this side
# or the publisher's.
#
# Sharing the name does **not** make the local prefix meaningful in the bucket.
# The published directory holds the shards at its own top level, so the
# publisher still re-scopes the weight manifest's paths against it; that is what
# the 404 above was, and it is unaffected by what the directory is called.
#
# Unifying cost a re-dump of every model: weights_hash feeds this prefix into
# the hash, so every weight directory published under the plural name has a
# hash nothing produces again. See semip_publish.weights_hash.
_WEIGHT_DIR = "weight"

# Several replicas in one pod each own <key>/replica<K>/, both locally and in a
# published skeleton; a pod holding one replica keeps the flat <key>/ layout.
# K is the pod-local slot ReplicaPool assigns (replica_pool._node_slots), so
# every pod of a multi-pod job resolves the same layout a single-pod dump
# wrote. The publisher holds a pinned copy of this spelling.
_REPLICA_DIR_PREFIX = "replica"
_REPLICA_ID_ENV = "SEMIP_REPLICA_ID"
_NUM_REPLICAS_ENV = "SEMIP_NUM_REPLICAS"

# One engine spanning N pods puts each node-partition under <key>/node<k>/. This is a
# different axis from replica<K> and the two do not nest in practice: a
# pod-spanning engine is a single replica by construction (its placement group
# holds the whole world_size), so a key carries node<k>/ or replica<K>/ and
# never both.
#
# Weights stay at the key level, not under node<k>/: the shards are named by
# global rank across the whole group (rank0..rank15), and a restore onto a
# different set of pods has to find all of them in one place. The publisher holds
# a pinned copy of this spelling, like _REPLICA_DIR_PREFIX.
_NODE_DIR_PREFIX = "node"

# The interface a pod-spanning group rendezvouses and runs NCCL on. dss pins
# NCCL_SOCKET_IFNAME=^lo into a multi-pod job's extra_env, which names no
# interface; semi-p needs one it can bind, account for in the socket census and
# close before the dump.
_MULTINODE_IFNAME_ENV = "SEMIP_IFNAME"
_DEFAULT_MULTINODE_IFNAME = "eth0"

# A multi-pod cold start has to outlast its slowest node-partition's weight load. On a
# pod whose page cache is cold that was 25 minutes for GLM-5.3, and gloo's own
# rendezvous timeout is 1800 s -- so a shorter wait here would fail jobs that
# were about to succeed, and one no longer than gloo's would race it.
_MULTINODE_INIT_TIMEOUT_S = 3600.0

# Everything else a node-partition does is local work on an image that already exists.
_MULTINODE_STEP_TIMEOUT_S = 900.0


def _multinode_ifname() -> str:
    return (os.environ.get(_MULTINODE_IFNAME_ENV)
            or _DEFAULT_MULTINODE_IFNAME).strip() or _DEFAULT_MULTINODE_IFNAME

# Stamped by the neutrino-model-cache DaemonSet once it has verified every file
# in a directory against the per-file SHA-256 in that directory's bucket
# manifest. Gating the copy on this rather than on image/meta.json existing is
# what keeps us from copying a sync in progress: the manifest is written last,
# but until then the directory can hold any subset of the payload. Observed
# live -- at 43 GB of a 77 GB first sync, image/meta.json was already present
# and readable with the right model_dir, and 34 GB of weights were still in
# flight.
_VERIFIED_MARKER = ".neutrino_verified"

# The marker carries {"manifest_digest": ..., "verified_at": <epoch>} and the
# daemon writes it *after* the payload, so nothing under the directory should be
# newer than that stamp. Anything newer means a re-sync has touched the
# directory since it was verified -- which marker *presence* alone cannot catch,
# because the previous pass's marker survives while new bytes land on top.
# Tolerance covers filesystem timestamp granularity, not clock skew: both the
# stamp and the mtimes come from this node.
_VERIFY_SKEW_S = 5.0

# Where a copy is assembled before being flipped into place. Dot-prefixed for
# the same reason as the dump lock: image/, weight/ and compilation/ are names
# Instance derives from model_dir, and a fourth entry must not look like one.
_INCOMING_NAME = ".incoming"

# What a materialize copies, in flip order.
#
# image/ is CRIU's -D dir, which a restore writes into (restore.log, pidfile);
# compilation/ is the vLLM child's live compile cache and the one directory
# whose absolute paths CRIU baked as mmaps. Both are bound to model_dir, so
# both have to be local. weight/ is absent on purpose -- it carries no baked
# path and is read in place. See _weights_dir_for_restore.
#
# image/ is last because it holds the hit predicate (image/meta.json), so a
# materialize interrupted anywhere leaves a directory that reads as a miss --
# which is why this order is spelled out here rather than reusing the
# publisher's SKELETON_DIRS, whose order is only its upload order.
_COPY_SUBDIRS = (_COMPILATION_DIR, _IMAGE_DIR)

# Written by _semip_save_weights beside the shards it indexes; TP>1 fans out
# into rank<N>/ subdirectories with one of these each.
_WEIGHTS_MANIFEST = "weights_meta.json"

# How many mismatching mappings to name in a log line before summarizing.
_ENV_FILES_REPORT = 5

# Both halves of the cache key are truncated to this many hex digits. Long
# enough that a collision is not a practical concern, short enough that the
# directory name stays readable in a log line.
_HASH_LEN = 12

# Where the pod's own identity is readable from inside the container. The
# ServiceAccount token authenticates a GET of this pod's object, whose
# containerStatuses carry the image *digest* -- unlike the tag, which moves.
_SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
_K8S_API = "https://kubernetes.default.svc"

# Which containerStatuses entry is ours. Overridable because the name is a
# deployment detail, not something this code can derive: a pod may carry
# sidecars whose images move independently of the one we run in.
_CONTAINER_NAME_ENV = "SEMIP_CONTAINER_NAME"
_DEFAULT_CONTAINER_NAME = "device-manager"

# The NVIDIA kernel module version, which is the driver the container toolkit
# bind-mounts libcuda.so.<version> in from. Read from procfs rather than NVML
# so the lookup costs nothing and needs no CUDA context.
_NVIDIA_VERSION_PROC = "/proc/driver/nvidia/version"
_NVRM_VERSION_RE = re.compile(r"NVRM version:.*?(\d+\.\d+(?:\.\d+)?)")

# One short generation between the cold start and the dump. Cheap, and it means
# a broken engine fails before we spend an image on it rather than after.
_DUMP_PROMPT = "Hi, can you introduce yourself?"
_DUMP_SAMPLING = {"max_tokens": 32, "temperature": 0.0, "ignore_eos": True}

# Guards one model_dir against two jobs dumping into it at once. Dot-prefixed so
# it cannot collide with image/, weight/ or compilation/, the three names
# Instance derives from model_dir; and it lives in model_dir rather than /tmp
# because model_dir is the resource being protected. Safe from the library's own
# cleanup, which only removes those subdirectories and never model_dir itself.
_DUMP_LOCK_NAME = ".dump.lock"

# How long to wait for another job's dump of the same model_dir before giving
# up. Generous on purpose: the holder may be paying a first-time hub download
# on top of a cold start and a dump.
_DUMP_LOCK_WAIT_S = 3600.0
_DUMP_LOCK_POLL_S = 5.0

# How long a lock file may stay empty before we call it abandoned. It is only
# unwritten for the microseconds between the O_EXCL create and the pid write, so
# anything past this means its creator died in between.
_DUMP_LOCK_GRACE_S = 30.0

# The PID namespace counter is advanced past this floor before a dump, so the
# image records task ids that nothing in a *fresh* pod can be sitting on.
#
# Every device-manager container gets its own PID namespace and runs the same
# workload, so pids land in the same low range every time: the Ray worker that
# drives a restore is reliably in the hundreds with threads reaching past 1100,
# and an image dumped in an identically shaped pod recorded 1105-1340. Those
# ranges overlap by construction, and the occupant cannot be evicted -- it is a
# thread of the actor performing the restore. Nor can an image's ids be
# rewritten afterwards: they live in the dumped processes' own memory (each
# glibc pthread descriptor holds its tid, robust and PI mutexes hold owner tids
# in the futex word) and in names such as /dev/shm/sem.*, not only in
# pstree.img. Choosing the ids before the processes exist is the only fix.
_PID_FLOOR_ENV = "SEMIP_PID_FLOOR"
_PID_FLOOR_DEFAULT = 100_000

# Room the floor must leave under pid_max for the dumped tree and for whatever
# the pod does afterwards. Nothing allocates it; it only decides whether a
# floor is worth applying at all. A node with the legacy pid_max of 32768 has
# no safe floor and is refused here rather than half-served.
_PID_FLOOR_HEADROOM = 100_000

# How many processes fork in parallel to advance the counter. The counter is
# namespace-global, so burners scale until they saturate the pod's CPU quota:
# measured on an H200 node against a quota of 8 CPUs, 20k ids cost 10.8s with
# one burner, 3.7s with 4 and 1.9s with 8, and 16 was no better than 8. A
# single fork costs ~540us there whatever the forking process's size, which is
# what makes a serial burn to a six-figure floor a minute-scale expense.
_PID_BURN_WORKERS_ENV = "SEMIP_PID_BURN_WORKERS"
_PID_BURN_WORKERS_DEFAULT = 8

# Forks a burner may spend beyond the ids it set out to consume. Every fork
# advances the counter by at least one, so this is only reached if the counter
# wraps -- and then the burn is abandoned rather than run unbounded.
_PID_BURN_SLACK = 10_000

# Bounds a burn that cannot make progress at all. The measured cost of a 100k
# burn is ~10s.
_PID_BURN_TIMEOUT_S = 300.0

# Written per PID namespace; read back to confirm a burn. Module constants so
# the tests can point them at ordinary files.
_NS_LAST_PID_PATH = "/proc/sys/kernel/ns_last_pid"
_PID_MAX_PATH = "/proc/sys/kernel/pid_max"

# The burn loop, run in a fresh interpreter and never in this process: the
# caller is a Ray actor with threads, and os.fork() runs CPython's at-fork
# handlers in the child, which deadlock if another thread held an internal lock
# at the moment of the fork. subprocess reaches a new process through
# fork+exec, which is safe from a threaded parent, and the counter is
# namespace-global so a burn there carries over to everything this process
# spawns afterwards. It also keeps each fork cheap, since a fresh interpreter
# has none of the actor's footprint to copy.
#
# Kept as source text rather than as a call into scripts/pidcheck.py --burn-to,
# which is the same operation: that directory is not in pyproject's packages
# and has no __init__.py, so it does not ship in the image. The script remains
# the manual form, for a node being investigated by hand.
_PID_BURN_SOURCE = r'''
import os, sys

target, workers, limit = (int(arg) for arg in sys.argv[1:4])


def burn():
    """Consume ids until one comes back past the target, or limit is spent."""
    pid = 0
    for _ in range(limit):
        pid = os.fork()
        if pid == 0:
            os._exit(0)
        os.waitpid(pid, 0)
        if pid >= target:
            break
    return pid


kids = []
for _ in range(workers - 1):
    child = os.fork()
    if child == 0:
        try:
            burn()
        finally:
            os._exit(0)          # never unwind back into the parent's code
    kids.append(child)
last = burn()
for child in kids:
    os.waitpid(child, 0)
print(last)
'''

# TCP_TIMEWAIT_LEN, hard-coded in the kernel (tcp_fin_timeout is a different
# timer). CRIU rebinds every recorded local port at restore, and a destructive
# dump -- or the teardown of a restored tree -- leaves those tuples in
# TIME_WAIT for this long. See _restore_with_port_retry.
_TIME_WAIT_S = 60.0
_RESTORE_RETRY_SLEEP_S = 10.0

# SamplingParams fields the Instance's vLLM child must not inherit: it reads
# cumulative outputs once a request finishes, whatever the caller streams.
_SAMPLING_PARAM_SKIP = frozenset({"output_kind", "stream_interval", "skip_clone"})


def _sampling_params_to_dict(params: Any) -> dict[str, Any]:
    """A vLLM ``SamplingParams`` as kwargs the child rebuilds it from.

    Every field a caller can set (those ``from_optional`` takes) is copied,
    ``None`` included: an explicit ``max_tokens=None`` means unbounded, where
    an omitted one means vLLM's default of 16.
    """
    if isinstance(params, dict):
        return dict(params)
    import inspect
    names = inspect.signature(type(params).from_optional).parameters
    return {name: getattr(params, name) for name in names
            if name not in _SAMPLING_PARAM_SKIP and hasattr(params, name)}


def _log_config_divergence(baked: dict[str, Any],
                           engine_kwargs: dict[str, Any],
                           model_dir: str,
                           requested: dict[str, Any] | None = None) -> None:
    """Record that the image's config wins over the job's ``vllm_config``.

    ``_restore`` hands ``baked`` to the ``Instance``, and ``criu_restore``
    compares it against the image by exact dict equality, so passing anything
    else would raise. The consequence is that a job's own ``vllm_config`` has no
    effect at all; log the divergence rather than let it pass unremarked.

    Since the image directory is derived from ``_config_hash``, a job whose
    config differs resolves to a *different* directory and never reaches this
    function -- so in normal operation it has nothing to report. What it still
    catches is the case where that is untrue: an image dumped before the
    directory encoded the config, a hash collision, or a hand-placed directory.
    Treat any output from it as evidence that the key is not doing its job,
    rather than as the routine drift it reported when a bare path was the whole
    cache key.

    Two deliberate choices, both corrections of an earlier version:

    * **Symmetric.** Iterating ``baked`` alone and requiring the key in both
      silently ignored every key the job asked for that the image had never
      heard of -- which was the common case, not the rare one, against a
      hand-dumped image whose baked config held two keys against a job JSON
      asking for six.
    * **``model`` included.** It used to be excluded because dss resolved it to
      a local path that could never equal the baked HF id. Under
      ``DSS_ALLOW_REMOTE_MODELS`` it stays an HF id on both sides -- what both
      the dump path here and the scripts bake -- so a mismatch is a real one (a
      job pointed at another model's image) rather than a formatting artefact.

    Keys present in the image but not requested are not reported: the image
    knowing more than the job asked is normal and not drift.

    ``requested`` is the job's config **after** ``_vllm_config_from_engine_kwargs``,
    which is the only thing comparable to ``baked``. Comparing the raw
    ``engine_kwargs`` instead made this fire on every single restore: the
    projection drops keys vLLM rejects at unset-like values, so
    ``fp32_lm_head=False`` was dumped as absent and then reported as "image has
    no such key" -- a message telling the reader to suspect the cache key, for
    a key that by construction can never be in the image. Callers pass the dict
    they already projected; the fallback re-projects rather than silently
    comparing the wrong shapes.
    """
    logger.info("semi_p: baked vllm_config=%s", baked)
    if requested is None:
        requested = _vllm_config_from_engine_kwargs(engine_kwargs)
    diverged = []
    for key in sorted(requested):
        if key not in baked:
            diverged.append(
                f"{key}: requested={requested[key]!r}, image has no such key")
        elif requested[key] != baked[key]:
            diverged.append(
                f"{key}: requested={requested[key]!r} baked={baked[key]!r}")
    if diverged:
        logger.warning(
            "semi_p: image config wins, ignoring requested %s. This should be "
            "unreachable -- %s is named after a hash of the config, so a "
            "differing config should have resolved elsewhere. Remove the "
            "directory to force a re-dump, and treat this as a bug in the key.",
            "; ".join(diverged), model_dir)


def _vllm_config_from_engine_kwargs(
        engine_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Project the job's ``engine_kwargs`` into a vLLM config the child accepts.

    Runs on **every** path, not just the dump. A restore still reads ``baked``
    out of meta.json to satisfy ``criu_restore``'s exact-equality check, but the
    result here is what ``_config_hash`` names the image directory after, so a
    lookup has to build it before it knows whether an image exists. The input
    needs projecting because ``engine_kwargs`` is not the right shape:
    ``_prepare_sampling_engine_kwargs`` can nest a whole dict under
    ``extra_engine_kwargs``, while the child builds the engine with
    ``LLM(**vllm_config)`` and would reject that as an unknown kwarg.

    The projection **must be reproducible**, which it was not required to be
    when only the dump path called it: a later job has to derive a byte-identical
    dict from its own ``engine_kwargs`` or it will not find the image. It also
    has one hard requirement inherited from before -- the child must be able to
    build an engine from the result -- which ``_strip_kwargs_vllm_rejects``
    enforces before we spend a cold start on it.

    Placement is deliberately *not* an input **to the projection**, which must
    stay equal to what ``meta.json`` records. The pod's device allocation is
    part of the key at TP>1 all the same, folded into the hash beside this
    dict by ``_device_binding`` -- see ``_config_hash``. Ray's GPU *ids* never
    enter either one; they are validated against ``tensor_parallel_size`` in
    ``_check_tp_matches_gpus`` and otherwise discarded, because an image
    restores onto any permutation of the devices it was dumped on.

    Deliberately *not* injected here:

    * ``worker_cls``. ``Instance.init`` adds it for TP>1 to a *copy* of the
      config, so it reaches the child without entering ``self.vllm_config``, and
      is correctly absent from meta.json. Adding it ourselves would bake it.
    * ``enable_sleep_mode`` and the TP>1 collective-path tweaks, which the child
      likewise applies to its own copy at init.
    """
    projected: dict[str, Any] = {}
    for key, val in engine_kwargs.items():
        if key == "extra_engine_kwargs":
            continue  # flattened below
        if key in _NON_VLLM_ENGINE_KEYS:
            logger.info("semi_p: dump omitting non-vLLM engine kwarg %r", key)
            continue
        projected[key] = val

    # Flatten. sampling.py moved these *out of* the user's vllm_config, so they
    # are genuine vLLM kwargs and the nesting is a dss transport detail.
    for key, val in (engine_kwargs.get("extra_engine_kwargs") or {}).items():
        if key in _NON_VLLM_ENGINE_KEYS:
            logger.info("semi_p: dump omitting non-vLLM engine kwarg %r "
                        "(from extra_engine_kwargs)", key)
            continue
        if key in projected and projected[key] != val:
            logger.warning(
                "semi_p: extra_engine_kwargs[%r]=%r overrides top-level %r",
                key, val, projected[key])
        projected[key] = val

    if not projected.get("model"):
        raise ValueError(
            "semi_p: cannot dump without a model; engine_kwargs carried no "
            "'model'. dss sets it from inference_config's model_name via "
            "resolve_model_path")

    projected = _strip_kwargs_vllm_rejects(projected)
    logger.info("semi_p: vllm_config=%s", projected)
    return projected


def _check_tp_matches_gpus(vllm_config: dict[str, Any],
                           gpus: list[int], nnodes: int = 1) -> None:
    """Reject a job whose TP degree disagrees with its GPU assignment.

    TP size comes from the config and the GPU list is placement only, so a
    disagreement is a job-config error rather than something to reconcile.
    ``Instance.init`` checks this too, but only after the zone is up and the
    worker is spawned; say it here, naming both fields the operator wrote.

    Split out of the projection so the *count* stays a job-config error rather
    than a cache outcome. The devices themselves do reach the key at TP>1, via
    ``_device_binding`` and not through ``vllm_config``, which has to keep
    matching ``meta.json`` byte for byte for ``criu_restore``. So an image is
    indeed found only by a job that landed on the same devices -- deliberately,
    because it is restorable only there.

    At ``nnodes > 1`` the TP group is split across pods, so what this pod
    holds is ``tp / nnodes`` GPUs. The split has to be exact: a TP group whose
    node-partitions have different rank counts deadlocks in its first collective rather
    than failing, which is the kind of error worth catching before the zone is
    even up.
    """
    tp = int(vllm_config.get("tensor_parallel_size", 1) or 1)
    nnodes = int(nnodes or 1)
    if nnodes > 1 and tp % nnodes:
        raise RuntimeError(
            f"semi_p: vllm_config.tensor_parallel_size={tp} does not divide "
            f"evenly over {nnodes} node-partitions, so they would hold different "
            f"numbers of ranks and deadlock in their first collective")
    expected = tp // nnodes
    if expected != len(gpus):
        where = (f" on this node (tensor_parallel_size={tp} over {nnodes} "
                 f"nodes)" if nnodes > 1 else "")
        raise RuntimeError(
            f"semi_p: expected {expected} GPU(s){where} but Ray assigned "
            f"this replica {len(gpus)} GPU(s) ({gpus}). Each replica gets "
            f"exactly tensor_parallel_size GPUs, so inference_config.n_gpus "
            f"must be a multiple of inference_config.vllm_config."
            f"tensor_parallel_size")


def _strip_kwargs_vllm_rejects(vllm_config: dict[str, Any]) -> dict[str, Any]:
    """Drop Arctic-only keys the semi-p child could not build an engine from.

    Dump path only, which is why the vLLM import sits here rather than at
    module scope: a restore never needs vLLM in this process (the engine lives
    out-of-process in the Instance child), and that stays true.

    ``engine_kwargs`` is a serialized ``server/config.py:ModelConfig``, and
    several of its fields are not vLLM engine args at all -- ``fp32_lm_head``,
    ``forest_cascade_attn_configs``, and whatever gets added next. The cold path
    copes by routing to an Arctic-aware ``EngineArgs`` subclass
    (``Fp32LmHeadAsyncEngineArgs``) or by popping the key *after* the semi_p
    early return. Neither helps us: the Instance child builds its engine with a
    plain ``LLM(**vllm_config)`` after only ``load_general_plugins()``, and on
    this pod that adds no Arctic fields whatsoever -- ``ARCTIC_INFERENCE_ENABLED``
    is unset, which also makes ``_ensure_arctic_vllm_patches()`` a no-op. So a
    key vLLM rejects here is a key that would break the child's cold start.

    **Drop what was not asked for; raise on what was.** A rejected key still
    holding its ``ModelConfig`` default is dropped and logged: the hand-written
    dump scripts never passed these, and no existing baked image records them.
    A rejected key holding a real value raises, because we cannot honour it and
    baking the config minus that key would quietly serve something other than
    what the job asked for -- permanently, since every later restore compares
    against what we write.

    Best effort about the rest: if vLLM will not import, or the constructor
    objects to something other than a bad kwarg name, log and let the child be
    the judge rather than fail a dump over a preflight. Only the constructor
    runs -- never ``create_engine_config()``, which inspects the model and would
    reach the network.
    """
    try:
        import vllm.plugins
        vllm.plugins.load_general_plugins()
        from vllm.engine.arg_utils import AsyncEngineArgs
    except Exception:
        logger.warning(
            "semi_p: could not import vLLM to preflight the dump config; the "
            "Instance child will validate it instead", exc_info=True)
        return vllm_config

    probe = dict(vllm_config)
    # Bounded: every pass removes exactly one key, so it cannot outlive the dict.
    for _ in range(len(probe) + 1):
        try:
            AsyncEngineArgs(**probe)
            return probe
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            key = match.group(1) if match else None
            if key is None or key not in probe:
                raise ValueError(
                    f"semi_p: refusing to dump a vllm_config vLLM will not "
                    f"accept: {exc}") from exc
            value = probe.pop(key)
            if not any(value is unset or value == unset for unset in _UNSET_LIKE):
                raise ValueError(
                    f"semi_p: cannot dump with {key}={value!r}. vLLM's engine "
                    f"args do not accept it, and the semi-p child builds its "
                    f"engine with a plain LLM(**vllm_config) -- no Arctic-aware "
                    f"EngineArgs subclass, and no Arctic plugin fields because "
                    f"ARCTIC_INFERENCE_ENABLED is unset. Dropping it would bake "
                    f"a config that silently does something other than what the "
                    f"job asked for. Remove it from inference_config, or run "
                    f"this as a cold (non-semi_p) job.") from exc
            logger.info(
                "semi_p: dropping %r=%r from the dump config -- not a vLLM "
                "engine arg, and left at its default", key, value)
        except Exception:
            logger.warning(
                "semi_p: dump config preflight raised something other than a "
                "bad kwarg name; proceeding and letting the child validate",
                exc_info=True)
            return probe
    return probe


def _config_hash(vllm_config: dict[str, Any],
                 device_nodes: list[str] | None = None) -> str:
    """Hash the engine config half of the cache key.

    Taken over the dict exactly as it goes to ``Instance(vllm_config, ...)``
    and ``LLM(**vllm_config)``, which is also what ``meta.json`` records -- so
    the config printed beside an image is the config that named its directory.

    Canonical *serialization* (sorted keys, stable separators) only, and no
    normalization of any value. In particular ``model`` keeps its absolute
    path. Rewriting it to a bare model name would assert that two
    differently-rooted paths are the same model, and every such claim of
    equivalence is a chance to alias two genuinely different configs onto one
    directory. A changed model root causing a miss is the harmless direction;
    two models sharing a directory is not.

    ``device_nodes`` is the pod's ``/dev`` allocation at TP>1 and ``None``
    below it -- ``_device_binding`` decides which, and says why. It is folded
    in here rather than added as a third component of the key so the directory
    name stays ``<cfg12>_<env12>``: criu validates that exact string, the
    compile caches are written under it, and ``semip_publish._DERIVED_KEY_RE``
    matches it. Widening the name would touch all three and buy nothing, since
    nothing reads the device set back out of it -- ``meta.json`` records it in
    full, and that is what ``_missing_device_nodes`` compares.

    **The set is sorted here rather than trusted from the caller**, because it
    is a set and not a sequence. ``_missing_device_nodes`` tests membership,
    and an image restores onto the same devices in a different order --
    measured, one dumped on ``[3, 1, 2, 0]`` came back on ``[3, 2, 0, 1]`` --
    because ``_gpu_migration_permutation`` builds the bijection. Hashing the
    caller's order would split one image into as many as ``TP!`` keys that a
    single one of them already satisfies.
    """
    canonical = json.dumps(vllm_config, sort_keys=True, separators=(",", ":"))
    if device_nodes is not None:
        # NUL-separated because no device node can contain one, so no device
        # set can collide with a config whose serialization ends in its text.
        canonical += "\0" + ",".join(sorted(device_nodes))
    return hashlib.sha256(canonical.encode()).hexdigest()[:_HASH_LEN]


def _device_binding(vllm_config: dict[str, Any],
                    replica_count: int = 1) -> list[str] | None:
    """The device nodes an image for this config will be bound to, or ``None``.

    An image is restorable only in a pod that can reopen every device node its
    captured state names, so at TP>1 the allocation is part of the image's
    identity and belongs in the key. Two consequences, both wanted:

    * A dump writes to a directory named for the devices it ran on, so two
      allocations of one config accumulate as separate images instead of the
      second overwriting the first. Coverage ratchets up, where before it was
      a lottery that exactly one placement could win.
    * A lookup matches a published skeleton only when this pod holds the
      devices that skeleton needs, which makes "found" and "restorable" the
      same statement. ``_check_device_visibility`` remains as a backstop for
      images that predate the recorded set, but on the published path it can
      no longer be the thing that fails.

    **``None`` for a lone TP=1 replica.** A TP=1 image has no communicator to
    rebuild and ``cuda_restore`` renumbers it onto whatever slot it lands on,
    so it restores anywhere and its allocation is not part of its identity.
    Binding it would split one universally restorable image into one key per
    GPU, turning a hit rate of 1 into 1/N.

    **Bound for TP=1 when the pod holds several replicas**, and for a different
    reason: not restorability but layout. Those replicas dump under
    ``<key>/replica<K>/`` where a lone replica dumps flat under ``<key>/``, and
    without the device set a 1-GPU and an 8-GPU pod would give both shapes one
    key. The pod's allocation tells them apart, as it already does at TP>1.
    ``_missing_device_nodes`` keeps exempting TP=1, since such an image still
    restores on any slot; at TP>1 the key and the check share
    ``_DEVICE_BOUND_MIN_TP``.
    """
    tp = int((vllm_config or {}).get("tensor_parallel_size", 1) or 1)
    if tp < _DEVICE_BOUND_MIN_TP and replica_count <= 1:
        return None
    return _visible_device_nodes()


def _pod_image_ref() -> str:
    """This container's image reference, including its registry digest.

    Read from the pod's own object through the in-cluster API, using the
    ServiceAccount token. ``status.containerStatuses[].imageID`` carries the
    content digest; ``.image`` carries the tag, which is mutable and would
    alias two genuinely different builds.

    Matches our container **by name and raises** when it is absent, rather than
    falling back to the first entry. A pod carries sidecars whose images move
    independently of ours, so a positional guess would silently key the cache
    on the wrong image -- a wrong key with no symptom, which is the one failure
    mode this design refuses.
    """
    token = _read_text(os.path.join(_SA_DIR, "token"))
    namespace = _read_text(os.path.join(_SA_DIR, "namespace"))
    pod = os.environ.get("HOSTNAME", "").strip()
    if not pod:
        raise RuntimeError(
            "semi_p: cannot identify this pod -- HOSTNAME is unset, so the "
            "image digest that keys the cache cannot be read")

    url = f"{_K8S_API}/api/v1/namespaces/{namespace}/pods/{pod}"
    context = ssl.create_default_context(cafile=os.path.join(_SA_DIR, "ca.crt"))
    request = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(request, context=context, timeout=30) as resp:
        body = json.load(resp)

    want = os.environ.get(_CONTAINER_NAME_ENV) or _DEFAULT_CONTAINER_NAME
    statuses = body.get("status", {}).get("containerStatuses") or []
    for status in statuses:
        if status.get("name") == want:
            image_id = status.get("imageID") or ""
            if "sha256:" not in image_id:
                raise RuntimeError(
                    f"semi_p: container {want!r} reports imageID {image_id!r}, "
                    f"which carries no digest; the cache cannot be keyed on a "
                    f"mutable tag")
            return image_id
    raise RuntimeError(
        f"semi_p: no container named {want!r} on this pod (found "
        f"{[s.get('name') for s in statuses]}). Set {_CONTAINER_NAME_ENV} to "
        f"the container this code runs in -- guessing would key the image "
        f"cache on a sidecar whose image moves independently")


def _driver_version() -> str:
    """The NVIDIA kernel module version, e.g. ``580.159.03``.

    Part of the environment key because the driver is **not** part of the
    container image: the NVIDIA container toolkit bind-mounts the host's
    ``libcuda.so.<version>`` into the container, and CRIU records that mapping
    like any other. Two nodes running one image with different drivers are two
    different environments, and without this term they would claim the same
    directory and overwrite each other's images indefinitely.
    """
    text = _read_text(_NVIDIA_VERSION_PROC)
    match = _NVRM_VERSION_RE.search(text)
    if not match:
        raise RuntimeError(
            f"semi_p: could not parse a driver version out of "
            f"{_NVIDIA_VERSION_PROC}: {text.splitlines()[:1]}")
    return match.group(1)


def _read_text(path: str) -> str:
    try:
        with open(path) as handle:
            return handle.read().strip()
    except OSError as exc:
        raise RuntimeError(
            f"semi_p: cannot read {path}, which the image cache key is derived "
            f"from: {exc}") from exc


def _env_hash() -> tuple[str, str, str]:
    """Return ``(hash, image_ref, driver_version)`` for this environment.

    The two inputs are exact identifiers, not samples: the image digest fixes
    every file the image ships (layers are content-addressed), and the driver
    version fixes the one mapped file it does not. Together they cover what
    CRIU re-validates at restore, so two distinct environments can never name
    the same directory.

    That exactness is what makes a miss *productive*. A key derived from a
    guessed sample of the filesystem could collide, and a collision has nowhere
    else to put the new image: the cold start would overwrite the image it just
    rejected, and two pods in that state would overwrite each other forever.

    There is deliberately no fallback. A partial key would reintroduce exactly
    that collision, so a pod that cannot identify its own environment fails
    instead of caching against a weaker name.
    """
    image_ref = _pod_image_ref()
    driver = _driver_version()
    digest = image_ref.split("sha256:", 1)[1]
    combined = hashlib.sha256(f"{digest}{driver}".encode()).hexdigest()
    return combined[:_HASH_LEN], image_ref, driver


class _ImagePaths(NamedTuple):
    """Everything the cache key names, resolved once.

    ``skeleton_root`` and ``weight_root`` are ``None`` when no mirror is
    configured, which is a supported deployment rather than an error.

    There is deliberately no ``source_dir`` here. A published skeleton's name
    carries the weight hash it was dumped with, and that hash is a content hash
    of the staged buffer -- so unlike the cfg and env halves it cannot be
    *derived*, only discovered by listing. ``_resolve_published_skeleton`` does
    that; this type stops at what a pure function of the config can name.

    ``image_ref`` and ``driver_version`` are carried into ``meta.json`` because
    the truncated hash in the path cannot be reversed into them, and a future
    mismatch should be reportable by name.

    ``replica_slot`` / ``replica_count`` are this replica's place in its pod
    (see ``_replica_slot``); with a count above 1 ``model_dir`` ends in
    ``replica<slot>``.

    ``key_dir`` is ``model_dir`` without any ``replica<K>`` / ``node<k>``
    level. A pod-spanning engine needs it because its node-partitions share one
    weight directory there, named by global rank rather than by pod.
    """
    model_dir: str
    key_prefix: str
    skeleton_root: str | None
    weight_root: str | None
    image_ref: str
    driver_version: str
    replica_slot: int = 0
    replica_count: int = 1
    key_dir: str = ""
    nnodes: int = 1
    node_rank: int = 0


def _replica_slot() -> tuple[int, int]:
    """``(slot, replicas in this pod)`` from the env ReplicaPool sets.

    A count of 1 (or unset) is a lone replica and the flat layout. Above 1 the
    slot is required: replicas guessing a shared one would restore one image
    twice in one pod, onto the same recorded task ids.
    """
    count = _env_int(_NUM_REPLICAS_ENV, 1)
    if count <= 1:
        return 0, 1
    raw = (os.environ.get(_REPLICA_ID_ENV) or "").strip()
    try:
        slot = int(raw)
    except ValueError:
        slot = -1
    if not 0 <= slot < count:
        raise RuntimeError(
            f"semi_p: {_NUM_REPLICAS_ENV}={count} but {_REPLICA_ID_ENV}={raw!r} "
            f"is not a slot in 0..{count - 1}. ReplicaPool sets both; a replica "
            f"without its own slot would share another's image directory")
    return slot, count


def _resolve_model_dir(vllm_config: dict[str, Any],
                       node_rank: int = 0) -> _ImagePaths:
    """Derive the image directory from the config and the environment.

    ``model_dir`` is derived rather than supplied, which is what lets a dump
    and a later restore agree on a path without an operator assigning one. It
    also means the path is identical in every pod for a given config and image,
    since the cache root is a fixed mount point.

    The read-only mirror's two roots are composed here too, from the *same*
    key, so nothing else has to recompute them. Two places deriving one key is
    the only way the roots could ever name different directories.
    """
    root = (os.environ.get(_IMAGE_CACHE_ENV)
            or _DEFAULT_IMAGE_CACHE).strip().rstrip("/")
    if not root:
        raise ValueError(
            f"semi_p: {_IMAGE_CACHE_ENV}={os.environ.get(_IMAGE_CACHE_ENV)!r} "
            f"names no directory to look for or write an image in; unset it "
            f"for the default {_DEFAULT_IMAGE_CACHE}")

    slot, count = _replica_slot()
    device_nodes = _device_binding(vllm_config, count)
    cfg_hash = _config_hash(vllm_config, device_nodes)
    env_hash, image_ref, driver = _env_hash()
    key = f"{cfg_hash}_{env_hash}"
    key_dir = os.path.join(root, key)
    model_dir = key_dir
    if count > 1:
        model_dir = os.path.join(model_dir, f"{_REPLICA_DIR_PREFIX}{slot}")
    # A pod-spanning engine puts each node-partition under ``node<k>/`` of one
    # shared key. The key is identical on every pod -- ``nnodes`` is in the config and
    # pod identity deliberately is not, and every pod exposes the same device
    # set (nvidia0-7, uverbs0-15) -- so this level is what keeps the
    # node-partitions of one image from writing over each other. Their weights stay at the key
    # level, because the shards are per rank across the whole group rather than
    # per pod.
    nnodes = int((vllm_config or {}).get("nnodes", 1) or 1)
    if nnodes > 1:
        model_dir = os.path.join(model_dir, f"{_NODE_DIR_PREFIX}{node_rank}")

    raw_source = os.environ.get(_IMAGE_SOURCE_ENV)
    if raw_source is None:
        raw_source = _DEFAULT_IMAGE_SOURCE
    source_root = raw_source.strip().rstrip("/")
    if source_root == root:
        # The mirror is read-only and the cache is written to, so one path
        # cannot be both. Copying a directory onto itself would at best waste
        # the work and at worst destroy the image mid-flip.
        logger.warning(
            "semi_p: %s and %s are both %s; ignoring the image source, since a "
            "directory cannot be materialized from itself",
            _IMAGE_SOURCE_ENV, _IMAGE_CACHE_ENV, root)
        source_root = ""

    # The model path groups a model's skeletons and weights together in the
    # bucket. Only grouping depends on it -- the hashes carry the identity -- but
    # it has to match what semip_publish wrote, which reads the same field out of
    # meta.json. A config with no model cannot be resolved against the mirror at
    # all, so treat it as "no mirror" rather than guessing a path.
    slug = _model_slug((vllm_config or {}).get("model"))
    if source_root and not slug:
        logger.warning(
            "semi_p: vllm_config carries no model, so the published tree cannot "
            "be addressed; treating this as a cache miss")
        source_root = ""

    skeleton_root = (os.path.join(source_root, slug, _SKELETON_DIR)
                     if source_root else None)
    weight_root = (os.path.join(source_root, slug, _WEIGHT_DIR)
                   if source_root else None)

    logger.info(
        "semi_p: resolved model_dir=%s (config=%s over %s, env=%s from image "
        "%s driver %s, replica slot %d of %d%s); published skeletons %s",
        model_dir, cfg_hash,
        f"devices {','.join(device_nodes)}" if device_nodes
        else "no device binding (TP=1)", env_hash, image_ref, driver,
        slot, count,
        f", node {node_rank} of {nnodes}" if nnodes > 1 else "",
        skeleton_root or "not configured")
    return _ImagePaths(model_dir, key, skeleton_root, weight_root,
                       image_ref, driver, slot, count, key_dir, nnodes,
                       node_rank)


def _model_slug(model: Any) -> str:
    """The published path component for a model: the **name** alone.

    ``vllm_config["model"]`` is an absolute path -- dss resolves it through
    ``resolve_model_path`` before the engine ever sees it -- so using it verbatim
    nests the mirror's own mount inside itself
    (``image-cache/mnt/neutrino/base-models/Qwen/...``), and worse, would fork the
    shared weight directory if that mount ever moved, re-uploading identical bytes
    under a second path.

    The org was dropped from the tail deliberately: one level reads better and
    the catalog's names are unique in practice. It costs nothing that was load-
    bearing. Only grouping and readability depend on this -- identity comes from
    the weight hash, and the skeleton key carries ``cfg12`` over the **full**
    absolute path -- so two orgs shipping one name would share a directory but
    could never share a key or a weight hash, and so never each other's bytes.

    **``semip_publish.model_slug`` must compute exactly this.** Publish writes
    where the engine reads, so a disagreement makes every job resolve a permanent
    miss -- silently, since a miss is a supported outcome. The two are duplicated
    rather than shared because ``semip_publish`` is a stdlib-only script that runs
    by path inside a pod, and ``test_publish_layout`` cross-checks them against a
    table of inputs so drift cannot merge unnoticed.
    """
    parts = [p for p in str(model or "").strip("/").split("/")
             if p and p not in (".", "..")]
    return parts[-1] if parts else ""


def _resolve_published_skeleton(paths: _ImagePaths) -> tuple[str | None,
                                                             str | None]:
    """Find the published skeleton for this config, and the weights it names.

    Returns ``(skeleton_dir, weight_hash)``, both ``None`` when there is nothing
    usable. The weight hash is read out of the directory *name*: a skeleton is
    published as ``<cfg12>_<env12>_<wt12>``, so the binding needs no file inside
    it, cannot be edited out from under a digest-verified manifest, and cannot
    disagree with the weights, since the dump that produced one produced both.

    Answered by listing rather than by deriving, because a content hash of the
    staged weight buffer is exactly what a restore does not have -- producing it
    is the cold start being avoided.

    **More than one match is a refusal.** Two weight versions of one config and
    backend is a deliberate state (it is how a rollback is kept), but which one a
    job should get is an operator's decision, not a tie-break to invent here. So
    this logs the candidates and reports a miss, which cold-starts.
    """
    root = paths.skeleton_root
    if not root:
        return None, None
    want = paths.key_prefix + "_"
    try:
        names = sorted(n for n in os.listdir(root) if n.startswith(want))
    except OSError:
        # No mirror directory for this model yet: an ordinary miss.
        return None, None
    if not names:
        return None, None
    if len(names) > 1:
        logger.warning(
            "semi_p: %d published skeletons match %s* (%s); refusing to guess "
            "which weights this job wants and cold-starting instead. Unpublish "
            "the stale one.", len(names), want, ", ".join(names))
        return None, None
    name = names[0]
    wt_hash = name[len(want):]
    if not wt_hash:
        logger.warning(
            "semi_p: published skeleton %s carries no weight hash in its name; "
            "it predates the skeleton/weight split and cannot be resolved",
            name)
        return None, None
    return os.path.join(root, name), wt_hash


def _replica_source(skeleton_dir: str | None,
                    paths: _ImagePaths) -> str | None:
    """The directory this replica materializes from inside a published skeleton.

    The skeleton itself for a lone replica; ``replica<slot>/`` inside it when the
    pod holds several. A skeleton with no ``replica<K>/`` at all under a
    multi-replica key is a miss rather than a refusal: every replica in the pod
    reads the same answer, so they all cold-start together.
    """
    if not skeleton_dir or paths.replica_count <= 1:
        return skeleton_dir
    try:
        names = os.listdir(skeleton_dir)
    except OSError:
        return None
    if not any(re.fullmatch(rf"{_REPLICA_DIR_PREFIX}\d+", n) for n in names):
        logger.warning(
            "semi_p: %s holds no %s<K>/ directories, but this pod runs %d "
            "replicas; cold-starting", skeleton_dir, _REPLICA_DIR_PREFIX,
            paths.replica_count)
        return None
    return os.path.join(skeleton_dir,
                        f"{_REPLICA_DIR_PREFIX}{paths.replica_slot}")


def _meta_json_text(meta: dict[str, Any]) -> str:
    """``meta.json``'s text, with ``env_files`` written a mapping per line.

    A dump records a few hundred ``[path, size, build_id]`` triples, and at
    ``indent=2`` each one costs five lines -- around two thousand lines of
    library paths, burying the ``vllm_config``, ``model_dir``, ``gpu_uuids``
    and ``pid_floor`` that anyone opening this file came for. Whitespace is
    nothing to ``json.load``, and every reader of this file goes through it.

    Done by splicing rather than by matching on the dumped text, so no path's
    contents can be mistaken for structure: everything else is dumped
    normally, and the field is appended before the closing brace. If that text
    is not the shape the splice needs, the ordinary dump is returned -- no
    formatting preference is worth failing a recorded dump over.
    """
    rows = meta.get("env_files")
    if not rows:
        return json.dumps(meta, indent=2)
    text = json.dumps({k: v for k, v in meta.items() if k != "env_files"},
                      indent=2)
    if not text.endswith("\n}"):
        return json.dumps(meta, indent=2)
    body = ",\n".join("    " + json.dumps(row) for row in rows)
    return f'{text[:-2]},\n  "env_files": [\n{body}\n  ]\n}}'


def _record_env_files(model_dir: str) -> None:
    """Append the image's file-backed mapping set to its ``meta.json``.

    CRIU re-validates the recorded *size* of every file-backed mapping when it
    reopens the file at restore, and aborts on the first mismatch from inside
    CRIU with no up-front check. Recording ``(path, size, build_id, mode)`` here
    means a later run can test that condition by stat-ing a few hundred paths,
    rather than decoding the image or spending a restore attempt to find out.

    ``mode`` is recorded for a different reason than the other three. CRIU
    checks it the same way it checks size -- ``bad mode 0100644 (expect
    0100755)`` and the restore is over -- but unlike size it does not survive
    the trip to the cluster: S3 objects carry no POSIX mode, so publishing an
    image and syncing it back flattens every file to the mirroring daemon's
    umask, and an executable mapping returns as 0644. After that round trip the
    image itself is the only surviving record of the original mode, which is
    what ``_apply_recorded_modes`` puts back.

    Written after the dump rather than with the rest of ``meta_extra``, because
    ``files.img`` is what we are reading and it does not exist until the dump
    has produced it.

    Best-effort: ``crit`` is part of the CRIU install and so is present wherever
    a dump just succeeded, but a failure here costs a diagnostic rather than the
    image, and must not discard a dump that already worked. Build IDs are
    recorded for diagnosis (they catch a same-size, different-build library that
    CRIU's size check would pass) and are not part of any key.
    """
    image_dir = os.path.join(model_dir, _IMAGE_DIR)
    meta_path = os.path.join(image_dir, "meta.json")
    try:
        result = subprocess.run(
            ["crit", "decode", "-i", os.path.join(image_dir, "files.img")],
            capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"crit decode failed (rc={result.returncode}): "
                f"{result.stderr.decode()[:300]}")
        entries = json.loads(result.stdout).get("entries", [])

        env_files = []
        for entry in entries:
            payload = entry.get(entry.get("type", "").lower()) or {}
            name = payload.get("name")
            if not name or "size" not in payload:
                continue
            words = payload.get("build_id") or []
            build_id = b"".join(
                struct.pack("<I", w & 0xFFFFFFFF) for w in words).hex() or None
            env_files.append([
                name if name.startswith("/") else "/" + name,
                payload["size"],
                build_id,
                payload.get("mode"),
            ])

        with open(meta_path) as handle:
            meta = json.load(handle)
        meta["env_files"] = sorted(env_files)
        with open(meta_path, "w") as handle:
            handle.write(_meta_json_text(meta))
        logger.info("semi_p: recorded %d file-backed mapping(s) in %s",
                    len(env_files), meta_path)
    except Exception:
        logger.warning(
            "semi_p: could not record the mapping set in %s; the image is fine, "
            "but a later run will have to validate it the expensive way",
            meta_path, exc_info=True)


def _check_env_files(meta: dict[str, Any], model_dir: str, *,
                     under_model_dir: bool) -> list[str]:
    """Test the image's recorded file-backed mappings against this pod.

    ``_record_env_files`` wrote ``(path, size, build_id)`` for every mapping in
    ``files.img`` at dump time and nothing has read them until now. CRIU
    re-validates the recorded **size** of each of those files when it reopens
    it at restore, and aborts from inside itself on the first mismatch with no
    up-front check -- so a few hundred ``stat`` calls answer in milliseconds a
    question that otherwise costs a whole restore attempt.

    ``under_model_dir`` splits the set in two, because the two halves fail for
    different reasons and are checkable at different times:

    - ``False`` -- the environment (``/usr/...``, the driver libraries). A
      mismatch here means this pod is not the environment the image was
      dumped in, despite ``env12`` agreeing. Checkable before copying anything.
    - ``True`` -- what a materialize just copied, almost all of it under
      ``compilation/``. A mismatch here means the copy is short or the
      published bytes are wrong.

    Returns the **size mismatches**, which are the fatal ones. Missing files
    are logged and not returned: a mapping can be missing for benign reasons
    (a deleted file CRIU ghosted, ``/tmp`` scratch), and refusing an image over
    one would trade a working restore for a cold start.
    """
    entries = meta.get("env_files") or []
    if not entries:
        logger.info(
            "semi_p: %s records no env_files, so its mappings cannot be "
            "checked up front; the restore will find out instead",
            os.path.join(model_dir, _IMAGE_DIR, "meta.json"))
        return []

    prefix = model_dir.rstrip("/") + os.sep
    mismatched: list[str] = []
    missing: list[str] = []
    checked = 0
    for entry in entries:
        try:
            path, size = entry[0], entry[1]
        except (IndexError, TypeError):
            continue
        if not isinstance(path, str) or not isinstance(size, int):
            continue
        if path.startswith(prefix) != under_model_dir:
            continue
        checked += 1
        try:
            actual = os.stat(path).st_size
        except OSError:
            missing.append(path)
            continue
        if actual != size:
            mismatched.append(f"{path} is {actual} B, image recorded {size} B")

    scope = "under" if under_model_dir else "outside"
    if missing:
        logger.warning(
            "semi_p: %d of %d mapping(s) %s %s are absent (%s); not fatal on "
            "its own -- CRIU re-validates size, and a ghosted or recreated "
            "file explains an absence -- but it is the first thing to look at "
            "if the restore fails to open a file",
            len(missing), checked, scope, model_dir,
            ", ".join(missing[:_ENV_FILES_REPORT]))
    if mismatched:
        logger.warning(
            "semi_p: %d of %d mapping(s) %s %s have the wrong size: %s%s",
            len(mismatched), checked, scope, model_dir,
            "; ".join(mismatched[:_ENV_FILES_REPORT]),
            " ..." if len(mismatched) > _ENV_FILES_REPORT else "")
    else:
        logger.info("semi_p: %d mapping(s) %s %s match the image",
                    checked, scope, model_dir)
    return mismatched


def _apply_recorded_modes(meta: dict[str, Any], scope: str) -> int:
    """Put back the file modes a publish round trip flattened, under ``scope``.

    CRIU compares the mode of every file-backed mapping against the one in the
    image and refuses the restore on a mismatch, so a mapping that was 0755
    when it was dumped has to be 0755 again when it is re-opened. Copying
    preserves mode faithfully, but the mirror is downstream of S3, which has no
    POSIX mode at all -- so what a materialize copies is already flattened to
    0644 and there is nothing left for ``copy2`` to preserve. The recorded mode
    from ``_record_env_files`` is the only place the original survives.

    ``scope`` is the subtree a materialize just put in place, and it is a
    safety boundary rather than an optimisation: ``env_files`` also lists the
    environment's own mappings (``/usr/lib``, the driver libraries), and this
    must never chmod one of those. Confining the walk to paths under the
    directory we just wrote makes that structurally impossible instead of
    merely unintended.

    Best-effort in the same sense as the rest of the materialize path: a mode
    we cannot set is logged and skipped, never raised. A restore that then
    fails on it cold-starts, which is the outcome we would have had anyway.
    Rows without a fourth element are images dumped before modes were
    recorded; they are skipped, and only a hand-patched mirror or a re-dump
    will help them.

    Returns the number of files actually changed.
    """
    prefix = scope.rstrip("/") + os.sep
    changed = 0
    failed: list[str] = []
    for entry in meta.get("env_files") or []:
        try:
            path, mode = entry[0], entry[3]
        except (IndexError, TypeError):
            continue
        if not isinstance(path, str) or not isinstance(mode, int):
            continue
        if not path.startswith(prefix):
            continue
        want = mode & 0o7777
        try:
            # lstat, and skip symlinks: chmod follows them, and the target of
            # one can sit outside `scope` -- which is the boundary this
            # function exists to respect.
            st = os.lstat(path)
            if stat.S_ISLNK(st.st_mode):
                continue
            if stat.S_IMODE(st.st_mode) == want:
                continue
            os.chmod(path, want)
            changed += 1
        except OSError as exc:
            failed.append(f"{path}: {exc}")

    if failed:
        logger.warning(
            "semi_p: could not set the recorded mode on %d path(s) under %s "
            "(%s%s); the restore will fail on the first one CRIU re-opens and "
            "fall back to a cold start",
            len(failed), scope, "; ".join(failed[:_ENV_FILES_REPORT]),
            " ..." if len(failed) > _ENV_FILES_REPORT else "")
    if changed:
        logger.info(
            "semi_p: restored the recorded mode on %d path(s) under %s that "
            "the publish round trip had flattened", changed, scope)
    return changed


def _published_weights_dir(weight_root: str | None,
                           weight_hash: str | None) -> str | None:
    """The published shard directory a skeleton names, if it is usable.

    Usable is two questions, and the second does not follow from the first. The
    directory must hold a shard manifest, *and* the node cache must have
    verified it -- ``weights_meta.json`` existing says only that one small file
    arrived, never that the 2 GiB shards beside it did.

    The daemon hands every payload file to s5cmd as one unordered batch and
    writes the marker last, so the manifest can land while shards are still in
    flight. It is the worst file to gate on: it sorts after ``shard_*`` and is
    thousands of times smaller, so it joins the final batch and finishes first.
    Reading shards in place inside that window yields whatever the unwritten
    regions hold, and ``_semip_load_weights`` validates the manifest against the
    model rather than the bytes, so nothing downstream would notice.

    Silent about a missing manifest -- both callers log their own verdict for
    that -- but not about a verification refusal, which neither can express:
    each otherwise reports "no shard manifest", which would be wrong here.
    """
    if not weight_root or not weight_hash:
        return None
    weights = os.path.join(weight_root, weight_hash)
    if not _holds_shards(weights):
        return None
    refusal = _verification_refusal(weights)
    if refusal:
        logger.warning(
            "semi_p: %s; cold-starting rather than reading shards in place out "
            "of a directory still being written", refusal)
        return None
    return weights


def _holds_shards(weights: str) -> bool:
    """Whether *weights* is a directory of shards with a manifest.

    TP1 writes ``weights_meta.json`` beside flat shards; TP>1 fans out into
    ``rank<N>/`` with a manifest each. Testing ``rank0`` rather than the rank
    this job would use keeps this a pure filesystem question, which is all the
    callers need -- ``_semip_load_weights`` validates the manifest it actually
    reads against the attached model anyway.
    """
    return (os.path.isfile(os.path.join(weights, _WEIGHTS_MANIFEST))
            or os.path.isfile(
                os.path.join(weights, "rank0", _WEIGHTS_MANIFEST)))


def _has_weights(root: str) -> bool:
    """Whether ``<root>/weights`` holds a shard manifest (the local layout)."""
    return _holds_shards(os.path.join(root, _WEIGHT_DIR))


def _weights_dir_for_restore(model_dir: str, weight_root: str | None,
                             weight_hash: str | None) -> str | None:
    """Where ``load_weights`` should read shards from; ``None`` for the default.

    Answered from the filesystem rather than remembered from whichever branch
    produced the image, so a hit, a fresh dump and a directory materialized
    from the mirror all get the right answer -- including on a second
    ``restore_and_wrap`` call in this pod, which has no memory of the first.

    A local dump's own ``weight/`` wins: this pod produced those shards, and
    they are the ones its image was dumped against. Otherwise the published
    ``weight/<wt12>/`` named by the skeleton is read **in place**, which is the
    point of the copy/read split. The shards carry no baked absolute path (0 of
    the measured image's 416 file-backed mappings live under them, because the
    dump writes them and then detaches before CRIU runs), ``_semip_load_weights``
    opens them read-only, and it reads each exactly once -- so copying 66 GB
    would double both the I/O and the disk to no end. It is also what lets one
    published directory serve every backend image that produces the same weights.

    Falls back to ``None`` when neither holds weights, which lets
    ``load_weights`` raise naming the local path it expected.
    """
    if _has_weights(model_dir):
        return None
    published = _published_weights_dir(weight_root, weight_hash)
    if published:
        logger.info("semi_p: reading weights in place from %s", published)
        return published
    if weight_root and weight_hash:
        # The skeleton named a hash whose directory is absent or half-synced.
        # A miss, not an error -- but a loud one, because it means a published
        # skeleton is unrestorable and every pod will cold-start silently.
        logger.warning(
            "semi_p: skeleton names weight hash %s but %s holds no shard "
            "manifest; cold-starting. The weights were unpublished while a "
            "skeleton still referenced them, or have not finished syncing",
            weight_hash, os.path.join(weight_root, weight_hash))
    return None


def _verification_refusal(directory: str) -> str | None:
    """Why *directory* is not safe to read off the mirror; ``None`` if it is.

    The DaemonSet writes the bucket manifest last and stamps the marker only
    after digest-verifying every file, so its absence means the directory may be
    a sync in progress -- or bytes that arrived by some route that never
    verified them. This is the check the whole read-off-the-mirror path rests on.

    Presence alone is not enough. A *re-sync* lands new bytes under the previous
    pass's marker, so the directory reads as verified while it is being
    overwritten; nothing under it may be newer than the stamp the marker carries.

    Applied to the skeleton and the weight directory alike. The daemon discovers
    and verifies the two independently -- a skeleton has been seen stamped while
    its own weights were still downloading -- so one directory's marker says
    nothing whatever about the other's.

    Returns a reason phrased to be logged as "semi_p: <reason>; <what we did>",
    so callers supply the consequence rather than restating the cause.
    """
    marker = os.path.join(directory, _VERIFIED_MARKER)
    if not os.path.isfile(marker):
        return (f"{directory} carries no {_VERIFIED_MARKER}, so the node cache "
                f"has not verified it")
    stamp = _verified_at(marker)
    if stamp is None:
        # An older daemon writes no timestamp. Keep the weaker gate rather than
        # none: presence is never wrong about a first sync.
        return None
    newest, newest_path = _newest_mtime(directory, skip=marker)
    if newest > stamp + _VERIFY_SKEW_S:
        return (f"{directory} was verified at {stamp:.0f} but {newest_path} is "
                f"newer ({newest:.0f}), so a re-sync is rewriting it")
    return None


def _verified_at(marker_path: str) -> float | None:
    """When the node cache last verified this directory, per its own marker.

    ``None`` when the marker will not parse or carries no timestamp, which is
    what an older daemon writes. Callers then fall back to gating on the
    marker's presence alone -- weaker, but never wrong about a first sync.
    """
    try:
        with open(marker_path) as handle:
            stamp = json.load(handle).get("verified_at")
    except (OSError, ValueError, AttributeError):
        return None
    return float(stamp) if isinstance(stamp, (int, float)) else None


def _newest_mtime(root: str, skip: str) -> tuple[float, str]:
    """``(mtime, path)`` of the most recently modified file under ``root``."""
    newest = 0.0
    where = root
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            path = os.path.join(dirpath, name)
            if path == skip:
                continue
            try:
                mtime = os.lstat(path).st_mtime
            except OSError:
                continue
            if mtime > newest:
                newest, where = mtime, path
    return newest, where


def _dir_bytes(root: str) -> tuple[int, int]:
    """``(file count, total bytes)`` under ``root``. Best effort."""
    files = 0
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            files += 1
            with contextlib.suppress(OSError):
                total += os.lstat(os.path.join(dirpath, name)).st_size
    return files, total


def _materialize_from_source(source_dir: str | None, model_dir: str,
                             weight_root: str | None = None,
                             weight_hash: str | None = None, *,
                             verified_dir: str | None = None,
                             strict: bool = False) -> bool:
    """Populate ``model_dir`` from a published skeleton. True if it now holds an image.

    The middle path between a hit and a cold start: another pod dumped this
    exact key, published it, and the DaemonSet has already put it on our
    node's read-only mirror. Copying the two path-bound directories beats
    cold-starting and leaves the weights where they are.

    ``source_dir`` is the published *skeleton* -- ``skeleton/<cfg12>_<env12>_
    <wt12>/`` -- which holds only ``image/`` and ``compilation/``. The weights it
    names live in a sibling ``weight/<wt12>/`` shared with every other skeleton
    that produced the same bytes, so they are checked through ``weight_root``
    and ``weight_hash`` rather than found inside the directory being copied.

    The copy scales with the model: ~6.7 GB for the small TP1 models this was
    first measured against, but ~90 GiB of ``image/`` for GLM-5.3 at TP8, where
    it costs ~30 s. The weights are what stay on the mirror, and that is the
    large half -- 760 GB for GLM-5.3.

    **Every refusal is a cache miss, not an error.** This returns ``False`` and
    logs, and the caller cold-starts -- which always works and only costs time.
    That asymmetry is the whole safety argument: nothing here can fail a job
    that would otherwise have run, so the checks can afford to be strict.

    **``strict`` (several replicas in the pod) narrows that.** The replicas
    cannot mix: one cold-starting beside siblings that restore puts its new
    processes on the task ids their images recorded. So the checks every
    replica answers identically -- the marker, the recorded ``model_dir``,
    uid, weights, the environment's libraries, devices -- stay misses, and the
    whole pod cold-starts together; anything that can fail one replica alone
    (its ``replica<K>/`` missing, its meta unreadable, its copy) raises and
    fails the job.

    ``verified_dir`` is where the node's verified marker sits when it is not
    ``source_dir`` itself: a multi-replica skeleton carries one marker at its
    top, over every ``replica<K>/``.

    Must be called under ``_dump_lock(model_dir)``: it writes the same
    directories a dump writes.
    """
    if not source_dir:
        return False
    if not os.path.isdir(source_dir):
        if strict:
            raise RuntimeError(
                f"semi_p: the published skeleton has no {source_dir}, so this "
                f"replica cannot restore beside the others in its pod")
        logger.info(
            "semi_p: no published image at %s; cold-starting. (Publish one "
            "with semi_persistence/scripts/semip_publish.py.)", source_dir)
        return False

    # The check the whole copy's safety rests on. The weight directory gets the
    # same one, through _published_weights_dir below -- the two are verified
    # independently, so the skeleton's marker vouches only for the skeleton.
    refusal = _verification_refusal(verified_dir or source_dir)
    if refusal:
        logger.warning(
            "semi_p: %s; cold-starting rather than copying a sync in flight",
            refusal)
        return False

    src_meta_path = os.path.join(source_dir, _IMAGE_DIR, "meta.json")
    try:
        with open(src_meta_path) as handle:
            meta = json.load(handle)
    except (OSError, ValueError) as exc:
        if strict:
            raise RuntimeError(
                f"semi_p: cannot read {src_meta_path} ({exc}), so this replica "
                f"cannot restore beside the others in its pod") from exc
        logger.warning("semi_p: cannot read %s (%s); cold-starting",
                       src_meta_path, exc)
        return False

    # image/ and compilation/ bake this absolute path -- criu_restore rejects
    # any other by name, in the parent process, before CRIU is spawned. Catch
    # it here rather than after spending the copy.
    recorded = meta.get("model_dir")
    if recorded != model_dir:
        logger.warning(
            "semi_p: %s was dumped with model_dir=%r but we resolved %r, so "
            "its baked paths would not match; cold-starting. A published "
            "directory whose key and recorded model_dir disagree was moved "
            "after its dump and is not restorable anywhere.",
            source_dir, recorded, model_dir)
        return False

    mismatch = _unprivileged_mismatch(meta)
    if mismatch:
        logger.warning("semi_p: %s %s; cold-starting", source_dir, mismatch)
        return False

    # The restored child keeps the dumping uid, and Instance.criu_restore
    # refuses the mix up front, so a foreign image is unusable however well it
    # copies.
    uid = meta.get("uid")
    if uid is not None and uid != os.getuid():
        logger.warning(
            "semi_p: %s was dumped by uid %s and we are uid %d; "
            "criu_restore would reject it, so cold-starting",
            source_dir, uid, os.getuid())
        return False

    # We do not copy weights, by design -- so a skeleton whose weights are not
    # on this node cannot be restored from. Checked before the copy rather than
    # after, since the copy is the expensive half and this is a filesystem test.
    if not _published_weights_dir(weight_root, weight_hash):
        logger.warning(
            "semi_p: %s names weight hash %s, but %s holds no %s (nor rank0/), "
            "so its shards cannot be read in place; cold-starting. The weights "
            "were unpublished while this skeleton still referenced them, or "
            "have not finished syncing.",
            source_dir, weight_hash,
            os.path.join(weight_root or "<no mirror>", weight_hash or "?"),
            _WEIGHTS_MANIFEST)
        return False

    if _check_env_files(meta, model_dir, under_model_dir=False):
        logger.warning(
            "semi_p: %s was dumped against different library bytes than this "
            "pod has, so CRIU would abort re-opening them; cold-starting. "
            "The environment hash matched, so this is worth understanding: "
            "run semi_persistence/scripts/imgdiff.py against the image.",
            source_dir)
        return False

    # A TP>1 image only restores into a pod holding the device nodes its
    # captured state names, and the scheduler assigns those slots freely -- so a
    # published TP=2 image is restorable in roughly one pod in four. Declining
    # here makes that a miss: the caller cold-starts, serves, and dumps an image
    # on *this* pod's slots, which is the same outcome the pod would have had
    # with no published image at all.
    #
    # It belongs here rather than only at the restore, because both alternatives
    # are worse. Raising later fails the job outright -- measured 2026-09-20,
    # the refusal propagates to HTTPException(500, "Failed to create job") and
    # nothing retries it -- and it raises *after* this copy has already run, so
    # it spends the materialize and then throws the job away anyway. Checking
    # before the copy costs neither.
    missing = _missing_device_nodes(meta)
    if missing and not _device_check_disabled():
        logger.warning("%s Cold-starting instead.",
                       _device_mismatch_detail(meta, missing, source_dir))
        return False

    incoming = os.path.join(model_dir, _INCOMING_NAME)
    t0 = time.perf_counter()
    files = total = 0
    try:
        shutil.rmtree(incoming, ignore_errors=True)
        for sub in _COPY_SUBDIRS:
            src = os.path.join(source_dir, sub)
            if not os.path.isdir(src):
                raise RuntimeError(f"{src} is missing from the published image")
            # copytree defaults to copy2, which preserves mode. Necessary but
            # not sufficient: CRIU re-validates the recorded mode of every path
            # it re-maps, and `src` is downstream of S3, which carries no POSIX
            # mode -- so these bytes arrive already flattened to the mirroring
            # daemon's umask and there is nothing left here to preserve.
            # _apply_recorded_modes repairs that after the flip. Do not shell
            # out to `cp -a` instead -- it implies -p, tries to chown as a
            # non-root user, and returns non-zero after copying fine.
            shutil.copytree(src, os.path.join(incoming, sub))
        files, total = _dir_bytes(incoming)

        for sub in _COPY_SUBDIRS:
            dest = os.path.join(model_dir, sub)
            # os.replace onto a non-empty directory fails, and leftovers from
            # an interrupted earlier attempt are exactly what we may find.
            shutil.rmtree(dest, ignore_errors=True)
            os.replace(os.path.join(incoming, sub), dest)
            # Per subdirectory rather than once after the loop, because
            # image/meta.json is the hit predicate and image/ is flipped last.
            # Repairing each subtree before the predicate lands keeps
            # "interrupted anywhere reads as a miss" true; a trailing fix-up
            # would leave a crash in between as a *hit* with flattened modes,
            # which fails a restore instead of cold-starting.
            _apply_recorded_modes(meta, dest)
            if sub == _COMPILATION_DIR:
                # Now that the copy is in place under its dump-time path, the
                # other half of env_files is checkable. Do it before image/
                # lands, so a bad copy leaves no image and reads as a miss.
                if _check_env_files(meta, model_dir, under_model_dir=True):
                    shutil.rmtree(dest, ignore_errors=True)
                    raise RuntimeError(
                        f"the copied compile cache under {dest} does not match "
                        f"the sizes the image recorded")
    except Exception as exc:
        if strict:
            raise RuntimeError(
                f"semi_p: materializing {model_dir} from {source_dir} failed "
                f"({exc}), and this replica cannot cold-start beside the "
                f"others in its pod that restore") from exc
        logger.warning(
            "semi_p: materializing %s from %s failed; falling back to a cold "
            "start", model_dir, source_dir, exc_info=True)
        return False
    finally:
        shutil.rmtree(incoming, ignore_errors=True)

    elapsed = time.perf_counter() - t0
    logger.info(
        "semi_p: materialized %s from %s: %d file(s), %.2f GiB in %.1fs "
        "(%.0f MB/s); weights stay on the mirror",
        model_dir, source_dir, files, total / 2**30, elapsed,
        total / 1e6 / elapsed if elapsed > 0 else 0.0)
    return True


def _resolve_physical_gpus() -> list[int]:
    """Physical GPU ids for the restore, in ``CUDA_VISIBLE_DEVICES`` order.

    Ray sets ``CUDA_VISIBLE_DEVICES`` to the assigned physical id(s); NVML
    (used by Instance) ignores CVD and enumerates physical devices, and the
    Instance's child sets its own CVD. So we (a) read every physical id out of
    CVD, then (b) clear CVD in this process so the child can address those
    physical ids directly.

    Every id matters, not just the first: ``cuda_restore`` validates the count
    against the image's ``tensor_parallel_size`` and raises on a mismatch, so a
    scalar would break every TP>1 restore.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    gpus: list[int] = []
    for token in cvd.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            gpus.append(int(token))
        except ValueError:
            logger.warning(
                "semi_p: ignoring non-numeric CUDA_VISIBLE_DEVICES entry %r",
                token)
    if not gpus:
        gpus = [0]
    # Let the Instance child address the physical ids directly.
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return gpus


def _local_gpu_uuids() -> list[str] | None:
    """This node's GPU UUIDs in physical index order, or None if unreadable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            out = []
            for idx in range(pynvml.nvmlDeviceGetCount()):
                uuid = pynvml.nvmlDeviceGetUUID(
                    pynvml.nvmlDeviceGetHandleByIndex(idx))
                out.append(uuid.decode() if isinstance(uuid, bytes) else uuid)
            return out
        finally:
            pynvml.nvmlShutdown()
    except Exception:  # pragma: no cover - best effort preflight only
        logger.debug("semi_p: could not read local GPU UUIDs", exc_info=True)
        return None


_DEVICE_GLOBS = ("/dev/nvidia[0-9]*", "/dev/infiniband/uverbs[0-9]*")
_DEVICE_CHECK_ENV = "SEMIP_REQUIRE_DEVICE_MATCH"

# The TP degree at and above which an image is bound to its device allocation.
# Read by both ``_device_binding``, which puts the devices in the cache key,
# and ``_missing_device_nodes``, which refuses a restore that lacks them. One
# constant because they must answer the same question: a key that bound the
# devices where the check did not would publish images nothing looks up, and a
# check that bound them where the key did not is the placement lottery this
# replaced.
_DEVICE_BOUND_MIN_TP = 2


def _visible_device_nodes() -> list[str]:
    """Sorted basenames of the GPU and EFA device nodes present in this pod.

    Mirrors ``worker._visible_device_nodes``, which records the same set at
    dump time. Kept as two small copies rather than one shared import because
    the two live on opposite sides of the CRIU boundary.

    **The two must stay byte-identical in behaviour.** They always had to agree
    or the check would refuse a restore that would have worked; now the cache
    key is derived from this one, so drift instead makes every TP>1 job compute
    a key no publish ever wrote -- a permanent silent miss rather than a loud
    refusal. ``test_image_cache_key`` pins the pair against a shared fixture
    for the same reason ``test_publish_layout`` pins ``model_slug``.
    """
    names = set()
    for pattern in _DEVICE_GLOBS:
        for path in glob.glob(pattern):
            names.add(os.path.basename(path))
    return sorted(names)


def _image_tp(meta: dict[str, Any]) -> int:
    """The tensor-parallel degree the image was dumped with.

    Read from the image rather than from the request: the constraint belongs to
    the captured state, and a request whose TP disagrees is rejected earlier by
    ``_check_tp_matches_gpus``. Unparseable or absent reads as 1, which is
    vLLM's own default and the degree that imposes no device constraint.
    """
    baked = meta.get("vllm_config") or {}
    try:
        return int(baked.get("tensor_parallel_size") or 1)
    except (TypeError, ValueError):
        return 1


def _device_check_disabled() -> bool:
    """Whether ``SEMIP_REQUIRE_DEVICE_MATCH`` opts out of the device check."""
    return (os.environ.get(_DEVICE_CHECK_ENV) or "").strip() in ("0", "false",
                                                                 "no")


def _missing_device_nodes(meta: dict[str, Any]) -> list[str]:
    """Device nodes the image needs that this pod does not have.

    Empty when the image can be restored here, when it predates
    ``device_nodes``, or when it is TP=1.

    **TP=1 is exempt, and not as a shortcut.** A TP=1 image has no communicator
    to rebuild -- ``reinit_nccl`` is a no-op -- and ``cuda_restore`` builds a
    device map, so renumbering alone is fine and a TP=1 image has always
    restored onto whatever slot it landed on. Applying the check to TP=1 refuses
    restores that demonstrably work. The constraint is specific to the
    multi-rank case, where ``SEMIP_GPU_MAP`` keeps every group GPU visible and
    each rank opens *all* of them, so the captured state names device paths that
    a differently-allocated pod does not have.

    Since ``_device_binding`` put the same set into the cache key, this can no
    longer fire for an image found under the current key scheme -- equal keys
    imply equal device sets. What it still catches is an image whose key
    predates that change, or one carrying no recorded set at all, and it is
    those two that keep it from being deletable.
    """
    if _image_tp(meta) < _DEVICE_BOUND_MIN_TP:
        return []
    baked = meta.get("device_nodes")
    if not baked:
        # Predates device_nodes. Absence is not evidence, and refusing these
        # would strand every image published before the field existed.
        return []
    present = set(_visible_device_nodes())
    return [d for d in baked if d not in present]


def _device_mismatch_detail(meta: dict[str, Any], missing: list[str],
                            where: str) -> str:
    """One message naming both device sets, for the miss path and the backstop."""
    baked = meta.get("device_nodes") or []
    present = sorted(_visible_device_nodes())
    return (
        f"semi_p: {where} was dumped in a pod holding "
        f"{', '.join(baked)}, but this pod has "
        f"{', '.join(present) or 'none of them'} -- missing "
        f"{', '.join(missing)}. At TP>1 the captured CUDA and NCCL state refers "
        f"to device nodes that do not exist here, so a restore fails inside "
        f"ncclCommInitRank as an unrelated-looking 'unhandled system error'. "
        f"This image needs a pod allocated the same device slots.")


def _check_device_visibility(meta: dict[str, Any], model_dir: str) -> None:
    """Backstop: refuse a restore whose image needs device nodes we lack.

    The mismatch that actually happens is caught one layer out, in
    ``_materialize_from_source``, where it costs a cold start instead of the
    job -- see the note there. By the time we are here the image is either one
    this pod dumped itself (so the slots are ours by construction) or one the
    miss path already vetted, which makes a mismatch at this point a bug rather
    than a placement. Raising is the right response to that.

    Note this does **not** save the copy, and never did: it runs inside
    ``_restore``, which the caller reaches only after ``_materialize_from_source``
    has finished. Measured 2026-09-20: a refused restore still paid the full
    materialize first, and ``create-job`` ran 27.4 s before refusing. The copy it
    was once described as preventing is only ``image/`` plus ``compilation/`` --
    model-dependent, from ~6.7 GB on the small TP1 models up to ~90 GiB for
    GLM-5.3 at TP8 -- and never the 708 GiB of ``weight/``, which are read in
    place off the mirror.

    ``SEMIP_REQUIRE_DEVICE_MATCH=0`` downgrades this to a warning, which is both
    the escape hatch if the check is ever wrong about a real placement and the
    way to reach NCCL deliberately on a known mismatch -- how the error behind
    'unhandled system error' was first read.
    """
    missing = _missing_device_nodes(meta)
    if not missing:
        return
    detail = _device_mismatch_detail(meta, missing, model_dir)
    if _device_check_disabled():
        logger.warning("%s (continuing: %s is set)", detail, _DEVICE_CHECK_ENV)
        return
    raise RuntimeError(detail)


def _check_gpu_placement(meta: dict[str, Any], gpus: list[int],
                         model_dir: str) -> None:
    """Report the placement this restore is about to perform.

    This used to **reject** a cross-node restore onto the dump's own GPU
    indices, because ``_worker_restore`` built its ``oldUuid=newUuid`` device
    map only when the *indices* changed -- so that one placement kept the
    capture node's baked UUIDs and failed inside
    ``cuCheckpointProcessRestore`` with an opaque ``CUDA_ERROR_INVALID_VALUE``.
    Refusing was right while an image could not travel: the only cure was to
    land somewhere else.

    ``_placement_changed`` now asks about UUIDs as well as indices, so the map
    is built whenever the node changed, and the case this refused is the
    supported one. It has to be, for a distributed cache: every node mounts the
    cache at one path, so a published image restores anywhere, and Ray is as
    likely to assign the dumped index as any other.

    What is left is the one shape that still cannot work -- an image with no
    recorded ``gpu_uuids`` restoring onto its own indices. Nothing can tell
    whether that is the capture node, and ``_placement_changed`` reports no
    change, so CRIU gets no map. Warn rather than raise: it is correct on the
    capture node, which is where such an old image is most likely to be.
    """
    baked_gpus = list(meta.get("gpus") or [])
    baked_uuids = set(meta.get("gpu_uuids") or [])
    if not baked_gpus or list(gpus) != baked_gpus:
        return          # indices differ: the map is built from them alone
    if not baked_uuids:
        logger.warning(
            "semi_p: %s records no gpu_uuids and is being restored onto its "
            "own indices (%s), so no device map can be built. This only works "
            "on the node that dumped it; re-dump to record them.",
            model_dir, gpus)
        return
    local = _local_gpu_uuids()
    if local is None or baked_uuids & set(local):
        return          # same node (or unreadable), nothing to remap
    logger.info(
        "semi_p: %s was dumped on another node and Ray assigned the same "
        "indices (%s), so the restore rebuilds the device map from the "
        "recorded UUIDs", model_dir, gpus)


def _staging_budget_bytes(inst: Any) -> int | None:
    """Mirror ``plan_restore_weights``' own budget computation, for reporting.

    This is the **outer bound**, not the effective chunk size: the number is
    predicted from config, and each worker clamps it against the free VRAM its
    own device reports before building a plan. Treat a large value here as
    permission, not as a measurement.

    Returns ``None`` exactly when the handle's state would make it fall back to
    the unbounded single-chunk path.
    """
    pinned = (getattr(inst, "max_pinned_bytes_per_worker", 0)
              or getattr(inst, "pinned_cpu_bytes", 0))
    total = getattr(inst, "total_gpu_bytes", 0)
    if total <= 0 or pinned <= 0:
        return None
    util = inst.vllm_config["gpu_memory_utilization"]
    return int(0.9 * min(pinned, int(total * util) - pinned))


def _check_staging_budget(inst: Any, model_dir: str) -> None:
    """Refuse to plan a weight restore that would be unbounded.

    ``plan_restore_weights`` degrades silently to a single unbounded chunk when
    the handle carries no pinned-buffer sizes: the staging buffer is then sized
    to the whole weight set and leaves nothing for ``wake_up_kv_cache``. The
    sizes reach the handle only with the ``attach`` acknowledgement -- an
    image's ``meta.json`` records zeros, because it is dumped after a detach --
    so this must run after ``attach()`` has been waited on.
    """
    budget = _staging_budget_bytes(inst)
    pinned = getattr(inst, "pinned_cpu_bytes", 0)
    per_worker = getattr(inst, "max_pinned_bytes_per_worker", 0)
    total = getattr(inst, "total_gpu_bytes", 0)
    if budget is None:
        raise RuntimeError(
            f"semi_p: refusing to plan an unbounded weight restore for "
            f"{model_dir}: pinned_cpu_bytes={pinned}, "
            f"max_pinned_bytes_per_worker={per_worker}, "
            f"total_gpu_bytes={total}. attach() must complete before "
            f"plan_restore_weights()")
    logger.info(
        "semi_p: staging budget <= %.2f GiB from config (pinned=%.2f GiB, "
        "per_worker=%.2f GiB, total_gpu=%.2f GiB, util=%s); each worker clamps "
        "this against its own free VRAM, so the effective chunk size is the one "
        "in the plan_restore_weights log",
        budget / 2**30, pinned / 2**30, per_worker / 2**30, total / 2**30,
        inst.vllm_config.get("gpu_memory_utilization"))


def _dump_lock_is_stale(lock_path: str) -> bool:
    """Whether ``lock_path``'s recorded pid is provably gone.

    An unreadable or empty file is normally just the microseconds between the
    ``O_EXCL`` create and the pid write, so it counts as held. Beyond the grace
    window it means whoever created it died in between and nothing will ever
    fill it in, which would otherwise block every future job for this model.
    """
    try:
        with open(lock_path) as f:
            raw = f.read().strip()
    except OSError:
        return False
    if not raw.isdigit():
        try:
            return (time.time() - os.path.getmtime(lock_path)) > _DUMP_LOCK_GRACE_S
        except OSError:
            return False
    try:
        os.kill(int(raw), 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # alive but not ours to signal
    return False


@contextlib.contextmanager
def _dump_lock(model_dir: str):
    """Hold an exclusive dump lock on ``model_dir``, waiting out any holder.

    A gateway has concurrent writers the dump scripts never had. Two jobs that
    resolve to the same directory -- same config, same container image, same
    driver -- would both miss, both cold-start, and both write into the same
    ``image/`` and ``weight/``, interleaving into an image that fails a later
    restore unrecognisably. Deriving the directory makes that collision the
    normal case rather than an operator error, which is what the lock is for.

    ``O_CREAT | O_EXCL`` is sufficient: both writers are the same uid in one
    pod, since ``Instance.criu_restore`` already forbids the cross-uid case.
    """
    lock_path = os.path.join(model_dir, _DUMP_LOCK_NAME)
    deadline = time.monotonic() + _DUMP_LOCK_WAIT_S
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            if _dump_lock_is_stale(lock_path):
                logger.warning(
                    "semi_p: removing stale dump lock %s (its pid is gone)",
                    lock_path)
                with contextlib.suppress(OSError):
                    os.unlink(lock_path)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"semi_p: another job has been dumping into {model_dir} for "
                    f"more than {_DUMP_LOCK_WAIT_S:.0f}s ({lock_path} still "
                    f"held). If nothing is dumping, remove that file.")
            logger.info(
                "semi_p: waiting for another job's dump of %s to finish (%s)",
                model_dir, lock_path)
            time.sleep(_DUMP_LOCK_POLL_S)
    try:
        os.write(fd, f"{os.getpid()}\n".encode())
    finally:
        os.close(fd)
    try:
        yield
    finally:
        with contextlib.suppress(OSError):
            os.unlink(lock_path)


def _read_sysctl(path: str) -> int | None:
    """An integer sysctl, or ``None`` when it cannot be read or parsed."""
    try:
        with open(path) as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _write_sysctl(path: str, value: int) -> bool:
    """Whether ``value`` could be written to ``path``.

    Failure is the ordinary case for ``ns_last_pid`` -- writing it needs
    CAP_SYS_ADMIN over the PID namespace's user namespace, which a
    reduced-capability container does not have -- so it is a return value
    rather than an exception.
    """
    try:
        with open(path, "w") as handle:
            handle.write(str(value))
        return True
    except OSError:
        return False


def _env_int(name: str, default: int) -> int:
    """An integer environment override, or ``default`` if unset or malformed.

    A malformed value is a warning and not an error: these knobs tune a dump,
    and none of them is worth failing a job over.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "semi_p: ignoring %s=%r, which is not an integer; using %d",
            name, raw, default)
        return default


def _raise_pid_floor() -> int | None:
    """Advance this PID namespace's counter past the floor, before a dump.

    Must run before anything that ends up inside the image exists, because the
    ids it is placing are the ones that tree will record. See
    ``_PID_FLOOR_ENV`` for why a low-id image cannot be restored into another
    pod and cannot be repaired afterwards.

    Three ways to get there, cheapest first: the counter may already be past
    the floor; a single write to ``ns_last_pid`` moves it instantly where the
    namespace allows one, which the reduced-capability containers this runs in
    normally do not; otherwise burners fork throwaway children until the
    counter clears the floor, which needs no privilege at all because it only
    consumes ids in the ordinary way.

    Returns the counter reached, or ``None`` when no floor was applied. That is
    never fatal and never raised: it leaves the image exactly as it would have
    been dumped before this existed, and the restore-side preflight in
    ``semi_persistence/worker.py`` still names the collision if one follows.
    """
    try:
        target = _env_int(_PID_FLOOR_ENV, _PID_FLOOR_DEFAULT)
        if target <= 0:
            logger.info("semi_p: PID floor disabled by %s=%d", _PID_FLOOR_ENV,
                        target)
            return None

        pid_max = _read_sysctl(_PID_MAX_PATH)
        if pid_max is None:
            logger.warning("semi_p: cannot read %s, so the PID floor is not "
                           "safe to place; dumping with the ids this "
                           "namespace hands out", _PID_MAX_PATH)
            return None
        if target + _PID_FLOOR_HEADROOM > pid_max:
            logger.warning(
                "semi_p: not raising the PID floor to %d: kernel.pid_max is "
                "%d, which leaves less than %d ids above the floor. Raise "
                "pid_max on the node, or lower %s -- but note that a floor "
                "close to pid_max is no floor at all, since the counter wraps "
                "back under it.",
                target, pid_max, _PID_FLOOR_HEADROOM, _PID_FLOOR_ENV)
            return None

        counter = _read_sysctl(_NS_LAST_PID_PATH)
        if counter is None:
            logger.warning("semi_p: cannot read %s, so the PID floor cannot "
                           "be placed or confirmed", _NS_LAST_PID_PATH)
            return None
        if counter >= target:
            logger.info("semi_p: PID floor already satisfied (%s is %d, floor "
                        "%d)", _NS_LAST_PID_PATH, counter, target)
            return counter

        if _write_sysctl(_NS_LAST_PID_PATH, target):
            written = _read_sysctl(_NS_LAST_PID_PATH)
            if written is not None and written >= target:
                logger.info("semi_p: PID floor raised %d -> %d by writing %s",
                            counter, written, _NS_LAST_PID_PATH)
                return written

        workers = max(1, _env_int(_PID_BURN_WORKERS_ENV,
                                  _PID_BURN_WORKERS_DEFAULT))
        limit = target - counter + _PID_BURN_SLACK
        logger.info("semi_p: burning pids to raise the floor from %d to %d "
                    "(%d burner(s))", counter, target, workers)
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-c", _PID_BURN_SOURCE,
             str(target), str(workers), str(limit)],
            capture_output=True, timeout=_PID_BURN_TIMEOUT_S)
        if result.returncode != 0:
            logger.warning("semi_p: the pid burner exited %d: %s",
                           result.returncode,
                           result.stderr.decode("utf-8", "replace")[:300])
            return None

        reached = _read_sysctl(_NS_LAST_PID_PATH)
        if reached is None or reached < target:
            logger.warning(
                "semi_p: the pid burner ran but %s is %s, short of the floor "
                "%d; dumping with the ids this namespace hands out",
                _NS_LAST_PID_PATH, reached, target)
            return None
        logger.info("semi_p: PID floor raised %d -> %d in %.1fs (%d burner(s))",
                    counter, reached, time.perf_counter() - started, workers)
        return reached
    except Exception:
        logger.warning(
            "semi_p: could not raise the PID floor. The dump will record the "
            "low task ids this namespace hands out, and restoring the image "
            "in another pod will collide with that pod's own launcher",
            exc_info=True)
        return None


def _dump(vllm_config: dict[str, Any], model_dir: str, gpus: list[int], *,
          image_ref: str, driver_version: str) -> None:
    """Cold-start vLLM, capture an image into ``model_dir``, and destroy it.

    The nine-call sequence from ``semi_persistence/scripts/test_weights.py``,
    whose images this adapter has restored repeatedly. It needs **no TP branch**:
    ``cuda_checkpoint()`` inserts ``destroy_nccl()`` itself when
    ``tensor_parallel_size > 1``, so unlike the restore side -- where the caller
    must add ``reinit_nccl()`` and ``rebind_graphs()`` by hand -- there is
    nothing here a TP1 test hides.
    See ``semi_persistence/skills/tp_DESIGN.md`` Section 2.

    The ordering is a contract rather than a style: ``save_weights()`` sits
    between ``stage()`` and ``detach()`` or there is nothing staged to write;
    ``detach()`` frees the pinned buffer so the image stays small, and zeroes the
    pinned counters, which is why the restore side must ``attach().wait()``
    before planning; and ``cuda_checkpoint()`` must release the GPU before
    ``criu_dump()`` runs.

    **Destructive**: ``criu_dump()`` kills the child, so this returns nothing
    servable and the caller must restore the image it just wrote.

    No ``filename`` / ``weights_dir`` overrides are passed. ``Instance`` derives
    ``<model_dir>/image`` and ``<model_dir>/weights``, which is exactly what
    ``restore_and_wrap`` looks for and what ``criu_restore``'s exact-string
    ``model_dir`` check requires; an override here could only disagree with the
    restore side.

    ``image_ref`` and ``driver_version`` are recorded in ``meta.json`` because
    the environment half of the directory name is a 12-hex truncation of their
    hash and cannot be reversed into them. Keeping the inputs means a future
    mismatch can be reported as "dumped under image X, restoring under Y"
    rather than as two unequal hashes. ``pid_floor`` joins them for a different
    reason: it is the one property of an image that decides whether it can be
    restored in a *different pod* at all, and an image that records ``null``
    there is one that predates ``_raise_pid_floor`` or was dumped where it
    could not be applied. See ``_PID_FLOOR_ENV``.
    """
    logger.info(
        "semi_p: cache miss -- cold-starting to dump model_dir=%s onto physical "
        "GPU(s) %s. Expect this to take far longer than a warm restore: it pays "
        "a cold start (plus a hub download on first use of the model), then the "
        "dump, then the restore below.", model_dir, gpus)
    t0 = time.perf_counter()
    # Before the Instance, because the ids being placed are the ones its child
    # tree will record, and the counter is namespace-global so both inherit it.
    pid_floor = _raise_pid_floor()
    inst = Instance(vllm_config, model_dir)
    try:
        inst.init(gpus=gpus)
        inst.generate([_DUMP_PROMPT], _DUMP_SAMPLING)
        inst.attach()
        inst.stage()
        inst.save_weights()
        inst.detach()
        inst.sleep()
        inst.cuda_checkpoint()  # TP>1: destroy_nccl inside
        inst.criu_dump(meta_extra={"image_ref": image_ref,
                                   "driver_version": driver_version,
                                   "pid_floor": pid_floor,
                                   "unprivileged": _unprivileged_mode()})
        inst.wait()             # criu_dump destroys the child
    finally:
        # Frees the Instance's worker process; the vLLM child is already gone.
        # Guarded because a failure before init() leaves no queues to send on.
        try:
            inst.teardown()
        except Exception:
            logger.warning("semi_p: teardown after dump failed", exc_info=True)
    # teardown is asynchronous and the restore spawns a fresh Instance worker;
    # example_full.py pauses here for the same reason, so the two do not overlap.
    time.sleep(2)
    _record_env_files(model_dir)
    logger.info("semi_p: dump complete model_dir=%s in %.1fs",
                model_dir, time.perf_counter() - t0)


def _unprivileged_mode() -> bool:
    """What ``semi_persistence/worker._unprivileged`` will read in this process."""
    return os.environ.get(_UNPRIVILEGED_ENV) == "1"


def _unprivileged_mismatch(meta: dict[str, Any]) -> str | None:
    """Why this image cannot restore under the current ``SEMIP_UNPRIVILEGED``.

    The mode fixes the capability level every dumped task records, and picks
    which restore path runs, so the two configurations are not interchangeable
    after the dump (CRIU_PLUMBING Complication 11). ``None`` when they agree or
    the image predates the field.
    """
    recorded = meta.get("unprivileged")
    if recorded is None or bool(recorded) == _unprivileged_mode():
        return None
    return (f"image was dumped with {_UNPRIVILEGED_ENV}="
            f"{'1' if recorded else '0'} but this process runs with "
            f"{os.environ.get(_UNPRIVILEGED_ENV)!r}; its recorded capability "
            f"level only restores under the same mode")


def _restore(model_dir: str, engine_kwargs: dict[str, Any],
             gpus: list[int], *,
             weights_dir: str | None = None,
             requested: dict[str, Any] | None = None) -> "_SemiPEngine":
    """Restore ``<model_dir>/image`` onto ``gpus`` and wrap it (blocking).

    Shared by all three entry paths -- an ordinary cache hit, the restore of an
    image this process has just dumped, and one materialized from the published
    mirror -- so a second run of a job exercises the same code the first run
    ended on, and a copied image is exercised by the code a local one is.

    ``weights_dir`` overrides where the shards are read from; ``None`` leaves
    ``Instance``'s own ``<model_dir>/weights``. See ``_weights_dir_for_restore``.

    ``requested`` is the caller's already-projected ``vllm_config``, forwarded
    so the divergence check compares it against ``baked`` rather than the raw
    ``engine_kwargs`` -- two different shapes. See ``_log_config_divergence``.
    """
    meta_path = os.path.join(model_dir, _IMAGE_DIR, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    baked = meta.get("vllm_config") or {}
    if not baked:
        raise ValueError(f"semi_p: image meta.json at {meta_path} has no vllm_config")

    _log_config_divergence(baked, engine_kwargs, model_dir, requested)
    tokenizer_path = engine_kwargs.get("model") or baked.get("model")
    mismatch = _unprivileged_mismatch(meta)
    if mismatch:
        raise RuntimeError(f"semi_p: {model_dir}: {mismatch}")
    _check_device_visibility(meta, model_dir)
    _check_gpu_placement(meta, gpus, model_dir)

    logger.info("semi_p: restoring image model_dir=%s (dumped on %s) onto "
                "physical GPU(s) %s", model_dir, meta.get("gpus"), gpus)
    inst = Instance(baked, model_dir)

    # Serving restore, following reproduce/example_full.py. The wait()
    # boundaries are part of the contract, not formatting: attach() is what
    # puts the real pinned-buffer sizes on the handle, and
    # plan_restore_weights() reads them synchronously when it is called.
    try:
        inst.criu_restore().wait()
        inst.cuda_restore(gpus=gpus).wait()
        inst.reinit_nccl().wait()
        inst.attach().load_weights(weights_dir=weights_dir).wait()
        inst.wake_up_weights().wait()
        _check_staging_budget(inst, model_dir)
        inst.repin().plan_restore_weights().wait()
        inst.restore_weights().wait()
        inst.wake_up_kv_cache().wait()
        inst.rebind_graphs().wait()
    except BaseException:
        # Do not leak the Instance's worker process (or a partially restored
        # tree) when the restore fails. This matters more than it looks:
        # _restore_with_port_retry may build another Instance seconds later, and
        # a leftover worker can hold the very ports and task ids the retry needs
        # -- turning one recoverable collision into a permanent one.
        try:
            inst.teardown()
        except Exception:
            logger.warning("semi_p: teardown after a failed restore also failed",
                           exc_info=True)
        raise
    logger.info("semi_p: restore complete model_dir=%s gpus=%s", model_dir, gpus)
    return _SemiPEngine(inst, tokenizer_path=tokenizer_path,
                        model=baked.get("model"))


# ---------------------------------------------------------------------------
# Multi-node: one engine whose TP group spans pods
# ---------------------------------------------------------------------------
#
# The leader owns ranks 0..local-1 and serves; each agent owns the next block
# and never serves. Every collective is the leader's, because its executor
# spans every node-partition -- so these functions only sequence the steps the
# leader cannot reach into the other pods to do: each pod's CRIU image, its CUDA state,
# and its end of the message-queue plane.
#
# Joint or nothing. A node-partition that cold-starts while another restores would
# rendezvous with a group that does not exist, and the failure mode is a
# deadlock in the first collective rather than an error. So a hit requires
# every node-partition to hold an image from the *same* dump, which is what
# ``dump_id`` identifies; anything else makes all of them cold-start and dump
# together.


def _leader_ip() -> str:
    """This pod's address on the network the other node-partitions reach it on."""
    import ray
    return ray.util.get_node_ip_address()


def _node_ranks(node_rank: int, local: int) -> list[int]:
    """The global ranks node-partition *node_rank* owns."""
    return list(range(node_rank * local, (node_rank + 1) * local))


def _joint_weights_dir(paths: _ImagePaths) -> str:
    """Where every node-partition puts its shards.

    At the key level rather than under ``node<k>/``: the shards are named by
    global rank (rank0..rank15), and a restore onto a different set of pods has
    to find all of them in one place.
    """
    return os.path.join(paths.key_dir or paths.model_dir, _WEIGHT_DIR)


def _check_digests(agents: list[Any], vllm_config: dict[str, Any]) -> None:
    """Refuse to cold-start node-partitions that would build different engines.

    The node-partitions profile their shapes independently and then meet in a
    collective. A disagreement there does not raise -- it deadlocks, for the
    full gloo timeout, with nothing in either log that names the field. This
    costs one Ray round trip to turn that into an error.
    """
    import ray
    mine = _config_hash(vllm_config)
    theirs = ray.get([a.config_digest.remote(vllm_config) for a in agents])
    bad = [(k + 1, d) for k, d in enumerate(theirs) if d != mine]
    if bad:
        raise RuntimeError(
            f"semi_p: the node-partitions of this engine disagree about its config: "
            f"leader={mine}, " + ", ".join(f"node{k}={d}" for k, d in bad) +
            ". Every node-partition must build the same engine or they deadlock in "
            "their first collective.")


def _prefetch_weights(weights_dir: str, ranks: list[int]) -> Any:
    """Warm the page cache for this pod's shards, in the background.

    A pod that did not dump reads its shards from disk cold, and that read is
    the whole difference between a same-pod restore and a swapped one (33 s
    against 8 s, measured). The CRIU, CUDA and NCCL steps that come first do
    not touch the disk, so the read is free if it happens under them.

    Best effort by construction: a failure here costs the speedup and nothing
    else, because ``load_weights`` reads the same files itself afterwards.
    """
    import threading

    def _read():
        for rank in ranks:
            rank_dir = os.path.join(weights_dir, f"rank{rank}")
            try:
                names = sorted(os.listdir(rank_dir))
            except OSError:
                continue
            for name in names:
                try:
                    with open(os.path.join(rank_dir, name), "rb") as handle:
                        while handle.read(8 << 20):
                            pass
                except OSError:
                    break

    thread = threading.Thread(target=_read, name="semip-prefetch", daemon=True)
    thread.start()
    return thread


def _dump_multinode(vllm_config: dict[str, Any], paths: _ImagePaths,
                    gpus: list[int], agents: list[Any], *,
                    master_port: int, ifname: str,
                    image_ref: str, driver_version: str) -> None:
    """Cold-start every node-partition together and dump them as one image.

    The ordering is the experiment's, and each step is where it is for a
    reason:

    * all node-partitions ``init`` concurrently, because they rendezvous inside
      vLLM and none returns until all have arrived;
    * the leader alone generates and stages, since its executor covers every
      rank;
    * the leader checkpoints CUDA first -- that is what drops the graphs and
      tears NCCL down across *all* ranks -- and the agents follow;
    * the leader's ``criu_dump`` parks every rank's message queue as its last
      collective step, so each agent's ``criu_dump`` waits for its own ranks
      to be parked before dumping. Those calls are issued first, so the wait
      overlaps the leader's dump rather than following it.
    """
    import ray
    from arctic_platform.inference.semi_persistence import Instance, MultiNode

    nnodes = paths.nnodes
    local = len(gpus)
    dump_id = uuid.uuid4().hex
    dump_ip = _leader_ip()
    weights_dir = _joint_weights_dir(paths)
    model_dir = paths.model_dir

    _check_digests(agents, vllm_config)

    t0 = time.perf_counter()
    # Each node-partition records its own pod's values: the floor is per PID namespace
    # and the mode is per process, so the leader's say nothing about an agent.
    per_node = {0: {"pid_floor": _raise_pid_floor(),
                    "unprivileged": _unprivileged_mode()}}

    def _meta(node_rank: int) -> dict[str, Any]:
        return {"image_ref": image_ref, "driver_version": driver_version,
                **per_node[node_rank],
                "nnodes": nnodes, "node_rank": node_rank,
                "dump_id": dump_id, "dump_ip": dump_ip}

    os.makedirs(model_dir, exist_ok=True)
    inst = Instance(vllm_config, model_dir,
                    multinode=MultiNode(node_rank=0, master_addr=dump_ip,
                                        master_port=int(master_port),
                                        ifname=ifname))
    logger.info("semi_p: multi-node cold start, %d pods, dump_id=%s, "
                "rendezvous %s:%d on %s", nnodes, dump_id, dump_ip,
                master_port, ifname)
    started = [
        agent.init.remote(
            vllm_config,
            os.path.join(paths.key_dir, f"{_NODE_DIR_PREFIX}{k + 1}"),
            _agent_gpus(local), k + 1, nnodes, dump_ip, int(master_port),
            ifname)
        for k, agent in enumerate(agents)]
    try:
        inst.init(gpus=gpus)
        # All node-partitions load weights from disk here, and in fresh pods that read
        # dominates: 25 min for GLM-5.3 on a cold page cache. gloo's own
        # rendezvous timeout is 1800 s, so this wait has to be longer than the
        # thing it is waiting for or it fails the job for being slow.
        for k, reply in enumerate(
                ray.get(started, timeout=_MULTINODE_INIT_TIMEOUT_S)):
            per_node[k + 1] = {"pid_floor": reply.get("pid_floor"),
                               "unprivileged": reply.get("unprivileged")}
        inst.wait()
        modes = {k: v["unprivileged"] for k, v in per_node.items()}
        if len(set(modes.values())) > 1:
            raise RuntimeError(
                f"semi_p: the node-partitions run different {_UNPRIVILEGED_ENV} modes "
                f"({modes}); each fixes the capability level its image "
                "records, so they could never restore under one mode")
        inst.generate([_DUMP_PROMPT], _DUMP_SAMPLING)
        inst.attach()
        inst.stage()
        inst.save_weights(weights_dir=weights_dir)
        inst.detach()
        inst.sleep()
        inst.arm_mq_park(ranks=list(range(local)))
        # Drops the graphs and tears NCCL down across every rank, including the
        # agents'. Theirs must follow, not lead.
        inst.cuda_checkpoint()
        inst.wait()
        ray.get([agent.cuda_checkpoint.remote() for agent in agents])
        # Issued before the leader's dump so the wait inside them overlaps it.
        dumping = [agent.criu_dump.remote(_meta(k + 1))
                   for k, agent in enumerate(agents)]
        inst.criu_dump(meta_extra=_meta(0))
        inst.wait()
        ray.get(dumping)
    except BaseException:
        try:
            inst.teardown()
        except Exception:  # noqa: BLE001
            logger.warning("semi_p: leader teardown after a failed multi-node "
                           "dump also failed", exc_info=True)
        for agent in agents:
            try:
                ray.get(agent.teardown.remote(), timeout=120)
            except Exception:  # noqa: BLE001
                logger.warning("semi_p: agent teardown after a failed "
                               "multi-node dump also failed", exc_info=True)
        raise
    time.sleep(2)
    _record_env_files(model_dir)
    logger.info("semi_p: multi-node dump complete in %.1fs, dump_id=%s",
                time.perf_counter() - t0, dump_id)


def _restore_multinode(engine_kwargs: dict[str, Any], paths: _ImagePaths,
                       gpus: list[int], agents: list[Any], *,
                       weights_dir: str | None, requested: dict[str, Any],
                       after: str) -> "_SemiPEngine":
    """Restore every node-partition of one image and return the leader's engine."""
    import ray
    from arctic_platform.inference.semi_persistence import Instance

    model_dir = paths.model_dir
    local = len(gpus)
    nnodes = paths.nnodes
    meta_path = os.path.join(model_dir, _IMAGE_DIR, "meta.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    baked = meta.get("vllm_config") or {}
    if not baked:
        raise ValueError(
            f"semi_p: image meta.json at {meta_path} has no vllm_config")
    _log_config_divergence(baked, engine_kwargs, model_dir, requested)
    tokenizer_path = engine_kwargs.get("model") or baked.get("model")
    mismatch = _unprivileged_mismatch(meta)
    if mismatch:
        raise RuntimeError(f"semi_p: {model_dir}: {mismatch}")
    _check_device_visibility(meta, model_dir)

    leader_ip = _leader_ip()
    remote_ranks = [r for k in range(1, nnodes) for r in _node_ranks(k, local)]
    local_ranks = list(range(local))
    logger.info("semi_p: multi-node restore of %s (%s), dump_id=%s, leader %s",
                model_dir, after, meta.get("dump_id"), leader_ip)

    prefetch = None
    if weights_dir:
        prefetch = _prefetch_weights(weights_dir, local_ranks)

    inst = Instance(baked, model_dir)
    restoring = [
        agent.criu_restore.remote(
            baked, os.path.join(paths.key_dir, f"{_NODE_DIR_PREFIX}{k + 1}"),
            _agent_gpus(local), k + 1)
        for k, agent in enumerate(agents)]
    try:
        inst.criu_restore().wait()
        ray.get(restoring, timeout=_MULTINODE_STEP_TIMEOUT_S)

        # The message-queue plane, rebuilt across pods. Every queue socket was
        # closed before the dump, so nothing here survived the image and the
        # node-partitions have to agree on a new one before any collective runs.
        inst.mq_begin_unpark(remote_ranks, leader_ip, local_ranks).wait()
        handle = inst.last_info["mq_begin_unpark"]["handle"]
        # One opaque blob per agent, each encoding its ranks' handles; the
        # leader's child decodes and merges them.
        handles: list[str] = []
        for k, agent in enumerate(agents):
            reply = ray.get(
                agent.mq_follower_unpark.remote(
                    handle, _node_ranks(k + 1, local)),
                timeout=_MULTINODE_STEP_TIMEOUT_S)
            handles.append(reply["handles"])
        inst.mq_finish_unpark(handles, local_ranks).wait()

        cuda = [agent.cuda_restore.remote() for agent in agents]
        inst.cuda_restore(gpus=gpus).wait()
        ray.get(cuda, timeout=_MULTINODE_STEP_TIMEOUT_S)

        if prefetch is not None:
            prefetch.join(timeout=0)

        # From here the leader's executor spans every rank again, so the rest
        # is ordinary single-engine work. reinit_nccl is the first time EFA
        # comes up: the cold start ran on sockets so that no EFA state would
        # have to survive CRIU.
        inst.reinit_nccl(master_addr=leader_ip,
                         ifname=_multinode_ifname()).wait()
        inst.attach().load_weights(weights_dir=weights_dir).wait()
        inst.wake_up_weights().wait()
        _check_staging_budget(inst, model_dir)
        inst.repin().plan_restore_weights().wait()
        inst.restore_weights().wait()
        inst.wake_up_kv_cache().wait()
        # Not rebind_graphs: the dump destroyed the graphs so that
        # ncclCommAbort could return, and these are captured fresh.
        inst.recapture_graphs().wait()
    except BaseException:
        try:
            inst.teardown()
        except Exception:  # noqa: BLE001
            logger.warning("semi_p: leader teardown after a failed multi-node "
                           "restore also failed", exc_info=True)
        for agent in agents:
            try:
                ray.get(agent.teardown.remote(), timeout=120)
            except Exception:  # noqa: BLE001
                logger.warning("semi_p: agent teardown after a failed "
                               "multi-node restore also failed", exc_info=True)
        raise
    logger.info("semi_p: multi-node restore complete model_dir=%s gpus=%s",
                model_dir, gpus)
    return _SemiPEngine(inst, tokenizer_path=tokenizer_path,
                        model=baked.get("model"), agents=agents)


def _agent_gpus(local: int) -> list[int]:
    """The physical GPUs an agent drives.

    Every pod in a zone exposes the same device set, and each node-partition takes
    the whole pod, so this is the identity list. It is a function rather than a
    literal because the agent resolves nothing itself -- the leader is the only
    place that knows the group's shape.
    """
    return list(range(local))


def _joint_hit(agents: list[Any], paths: _ImagePaths,
               local: int) -> tuple[bool, str | None]:
    """Whether every node-partition holds an image from the same dump.

    Returns ``(hit, dump_id)``. Node-partitions each holding *an* image prove
    nothing: they could be from different dumps, and restoring mismatched
    node-partitions deadlocks rather than failing.
    """
    import ray
    meta_path = os.path.join(paths.model_dir, _IMAGE_DIR, "meta.json")
    try:
        with open(meta_path) as handle:
            mine = json.load(handle)
    except (OSError, ValueError):
        return False, None
    my_id = mine.get("dump_id")
    probes = ray.get([
        agent.probe.remote(
            os.path.join(paths.key_dir, f"{_NODE_DIR_PREFIX}{k + 1}"))
        for k, agent in enumerate(agents)])
    missing = [k + 1 for k, p in enumerate(probes) if not p.get("hit")]
    if missing:
        logger.info("semi_p: node-partition(s) %s hold no image for this key; every "
                    "node-partition will cold-start and dump together", missing)
        return False, None
    mismatched = [(k + 1, p.get("dump_id")) for k, p in enumerate(probes)
                  if p.get("dump_id") != my_id]
    if mismatched:
        logger.warning(
            "semi_p: the node-partitions hold images from different dumps "
            "(leader=%s, %s); cold-starting rather than restoring a set that "
            "would deadlock in its first collective", my_id,
            ", ".join(f"node{k}={d}" for k, d in mismatched))
        return False, None
    return True, my_id


def _is_address_in_use(exc: BaseException) -> bool:
    """Whether ``exc`` is CRIU failing to rebind a port still in ``TIME_WAIT``.

    Two probes, because the failure can arrive in two shapes. ``worker.py``'s
    helper walks ``__cause__`` / ``__context__`` / exception groups for a real
    ``OSError``, which covers the chained case; but the ``Instance`` reports a
    worker-side failure as ``RuntimeError("command 'criu_restore' failed: ...")``
    carrying CRIU's stderr as *text*, with nothing to walk. Match the string too.
    """
    with contextlib.suppress(Exception):
        from arctic_platform.inference.server.worker import _is_address_in_use_error
        if _is_address_in_use_error(exc):
            return True
    return "address already in use" in str(exc).lower()


def _restore_with_port_retry(model_dir: str, engine_kwargs: dict[str, Any],
                             gpus: list[int], *, after: str,
                             weights_dir: str | None = None,
                             requested: dict[str, Any] | None = None
                             ) -> "_SemiPEngine":
    """``_restore``, retried while a recorded port is still in ``TIME_WAIT``.

    CRIU records every inet socket's local port and *rebinds* it at restore.
    ``socket:`` is on the child's dump keep-list, so those sockets ride into the
    image; a destructive dump -- or the teardown of a restored tree -- closes
    them with a FIN and leaves the tuples in ``TIME_WAIT`` for
    ``TCP_TIMEWAIT_LEN``. A restore inside that window dies with::

        Error (criu/sk-inet.c): Can't bind inet socket (id 0x...): Address already in use

    The library mitigates this by marking sockets ``SO_LINGER(1, 0)`` so they RST
    instead, but the call sits after the ``tp_size <= 1`` early return in
    ``_destroy_nccl`` and only walks worker ranks -- so nothing is marked at TP1,
    and the child's own sockets are never marked at any TP size. CRIU_PLUMBING.md
    Complication 12 records both gaps as "benign today", which held only because
    nothing restored within a minute of its own dump.

    Both of our paths do. The post-dump restore follows its dump by seconds; and
    with ``DESTROY=1``, re-running the same job puts the *cache-hit* restore
    seconds after the previous run's teardown. Sockets that were marked get the
    option replayed from the image and RST cleanly on teardown; unmarked ones
    re-block their ports for a minute every time.

    Retrying rather than sleeping unconditionally costs nothing when there is no
    collision, and it is the only thing that helps for an image already dumped
    without the marking -- the condition is time-bound, not baked in, so the
    same image restores fine once the window drains.
    """
    deadline = time.monotonic() + _TIME_WAIT_S + 15.0
    attempt = 0
    while True:
        attempt += 1
        try:
            return _restore(model_dir, engine_kwargs, gpus,
                            weights_dir=weights_dir, requested=requested)
        except Exception as exc:
            if not _is_address_in_use(exc) or time.monotonic() >= deadline:
                raise
            remaining = deadline - time.monotonic()
            logger.warning(
                "semi_p: restore of %s hit a port still in TIME_WAIT from the "
                "%s (attempt %d, ~%.0fs of the %.0fs window left); retrying in "
                "%.0fs. See CRIU_PLUMBING.md Complication 12.",
                model_dir, after, attempt, remaining, _TIME_WAIT_S,
                _RESTORE_RETRY_SLEEP_S)
            time.sleep(_RESTORE_RETRY_SLEEP_S)


def restore_and_wrap(engine_kwargs: dict[str, Any],
                     agents: list[Any] | None = None) -> "_SemiPEngine":
    """Bring a semi-p engine up and return a ``_SemiPEngine`` (blocking).

    Intended to be called via ``asyncio.to_thread`` from the worker.

    ``agents`` are the ``SemipNodeAgent`` handles for the other pods when this
    engine's TP group spans pods; ``None`` or empty is the single-pod case,
    which is every TP <= 8 deployment and takes exactly the path it always
    did. With agents, the hit decision, the dump and the restore all become
    joint: see ``_joint_hit``, ``_dump_multinode`` and ``_restore_multinode``.

    On a cache hit this restores the image and serves from it. On a **miss**
    it first tries to materialize the image from the read-only published
    mirror; failing that it cold-starts vLLM and dumps an image into the
    derived directory. Either way it then restores and serves from that
    restore: the cold-started engine cannot serve directly because
    ``criu_dump()`` is destructive, and since a restore is needed regardless,
    doing it here makes "the dump worked" and "the image is restorable" a
    single observation.

    The image directory is **derived, not supplied**:
    ``$SEMIP_IMAGE_CACHE/<config hash>_<environment hash>``, plus
    ``replica<slot>`` when the pod holds several replicas. That is what lets
    a dump and a later restore agree on a path with nothing assigning one, and
    it satisfies ``criu_restore``'s exact-string ``model_dir`` check for free,
    since the cache root is a fixed mount point and the hashes are a pure
    function of the config and the environment.

    Deriving it also fixes what a bare path could not: the directory name now
    *is* the cache key, so a changed ``vllm_config`` or a rebuilt container
    image resolves somewhere else rather than silently reusing an image that
    answers a different question.
    """
    # Before any Instance exists: its worker inherits this environment. A job's
    # extra_env, applied to os.environ before this call, still overrides it.
    os.environ.setdefault(_UNPRIVILEGED_ENV, "1")

    # On sys.path once the Instance import above has run. The actor's stdout is
    # Ray's per-worker file, so this is what puts the engine's lines in the pod
    # log beside the worker's and child's; they still propagate to Ray too.
    import semip_logging
    semip_logging.attach_pod_log([logger.name])

    vllm_config = _vllm_config_from_engine_kwargs(engine_kwargs)

    # Topology goes in the key, node identity does not. ``nnodes`` changes the
    # image -- a node-partition of a TP=16 group is not a TP=8 engine -- so it is hashed;
    # node_rank, the rendezvous address and the interface travel outside the
    # config so that all node-partitions derive the same key and a restored engine is
    # free to meet somewhere new.
    agents = list(agents or ())
    if agents:
        vllm_config = dict(vllm_config)
        vllm_config["nnodes"] = len(agents) + 1

    paths = _resolve_model_dir(vllm_config)
    model_dir = paths.model_dir

    # Resolve once: this clears CUDA_VISIBLE_DEVICES as a side effect, so a
    # second call would find it empty and fall back to [0]. The dump and the
    # restore both take the list resolved here.
    gpus = _resolve_physical_gpus()
    _check_tp_matches_gpus(vllm_config, gpus, paths.nnodes)

    if agents:
        return _multinode_restore_and_wrap(
            engine_kwargs, vllm_config, paths, gpus, agents)

    # Resolved before the lock and re-used after it: the published tree is
    # read-only, so listing it twice could only produce the same answer at extra
    # cost. The weight hash comes from the skeleton's name, so a miss here also
    # means no published weights to read.
    skeleton_dir, weight_hash = _resolve_published_skeleton(paths)
    source_dir = _replica_source(skeleton_dir, paths)

    meta_path = os.path.join(model_dir, _IMAGE_DIR, "meta.json")
    after = "previous run's teardown"
    if not os.path.isfile(meta_path):
        # Creates the parent too: the image-cache root need not exist yet.
        os.makedirs(model_dir, exist_ok=True)
        with _dump_lock(model_dir):
            # Re-check under the lock. If we waited for another job's dump of
            # this same model_dir, the image we wanted now exists and dumping
            # again would only throw away its work.
            if os.path.isfile(meta_path):
                logger.info(
                    "semi_p: another job dumped %s while we waited; restoring "
                    "its image instead of dumping again", model_dir)
            elif _materialize_from_source(source_dir, model_dir,
                                         paths.weight_root, weight_hash,
                                         verified_dir=skeleton_dir,
                                         strict=paths.replica_count > 1):
                # A pod that never ran this config can still serve it warm.
                after = "copy from the image source"
            else:
                _dump(vllm_config, model_dir, gpus,
                      image_ref=paths.image_ref,
                      driver_version=paths.driver_version)
                after = "dump we just took"

    # Nothing is cleaned up here on the miss path: _worker_criu_save rmtree's
    # image/ before writing, and _semip_save_weights does the same for its rank
    # dir, both with a targeted message when the leftovers belong to another uid.
    # Repeating that would duplicate the work and swallow the diagnostic.
    return _restore_with_port_retry(
        model_dir, engine_kwargs, gpus, after=after,
        weights_dir=_weights_dir_for_restore(
            model_dir, paths.weight_root, weight_hash),
        requested=vllm_config)


def _multinode_restore_and_wrap(engine_kwargs: dict[str, Any],
                                vllm_config: dict[str, Any],
                                paths: _ImagePaths, gpus: list[int],
                                agents: list[Any]) -> "_SemiPEngine":
    """``restore_and_wrap`` for an engine whose TP group spans pods.

    Same shape as the single-pod path -- hit, else materialize, else dump,
    then restore -- with every decision made for the group rather than for this
    pod. The dump lock is taken on the shared key directory, not on this pod's
    node-partition, because the node-partitions dump together and two leaders dumping one key at
    once would interleave their images.
    """
    local = len(gpus)
    key_dir = paths.key_dir or paths.model_dir
    hit, dump_id = _joint_hit(agents, paths, local)
    after = "previous run's teardown"

    if not hit:
        os.makedirs(paths.model_dir, exist_ok=True)
        with _dump_lock(key_dir):
            # Re-probe under the lock: another job may have dumped this key
            # while we waited, and dumping again would throw its work away.
            hit, dump_id = _joint_hit(agents, paths, local)
            if hit:
                logger.info(
                    "semi_p: another job dumped %s while we waited; restoring "
                    "its image instead of dumping again", key_dir)
            else:
                materialized = _multinode_materialize(agents, paths)
                if materialized:
                    after = "copy from the image source"
                else:
                    _dump_multinode(
                        vllm_config, paths, gpus, agents,
                        master_port=_pick_master_port(),
                        ifname=_multinode_ifname(),
                        image_ref=paths.image_ref,
                        driver_version=paths.driver_version)
                    after = "dump we just took"
                hit, dump_id = _joint_hit(agents, paths, local)
                if not hit:
                    raise RuntimeError(
                        "semi_p: after a multi-node dump the node-partitions still do "
                        "not agree on a dump_id; refusing to restore a set "
                        "that would deadlock in its first collective")

    skeleton_dir, weight_hash = _resolve_published_skeleton(paths)
    weights_dir = _weights_dir_for_restore(
        key_dir, paths.weight_root, weight_hash)
    if weights_dir is None:
        weights_dir = _joint_weights_dir(paths)

    deadline = time.monotonic() + _TIME_WAIT_S + 15.0
    attempt = 0
    while True:
        attempt += 1
        try:
            return _restore_multinode(
                engine_kwargs, paths, gpus, agents,
                weights_dir=weights_dir, requested=vllm_config, after=after)
        except Exception as exc:
            if not _is_address_in_use(exc) or time.monotonic() >= deadline:
                raise
            # All node-partitions retry together: the ports CRIU rebinds are
            # recorded per image, so a collision on any of them means no node-partition's
            # restore completed. See _restore_with_port_retry.
            logger.warning(
                "semi_p: multi-node restore hit a port still in TIME_WAIT "
                "from the %s (attempt %d); retrying in %.0fs",
                after, attempt, _RESTORE_RETRY_SLEEP_S)
            time.sleep(_RESTORE_RETRY_SLEEP_S)


def _pick_master_port() -> int:
    """One rendezvous port for the whole group, chosen by the leader.

    Every node-partition has to name the same port, and only the leader is in a
    position to choose: a port that is free on a follower says nothing about
    this pod, which is the one that binds it.
    """
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _multinode_materialize(agents: list[Any], paths: _ImagePaths) -> bool:
    """Copy every node-partition of a published skeleton into place, or none.

    A pod that materializes while another pod of the engine cold-starts is the
    mismatch this whole protocol exists to avoid, so a partial result is
    treated as a miss and every node-partition cold-starts.
    """
    import ray

    skeleton_dir, weight_hash = _resolve_published_skeleton(paths)
    if not skeleton_dir:
        return False
    key_dir = paths.key_dir or paths.model_dir
    mine = _materialize_from_source(
        _node_source(skeleton_dir, 0), paths.model_dir,
        paths.weight_root, weight_hash, verified_dir=skeleton_dir, strict=True)
    if not mine:
        return False
    theirs = ray.get([
        agent.materialize.remote(
            _node_source(skeleton_dir, k + 1),
            os.path.join(key_dir, f"{_NODE_DIR_PREFIX}{k + 1}"),
            paths.weight_root, weight_hash, skeleton_dir)
        for k, agent in enumerate(agents)])
    if all(theirs):
        return True
    logger.warning(
        "semi_p: only some node-partitions could be materialized from the "
        "published skeleton (%s); cold-starting every node-partition instead",
        [True] + list(theirs))
    return False


def _node_source(skeleton_dir: str | None, node_rank: int) -> str | None:
    """A published skeleton's ``node<k>/`` node-partition, beside ``_replica_source``."""
    if not skeleton_dir:
        return None
    return os.path.join(skeleton_dir, f"{_NODE_DIR_PREFIX}{node_rank}")


def _dense_from_numpy(prompt_logprobs: Any) -> Any:
    """Dense prompt logprobs cross the child's pipe as numpy; the worker wants
    the tensors the dense patch produced. Anything else passes through."""
    if not isinstance(prompt_logprobs, dict):
        return prompt_logprobs
    import torch
    return {key: torch.from_numpy(value) if hasattr(value, "dtype") else value
            for key, value in prompt_logprobs.items()}


class _SemiPEngine:
    """Stands in for ``self.llm`` (a vLLM ``AsyncLLM``) over a semi-p Instance.

    Generates run concurrently, as they do on ``AsyncLLM``: each is its own
    ``generate`` command, and the child batches every request it holds into
    one engine step. Everything else (sleep, wake, pause, resume) runs alone,
    after the in-flight generates finish and before any new one starts.
    """

    def __init__(self, inst: Any, *, tokenizer_path: str | None,
                 model: str | None, agents: list[Any] | None = None):
        self._inst = inst
        self._tokenizer_path = tokenizer_path
        self._model = model
        self._tokenizer: Any = None
        # Held by control operations for their whole run, and by a generate
        # only while it submits.
        self._lock = asyncio.Lock()
        # req_id -> (loop, future) of each generate the child still holds.
        # Written on the event loop, popped on the demuxer thread.
        self._waiters: dict[str, tuple[asyncio.AbstractEventLoop,
                                       asyncio.Future]] = {}
        self._waiters_lock = threading.Lock()
        self._inflight = 0
        self._drained = asyncio.Event()
        self._drained.set()
        inst.add_cmd_listener("generate", self._on_generate_done)
        # The other node-partitions of a pod-spanning engine. They hold GPUs and a
        # restored process tree, so they have to come down with this engine;
        # nothing else owns them.
        self._agents = list(agents or ())

    # -- generate -----------------------------------------------------------

    async def generate(self, prompt_input: Any, params: Any,
                       request_id: str | None = None,
                       reasoning_ended: bool | None = None):
        """Async generator yielding a single vLLM-``RequestOutput``-shaped final."""
        sp = _sampling_params_to_dict(params)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        async with self._lock:
            with self._waiters_lock:
                self._inst.generate([prompt_input], sp,
                                    reasoning_ended=reasoning_ended)
                self._waiters[self._inst.last_req_id] = (loop, future)
            self._inflight += 1
            self._drained.clear()
        yield self._to_request_output(await future, sp)

    def _on_generate_done(self, cmd: str, elapsed: float,
                          error: object | None, info: Any) -> None:
        """Demuxer listener: hand each finished generate to its waiter."""
        rid = info.get("req_id") if isinstance(info, dict) else None
        with self._waiters_lock:
            if rid is None:
                # The worker reports a dead child without a req_id, and
                # nothing the child held will ever finish.
                done = list(self._waiters.items())
                self._waiters.clear()
            else:
                waiter = self._waiters.pop(rid, None)
                done = [(rid, waiter)] if waiter is not None else []
        for done_rid, (loop, future) in done:
            result = self._inst.generate_results.pop(done_rid, None) or {}
            loop.call_soon_threadsafe(self._settle, future, result, error)

    def _settle(self, future: asyncio.Future, result: dict[str, Any],
                error: object | None) -> None:
        # Counted here, not where the caller awaits: a caller that gives up
        # leaves its request running in the child all the same.
        self._inflight -= 1
        if not self._inflight:
            self._drained.set()
        if future.done():
            return
        if error is not None:
            future.set_exception(
                RuntimeError(f"semi_p: generate failed: {error}"))
        else:
            future.set_result(result)

    @contextlib.asynccontextmanager
    async def _exclusive(self):
        """Run a control operation with no generate in flight."""
        async with self._lock:
            await self._drained.wait()
            # Every generate failure already reached its caller through
            # _settle, but the demuxer also latches the first one, and the
            # control operation's own wait() would raise it as its own.
            try:
                self._inst.wait()
            except RuntimeError as exc:
                logger.debug("semi_p: dropping a reported generate error: %s",
                             exc)
            yield

    @staticmethod
    def _first(seq, i=0):
        if isinstance(seq, (list, tuple)) and len(seq) > i:
            return seq[i]
        return None

    def _to_request_output(self, res: dict[str, Any],
                           sp: dict[str, Any] | None = None) -> SimpleNamespace:
        # outputs: [[text per sample] per prompt]; take prompt 0, sample 0.
        prompt_texts = self._first(res.get("outputs"))
        text = self._first(prompt_texts) or ""

        # completion_token_ids: [[ids per sample] per prompt].
        token_ids = self._first(self._first(res.get("completion_token_ids")))
        prompt_ids = self._first(res.get("prompt_token_ids"))
        if token_ids is None or prompt_ids is None:
            # Callers train on these ids, so a stand-in would be silently
            # wrong rather than merely missing.
            raise RuntimeError(
                "semi_p: the restored engine reported no token ids for this "
                f"request (result keys: {sorted(res)})")

        logprobs = self._first(self._first(res.get("completion_logprobs")))
        prompt_logprobs = _dense_from_numpy(
            self._first(res.get("prompt_logprobs")))
        sp = sp or {}
        for name, value in (("logprobs", logprobs),
                            ("prompt_logprobs", prompt_logprobs)):
            if sp.get(name) is not None and value is None:
                raise RuntimeError(
                    f"semi_p: {name}={sp[name]} was requested but the "
                    f"restored engine returned none")

        finish_reasons = res.get("finish_reasons") or []
        finish_reason = finish_reasons[0] if finish_reasons else "stop"

        choice = SimpleNamespace(
            index=0,
            text=text,
            token_ids=token_ids,
            finish_reason=finish_reason,
            logprobs=logprobs,
            cumulative_logprob=None,
        )
        return SimpleNamespace(
            outputs=[choice],
            prompt_token_ids=prompt_ids,
            prompt_logprobs=prompt_logprobs,
            num_cached_tokens=res.get("num_cached_tokens"),
        )

    # -- tokenizer ----------------------------------------------------------

    def get_tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tokenizer_path, trust_remote_code=True
            )
        return self._tokenizer

    # -- sleep / wake / pause / resume / prefix-cache -----------------------

    async def collective_rpc(self, method: str, args: tuple = (),
                             kwargs: dict | None = None):
        if method == "sleep":
            async with self._exclusive():
                await asyncio.to_thread(self._sleep_blocking)
            return [None]
        if method == "wake_up":
            async with self._exclusive():
                await asyncio.to_thread(self._wake_blocking)
            return [None]
        if method == "_arl_cuda_sync":
            # CUDA barrier only needed for weight-sync; no-op for sampling.
            return []
        if method in _WEIGHT_SYNC_RPCS:
            raise NotImplementedError(
                f"semi_p: weight-sync not implemented yet ({method})"
            )
        raise NotImplementedError(f"semi_p: collective_rpc({method}) not supported")

    def _sleep_blocking(self):
        self._inst.sleep()
        self._inst.wait()

    def _wake_blocking(self):
        # restore_weights() is not optional here: sleep() is level 2, which
        # frees the weight memory, and wake_up_weights() only re-allocates the
        # parameter tensors without repopulating them. Without it the model
        # generates from uninitialized weights and never errors. Matches
        # Orchestrator's wake sequence; no repin(), since the buffer stays
        # pinned and the chunk plan cached at restore time still applies.
        self._inst.wake_up_weights()
        self._inst.restore_weights()
        self._inst.wake_up_kv_cache()
        self._inst.wait()

    async def pause_generation(self, mode: str = "keep", clear_cache: bool = False):
        async with self._exclusive():
            await asyncio.to_thread(self._pause_blocking)

    def _pause_blocking(self):
        self._inst.pause()
        self._inst.wait()

    async def resume_generation(self):
        # Not _exclusive: a generate submitted while paused is parked in the
        # child until this resume, so waiting for it to drain would deadlock.
        async with self._lock:
            await asyncio.to_thread(self._resume_blocking)

    def _resume_blocking(self):
        self._inst.resume()
        self._inst.wait()

    async def reset_prefix_cache(self) -> bool:
        # No-op: the published Instance has no reset_prefix_cache primitive.
        # The only live caller is replica_pool's post-weight-sync reset, and
        # weight sync already raises NotImplementedError above, so returning
        # False (reset did not happen) is honest and unreachable in practice.
        # Restoring the primitive across instance/worker/vllm_child is the
        # alternative if weight sync ever lands.
        logger.debug("semi_p: reset_prefix_cache is a no-op")
        return False

    # -- teardown -----------------------------------------------------------

    def close(self) -> None:
        inst = getattr(self, "_inst", None)
        if inst is not None:
            try:
                inst.teardown()
                # teardown() is a non-blocking _send, and the actual kill of the
                # vLLM child happens in the Instance's worker process when it
                # picks the command up (_kill_process_tree in worker.py's
                # teardown handler). Without this wait, /destroy lets the actor
                # exit first, the worker dies before draining the command, and
                # the child survives -- reparented to init, invisible to a
                # `ps | grep vllm` because its comm is plain "python", and
                # holding the entire GPU. That is exactly what a restored tree
                # does: criu_restore sets child_proc=None, so worker_loop's
                # force-kill branch is skipped and this handler is the only
                # thing that reaps it. Bounded by the worker's own 30s/5s joins.
                inst.wait()
            except Exception:  # pragma: no cover - best effort
                logger.warning("semi_p: teardown failed", exc_info=True)
            self._inst = None
            with self._waiters_lock:
                waiters = list(self._waiters.values())
                self._waiters.clear()
            for loop, future in waiters:
                loop.call_soon_threadsafe(
                    self._settle, future, {}, "engine closed")
        # Fan out to the other node-partitions. After the leader is down they hold a
        # process tree that can never be driven again -- its executor was the
        # leader's -- so leaving them alive would pin a pod's GPUs until the
        # job ended.
        agents, self._agents = list(getattr(self, "_agents", ())), []
        if agents:
            import ray
            for agent in agents:
                try:
                    ray.get(agent.teardown.remote(), timeout=120)
                except Exception:  # pragma: no cover - best effort
                    logger.warning("semi_p: agent teardown failed",
                                   exc_info=True)

    def __del__(self):  # pragma: no cover - GC path
        try:
            self.close()
        except Exception:
            pass
