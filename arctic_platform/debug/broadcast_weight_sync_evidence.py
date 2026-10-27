# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in evidence for the native DSS-to-AP broadcast weight-transfer route."""

from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy

from arctic_platform.debug.weight_sync_evidence import VERIFY_ENV
from arctic_platform.debug.weight_sync_evidence import ShadowVerifier
from arctic_platform.debug.weight_sync_evidence import enabled
from arctic_platform.debug.weight_sync_evidence import source_files
from arctic_platform.debug.weight_sync_evidence import tensor_manifest
from arctic_platform.debug.weight_sync_evidence import verify_sources

_SEND = ContextVar("native_broadcast_weight_evidence", default=None)
TRAINING_OPTION = "arctic_platform_weight_sync_verify"
SHADOW_MAX_BYTES_ENV = "ARCTIC_PLATFORM_WEIGHT_SYNC_SHADOW_MAX_BYTES"


def maybe_configure_broadcast_training(ds_config):
    """Consume the opt-in before DeepSpeed sees it in a dedicated trainer process."""
    if TRAINING_OPTION not in ds_config:
        return ds_config
    configured = ds_config[TRAINING_OPTION]
    if type(configured) is not bool:
        raise ValueError(f"{TRAINING_OPTION} must be a boolean")
    result = {key: value for key, value in ds_config.items() if key != TRAINING_OPTION}
    if configured:
        os.environ[VERIFY_ENV] = "1"
    return result


def maybe_configure_broadcast_sampling(sampling, *, verify):
    """Carry verification through the native sampler actor and vLLM child environments."""
    if not verify:
        return sampling
    vllm = deepcopy(sampling.vllm)
    extra_env = vllm.get("extra_env")
    if extra_env is not None and not isinstance(extra_env, dict):
        raise ValueError("weight-sync extra_env must be an object")
    extra_env = dict(extra_env or {})
    if extra_env.get(VERIFY_ENV, "1") != "1":
        raise ValueError("conflicting weight-sync verification environment")
    extra_env[VERIFY_ENV] = "1"
    limits = []
    for value in (extra_env.get(SHADOW_MAX_BYTES_ENV), os.environ.get(SHADOW_MAX_BYTES_ENV)):
        if value is None:
            continue
        try:
            limit = int(value) if isinstance(value, str) else 0
        except ValueError:
            limit = 0
        if limit <= 0:
            raise ValueError(f"{SHADOW_MAX_BYTES_ENV} must be a positive integer byte limit")
        limits.append(limit)
    if len(set(limits)) > 1:
        raise ValueError("conflicting weight-sync shadow byte limits")
    if limits:
        extra_env[SHADOW_MAX_BYTES_ENV] = str(limits[0])
    vllm["extra_env"] = extra_env
    return sampling.model_copy(update={"vllm": vllm}, deep=True)


@contextmanager
def weight_sync_verification_environment():
    """The legacy local client inherits this opt-in; never leak it to later checks."""
    previous = os.environ.get(VERIFY_ENV)
    os.environ[VERIFY_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(VERIFY_ENV, None)
        else:
            os.environ[VERIFY_ENV] = previous


def maybe_send_broadcast_weights(sender, weights, send):
    """Observe the existing stream once, without replacing its exporter or transport."""
    if not enabled():
        return send(weights)
    if _SEND.get() is not None:
        raise RuntimeError("nested broadcast weight verification is unsupported")
    state = {"trainer_parameters": None, "trainer_inventory": None}
    token = _SEND.set(state)
    sources = source_files(send, type(sender), maybe_send_broadcast_weights)
    manifest = {}

    def observed():
        for name, value in weights:
            if not isinstance(name, str) or not name or name in manifest:
                raise ValueError("broadcast tensor names must be nonempty and unique")
            entry = tensor_manifest(value)
            manifest[name] = entry
            yield name, value
            if tensor_manifest(value) != entry:
                raise RuntimeError("actually sent broadcast tensor changed during transfer")

    try:
        verify_sources(sources)
        result = send(observed())
        verify_sources(sources)
        if not isinstance(result, dict):
            raise TypeError("native broadcast sender did not return a dictionary")
        if not manifest or state["trainer_parameters"] is None:
            raise RuntimeError("broadcast sender has no complete collective trainer inventory")
        sources.update(state["evidence_sources"])
        verify_sources(sources)
        return {
            **result,
            "sent_manifest": manifest,
            "trainer_parameters": state["trainer_parameters"],
            "trainer_inventory": state["trainer_inventory"],
            "evidence_sources": sources,
        }
    finally:
        _SEND.reset(token)


def maybe_bind_broadcast_trainer_inventory(model, *, is_master, is_zero3):
    """All exporter ranks bind actual EP holdings; count DP replicas only once."""
    if not enabled():
        return
    import torch.distributed as dist

    from arctic_platform.debug.ep_weight_sync_evidence import global_inventory

    if is_zero3:
        raise ValueError("native broadcast evidence does not admit ZeRO3 tensor views")
    if not dist.is_initialized():
        raise RuntimeError("native broadcast evidence requires the actual training process group")
    rank, world = dist.get_rank(), dist.get_world_size()
    sources = source_files(
        maybe_bind_broadcast_trainer_inventory,
        type(model).named_parameters,
        dist.all_gather_object,
        global_inventory,
    )
    parameters = {}
    error = None
    try:
        if is_master != (rank == 0):
            raise ValueError("exporter master does not match the actual training rank")
        for name, value in model.named_parameters():
            if hasattr(value, "ds_id"):
                raise ValueError("native broadcast evidence does not admit ZeRO3 tensor views")
            members = None
            if getattr(value, "allreduce", True) is False:
                import deepspeed.utils.groups as groups

                group_name = getattr(value, "group_name", None)
                if not isinstance(group_name, str) or not group_name:
                    raise ValueError("expert shard has no actual named process group")
                group = groups._get_expert_parallel_group(group_name)
                members = list(dist.get_process_group_ranks(group))
                sources.update(source_files(groups._get_expert_parallel_group, dist.get_process_group_ranks))
            parameters[name] = {
                **tensor_manifest(value),
                "requires_grad": value.requires_grad,
                "ep_ranks": members,
            }
        if not parameters:
            raise ValueError("actual trainer has no registered parameters")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    verify_sources(sources)
    local = {"rank": rank, "world_size": world, "parameters": parameters, "buffers": {}, "error": error}
    records = [None] * world
    dist.all_gather_object(records, local)
    errors = [record.get("error") for record in records if isinstance(record, dict) and record.get("error")]
    if errors:
        raise ValueError("collective trainer inventory failed: " + "; ".join(errors))
    inventory = global_inventory(records)
    _verify_trainer_replicas(records, inventory)
    verify_sources(sources)
    state = _SEND.get()
    if is_master:
        if state is None or state["trainer_parameters"] is not None:
            raise RuntimeError("trainer inventory is not bound to exactly one active broadcast")
        state.update(
            trainer_parameters={
                "numel": sum(entry["global_numel"] for entry in inventory["parameters"].values()),
                "count": len(inventory["parameters"]),
            },
            trainer_inventory=inventory,
            evidence_sources=sources,
        )


def _verify_trainer_replicas(records, inventory):
    by_rank = {record["rank"]: record for record in records}
    for name, entry in inventory["parameters"].items():
        groups = entry["ep_groups"] or [[rank] for rank in sorted(by_rank)]
        for shards in zip(*groups):
            if len({by_rank[rank]["parameters"][name]["sha256"] for rank in shards}) != 1:
                raise ValueError(f"actual trainer replica bytes disagree: {name}")


def maybe_admit_broadcast_receiver(receiver, *, engine_only):
    if not enabled():
        return
    if not engine_only and receiver.model_config.quantization:
        raise ValueError("quantized native broadcast shadow evidence is not implemented")


def maybe_begin_broadcast_load(receiver, model, engine):
    if not enabled():
        return None
    if receiver.model_config.quantization:
        raise ValueError("quantized native broadcast shadow evidence is not implemented")
    limit = int(os.environ.get(SHADOW_MAX_BYTES_ENV, str(1 << 30)))
    verifier = ShadowVerifier(model, max_shadow_bytes=limit)
    verifier.sources.update(source_files(maybe_begin_broadcast_load, type(engine).receive_weights))
    verify_sources(verifier.sources)
    return verifier


def maybe_bind_broadcast_loader(verifier, writer):
    if verifier is not None:
        verify_sources(verifier.sources)
        verifier.sources.update(source_files(type(writer)))
        verify_sources(verifier.sources)


def maybe_load_broadcast_tensor(verifier, loader, name, value):
    if verifier is None:
        return loader(name, value)
    return verifier.load(name, value, loader=loader)


def maybe_load_broadcast_strategy(loader, model, engine, verifier):
    if verifier is None:
        return loader(model, engine)
    return loader(model, engine, verifier=verifier)


def maybe_broadcast_receiver_result(result, verifier, *, rank):
    if verifier is None:
        return result
    return {
        **result,
        "rank": rank,
        "weight_sync_verification": {**verifier.finish(), "rank": rank},
    }


def maybe_collect_broadcast_verification(results, *, engine_only, llm, rank_offset):
    """Preserve native results and reject missing or duplicated sampler ranks."""
    if not enabled() or engine_only:
        return results
    expected_ranks = llm.vllm_config.parallel_config.world_size
    if (
        type(expected_ranks) is not int
        or expected_ranks < 1
        or type(rank_offset) is not int
        or not isinstance(results, list)
        or len(results) != expected_ranks
        or any(
            not isinstance(row, dict) or not isinstance(row.get("weight_sync_verification"), dict) for row in results
        )
        or any(type(row.get("rank")) is not int for row in results)
        or {row["rank"] for row in results} != set(range(rank_offset, rank_offset + expected_ranks))
        or any(
            type(row["weight_sync_verification"].get("rank")) is not int
            or row["weight_sync_verification"]["rank"] != row["rank"]
            for row in results
        )
    ):
        raise RuntimeError("native broadcast did not return every actual sampler rank's verification")
    return [
        {
            **results[0],
            "all_rank_weight_sync_verification": [row["weight_sync_verification"] for row in results],
        },
        *results[1:],
    ]
