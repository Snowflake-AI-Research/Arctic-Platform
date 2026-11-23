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

"""Which skeleton a semi_p job body would publish, and which nodes already verified it.

A skeleton is ``<model>/skeleton/<cfg12>_<env12>_<wt12>``. ``cfg12`` hashes the
vLLM config the engine derives from the job plus what the image is bound to
(``semip_engine._config_hash``). It is computed here offline by replaying every
step between a job body and that hash: dss's ``InferenceConfig`` defaults and
``_prepare_sampling_engine_kwargs``, ``ModelConfig.to_engine_kwargs``, the
replica pool, ``InferenceWorker.initialize``, ``semip_frontend.restore_and_attach``
and ``semip_engine``'s projection, device binding and hash. The engine modules
import vLLM, Ray and NVML, so the steps are restated rather than imported;
``tests/test_keys.py`` holds them to the engine's source and to published keys.

One input is not in the job body: a job that names no ``reasoning_parser`` gets
the one chat pairs with the checkpoint's architecture (``chat.CHAT_MODELS``), or
none. ``Keying.candidates`` then holds one cfg12 per possible reasoner, and the
node labels pick the one that exists.

``env12`` hashes the pod's image digest and GPU driver, and ``wt12`` the staged
weights, so neither is computed. The model-cache DaemonSet labels each node
``model.neutrino.snowflake.com/<segment>=ready`` once it has verified a copy;
``<segment>`` is the directory under the bucket, sanitized and, past 63
characters, cut to 55 plus ``-`` and 7 hex of the directory's sha256. A cut label
still shows the model, the start of cfg12 and usually of env12, and a key whose
env12 and wt12 are known (from other labels) is confirmed by recomputing it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

from arctic_platform.common.peft import normalize_lora_peft_config
from arctic_platform.inference.semi_persistence.onboarding import verify
from arctic_platform.inference.semi_persistence.onboarding.jobs import GPUS_PER_NODE

HASH_LEN = 12

# dss.model_paths.resolve_model_path: an <org>/<model> id is baked under this root.
BAKED_MODEL_ROOT = "/mnt/neutrino/base-models"
# dss InferenceConfig.max_num_batched_tokens, which always reaches the engine.
DSS_MAX_NUM_BATCHED_TOKENS = 8192
# dss.ray_dss.peft.VLLM_SUPPORTED_LORA_RANKS: max_lora_rank is the first that covers r.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
# Keys dss moves from vllm_config into extra_engine_kwargs (_prepare_sampling_engine_kwargs).
DSS_EXTRA_KEYS = ("enforce_eager", "enable_return_routed_experts", "enable_prefix_caching",
                  "router_replay_max_cache_bytes", "router_replay_cache_device",
                  "distributed_executor_backend", "enable_flashinfer_autotune", "speculative_config")

# server/config.py ModelConfig: every field model_dump() emits (those not exclude=True), with
# its type and default. to_engine_kwargs() keeps the ones that are not None.
MODEL_CONFIG_FIELDS: dict[str, tuple[type, Any]] = {
    "tensor_parallel_size": (int, 1),
    "pipeline_parallel_size": (int, 1),
    "distributed_executor_backend": (str, None),
    "max_model_len": (int, 32768),
    "max_num_seqs": (int, 1024),
    "gpu_memory_utilization": (float, 0.95),
    "quantization": (str, None),
    "enable_chunked_prefill": (bool, True),
    "kv_cache_dtype": (str, None),
    "enable_expert_parallel": (bool, False),
    "trust_remote_code": (bool, True),
    "fp32_lm_head": (bool, False),
    "gdn_prefill_backend": (str, None),
    "structured_outputs_config": (dict, None),
    "speculative_config": (dict, None),
    "reasoning_parser": (str, None),
    "tool_call_parser": (str, None),
    "chat_reasoning_parser": (str, None),
    "return_reasoning_content": (bool, False),
    "forest_cascade_attn_configs": (str, "{}"),
    "semi_p": (bool, False),
}
# ModelConfig fields with exclude=True: accepted from the job, never in the engine kwargs.
MODEL_CONFIG_EXCLUDED = ("ulysses_sequence_parallel_size", "text_config_overrides", "extra_engine_kwargs",
                         "extra_env", "ray_num_gpus", "lora_sync_staging", "clear_cache_on_weight_sync")
# Popped by InferenceWorker.initialize before the restore (forest_cascade_attn_configs because
# ARCTIC_INFERENCE_ENABLED is unset on the pods) and by _pop_router_replay_engine_kwargs.
WORKER_POPPED = ("tool_call_parser", "chat_reasoning_parser", "return_reasoning_content", "lora_adapter_path",
                 "lora_peft_config", "semi_p", "semi_p_model_dir", "semi_p_agents", "forest_cascade_attn_configs",
                 "router_replay_max_cache_bytes", "router_replay_cache_device", "router_replay_staging_bytes")
WORKER_EXTENSION_CLS = "arctic_platform.inference.server.weight_sync.WeightSyncExtension"
# semip_engine._NON_VLLM_ENGINE_KEYS.
NON_VLLM_ENGINE_KEYS = ("ray_num_gpus", "router_replay_max_cache_bytes", "router_replay_cache_device",
                        "router_replay_staging_bytes")
# ModelConfig fields vLLM's AsyncEngineArgs rejects; semip_engine._strip_kwargs_vllm_rejects
# drops them at their default and refuses the job otherwise.
VLLM_REJECTED = ("fp32_lm_head",)
# Every reasoner chat.CHAT_MODELS can add to an engine whose job names none, and none.
CHAT_REASONERS = (None, "qwen3", "glm47", "deepseek_v4", "openai_gptoss", "minimax_m2", "nemotron_v3", "inkling")

# semip_engine._device_binding: at TP >= 2 an image is bound to the pod's /dev/nvidia* and
# /dev/infiniband/uverbs* nodes, which a whole-node pod always holds in full.
DEVICE_BOUND_MIN_TP = 2
WHOLE_NODE_DEVICES = {
    "H200": sorted([f"nvidia{i}" for i in range(GPUS_PER_NODE)] + [f"uverbs{i}" for i in range(16)]),
}
TP1_REPLICA_LAYOUT = "replica-layout"

LABEL_PREFIX = "model.neutrino.snowflake.com/"
_CUT = 55
_LABEL_CUT = re.compile(r"(.+)-[0-9a-f]{7}")
_HEX12 = re.compile(r"[0-9a-f]{12}")
# A cut label must show at least this much of cfg12 to be told apart from the model's other skeletons.
MIN_CFG_CHARS = 8
# An env12 shown in full in any skeleton label: a candidate for confirming cut ones.
_SHOWN_ENV = re.compile(r"\.skeleton\.[0-9a-f]{12}_([0-9a-f]{12})(?:_[0-9a-f]{0,12})?(?:-[0-9a-f]{7})?")


class Unsupported(ValueError):
    """A job whose cfg12 cannot be computed offline; the message says why."""


@functools.cache
def _publisher() -> Any:
    return verify._publisher()


@dataclass
class Keying:
    """A job's skeleton key as far as the job body determines it.

    ``cfg12`` is None when the job names no reasoner (then ``candidates`` maps each
    possible reasoner to its cfg12) or when it cannot be computed (``reason``).
    """

    slug: str
    cfg12: str | None
    reasoning_parser: str | None = None
    candidates: dict = field(default_factory=dict)
    reason: str | None = None


def resolve_model_path(model_name: str) -> str:
    if model_name.startswith(("/", "./", "../")):
        return model_name
    parts = model_name.split("/")
    return f"{BAKED_MODEL_ROOT}/{parts[0]}/{parts[1]}" if len(parts) == 2 else model_name


def lora_rank(peft_config: dict) -> int:
    r = normalize_lora_peft_config(peft_config, location="inference_config.peft_config")["r"]
    for rank in VLLM_LORA_RANKS:
        if r <= rank:
            return rank
    raise Unsupported(f"peft_config.r={r} is above vLLM's largest LoRA rank {VLLM_LORA_RANKS[-1]}")


def _coerce(kind: type, value: Any) -> Any:
    """What pydantic's lax mode makes of a job value for a ModelConfig field of ``kind``."""
    if value is None or isinstance(value, bool):
        return value
    if kind is int and isinstance(value, (int, float, str)):
        return int(float(value)) if isinstance(value, str) else int(value)
    if kind is float and isinstance(value, (int, float, str)):
        return float(value)
    return value


def dss_kwargs(sub_job: dict) -> dict:
    """dss's model kwargs for a sampling sub-job (``InferenceConfig`` validation, then
    ``_prepare_sampling_engine_kwargs``)."""
    ic = sub_job["inference_config"]
    for key in ("source_checkpoint_info", "resume_checkpoint_tag"):
        if sub_job.get(key):
            raise Unsupported(f"{key} loads a checkpoint, whose path is only known on the pods")
    for key in ("text_config_overrides", "model_options"):
        if ic.get(key) is not None:
            raise Unsupported(f"inference_config.{key} is not replayed offline")
    kwargs = dict(ic.get("vllm_config") or {})
    if "max_model_len" not in kwargs and "max_seq_len" in ic:
        kwargs["max_model_len"] = int(ic["max_seq_len"])
    extra = dict(kwargs.get("extra_engine_kwargs") or {})
    for key in ("structured_outputs_config", "speculative_config", "text_config_overrides"):
        if kwargs.get(key) is not None or extra.get(key) is not None:
            raise Unsupported(f"{key} is not replayed offline")
    mnbt = ic.get("max_num_batched_tokens", DSS_MAX_NUM_BATCHED_TOKENS)
    if mnbt is None:
        raise Unsupported("max_num_batched_tokens=null leaves it to vLLM's default")
    kwargs.pop("max_num_batched_tokens", None)
    extra["max_num_batched_tokens"] = int(mnbt)
    replay = ic.get("router_replay") or {}
    if replay.get("enabled"):
        kwargs["enable_return_routed_experts"] = True
        kwargs["enable_prefix_caching"] = True
        kwargs["router_replay_max_cache_bytes"] = int(replay.get("max_cache_bytes", 16 * 1024**3))
        if replay.get("cache_device") is not None:
            kwargs["router_replay_cache_device"] = replay["cache_device"]
    if ic.get("peft_config"):
        peft = normalize_lora_peft_config(ic["peft_config"], location="inference_config.peft_config")
        extra.update(enable_lora=True, max_lora_rank=lora_rank(peft), max_loras=1, max_cpu_loras=2)
        if peft.get("target_parameters"):
            extra["enable_mixed_moe_lora_format"] = True
    for key in DSS_EXTRA_KEYS:
        if key in kwargs:
            extra[key] = kwargs.pop(key)
    if extra:
        kwargs["extra_engine_kwargs"] = extra
    kwargs["semi_p"] = True
    kwargs["model"] = resolve_model_path(sub_job["model_name"])
    return kwargs


def engine_kwargs(kwargs: dict, multi_node: bool) -> dict:
    """``ModelConfig(**kwargs).to_engine_kwargs()`` and the replica pool's multi-node default."""
    unknown = set(kwargs) - set(MODEL_CONFIG_FIELDS) - set(MODEL_CONFIG_EXCLUDED) - {"model"}
    if unknown:
        raise Unsupported(f"vllm_config.{sorted(unknown)[0]} is not a ModelConfig field (the job would be refused)")
    if str((kwargs.get("extra_env") or {}).get("ARCTIC_INFERENCE_ENABLED", "0")) == "1":
        raise Unsupported("ARCTIC_INFERENCE_ENABLED=1 keeps Arctic-only engine args, which is not replayed offline")
    out = {"model": kwargs["model"]}
    for name, (kind, default) in MODEL_CONFIG_FIELDS.items():
        value = _coerce(kind, kwargs.get(name, default))
        if value is not None:
            out[name] = value
    out.update(kwargs.get("extra_engine_kwargs") or {})
    out.pop("forest_cascade_attn_configs", None)
    if multi_node:
        out.setdefault("distributed_executor_backend", "mp")
    return out


def restore_kwargs(kwargs: dict, reasoner: str | None) -> dict:
    """What ``restore_engine`` receives: the worker's pops and defaults, then the front end's."""
    out = {k: v for k, v in kwargs.items() if k not in WORKER_POPPED}
    out.setdefault("worker_extension_cls", WORKER_EXTENSION_CLS)
    out.setdefault("disable_log_stats", False)
    if reasoner and not out.get("reasoning_parser"):
        out["reasoning_parser"] = reasoner
    return out


def vllm_config(kwargs: dict, nnodes: int = 1) -> dict:
    """``semip_engine._vllm_config_from_engine_kwargs`` with vLLM's rejections, then ``nnodes``."""
    out = {k: v for k, v in kwargs.items() if k != "extra_engine_kwargs" and k not in NON_VLLM_ENGINE_KEYS}
    for key, value in (kwargs.get("extra_engine_kwargs") or {}).items():
        if key not in NON_VLLM_ENGINE_KEYS:
            out[key] = value
    if kwargs.get("gdn_prefill_backend") is not None:
        raise Unsupported("gdn_prefill_backend: whether vLLM takes it is not known offline")
    for key in VLLM_REJECTED:
        if key in out:
            if out[key] not in (None, False, 0, "", "{}"):
                raise Unsupported(f"{key}={out[key]!r} is refused by the semi_p dump")
            del out[key]
    if nnodes > 1:
        out["nnodes"] = nnodes
    return out


def device_binding(config: dict, n_gpus: int, hardware: str | None) -> list[str] | None:
    """``semip_engine._device_binding`` for a pod of ``min(n_gpus, 8)`` GPUs."""
    tp = int(config.get("tensor_parallel_size", 1) or 1)
    world = tp * int(config.get("pipeline_parallel_size", 1) or 1)
    if tp >= DEVICE_BOUND_MIN_TP:
        if n_gpus % GPUS_PER_NODE:
            raise Unsupported(f"TP={tp} on {n_gpus} GPUs binds the image to whichever GPUs the pod gets")
        if hardware not in WHOLE_NODE_DEVICES:
            raise Unsupported(f"the device set of a whole {hardware or 'unnamed'} node is not known")
        return WHOLE_NODE_DEVICES[hardware]
    replicas = min(n_gpus, GPUS_PER_NODE) // world
    if replicas <= 1:
        return None
    return [TP1_REPLICA_LAYOUT if replicas == GPUS_PER_NODE else f"{TP1_REPLICA_LAYOUT}-x{replicas}"]


def config_hash(config: dict, binding: list[str] | None = None) -> str:
    """``semip_engine._config_hash``."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    if binding is not None:
        canonical += "\0" + ",".join(sorted(binding))
    return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LEN]


def topology(sub_job: dict) -> tuple[int, int]:
    """``(n_gpus, nnodes)``: one engine spans ``nnodes`` whole-node pods when its world size
    exceeds a node (dss builds its placement group, and restore_engine adds ``nnodes``)."""
    ic = sub_job["inference_config"]
    vc = ic.get("vllm_config") or {}
    n_gpus = int(ic["n_gpus"])
    world = int(vc.get("tensor_parallel_size", 1) or 1) * int(vc.get("pipeline_parallel_size", 1) or 1)
    if n_gpus % world:
        raise Unsupported(f"n_gpus={n_gpus} is not a multiple of the world size {world}")
    if world <= GPUS_PER_NODE:
        return n_gpus, 1
    if world % GPUS_PER_NODE or n_gpus != world:
        raise Unsupported(f"a world size of {world} on {n_gpus} GPUs is not one engine over whole nodes")
    return n_gpus, world // GPUS_PER_NODE


def reasoner_of(sub_job: dict) -> tuple[bool, str | None]:
    """``(known, reasoner)``: the job's own reasoning_parser, else its chat_reasoning_parser
    (chat adds it), else not known from the body."""
    vc = sub_job["inference_config"].get("vllm_config") or {}
    for key in ("reasoning_parser", "chat_reasoning_parser"):
        if vc.get(key):
            return True, vc[key]
    return False, None


def keying(sub_job: dict, hardware: str | None) -> Keying:
    """The job's model slug and cfg12 (or its candidates, or why it is unknown)."""
    slug = _publisher().model_slug({"vllm_config": {"model": resolve_model_path(sub_job.get("model_name") or "")}})
    try:
        n_gpus, nnodes = topology(sub_job)
        kwargs = engine_kwargs(dss_kwargs(sub_job), multi_node=nnodes > 1)
        known, reasoner = reasoner_of(sub_job)
        cfgs = {}
        for r in ([reasoner] if known else CHAT_REASONERS):
            config = vllm_config(restore_kwargs(kwargs, r), nnodes)
            cfgs[r] = config_hash(config, device_binding(config, n_gpus, hardware))
    except Unsupported as exc:
        return Keying(slug, None, reason=str(exc))
    except (KeyError, TypeError, ValueError) as exc:
        return Keying(slug, None, reason=f"job body not understood: {exc!r}")
    if known:
        return Keying(slug, cfgs[reasoner], reasoner)
    return Keying(slug, None, candidates=cfgs)


def _segment_prefix(name: str) -> str:
    """The start of a label segment for a directory starting with ``name`` (before any cut)."""
    pub = _publisher()
    return re.sub(r"^[^A-Za-z0-9]+", "", pub._LABEL_DISALLOWED_RUN.sub(".", name))


@dataclass(frozen=True)
class Found:
    """Skeleton labels for one model and cfg12 under one env12 (``env_exact``) or the start
    of one, and how many nodes carry at least one of them as ``ready``."""

    env: str
    env_exact: bool
    nodes: int


@dataclass
class Lookup:
    found: list
    unreadable: int = 0


class NodeCache:
    """The model-cache nodes' ``ready`` labels, read once from ``kubectl get nodes -o json``."""

    def __init__(self, items: list[dict], env_hash: str | None = None):
        labels = [n.get("metadata", {}).get("labels") or {} for n in items]
        cache = [{k[len(LABEL_PREFIX):] for k, v in node.items() if k.startswith(LABEL_PREFIX) and v == "ready"}
                 for node in labels if any(k.startswith(LABEL_PREFIX) for k in node)]
        self.ready = cache
        self.nodes = len(cache)
        self.all = set().union(*cache) if cache else set()
        self.envs = {m.group(1) for s in self.all for m in [_SHOWN_ENV.search(s)] if m}
        if env_hash:
            self.envs.add(env_hash)

    def weights(self, slug: str) -> set[str]:
        """Weight hashes whose label for ``slug`` is short enough to show them whole."""
        start = _segment_prefix(f"{_publisher().DEFAULT_PREFIX}/{slug}/{_publisher().WEIGHT_DIR}/")
        return {s[len(start):] for s in self.all if s.startswith(start) and _HEX12.fullmatch(s[len(start):])}

    def lookup(self, slug: str, cfg12: str) -> Lookup:
        """The skeletons of ``slug`` with ``cfg12`` that nodes verified, per env12 (or its visible start)."""
        pub = _publisher()
        root = f"{pub.DEFAULT_PREFIX}/{slug}/{pub.SKELETON_DIR}/"
        exact = {pub.sanitize_label_segment(f"{root}{cfg12}_{env}_{wt}"): env
                 for env in self.envs for wt in self.weights(slug)}
        before = _segment_prefix(root)
        start = before + cfg12 + "_"
        shown_cfg = min(HASH_LEN, max(0, _CUT - len(before)))
        cut_at = start[:_CUT].rstrip(".-_") if len(start) >= _CUT else None

        def env_of(segment: str) -> str | None | bool:
            """The env12 (or its start) a label of this cfg12 names; None if it is another
            skeleton's; False if it is cut too short to tell."""
            if segment in exact:
                return exact[segment]
            if segment.startswith(start) and re.fullmatch(r"[0-9a-f]{12}_[0-9a-f]{12}", segment[len(start):]):
                return segment[len(start):len(start) + HASH_LEN]
            m = _LABEL_CUT.fullmatch(segment)
            if not m:
                return None
            shown = m.group(1)
            if cut_at is not None:
                if shown != cut_at:
                    return None
                return "" if shown_cfg >= MIN_CFG_CHARS else False
            if not shown.startswith(start):
                return None
            rest = shown[len(start):]
            return rest.split("_")[0] if re.fullmatch(r"[0-9a-f]{1,12}(_[0-9a-f]*)?", rest) else None

        per_env: dict[tuple[str, bool], int] = {}
        unreadable = 0
        for node in self.ready:
            seen = set()
            for segment in node:
                env = env_of(segment)
                if env is False:
                    unreadable = max(unreadable, 1)
                elif env is not None:
                    seen.add((env, len(env) == HASH_LEN))
            for key in seen:
                per_env[key] = per_env.get(key, 0) + 1
        if unreadable:
            unreadable = len({s for s in self.all if env_of(s) is False})
        found = sorted((Found(env, is_exact, n) for (env, is_exact), n in per_env.items()),
                       key=lambda f: (-f.nodes, f.env))
        return Lookup(found, unreadable)
