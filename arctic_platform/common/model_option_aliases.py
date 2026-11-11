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

"""Validate a job config's ``model_options`` map and rewrite it onto the ``ModelSettings`` fields.

``model_options`` names each model option once, by its key in Arctic Platform's option registry
(``arctic_platform.common.option_registry``). The map nests by key component, so
``{"lm_head": {"fp32": true}}`` sets ``lm_head.fp32``. Validation checks every key against the registry,
checks that it belongs in this job config, checks the value's type, and then writes the value onto the legacy
field or fields that the training and sampling code already read (``ALIASES``). The loaders read only those
fields, never ``model_options`` itself, so a config that does not send it validates exactly as before.

A legacy field and the ``model_options`` key that sets it may both be given only with the same value.

A server reads the map itself only at start-up: ``check_profile`` checks the requested keys against the training
loader's option profile before any weights load, and ``describe_resolved`` gives the line each job logs.

The module also holds the data of the deprecated ``prime_rl`` block (``PRIME_RL_BLOCK_KEYS`` and
``PRIME_RL_BLOCK_OTHER``, the replacement of each entry), ``record_prime_rl_request_entries``, which records the
entries a request's own block carried, and ``prime_rl_block_warning``, the warning a job that sent it logs.

Only the registry is imported; it has no torch or model imports. Cortex (DSS) and the on-prem server both use it.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Optional

from pydantic import TypeAdapter
from pydantic import ValidationError

from arctic_platform.common import option_registry
from arctic_platform.common.option_registry import Registry

TRAINING = "training_config"
INFERENCE = "inference_config"

# Where each sub-job config accepts a key, by the registry's ``runs_in``.
_RUNS_IN_BY_LOCATION = {TRAINING: ("trainer", "both"), INFERENCE: ("sampler", "both")}
_LOCATION_BY_SIDE = {"trainer": TRAINING, "sampler": INFERENCE}

# Registered keys that are deliberately not accepted, with what to do instead. ``{location}`` is the sub-job
# config the key was sent in.
NOT_ACCEPTED: dict[str, str] = {
    "peft": "LoRA is a training mode, not a model option; set {location}.peft_config instead",
    "zorro_train": (
        "ZoRRo Train is a training mode, not a model option; on-prem it is ds_worker_config.zorro_train_enable, "
        "and Cortex has no setting for it"
    ),
    "numerics.reduce_dtype": (
        "gradient reduction always runs in fp32 (DeepSpeed communication_data_type), so it is not a job setting"
    ),
}

# The names ``TrainingConfig`` accepts for ``activation_checkpointing``, in the order its alias
# normalization picks them. ``TrainingConfig`` reads this tuple for both.
ACTIVATION_CHECKPOINTING_NAMES = (
    "activation_checkpointing",
    "activation_checkpointing_freq",
    "gradient_checkpointing_freq",
    "gradient_checkpointing",
)
# Legacy fields that a client may send under another name, with every name in order of precedence.
_FIELD_NAMES = {"activation_checkpointing": ACTIVATION_CHECKPOINTING_NAMES}

_MISSING = object()

Writes = list[tuple[str, Any]]
# ``(value, config, options)``: the key's value, the sub-job config the client sent, and all of its options by key.
Writer = Callable[[Any, Mapping[str, Any], Mapping[str, Any]], Writes]


class ModelOptionsRegistryMismatch(ValueError):
    """A key the registry has that the alias table cannot handle yet: the two disagree.

    Its own type, so the zone's create-error classifier reports it by its type rather than guess from the text,
    which names the key.
    """


class _Conflict(Exception):
    """A writer found a legacy field whose value rules out the requested one."""

    def __init__(self, legacy_path: str, legacy_value: Any):
        super().__init__(legacy_path)
        self.legacy_path = legacy_path
        self.legacy_value = legacy_value


@dataclass(frozen=True)
class Alias:
    """How one accepted key is checked and which legacy fields it sets in each sub-job config.

    ``check`` raises ``ValueError`` for a value of the wrong type. A writer returns ``(dotted path, value)``
    pairs relative to the sub-job config; the config it receives is the one the client sent, and the options are
    the sub-job's whole ``model_options`` map by key.
    """

    check: Callable[[str, Any], None]
    training: Writer | None = None
    inference: Writer | None = None

    def writer(self, location: str) -> Writer | None:
        return self.training if location == TRAINING else self.inference


# Value checks.


def _boolean(path: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be true or false, got {value!r}")


def _string(path: str, value: Any) -> None:
    if not isinstance(value, str) or len(value) == 0:
        raise ValueError(f"{path} must be a non-empty string, got {value!r}")


def _string_list(path: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{path} must be a list of strings, got {value!r}")


def _is_positive_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_integer(path: str, value: Any) -> None:
    if not _is_positive_integer(value):
        raise ValueError(f"{path} must be a positive integer, got {value!r}")


# The dtypes the custom MoE loaders accept as the optimization dtype (their ModelSpec.dtype check).
_OPTIMIZATION_DTYPES = ("bfloat16", "float32")


def _optimization_dtype(path: str, value: Any) -> None:
    if value not in _OPTIMIZATION_DTYPES:
        raise ValueError(f"{path} must be one of {', '.join(map(repr, _OPTIMIZATION_DTYPES))}, got {value!r}")


def _boolean_or_string(path: str, value: Any) -> None:
    if not isinstance(value, (bool, str)) or value == "":
        raise ValueError(f"{path} must be true, false or a backend name such as 'liger', got {value!r}")


def _tiled_mlp_chunk_size(path: str, value: Any) -> None:
    """The rule of ``tiled_mlp_token_chunk_size``, with its message, named after the ``model_options`` path."""
    if not _is_positive_integer(value):
        raise ValueError(f"{path}: tiled_mlp_token_chunk_size must be a positive integer, got {value!r}")


# Writers.


# The loader every job off the MoE route builds with (``dense_model_spec``).
DENSE_LOADER = "huggingface"

# ``training_config.model_provider`` of a DFlash2 draft-training job. It names the Arctic Platform loader it
# builds with as well: the frozen target is ``model_name`` and the draft is configured by ``training_config.dflash2``.
DFLASH2_PROVIDER = "dflash2"


def moe_route(ep_size: int) -> bool:
    """Whether a training job with this expert-parallel degree takes the MoE route: ``load_ap_moe`` and the
    ``prime_rl`` block.

    Every other job loads through ``huggingface``. On the MoE route a model type that no MoE loader matches falls
    back to ``huggingface`` as well.
    """
    return ep_size > 1


# ``TrainingConfig.ep_size``'s own type, so the rewrite reads a raw value exactly as the field will.
_EP_SIZE: TypeAdapter[Optional[int]] = TypeAdapter(Optional[int])


def _moe_route(config: Mapping[str, Any]) -> bool:
    """``moe_route`` for a raw sub-job config. A value ``ep_size`` cannot take counts as not MoE; the field's
    own validation reports it."""
    try:
        ep_size = _EP_SIZE.validate_python(config.get("ep_size"))
    except ValidationError:
        return False
    return ep_size is not None and moe_route(ep_size)


def _field(field: str) -> Writer:
    """One legacy field, by its dotted path in the sub-job config."""
    return lambda value, config, options: [(field, value)]


def _field_and_moe(field: str, prime_rl_field: str) -> Writer:
    """A field the dense route reads at the top level and the MoE route reads from ``prime_rl``."""

    def write(value, config, options):
        writes = [(field, value)]
        if _moe_route(config):
            writes.append((f"prime_rl.{prime_rl_field}", value))
        return writes

    return write


def _ac_config(field: str) -> Writer:
    """An ``ac_config`` field, also written into ``prime_rl.ac_config`` when that block replaces ``ac_config``."""

    def write(value, config, options):
        writes = [(f"ac_config.{field}", value)]
        prime_rl = config.get("prime_rl")
        if isinstance(prime_rl, Mapping) and isinstance(prime_rl.get("ac_config"), Mapping):
            writes.append((f"prime_rl.ac_config.{field}", value))
        return writes

    return write


def _checkpointing_offload(value, config, options):
    # ``false`` asks for no change, so it writes nothing when the rewrite leaves no ac_config block: on the MoE
    # route any ac_config block turns activation checkpointing on.
    if value is False and not _has_ac_config(config, options):
        return []
    return _ac_config("offload_config.enabled")(value, config, options)


# The checkpointing keys that write into ac_config whatever their value.
_AC_CONFIG_KEYS = ("checkpointing.mode", "checkpointing.freq", "checkpointing.targets")


def _has_ac_config(config: Mapping[str, Any], options: Mapping[str, Any]) -> bool:
    """Whether the rewrite leaves an ac_config block: the client sent one, or another checkpointing key writes one.

    Counting the other keys makes the rewrite a fixed point: validating its output again writes the same.
    """
    prime_rl = config.get("prime_rl")
    nested = prime_rl.get("ac_config") if isinstance(prime_rl, Mapping) else None
    if config.get("ac_config") is not None or nested is not None:
        return True
    return any(key in options for key in _AC_CONFIG_KEYS)


def _moe_only(prime_rl_field: str) -> Writer:
    """A loader option only the MoE loaders read, from ``prime_rl``; the dense route has no field for it."""

    def write(value, config, options):
        if _moe_route(config):
            return [(f"prime_rl.{prime_rl_field}", value)]
        return []

    return write


def _dense_field_or_moe(field: str, prime_rl_field: str) -> Writer:
    """A loader option the dense route reads from a top-level field and the MoE route from ``prime_rl``."""

    def write(value, config, options):
        if _moe_route(config):
            return [(f"prime_rl.{prime_rl_field}", value)]
        return [(field, value)]

    return write


def _checkpointing_freq(value, config, options):
    # The dense route reads the frequency from ``activation_checkpointing``, the MoE route from ``ac_config``.
    return [*_ac_config("freq")(value, config, options), ("activation_checkpointing", value)]


def _attention_backend(value, config, options):
    # ``attn_impl`` and ``prime_rl.attn`` take precedence over ``attn_implementation`` when present, so they
    # must agree with it. Every route reads ``prime_rl.attn`` first, then ``attn_impl``, then
    # ``attn_implementation`` (``resolve_attn_implementation``); ``attn_impl`` comes first here so that, without
    # ``prime_rl.attn``, it is the config report's target (see ``_normalization``).
    writes = []
    if _get(config, "attn_impl") is not _MISSING:
        writes.append(("attn_impl", value))
    writes.append(("attn_implementation", value))
    prime_rl = config.get("prime_rl")
    if isinstance(prime_rl, Mapping) and prime_rl.get("attn") is not None:
        writes.append(("prime_rl.attn", value))
    return writes


def _liger(value, config, options):
    # The dense loader reads ``use_liger``. ``model_provider`` stays ``huggingface``.
    use_liger = _get(config, "use_liger")
    if value:
        return [("use_liger", True)]
    if use_liger is True:
        raise _Conflict("use_liger", use_liger)
    return []


ALIASES: dict[str, Alias] = {
    "checkpointing.mode": Alias(_string, training=_ac_config("mode")),
    "checkpointing.freq": Alias(_positive_integer, training=_checkpointing_freq),
    "checkpointing.targets": Alias(_string_list, training=_ac_config("targets")),
    "checkpointing.offload": Alias(_boolean, training=_checkpointing_offload),
    "tiled_mlp.token_chunk_size": Alias(_tiled_mlp_chunk_size, training=_field("tiled_mlp_token_chunk_size")),
    "lm_head.fp32": Alias(
        _boolean,
        training=_field_and_moe("fp32_lm_head", "fp32_lm_head"),
        inference=_field("vllm_config.fp32_lm_head"),
    ),
    "lm_head.token_chunk_size": Alias(
        _positive_integer,
        training=_field_and_moe("fused_lm_head_token_chunk_size", "fused_lm_head_token_chunk_size"),
    ),
    "lm_head.vocab_chunk_size": Alias(_positive_integer, training=_field("fused_lm_head_vocab_chunk_size")),
    "lm_head.cross_entropy": Alias(_boolean_or_string, training=_field("prime_rl.fused_cross_entropy")),
    "liger": Alias(_boolean, training=_liger),
    "moe.grouped_mm": Alias(_boolean, training=_field("prime_rl.moe_use_grouped_mm")),
    "moe.sonicmoe": Alias(_boolean, training=_field("prime_rl.moe_use_sonicmoe")),
    "moe.comm_backend": Alias(_string, training=_field("prime_rl.ep_comm_backend")),
    "moe.comm_sms": Alias(_positive_integer, training=_field("prime_rl.deepep_num_sms")),
    "moe.comm_token_chunk": Alias(_positive_integer, training=_field("prime_rl.deepep_token_chunk_size")),
    "attention.backend": Alias(_string, training=_attention_backend),
    "attention.sparse_mla": Alias(_string, training=_field("prime_rl.sparse_mla_backend")),
    "compile.fullgraph": Alias(_boolean, training=_field("compile.fullgraph")),
    # The Qwen3.5-MoE loader reads it from prime_rl on the MoE route; on the dense route dense_model_spec passes the
    # top-level field to the huggingface loader as loader_options.fp32_parity.
    "numerics.fp32_parity": Alias(_boolean, training=_dense_field_or_moe("fp32_parity", "fp32_parity")),
    # Experimental, and only the huggingface loader has it: dense_model_spec passes the top-level field on as
    # loader_options.fp32_residual. The MoE loaders' profiles reject the key when the sub-job starts.
    "numerics.fp32_residual": Alias(_boolean, training=_field("fp32_residual")),
    # Only the glm5_next and qwen4_exp loaders have it, from prime_rl; the other loaders' profiles reject the key
    # when the sub-job starts.
    "numerics.fp32_routed_weighting": Alias(_boolean, training=_moe_only("fp32_routed_weighting")),
    # The MoE loaders take it from prime_rl; on the dense route the job-level dtype decides instead.
    "numerics.optimization_dtype": Alias(_optimization_dtype, training=_moe_only("optimization_dtype")),
}


# The deprecated ``training_config.prime_rl`` block: each entry it can carry, and what replaces it. An entry in
# ``PRIME_RL_BLOCK_KEYS`` is replaced by those ``model_options`` keys. One in ``PRIME_RL_BLOCK_OTHER`` has no key: it
# is set by the server, internal, the only spelling of its setting (``trust_remote_code``), or, for ``reduce_dtype``
# and ``tiled_mlp_token_chunk_size``, rejected at validation, so the warning never names those two.
# Before the block itself can be rejected, two settings need another spelling: ``trust_remote_code`` on every route
# and, with ep_size 1, ``fused_cross_entropy``, whose block entries are their only spelling today.
PRIME_RL_BLOCK_KEYS: dict[str, tuple[str, ...]] = {
    "ac_config": ("checkpointing.mode", "checkpointing.freq", "checkpointing.targets", "checkpointing.offload"),
    "ep_comm_backend": ("moe.comm_backend",),
    "deepep_num_sms": ("moe.comm_sms",),
    "deepep_token_chunk_size": ("moe.comm_token_chunk",),
    "moe_use_grouped_mm": ("moe.grouped_mm",),
    "moe_use_sonicmoe": ("moe.sonicmoe",),
    "fused_cross_entropy": ("lm_head.cross_entropy",),
    "fused_lm_head_token_chunk_size": ("lm_head.token_chunk_size",),
    "fp32_lm_head": ("lm_head.fp32",),
    "fp32_parity": ("numerics.fp32_parity",),
    "fp32_routed_weighting": ("numerics.fp32_routed_weighting",),
    "sparse_mla_backend": ("attention.sparse_mla",),
    "attn": ("attention.backend",),
    "optimization_dtype": ("numerics.optimization_dtype",),
}
PRIME_RL_BLOCK_OTHER: dict[str, str] = {
    "trust_remote_code": (
        "no key: this entry stays the only documented spelling, which the expert-parallel loaders read; an "
        "external-checkpoint import always loads with false"
    ),
    "attention_query_chunk_size": (
        "no key: this entry stays the only documented spelling, which the expert-parallel loaders read"
    ),
    "weight_conversion_cache_dir": (
        "set by the server: an external-checkpoint import always uses its per-job directory; for other "
        "expert-parallel training a value sent here is used, and a future server may ignore or reject it; on the "
        "dense route it has no effect"
    ),
    "reduce_dtype": "rejected: gradient reduction always runs in fp32",
    "tiled_mlp_token_chunk_size": "rejected here: use tiled_mlp.token_chunk_size",
    "seq_len": "set by the server from max_seq_len for expert-parallel training; no effect on the dense route",
    "debug": "internal test settings, which stay in the block",
}
# The ``ac_config`` fields no ``checkpointing.*`` key replaces, and the warning's text for them.
AC_CONFIG_FIELDS_WITHOUT_KEYS = (
    "layers",
    "router_replay_recompute",
    "combine_replay",
    "offload_config.keep_last_n",
    "offload_config.use_streams",
    "offload_config.tensor_size_threshold",
    "offload_config.pin_memory_*",
)
AC_CONFIG_WITHOUT_KEYS = (
    "layers, router_replay_recompute, combine_replay and offload_config.keep_last_n, .use_streams, "
    ".tensor_size_threshold and .pin_memory_* have no key and stay under the top-level ac_config"
)
# The carved MoE loaders' own ``seq_len`` default (``seq_len`` of their options models in Arctic Platform). The
# warning names it when a block without ``seq_len`` gives the loaders ``max_seq_len`` instead.
LOADER_DEFAULT_SEQ_LEN = 4096
# The internal field of ``TrainingConfig`` that records the block entries the request itself carried.
PRIME_RL_REQUEST_ENTRIES = "prime_rl_request_entries"


def record_prime_rl_request_entries(data: Any) -> Any:
    """``data`` with the ``prime_rl`` entries it carries recorded, when it also sends ``model_options``.

    The ``model_options`` rewrite writes block entries of its own, so after it the block no longer tells what the
    request sent. Recorded once: validating a dumped config again keeps the record. Without ``model_options`` the
    block is the request's own and nothing is recorded.
    """
    if not isinstance(data, dict) or data.get("model_options") is None or PRIME_RL_REQUEST_ENTRIES in data:
        return data
    block = data.get("prime_rl")
    return {**data, PRIME_RL_REQUEST_ENTRIES: list(block) if isinstance(block, Mapping) else []}


def prime_rl_block_warning(training_config: Mapping[str, Any]) -> str | None:
    """The deprecation warning for a training config whose request sent a non-empty ``prime_rl`` block, or ``None``.

    Names only the entries the request sent, not those the ``model_options`` rewrite wrote.
    """
    block = training_config.get("prime_rl")
    if not isinstance(block, Mapping):
        return None
    sent = training_config.get(PRIME_RL_REQUEST_ENTRIES)
    entries = [entry for entry in (block if sent is None else sent) if entry in block]
    if not entries:
        return None
    moe = _moe_route(training_config)
    max_seq_len = training_config.get("max_seq_len")
    parts = []
    for entry in entries:
        if not moe and entry in PRIME_RL_BLOCK_KEYS and entry != "attn":
            part = _dense_entry_part(entry, training_config.get("model_provider"))
        elif entry in PRIME_RL_BLOCK_KEYS:
            part = f"{entry} -> {', '.join(PRIME_RL_BLOCK_KEYS[entry])}"
            if entry == "ac_config":
                part += f" ({AC_CONFIG_WITHOUT_KEYS})"
        else:
            part = f"{entry} -> {PRIME_RL_BLOCK_OTHER.get(entry, 'not a known entry')}"
        if entry == "seq_len" and moe and block[entry] != max_seq_len:
            part += f" (this block's seq_len {block[entry]!r} differs from max_seq_len {max_seq_len!r} and is used)"
        parts.append(part)
    message = (
        f"{TRAINING}.prime_rl is deprecated; set {TRAINING}.model_options instead. The server keeps reading the "
        f"block as it did before. Replacements for its entries: {'; '.join(parts)}"
    )
    if moe and "seq_len" not in entries and max_seq_len != LOADER_DEFAULT_SEQ_LEN:
        message += (
            f". This block has no seq_len, so the MoE loaders get max_seq_len {max_seq_len!r} instead of their "
            f"default {LOADER_DEFAULT_SEQ_LEN}"
        )
    return message


def _dense_entry_part(entry: str, model_provider: Any) -> str:
    """The warning's text for a block entry with a replacement key on a job with ``ep_size`` 1.

    That job loads a plain model through the dense loader, which reads none of the MoE loaders' entries; the text says
    what the entry does there and which of its keys the dense loader's profile rejects.
    """
    keys = PRIME_RL_BLOCK_KEYS[entry]
    cells = option_registry.settle(DENSE_LOADER)
    rejected = [key for key in keys if cells[key].status != option_registry.SUPPORTS]
    taken = [key for key in keys if key not in rejected]
    if entry == "fused_cross_entropy":
        if model_provider == "arctic":
            effect = (
                "with ep_size 1 the model has no fused head; this entry, the only spelling there, only tells the"
                " server whether to refuse requests a fused head could not serve"
            )
        else:
            effect = "with ep_size 1 it has no effect"
    else:
        effect = "with ep_size 1 it has no effect"
    part = f"{entry}: {effect}"
    if rejected:
        part += f"; {', '.join(rejected)} {'is' if len(rejected) == 1 else 'are'} rejected there"
    if taken:
        part += f"; {', '.join(taken)} {'sets' if len(taken) == 1 else 'set'} the dense route's own fields"
    return part


def accepted_keys(location: str, registry: Registry | None = None) -> tuple[str, ...]:
    """The registered keys ``location``'s ``model_options`` accepts, sorted."""
    runs_in = _RUNS_IN_BY_LOCATION[location]
    return tuple(
        entry.key
        for entry in option_registry.options(registry)
        if entry.runs_in in runs_in and entry.key not in NOT_ACCEPTED
    )


def alias_table_problems(registry: Registry | None = None) -> list[str]:
    """Every mismatch between ``ALIASES`` / ``NOT_ACCEPTED`` and the registry. Empty means they agree."""
    entries = {entry.key: entry for entry in option_registry.options(registry)}
    problems = []
    for key in sorted(set(ALIASES) | set(NOT_ACCEPTED)):
        if key not in entries:
            problems.append(f"{key}: listed in the alias table but not registered")
    for key in sorted(set(ALIASES) & set(NOT_ACCEPTED)):
        problems.append(f"{key}: both mapped and listed as not accepted")
    for key, entry in sorted(entries.items()):
        if key in NOT_ACCEPTED:
            continue
        alias = ALIASES.get(key)
        if alias is None:
            problems.append(f"{key}: registered but neither mapped in ALIASES nor listed in NOT_ACCEPTED")
            continue
        for location, runs_in in _RUNS_IN_BY_LOCATION.items():
            accepted_here = entry.runs_in in runs_in
            if accepted_here and alias.writer(location) is None:
                problems.append(f"{key}: runs in {entry.runs_in} but has no {location} target")
            if not accepted_here and alias.writer(location) is not None:
                problems.append(f"{key}: runs in {entry.runs_in} but has a {location} target")
    return problems


def flatten(options: Any, *, location: str, registry: Registry | None = None) -> dict[str, Any]:
    """The map as ``{dotted key: value}``. Raises ``ValueError`` for anything that is not a registered key.

    Nested objects descend until the path is a registered key; that key's value is taken as is. A dotted name
    inside the map (``{"lm_head.fp32": true}``) names the same key as the nested form.
    """
    root = f"{location}.model_options"
    if not isinstance(options, Mapping):
        raise ValueError(f"{root} must be an object mapping option keys to values, got {options!r}")
    registered = {entry.key: entry for entry in option_registry.options(registry)}
    flat: dict[str, Any] = {}
    pending = [(None, options)]
    while pending:
        prefix, mapping = pending.pop(0)
        for name, value in mapping.items():
            key = name if prefix is None else f"{prefix}.{name}"
            if key in registered:
                if key in flat:
                    raise ValueError(f"{root}.{key} is set twice")
                flat[key] = value
                continue
            children = sorted(k for k in registered if k.startswith(f"{key}."))
            if isinstance(value, Mapping) and len(children) > 0:
                pending.append((key, value))
            elif len(children) > 0:
                accepted = [child for child in children if child in accepted_keys(location, registry)]
                remedy = (
                    f"set one of {', '.join(accepted)}"
                    if len(accepted) > 0
                    else f"none of its options is accepted in {root}"
                )
                raise ValueError(
                    f"{root}.{key} groups several options and cannot be set to {value!r} as a whole; {remedy}"
                )
            else:
                raise ValueError(f"{root}.{key} is not a known model option; {_suggestion(key, location, registry)}")
    return flat


def _suggestion(key: str, location: str, registry: Registry | None) -> str:
    candidates = accepted_keys(location, registry)
    close = difflib.get_close_matches(key, candidates, n=3, cutoff=0.5)
    if len(close) > 0:
        return f"did you mean {', '.join(close)}?"
    return f"{location}.model_options accepts {', '.join(candidates)}"


def _check_placement(key: str, *, location: str, registry: Registry | None) -> None:
    path = f"{location}.model_options.{key}"
    if key in NOT_ACCEPTED:
        raise ValueError(f"{path} is not accepted in model_options: {NOT_ACCEPTED[key].format(location=location)}")
    runs_in = next(entry.runs_in for entry in option_registry.options(registry) if entry.key == key)
    if runs_in not in _RUNS_IN_BY_LOCATION[location]:
        side = "trainer" if runs_in == "trainer" else "sampler"
        raise ValueError(
            f"{path} runs only in the {side}; set it in {_LOCATION_BY_SIDE[runs_in]}.model_options instead"
        )


def _given_name(config: Mapping[str, Any], dotted: str) -> str:
    """The name the client used for ``dotted``: the first of its names in ``_FIELD_NAMES`` that it sent."""
    for name in _FIELD_NAMES.get(dotted, ()):
        if config.get(name) is not None:
            return name
    return dotted


def _get(config: Mapping[str, Any], dotted: str) -> Any:
    """The value at ``dotted`` in ``config``, or ``_MISSING`` when it is absent or ``None``."""
    dotted = _given_name(config, dotted)
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or node.get(part) is None:
            return _MISSING
        node = node[part]
    return node


def _set(config: dict[str, Any], dotted: str, value: Any, *, location: str) -> None:
    """Set ``dotted`` in ``config``, copying every nested object on the way so the input is never changed."""
    parts = dotted.split(".")
    node = config
    for depth, part in enumerate(parts[:-1]):
        child = node.get(part)
        if child is None:
            child = {}
        elif not isinstance(child, Mapping):
            parent = ".".join(parts[: depth + 1])
            raise ValueError(f"{location}.{parent} must be an object to take {location}.{dotted}, got {child!r}")
        node[part] = dict(child)
        node = node[part]
    node[parts[-1]] = value


def _conflict_message(path: str, value: Any, location: str, legacy_path: str, legacy_value: Any) -> str:
    return (
        f"{path} is {value!r} but {location}.{legacy_path} is {legacy_value!r}; "
        "set the option once, or give both the same value"
    )


def apply(data: Any, *, location: str, registry: Registry | None = None, provenance: Any = None) -> Any:
    """Validate ``data["model_options"]`` and return a copy of ``data`` with each option on its legacy fields.

    ``data`` is the raw sub-job config a ``mode="before"`` validator receives. It is returned unchanged when it
    is not a dict or carries no ``model_options``. With a ``provenance`` (the config report's
    ``ConfigProvenance``), each key that writes a legacy field is recorded in ``provenance.normalized``, naming the
    field the route reads (see ``_normalization``).
    """
    if not isinstance(data, dict) or data.get("model_options") is None:
        return data
    flat = flatten(data["model_options"], location=location, registry=registry)
    writes: list[tuple[str, Any]] = []
    for key, value in flat.items():
        path = f"{location}.model_options.{key}"
        _check_placement(key, location=location, registry=registry)
        alias = ALIASES.get(key)
        if alias is None:
            raise ModelOptionsRegistryMismatch(f"{path} is registered but the server does not accept it yet")
        alias.check(path, value)
        writer = alias.writer(location)
        if writer is None:
            raise ModelOptionsRegistryMismatch(
                f"{path} is registered but the server does not accept it yet in {location}"
            )
        try:
            key_writes = writer(value, data, flat)
        except _Conflict as conflict:
            raise ValueError(
                _conflict_message(path, value, location, conflict.legacy_path, conflict.legacy_value)
            ) from None
        for legacy_path, legacy_value in key_writes:
            current = _get(data, legacy_path)
            if current is not _MISSING and current != legacy_value:
                legacy_name = _given_name(data, legacy_path)
                raise ValueError(_conflict_message(path, value, location, legacy_name, current))
        writes.extend(key_writes)
        if provenance is not None and key_writes:
            provenance.normalized.append(_normalization(path, value, location, key_writes, _moe_route(data)))
    if provenance is not None:
        provenance.normalized.extend(_renames(data, writes, location))
    rewritten = dict(data)
    for legacy_path, legacy_value in writes:
        _set(rewritten, legacy_path, legacy_value, location=location)
    if location == TRAINING:
        _check_fullgraph_with_tiled_mlp(flat, rewritten)
        _check_chunked_lm_head(flat, rewritten)
    return rewritten


def _renames(data: Mapping[str, Any], writes: Writes, location: str) -> list[dict[str, Any]]:
    """The report entries for a field the client sent under another name and a key writes by its own name.

    Writing the canonical name hides the client's alias from ``TrainingConfig``'s own alias normalization,
    which would otherwise record the rename, so it is recorded here in the same shape.
    """
    entries = []
    for legacy_path in dict.fromkeys(path for path, _ in writes):
        given = _given_name(data, legacy_path)
        if given != legacy_path:
            entries.append(
                {
                    "field": f"{location}.{given}",
                    "submitted_value": data[given],
                    "effective_field": f"{location}.{legacy_path}",
                    "reason": f"'{given}' is another name for '{legacy_path}'",
                }
            )
    return entries


def _check_fullgraph_with_tiled_mlp(flat: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Name the ``model_options`` keys in the fullgraph and tiled-MLP conflict when the keys set either side.

    ``TrainingConfig`` refuses the pair by its field names; this runs first, so a request that used the keys
    hears about the keys.
    """
    if "compile.fullgraph" not in flat and "tiled_mlp.token_chunk_size" not in flat:
        return
    compile_config = config.get("compile")
    fullgraph = compile_config.get("fullgraph", False) if isinstance(compile_config, Mapping) else False
    # An invalid chunk size is the field's own error, which its validator reports; only a valid one conflicts.
    if not _is_positive_integer(config.get("tiled_mlp_token_chunk_size")) or not fullgraph:
        return
    root = f"{TRAINING}.model_options"
    compile_path = f"{root}.compile.fullgraph" if "compile.fullgraph" in flat else f"{TRAINING}.compile.fullgraph"
    tiled_path = (
        f"{root}.tiled_mlp.token_chunk_size"
        if "tiled_mlp.token_chunk_size" in flat
        else f"{TRAINING}.tiled_mlp_token_chunk_size"
    )
    raise ValueError(f"{compile_path} cannot be combined with {tiled_path}")


def _normalization(path: str, value: Any, location: str, writes: Writes, moe: bool) -> dict[str, Any]:
    """The config report's entry for a key that wrote ``writes``, in the shape of the other normalized fields.

    It names the field the route reads: ``prime_rl.attn`` when the key writes it, which every route reads first
    (``resolve_attn_implementation``); on the MoE route the ``prime_rl`` entry when the key writes one; otherwise a
    top-level field when it writes one (the dense loader reads those), otherwise its first field.
    """
    paths = [legacy_path for legacy_path, _ in writes]
    if "prime_rl.attn" in paths:
        preferred = ["prime_rl.attn"]
    elif moe:
        preferred = [path for path in paths if path.startswith("prime_rl.")]
    else:
        preferred = [path for path in paths if "." not in path]
    target = (preferred or paths)[0]
    return {
        "field": path,
        "submitted_value": value,
        "effective_field": f"{location}.{target}",
        "reason": "set through model_options",
    }


# The MoE loaders' fused cross-entropy when the prime_rl block names none (Arctic Platform's
# ``effective_fused_cross_entropy``). Kept here so validation imports no model code; a test pins the two together.
MOE_DEFAULT_FUSED_CROSS_ENTROPY = "liger"


def _check_chunked_lm_head(flat: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """On the MoE route, a chunked LM head needs fused cross-entropy off.

    Both select the LM head the MoE loaders install, so the worker refuses the pair when it loads the model. When
    either key of the pair is sent, this refuses it at creation instead, with the worker's rule: the chunk size and
    the fused cross-entropy the loaders read from the ``prime_rl`` block, whichever spelling set them.
    """
    if not ("lm_head.token_chunk_size" in flat or "lm_head.cross_entropy" in flat) or not _moe_route(config):
        return
    prime_rl = config.get("prime_rl")
    block = prime_rl if isinstance(prime_rl, Mapping) else {}
    chunk_size = block.get("fused_lm_head_token_chunk_size")
    fused_cross_entropy = block.get("fused_cross_entropy", MOE_DEFAULT_FUSED_CROSS_ENTROPY)
    # As the worker rule does: any int, so a boolean chunk size counts too.
    if not isinstance(chunk_size, int) or not fused_cross_entropy:
        return
    if "lm_head.token_chunk_size" in flat:
        chunk = f"{TRAINING}.model_options.lm_head.token_chunk_size"
    else:
        chunk = f"{TRAINING}.prime_rl.fused_lm_head_token_chunk_size"
    if "lm_head.cross_entropy" not in flat and "fused_cross_entropy" in block:
        raise ValueError(
            f"{chunk} needs fused cross-entropy off when ep_size is above 1, but the block entry "
            f"{TRAINING}.prime_rl.fused_cross_entropy is {fused_cross_entropy!r}: set it to false, or remove it and "
            f"set {TRAINING}.model_options.lm_head.cross_entropy to false"
        )
    raise ValueError(
        f"{chunk} needs {TRAINING}.model_options.lm_head.cross_entropy set to false when ep_size is above 1: "
        f"the chunked LM head and fused cross-entropy ({fused_cross_entropy!r}) cannot both be installed"
    )


def requested(config: Mapping[str, Any], *, location: str, registry: Registry | None = None) -> dict[str, Any]:
    """The validated sub-job config's ``model_options`` as ``{dotted key: value}``; empty when none was sent."""
    options = config.get("model_options")
    if options is None:
        return {}
    return flatten(options, location=location, registry=registry)


class ModelOptionsUnsupported(ValueError):
    """A requested model option that the training job's loader cannot take.

    Its own type, so the zone's create-error classifier reports it as a configuration error by its type rather
    than guess from the text, which names the rejected keys.
    """


# Values that ask the loader for nothing, so the profile check skips the key. ``compile.fullgraph: false`` is not
# here: the presence of ``compile`` turns per-layer compile on.
_NOT_A_REQUEST: dict[str, Any] = {"liger": False}


# Keys a profile supports that a job with this model_provider still cannot send: ``liger`` applies kernels on a
# huggingface model, which conflicts with "arctic" at creation. Left out of the message's list of supported keys.
_KEYS_NOT_FOR_PROVIDER: dict[str, tuple[str, ...]] = {"arctic": ("liger",)}


def check_profile(
    options: Mapping[str, Any],
    profile: str,
    *,
    location: str,
    registry: Registry | None = None,
    ep_size: int = 1,
    model_provider: str | None = None,
    expert_parallel_loader: str | None = None,
) -> None:
    """Fail unless the ``profile`` loader's option profile supports every requested key in ``options``.

    ``options`` maps dotted keys to values. A key counts as requested whatever its value, except a value in
    ``_NOT_A_REQUEST``. The message gives the profile's reason for each key it does not support, and the
    ``model_options`` keys the profile supports that a job with this ``model_provider`` can send. When the loader
    is ``DENSE_LOADER`` it also says why, from the job's ``ep_size``, and, with ``ep_size`` 1, what ep_size above
    1 would give: ``expert_parallel_loader`` is the loader the model would use then, or ``None`` when no
    expert-parallel loader takes its model type. Every registered profile settles every key, which Arctic
    Platform's coverage check enforces; a profile Arctic Platform does not know, or a key it does not settle,
    fails with ``ModelOptionsRegistryMismatch``.
    """
    try:
        cells = option_registry.settle(profile, registry)
    except ValueError as error:
        raise ModelOptionsRegistryMismatch(
            f"{location}.model_options cannot be checked: the {profile} loader has no usable option profile in this "
            f"Arctic Platform ({error})"
        ) from None
    problems = []
    for key, value in sorted(options.items()):
        if key in _NOT_A_REQUEST and value is _NOT_A_REQUEST[key]:
            continue
        cell = cells.get(key)
        if cell is None:
            raise ModelOptionsRegistryMismatch(
                f"{location}.model_options.{key} is registered but the {profile} loader's option profile does not "
                "settle it, so the server does not accept it yet"
            )
        if cell.status != option_registry.SUPPORTS:
            status = cell.status.replace("_", " ")
            problems.append(f"{location}.model_options.{key} is {status} with the {profile} loader: {cell.reason}")
    if len(problems) > 0:
        excluded = _KEYS_NOT_FOR_PROVIDER.get(model_provider or "", ())
        supported = [
            key
            for key in accepted_keys(location, registry)
            if key in cells and cells[key].status == option_registry.SUPPORTS and key not in excluded
        ]
        message = f"{'; '.join(problems)}. The {profile} loader supports: {', '.join(supported)}"
        if profile == DENSE_LOADER:
            message += _dense_loader_hint(
                options,
                location,
                ep_size=ep_size,
                model_provider=model_provider,
                expert_parallel_loader=expert_parallel_loader,
            )
        raise ModelOptionsUnsupported(message)


def _dense_loader_hint(
    options: Mapping[str, Any],
    location: str,
    *,
    ep_size: int,
    model_provider: str | None,
    expert_parallel_loader: str | None,
) -> str:
    """Why the job loads through ``DENSE_LOADER``, and what ep_size above 1 would give it."""
    if moe_route(ep_size):
        return (
            f". This job has ep_size {ep_size}, but no expert-parallel loader takes its model type, so it falls back "
            f"to the {DENSE_LOADER} loader, which cannot run with expert parallelism: this model cannot train with "
            "ep_size above 1"
        )
    hint = f". This job has ep_size 1 or omitted, so it loads through the {DENSE_LOADER} loader; "
    if expert_parallel_loader is None:
        hint += "no expert-parallel loader takes this model type, so it cannot train with ep_size above 1"
    else:
        hint += (
            'with ep_size above 1 and model_provider "arctic" this model loads through the '
            f"{expert_parallel_loader} loader"
        )
    if "lm_head.cross_entropy" in options and model_provider == "arctic":
        hint += (
            f"; with ep_size 1 the block entry {location}.prime_rl.fused_cross_entropy is the only spelling of that "
            "setting"
        )
    return hint


def _value_at(config: Mapping[str, Any], dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        node = node[part]
    return node


def describe_resolved(config: Mapping[str, Any], *, location: str) -> str | None:
    """One line naming each requested option, by key, and the legacy field values it resolved to, or ``None``.

    ``config`` is the validated sub-job config, in which the options have already been written onto their
    fields. The rewrite is a fixed point, so a writer run on it again names only fields that hold a value. The
    line is only a log, so it never raises: a key whose fields cannot be read back is named with its value
    alone, and a map that cannot be read at all gives a line that says so.
    """
    try:
        options = requested(config, location=location)
    except Exception:  # noqa: BLE001 - a log line must not stop the job
        return f"{location}.model_options could not be described"
    parts = []
    for key, value in sorted(options.items()):
        try:
            writer = ALIASES[key].writer(location)
            assert writer is not None
            writes = writer(value, config, options)
            fields = ", ".join(f"{path}={_value_at(config, path)!r}" for path, _ in writes) or "no field"
        except Exception:  # noqa: BLE001 - a log line must not stop the job
            fields = "fields not resolved"
        parts.append(f"{key}={value!r} -> {fields}")
    if len(parts) == 0:
        return None
    return f"{location}.model_options resolved: {'; '.join(parts)}"
