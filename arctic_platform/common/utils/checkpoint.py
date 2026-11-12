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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint directory helpers (no Ray / DeepSpeed imports)."""

from __future__ import annotations

import contextlib
import os
import re
import shutil
from collections.abc import Iterator


def resolve_checkpoint_save_paths(root: str, step: int | None) -> tuple[str, str]:
    """Return ``(save_dir, prune_root)`` for a training checkpoint save.

    ``prune_root`` is the job checkpoint directory (the parent of ``checkpoint-*``
    children). It is always ``root``, including when ``step`` is omitted.
    """
    if step is not None:
        save_dir = os.path.join(root, f"checkpoint-{int(step)}")
    else:
        save_dir = root
    return save_dir, root


def has_ep_sharded_parameters(module) -> bool:
    from arctic_platform.model.weight_export import _has_ep_sharded_parameters

    return _has_ep_sharded_parameters(module)


@contextlib.contextmanager
def omit_frozen_parameters(module) -> Iterator[None]:
    """Leave frozen parameters out of ``module.state_dict``, which every DeepSpeed save path reads.

    DeepSpeed's ``exclude_frozen_parameters`` matches parameter names against state-dict keys, which drop
    activation-checkpoint wrapper segments (``_checkpoint_wrapped_module.``), so it keeps wrapped frozen weights.
    """
    from torch import nn

    frozen = {
        key
        for key, value in module.state_dict(keep_vars=True).items()
        if isinstance(value, nn.Parameter) and not value.requires_grad
    }
    original = module.state_dict

    def state_dict(*args, **kwargs):
        state = original(*args, **kwargs)
        prefix = kwargs.get("prefix", "")
        for key in frozen:
            state.pop(prefix + key, None)
        return state

    module.state_dict = state_dict
    try:
        yield
    finally:
        del module.state_dict


def load_module_state_skipping_ep_shards(src: dict, dst) -> None:
    """``custom_load_fn`` for ``engine.load_checkpoint`` on expert-parallel models.

    DeepSpeed writes module state from rank 0 only, so its EP-sharded tensors are rank 0's experts and must not be
    loaded elsewhere. Frozen parameters may be absent (saved with ``exclude_frozen_parameters``); every other key must
    load.
    """
    from arctic_platform.model.weight_export import _is_ep_sharded_parameter
    from arctic_platform.model.weight_export import strip_activation_checkpoint_segments as strip

    parameters = {strip(name): p for name, p in dst.named_parameters(remove_duplicate=False)}
    skipped = {name for name, parameter in parameters.items() if _is_ep_sharded_parameter(parameter)}
    frozen = {name for name, parameter in parameters.items() if not parameter.requires_grad}
    result = dst.load_state_dict({k: v for k, v in src.items() if strip(k) not in skipped}, strict=False)
    missing = [k for k in result.missing_keys if strip(k) not in skipped and strip(k) not in frozen]
    if missing or result.unexpected_keys:
        raise RuntimeError(f"Checkpoint module state mismatch: missing={missing} unexpected={result.unexpected_keys}")


def restore_trainable_ep_shards(engine) -> None:
    """After ``load_module_state_skipping_ep_shards``, set this rank's trainable EP-sharded parameters (e.g. expert
    LoRA) from its restored optimizer state.

    ZeRO-1/2 restores every rank's own fp32 partitions, expert groups included, but copies them into the module's
    parameters only on universal-checkpoint loads.
    """
    from arctic_platform.model.weight_export import _is_ep_sharded_parameter

    if not any(p.requires_grad and _is_ep_sharded_parameter(p) for p in engine.module.parameters()):
        return
    update_lp_params = getattr(engine.optimizer, "update_lp_params", None)
    if update_lp_params is None:
        raise NotImplementedError(
            f"Resuming trainable expert-parallel parameters is not supported with {type(engine.optimizer).__name__}"
        )
    update_lp_params()


def prune_checkpoint_dirs(parent_dir: str, keep: int) -> int:
    """Keep the newest ``keep`` ``checkpoint-*`` dirs under ``parent_dir``; remove older.

    Returns the number of directories removed. No-op when ``keep <= 0``.
    """
    if keep is None or int(keep) <= 0:
        return 0
    keep = int(keep)
    if not os.path.isdir(parent_dir):
        return 0
    pat = re.compile(r"^checkpoint-(\d+)$")
    found = []
    for name in os.listdir(parent_dir):
        m = pat.match(name)
        if m:
            found.append((int(m.group(1)), os.path.join(parent_dir, name)))
    found.sort(key=lambda x: x[0])
    to_remove = found[:-keep] if len(found) > keep else []
    removed = 0
    for _, path in to_remove:
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    return removed
