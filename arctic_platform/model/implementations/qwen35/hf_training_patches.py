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

"""HF-path Qwen3.5 training patches (LM head + unused vision freeze).

The MoE loader injects the Prime LM head inside ``deepspeed_integration``. The
dense HF path used by OPD must apply the same knobs without importing
``qwen35.models`` (that package pulls flash-attn / cute).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)


def import_inject_prime_lm_head() -> Callable[..., None]:
    """Load ``inject_prime_lm_head`` without importing ``qwen35.models`` (flash-attn cute)."""
    import importlib.util
    import sys
    import types
    from pathlib import Path

    name = "arctic_platform.model.implementations.qwen35.models.layers.lm_head"
    cached = sys.modules.get(name)
    if cached is not None and hasattr(cached, "inject_prime_lm_head"):
        return cached.inject_prime_lm_head
    qwen35 = Path(__file__).resolve().parent
    models_pkg = "arctic_platform.model.implementations.qwen35.models"
    layers_pkg = f"{models_pkg}.layers"
    if models_pkg not in sys.modules:
        models = types.ModuleType(models_pkg)
        models.__path__ = [str(qwen35 / "models")]
        models.__package__ = models_pkg
        sys.modules[models_pkg] = models
    if layers_pkg not in sys.modules:
        layers = types.ModuleType(layers_pkg)
        layers.__path__ = [str(qwen35 / "models" / "layers")]
        layers.__package__ = layers_pkg
        sys.modules[layers_pkg] = layers
    spec = importlib.util.spec_from_file_location(name, qwen35 / "models" / "layers" / "lm_head.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {qwen35}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.inject_prime_lm_head


def apply_lm_head_from_config(model: nn.Module, cfg: Mapping[str, Any]) -> None:
    """Replace the HF ``lm_head`` using ``fp32_lm_head`` / chunk / fused-CE knobs."""
    chunk = cfg.get("fused_lm_head_token_chunk_size")
    chunk_size = chunk if isinstance(chunk, int) else None
    import_inject_prime_lm_head()(
        model,
        chunk_size=chunk_size,
        fused_cross_entropy=cfg.get("fused_cross_entropy", False),
        fp32_lm_head=bool(cfg.get("fp32_lm_head", False)),
    )


def apply_qwen35_training_patches(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    rank: int = 0,
) -> int:
    """Apply LM-head injection (if requested) and freeze an unused vision tower.

    Reads ``fp32_lm_head``, ``fused_lm_head_token_chunk_size``, ``fused_cross_entropy``,
    and ``freeze_unused_vision`` (default True). Returns the number of vision
    parameters frozen.
    """
    want_fp32 = bool(config.get("fp32_lm_head", False))
    chunk = config.get("fused_lm_head_token_chunk_size")
    chunk_size = chunk if isinstance(chunk, int) else None
    if want_fp32 or chunk_size is not None:
        apply_lm_head_from_config(model, config)
        logger.info(
            "rank=%d injected lm_head fp32=%s chunk_size=%s fused_cross_entropy=%s",
            rank,
            want_fp32,
            chunk_size,
            config.get("fused_cross_entropy", False),
        )

    if not config.get("freeze_unused_vision", True):
        return 0
    from arctic_platform.model.implementations.qwen35.vlm import freeze_unused_vision_tower

    frozen = freeze_unused_vision_tower(model, rank)
    if frozen:
        logger.info(
            "rank=%d froze %d vision-tower params (text-only job; unused params "
            "produce no grads and stall ZeRO-3 reduction)",
            rank,
            frozen,
        )
    return frozen
