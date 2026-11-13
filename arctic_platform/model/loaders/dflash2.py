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
"""Loader for online DFlash2 draft training: a frozen target plus a trainable DFlash2 draft.

``model_path_or_name`` is the *target* checkpoint. The draft architecture comes from
``loader_options.draft_config`` (the content of a DFlash2 draft ``config.json``). A target that is a LoRA
fine-tune arrives as its base checkpoint with the PEFT adapter pack in its ``default/`` folder; the loader finds the
pack there and merges it in first.
"""

from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator
from transformers import AutoModelForCausalLM
from typing_extensions import Self

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.loader import LoadedModel
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.loader import register_loader


class DFlash2Options(BaseModel):
    """Validated ``loader_options`` for the dflash2 loader."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    draft_config: dict = Field(
        ..., description="Content of the DFlash2 draft config.json (architectures + dflash_config)."
    )
    init_from: str | None = Field(
        None, description="Existing draft directory (config.json + model.safetensors) to start from."
    )
    attention_backend: Literal["sdpa", "flex_attention", "eager"] = Field(
        "sdpa", description="Draft attention implementation (SpecForge's training.attention_backend)."
    )
    num_anchors: int = Field(512, gt=0, description="Anchor blocks sampled per sequence.")
    loss_decay_gamma: float | None = Field(
        7.0, gt=0, description="Per-slot loss decay exp(-(k-1)/gamma); null disables."
    )
    selector_loss_alpha: float = Field(
        1.0,
        ge=0,
        allow_inf_nan=False,
        description="Default weight of the selector CE (the client may override per step).",
    )
    selector_stop_gradient: bool = Field(
        False, description="Detach the selector's inputs so only its own parameters train on its CE."
    )
    lk_loss_type: Literal["alpha", "lambda", "tv"] | None = Field(
        None,
        description=(
            "Acceptance-aware token objective in place of CE: 'alpha' equals CE on hard labels, 'tv' is "
            "1 - p(label), 'lambda' mixes CE and TV with the CE weight kl_scale * exp(-kl_decay * acceptance)."
        ),
    )
    kl_scale: float = Field(1.0, allow_inf_nan=False, description="Scale of the 'lambda' LK objective's CE weight.")
    kl_decay: float = Field(
        1.0, allow_inf_nan=False, description="Decay of the 'lambda' LK objective's CE weight with the acceptance."
    )
    objective_chunk_blocks: int = Field(
        128,
        ge=0,
        description="Anchor blocks per vocabulary-projection chunk (bounds memory); 0 projects all blocks at once.",
    )
    teacher_metrics: bool = Field(
        True,
        description=(
            "Report the teacher-agreement diagnostics (dflash/teacher/*), which read the target's final hidden "
            "state; they never enter the loss."
        ),
    )
    seed: int = Field(0, description="Seed for draft initialization and anchor sampling.")

    @model_validator(mode="after")
    def _check_draft_config(self) -> Self:
        from arctic_platform.model.implementations.dflash2.config import build_draft_config

        build_draft_config(self.draft_config)  # raises ValueError on a malformed or serving-incompatible draft
        return self


def _validate_spec(spec: ModelSpec) -> None:
    if spec.dtype not in ("bfloat16", "float32", "float16"):
        raise ValueError("dflash2 dtype must be bfloat16, float16 or float32")
    unsupported = (
        spec.patches.liger,
        spec.patches.peft,
        spec.patches.gradient_checkpointing,
        spec.patches.activation_offload,
        spec.patches.compile,
        spec.patches.tiled_mlp,
        spec.patches.lm_head,
        spec.patches.zorro_train,
    )
    if any(unsupported):
        raise ValueError("the dflash2 loader does not support generic model patches; use loader_options instead")
    if spec.parallelism.sequence_parallel != 1 or spec.parallelism.expert_parallel != 1:
        raise ValueError("the dflash2 loader supports data parallelism only (sequence/expert parallel size must be 1)")


def merge_target_adapter(target: torch.nn.Module, adapter_dir: str) -> torch.nn.Module:
    """``target`` with the LoRA adapter at ``adapter_dir`` merged into its weights, as a plain model again."""
    from peft import PeftModel

    return PeftModel.from_pretrained(target, adapter_dir, is_trainable=False).merge_and_unload()


@register_loader("dflash2", options=DFlash2Options, validate_spec=_validate_spec)
def load_dflash2(ctx: LoaderContext) -> LoadedModel:
    from arctic_platform.model.implementations.dflash2.online import build_online_model
    from arctic_platform.model.weight_export import checkpoint_peft_adapter_dir

    options = DFlash2Options.model_validate(ctx.spec.loader_options)
    dtype = getattr(torch, ctx.spec.dtype)
    target = AutoModelForCausalLM.from_pretrained(
        ctx.spec.model_path_or_name,
        dtype=dtype,
        attn_implementation=ctx.spec.attn_implementation,
    )
    # A LoRA fine-tune's checkpoint is its base weights plus the adapter pack; merging it before the target is frozen
    # makes the draft read the fine-tune's hidden states, not the base's.
    adapter_dir = checkpoint_peft_adapter_dir(ctx.spec.model_path_or_name)
    if adapter_dir is not None:
        target = merge_target_adapter(target, adapter_dir)
    model = build_online_model(
        target,
        options.draft_config,
        dtype=dtype,
        attention_backend=options.attention_backend,
        num_anchors=options.num_anchors,
        loss_decay_gamma=options.loss_decay_gamma,
        objective_chunk_blocks=options.objective_chunk_blocks,
        lk_loss_type=options.lk_loss_type,
        kl_scale=options.kl_scale,
        kl_decay=options.kl_decay,
        selector_loss_alpha=options.selector_loss_alpha,
        selector_stop_gradient=options.selector_stop_gradient,
        teacher_metrics=options.teacher_metrics,
        seed=options.seed,
        init_from=options.init_from,
    )
    return LoadedModel(model=model)
