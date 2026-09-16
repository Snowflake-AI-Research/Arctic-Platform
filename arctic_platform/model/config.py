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
"""Declarative, JSON-serializable spec that describes a model to build."""

from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from typing_extensions import Self


class ParallelismConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_default=True)

    expert_parallel: int = Field(1, description="Expert-parallel degree.")
    sequence_parallel: int = Field(1, description="Ulysses sequence-parallel degree.")


class ZorroTrainPatch(BaseModel):
    """ZoRRo Train forward-patch knobs. Non-None enables the patch."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    response_len: int | None = Field(None, description="Response length per rollout.")
    max_token_len: int | None = Field(None, description="Per-GPU train token budget.")
    rollout_n: int | None = Field(None, description="GRPO rollout group size.")
    temperature: float | None = Field(None, description="Sampling temperature for logprob/entropy.")
    use_unpad: bool = Field(True, description="Use unpadded (packed) sequences.")
    world_size: int | None = Field(None, description="Distributed world size (injected by worker).")
    logits_optimization: str = Field("none", description='One of "none" | "memory" | "compute".')
    logits_optimization_peak_mem_size_in_gib: int = Field(4, description="Peak mem budget (GiB) for memory/compute.")
    logits_compute_from_fp32_inputs: bool = Field(False, description="Logits from fp32 hiddens.")
    logits_compute_in_fp32: bool = Field(False, description="Compute logits in fp32.")


class Patches(BaseModel):
    """Optional features applied to the model after it is loaded."""

    model_config = ConfigDict(extra="forbid", validate_default=True)

    liger: bool = Field(False, description="Apply Liger kernels.")
    zorro_train: ZorroTrainPatch | None = Field(None, description="ZoRRo Train patch (None disables).")
    gradient_checkpointing: bool = Field(False, description="HF gradient checkpointing.")


class ModelSpec(BaseModel):
    """Everything needed to build a model, fully JSON-serializable.

    Non-serializable runtime handles (process groups, device mesh) are NOT in the
    spec; they are passed separately to ``build_model(spec, parallel_groups=...)``.
    """

    model_config = ConfigDict(extra="forbid", validate_default=True)

    model_path_or_name: str = Field(..., description="HF model path or hub name.")
    dtype: str = Field("bfloat16", description="Parameter dtype.")
    attn_implementation: str | None = Field(None, description="Attention implementation to request from HF.")
    loader: str | None = Field(None, description="Loader name; auto-resolved at construction when not set.")
    parallelism: ParallelismConfig = Field(
        default_factory=ParallelismConfig, description="Loader-specific parallelism."
    )
    patches: Patches = Field(default_factory=Patches, description="Post-load patches.")
    loader_options: dict = Field(default_factory=dict, description="JSON-only loader-specific extras.")

    @classmethod
    def from_ds_worker_config(cls, model_name: str, ds_worker_config: dict) -> "ModelSpec":
        """Transitional bridge: map flat verl ``ds_worker_config`` into a ``ModelSpec``.

        Callers (adapter / SFT demos) own the knobs on the flat dict today. Longer-term,
        ``ModelSpec`` should be constructed upstream as part of the main client config
        instead of being inferred here.
        """
        cfg = ds_worker_config or {}

        # Require an explicit attention backend. Packed varlen / ZoRRo Train need
        # some flash-attention implementation (FA2 / FA3 / FA4); do not invent a
        # default here (ModelSpec leaves it None).
        if "attn_implementation" not in cfg or cfg["attn_implementation"] is None:
            raise ValueError(
                "from_ds_worker_config requires attn_implementation "
                "(flash_attention_2 / flash_attention_3 / flash_attention_4)."
            )

        zorro_train_patch = None
        if cfg.get("zorro_train_enable", False):
            # Only forward keys present in cfg; ZorroTrainPatch pydantic defaults fill the rest.
            zorro_keys = (
                "response_len",
                "max_token_len",
                "rollout_n",
                "temperature",
                "use_unpad",
                "world_size",
                "logits_optimization",
                "logits_optimization_peak_mem_size_in_gib",
                "logits_compute_from_fp32_inputs",
                "logits_compute_in_fp32",
            )
            zorro_train_patch = ZorroTrainPatch(**{k: cfg[k] for k in zorro_keys if k in cfg})

        # Worker bridge defaults GC on (historical DeepSpeedWorker behavior). Generic
        # ``Patches.gradient_checkpointing`` stays False for direct ModelSpec users.
        return cls(
            model_path_or_name=model_name,
            dtype="bfloat16",
            attn_implementation=cfg["attn_implementation"],
            patches=Patches(
                liger=cfg.get("use_liger", False),
                zorro_train=zorro_train_patch,
                gradient_checkpointing=cfg.get("enable_gradient_checkpointing", True),
            ),
        )

    @field_validator("dtype")
    @classmethod
    def _validate_dtype(cls, value: str) -> str:
        import torch

        assert value == "auto" or isinstance(getattr(torch, value, None), torch.dtype), f"unknown dtype {value!r}"
        return value

    @model_validator(mode="after")
    def _resolve_loader(self) -> Self:
        from arctic_platform.model.loader import get_loader_options_model
        from arctic_platform.model.loader import is_registered_loader
        from arctic_platform.model.loader import resolve_loader_name

        if self.loader is None:
            self.loader = resolve_loader_name(self)
        else:
            assert is_registered_loader(self.loader), f"unknown loader {self.loader!r}"

        # Validate loader_options against the resolved loader's schema (if it has one),
        # storing the fully-defaulted dict back so the spec records the effective values.
        options_model = get_loader_options_model(self.loader)
        if options_model is not None:
            self.loader_options = options_model.model_validate(self.loader_options).model_dump()
        return self
