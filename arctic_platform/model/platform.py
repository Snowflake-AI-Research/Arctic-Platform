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

"""Runtime capabilities used to resolve model-loader defaults."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Literal

AcceleratorFamily = Literal["cpu", "ampere", "hopper", "blackwell", "unknown"]


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


@dataclass(frozen=True)
class PlatformCapabilities:
    accelerator: AcceleratorFamily
    compute_capability: tuple[int, int] | None
    attention_backends: frozenset[str]
    ep_comm_backends: frozenset[str]

    @classmethod
    def detect(cls) -> "PlatformCapabilities":
        import torch

        compute_capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else None
        if compute_capability is None:
            accelerator: AcceleratorFamily = "cpu"
        elif compute_capability[0] >= 10:
            accelerator = "blackwell"
        elif compute_capability[0] == 9:
            accelerator = "hopper"
        elif compute_capability[0] == 8:
            accelerator = "ampere"
        else:
            accelerator = "unknown"

        attention_backends = {"sdpa"}
        if _module_available("flash_attn"):
            attention_backends.add("flash_attention_2")
        if _module_available("flash_attn_interface"):
            attention_backends.add("flash_attention_3")
        if _module_available("flash_attn.cute"):
            attention_backends.add("flash_attention_4")

        ep_comm_backends = set()
        if _module_available("deep_ep"):
            ep_comm_backends.add("deepep")
        if _module_available("uccl.ep"):
            ep_comm_backends.add("uccl")

        return cls(
            accelerator=accelerator,
            compute_capability=compute_capability,
            attention_backends=frozenset(attention_backends),
            ep_comm_backends=frozenset(ep_comm_backends),
        )

    @classmethod
    def for_accelerator(
        cls,
        accelerator: AcceleratorFamily,
        *,
        attention_backends: frozenset[str] | None = None,
        ep_comm_backends: frozenset[str] = frozenset({"deepep", "uccl"}),
    ) -> "PlatformCapabilities":
        compute_capability = {
            "ampere": (8, 0),
            "hopper": (9, 0),
            "blackwell": (10, 0),
        }.get(accelerator)
        if attention_backends is None:
            default_attention = {
                "ampere": "sdpa",
                "hopper": "flash_attention_3",
                "blackwell": "flash_attention_4",
            }.get(accelerator, "sdpa")
            attention_backends = frozenset({default_attention})
        return cls(
            accelerator=accelerator,
            compute_capability=compute_capability,
            attention_backends=attention_backends,
            ep_comm_backends=ep_comm_backends,
        )
