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


"""Canonical tensor names shared by the PEFT adapter checks."""


def canonical_name(name: str) -> str:
    """Strip only activation-checkpoint segments and the terminal default adapter."""
    if not isinstance(name, str) or not name or any(character.isspace() for character in name):
        raise ValueError("LoRA tensor name must be a non-empty dotted name without whitespace")
    parts = [part for part in name.split(".") if part != "_checkpoint_wrapped_module"]
    if any(not part for part in parts):
        raise ValueError(f"invalid LoRA tensor name: {name!r}")
    if len(parts) >= 4 and parts[-3] in {"lora_A", "lora_B"} and parts[-2] == "default":
        del parts[-2]
    if len(parts) < 3 or parts[-2] not in {"lora_A", "lora_B"} or parts[-1] != "weight":
        raise ValueError(f"unsupported LoRA tensor name: {name!r}; only default A/B weights are supported")
    if any(part.startswith("lora_") for part in parts[:-2]):
        raise ValueError(f"ambiguous LoRA tensor name: {name!r}")
    return ".".join(parts)
