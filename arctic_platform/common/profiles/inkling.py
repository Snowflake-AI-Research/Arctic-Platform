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

"""Options supported by the Inkling text training loader."""

from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import register_profile


def register(registry: Registry) -> None:
    register_profile(
        "inkling",
        extends="generic_moe",
        supports=["peft"],
        unsupported={
            "checkpointing.targets": "only full activation checkpointing is supported",
            "tiled_mlp.token_chunk_size": "native Inkling dense and shared experts are used",
            "lm_head.fp32": "native Inkling LM head is used",
            "lm_head.token_chunk_size": "native Inkling LM head is used",
            "lm_head.cross_entropy": "native Inkling causal LM loss is used",
            "moe.sonicmoe": "routed experts use grouped GEMM or the expert loop",
        },
        registry=registry,
    )
