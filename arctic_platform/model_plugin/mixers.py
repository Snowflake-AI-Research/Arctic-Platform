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

"""Mixers: per-layer-type objects that tell the platform how to run a sequence-mixing layer it has never seen."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Protocol
from typing import runtime_checkable

from torch import nn

from arctic_platform.model_plugin.attention import AttentionModule
from arctic_platform.model_plugin.parallel import head_parallel

if TYPE_CHECKING:
    from arctic_platform.model_plugin.plugin import ParallelismInfo


@runtime_checkable
class Mixer(Protocol):
    """One layer type's integration with the platform.

    ``is_instance`` claims the modules of this layer type in a built model, ``validate_sp`` rejects a
    sequence-parallel degree the layer cannot run at, and ``apply_sequence_parallel`` makes a claimed module run
    with its sequence sharded across ``group``.
    """

    layer_type: str

    def is_instance(self, module: nn.Module) -> bool: ...

    def validate_sp(self, info: ParallelismInfo, sp_size: int) -> None: ...

    def apply_sequence_parallel(self, module: nn.Module, group) -> None: ...


class HeadParallelMixer:
    """A mixer for modules that expose ``mix(*tensors, cu_seqlens, shard)`` over head-major tensors.

    Sequence parallelism exchanges sequence for heads (:func:`head_parallel`), so every head count the core
    exchanges must divide by the sequence-parallel degree.
    """

    module_class: type[nn.Module] = nn.Module

    def __init__(self, layer_type: str, module_class: type[nn.Module] | None = None):
        self.layer_type = layer_type
        if module_class is not None:
            self.module_class = module_class

    def is_instance(self, module: nn.Module) -> bool:
        return isinstance(module, self.module_class)

    def validate_sp(self, info: ParallelismInfo, sp_size: int) -> None:
        for heads in info.heads.get(self.layer_type, ()):
            if heads % sp_size != 0:
                raise ValueError(
                    f"layer type {self.layer_type!r} has {heads} heads, not divisible by sp_size {sp_size}"
                )

    def apply_sequence_parallel(self, module: nn.Module, group) -> None:
        head_parallel(module, group)


class AttentionMixer(HeadParallelMixer):
    """The platform mixer behind ``layer_types.<name> = "platform:attention"``."""

    module_class = AttentionModule
