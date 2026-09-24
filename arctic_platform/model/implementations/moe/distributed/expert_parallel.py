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
from typing import Optional

import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.distributed.tensor import DeviceMesh
from torch.distributed.tensor import Shard
from torch.distributed.tensor import distribute_module
from torch.distributed.tensor import distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle

# When set (by DeepSpeed integration), `get_ep_group` returns the DeepSpeed
# expert-parallel group registered under this name instead of the per-module
# group cached by `DeepEPExpertParallel`. Lets DeepSpeed own EP/DP topology
# while Prime-RL's MoE forward keeps using DeepEP dispatch/combine.
_deepspeed_ep_group_name: Optional[str] = None


def set_deepspeed_ep_group_name(group_name: Optional[str]) -> None:
    global _deepspeed_ep_group_name
    _deepspeed_ep_group_name = group_name


class DeepEPExpertParallel(ParallelStyle):
    """Expert-parallel style backed by DeepEP dispatch/combine.

    Only handles weight sharding (Shard(0) on expert dim) and stores the EP
    process group on the module. PrimeRL drives DeepEP dispatch/combine from
    `MoE.forward()` so communication stays outside the selective-AC checkpoint
    boundary while local expert matmuls remain checkpointable.
    """

    @staticmethod
    def _partition_fn(name: str, mod: nn.Module, device_mesh: DeviceMesh) -> None:
        for param_name, param in mod.named_parameters(recurse=False):
            mod.register_parameter(param_name, nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])))
        mod._ep_group = device_mesh.get_group()

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(module, device_mesh, partition_fn=self._partition_fn)


def get_ep_group(experts: nn.Module) -> ProcessGroup:
    if _deepspeed_ep_group_name is not None:
        import deepspeed.utils.groups as ds_groups

        return ds_groups._get_expert_parallel_group(_deepspeed_ep_group_name)
    return experts._ep_group
