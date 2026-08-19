# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Modifications copyright (c) 2025 Prime Intellect, Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from functools import cached_property

from torch._utils import _get_available_device_type
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .logging_utils import get_logger

device_type = _get_available_device_type() or "cuda"

__all__ = ["ParallelDims"]


@dataclass
class ParallelDims:
    dp_replicate: int
    dp_shard: int
    cp: int
    pp: int
    ep: int
    world_size: int

    _world_mesh: DeviceMesh = None
    _submeshes: dict = None

    def __post_init__(self):
        self._submeshes = {}
        self._validate()

    def _validate(self):
        dp_replicate, dp_shard, cp, pp, ep = (
            self.dp_replicate,
            self.dp_shard,
            self.cp,
            self.pp,
            self.ep,
        )
        for d in (dp_replicate, cp, pp, ep):
            assert d >= 1, "Parallelism degree should be >= 1, except for dp_shard"

        assert dp_shard == -1 or dp_shard >= 1, " dp_shard must -1 or >=1."
        if dp_shard < 0:
            self.dp_shard = dp_shard = self.world_size // (dp_replicate * cp * pp)
        assert dp_shard >= 1

        assert dp_replicate * dp_shard * cp * pp == self.world_size, (
            f"Invalid parallel dims: dp_replicate({dp_replicate}) * dp_shard({dp_shard}) * "
            f"cp({cp}) * pp({pp}) != WORLD_SIZE({self.world_size})"
        )

        if ep > 1:
            # EP would borrow all cp and some dp_shard degree
            assert ep % cp == 0 and (dp_shard * cp) % ep == 0

    def build_mesh(self) -> DeviceMesh:
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_with_ep(self) -> DeviceMesh:
        # With ep, dp_shard and ep are derived submeshes:
        # dp_shard = dp_shard_mod_ep * dp_shard_in_ep
        # ep = dp_shard_in_ep * cp
        dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
        dp_shard_in_ep = self.ep // self.cp

        dims = []
        names = []
        for d, name in zip(
            [
                self.pp,
                self.dp_replicate,
                dp_shard_mod_ep,
                dp_shard_in_ep,
                self.cp,
            ],
            ["pp", "dp_replicate", "dp_shard_mod_ep", "dp_shard_in_ep", "cp"],
        ):
            if d > 1 or name == "dp_shard_mod_ep":
                dims.append(d)
                names.append(name)

        self.logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        dp_mesh_dim_names = []
        dp_shard_cp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []
        ep_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        dp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_shard_cp_mesh_dim_names.append("dp_shard_mod_ep")
        dp_cp_mesh_dim_names.append("dp_shard_mod_ep")
        if "dp_shard_in_ep" in names:
            dp_mesh_dim_names.append("dp_shard_in_ep")
            dp_shard_cp_mesh_dim_names.append("dp_shard_in_ep")
            dp_cp_mesh_dim_names.append("dp_shard_in_ep")
            ep_mesh_dim_names.append("dp_shard_in_ep")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")
            ep_mesh_dim_names.append("cp")

        self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        self._submeshes["dp_shard_cp"] = mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_shard_cp")
        self._submeshes["dp_cp"] = mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")
        self._submeshes["ep"] = mesh[tuple(ep_mesh_dim_names)]._flatten(mesh_dim_name="ep")

        if self.dp_replicate_enabled:
            parent = mesh[tuple(["dp_replicate"] + dp_shard_cp_mesh_dim_names)]
            hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
            self._submeshes["hsdp"] = DeviceMesh(
                device_type, hsdp_tensor, mesh_dim_names=("dp_replicate", "dp_shard_cp")
            )
        else:
            self._submeshes["hsdp"] = self._submeshes["dp_shard_cp"]

        return mesh

    def _build_mesh_without_ep(self) -> DeviceMesh:
        dims = []
        names = []
        for d, name in zip(
            [self.pp, self.dp_replicate, self.dp_shard, self.cp],
            ["pp", "dp_replicate", "dp_shard", "cp"],
        ):
            if d > 1 or name == "dp_shard":
                dims.append(d)
                names.append(name)

        self.logger.info(f"Building {len(dims)}-D device mesh with {names}, {dims}")
        mesh = init_device_mesh(device_type, dims, mesh_dim_names=names)

        dp_mesh_dim_names = []
        dp_shard_cp_mesh_dim_names = []
        dp_cp_mesh_dim_names = []

        if self.dp_replicate_enabled:
            dp_mesh_dim_names.append("dp_replicate")
            dp_cp_mesh_dim_names.append("dp_replicate")
        dp_mesh_dim_names.append("dp_shard")
        dp_shard_cp_mesh_dim_names.append("dp_shard")
        dp_cp_mesh_dim_names.append("dp_shard")
        if self.cp_enabled:
            dp_shard_cp_mesh_dim_names.append("cp")
            dp_cp_mesh_dim_names.append("cp")

        if dp_mesh_dim_names != []:
            self._submeshes["dp"] = mesh[tuple(dp_mesh_dim_names)]._flatten(mesh_dim_name="dp")
        if dp_shard_cp_mesh_dim_names != []:
            self._submeshes["dp_shard_cp"] = mesh[tuple(dp_shard_cp_mesh_dim_names)]._flatten(
                mesh_dim_name="dp_shard_cp"
            )
        if dp_cp_mesh_dim_names != []:
            self._submeshes["dp_cp"] = mesh[tuple(dp_cp_mesh_dim_names)]._flatten(mesh_dim_name="dp_cp")

        if self.dp_replicate_enabled:
            parent = mesh[tuple(["dp_replicate"] + dp_shard_cp_mesh_dim_names)]
            hsdp_tensor = parent.mesh.reshape(self.dp_replicate, -1)
            self._submeshes["hsdp"] = DeviceMesh(
                device_type, hsdp_tensor, mesh_dim_names=("dp_replicate", "dp_shard_cp")
            )
        else:
            self._submeshes["hsdp"] = self._submeshes["dp_shard_cp"]

        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    def get_mesh(self, name: str) -> DeviceMesh:
        mesh = self.world_mesh  # ensure lazy init has run
        if name in self._submeshes:
            return self._submeshes[name]
        return mesh[name]

    @property
    def dp_enabled(self):
        return self.dp_replicate > 1 or self.dp_shard > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard > 1

    @property
    def cp_enabled(self):
        return self.cp > 1

    @property
    def dp_cp_enabled(self):
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self):
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def pp_enabled(self):
        return self.pp > 1

    @property
    def ep_enabled(self):
        return self.ep > 1

    @cached_property
    def fsdp_gradient_divide_factor(self) -> int:
        return self.dp_replicate * self.dp_shard * self.cp

    @cached_property
    def non_data_parallel_size(self):
        return self.cp * self.pp

    @cached_property
    def seq_len_divisor(self):
        return self.cp * 2

    @cached_property
    def logger(self):
        return get_logger()
