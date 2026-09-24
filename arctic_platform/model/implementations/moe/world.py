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
import os


class World:
    """This class stores topology information for distributed training and inference settings by parsing environment variables set by torchrun."""

    def __init__(self):
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        self._check_world()
        self.num_nodes = self.world_size // self.local_world_size

    def _check_world(self):
        assert 0 <= self.local_rank < self.local_world_size
        assert 0 <= self.rank < self.world_size
        assert self.local_world_size <= self.world_size
        assert self.world_size % self.local_world_size == 0

    @property
    def is_master(self):
        return self.rank == 0

    def __repr__(self):
        return (
            f"World(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank},"
            f" local_world_size={self.local_world_size}, num_nodes={self.num_nodes})"
        )


# Singleton instance of World
_WORLD: World | None = None


def get_world() -> World:
    """Returns the World. If not initialized, it will initialize."""
    global _WORLD
    if _WORLD is None:
        _WORLD = World()
    return _WORLD


def reset_world():
    global _WORLD
    _WORLD = None
