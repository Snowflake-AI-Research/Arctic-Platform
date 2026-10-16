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

"""Parse DeepSpeed hostfiles for correctness placement and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostfileEntry:
    host: str
    slots: int


def parse_hostfile(path: str | Path) -> list[HostfileEntry]:
    entries: list[HostfileEntry] = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        host = fields[0]
        slots = 1
        for field in fields[1:]:
            if field.startswith("slots="):
                slots = int(field.partition("=")[2])
        if slots < 1:
            raise ValueError(f"{path}:{number}: slots must be positive, got {slots}")
        entries.append(HostfileEntry(host=host, slots=slots))
    return entries
