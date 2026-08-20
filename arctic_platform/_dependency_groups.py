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
"""Report which dependency group to install when dependencies are absent.

The base install carries config models only; each backend lives behind an extra
(see ``pyproject.toml``). Subpackages call `require_any_dep_group` so an import
under, say, ``arctic_platform.common`` names the extra to install instead of
failing with a bare ``ModuleNotFoundError`` on whichever dependency happened to
be first.

The requirement lists are read from installed package metadata, which is
generated from ``pyproject.toml``, so nothing here has to be kept in sync.
"""

from __future__ import annotations

import re
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import distribution
from importlib.metadata import metadata

_DIST = "arctic-platform"
# Strip a requirement down to its distribution name: "foo[bar]>=1.2" -> "foo[bar]".
_VERSION_SPEC = re.compile(r"[<>=!~\s(]")
_NAME_SEP = re.compile(r"[-_.]+")


@cache
def _requires_dist() -> tuple[str, ...]:
    # Absent when running from an uninstalled checkout; skip the check rather
    # than mis-report, and let the real ImportError surface.
    try:
        return tuple(metadata(_DIST).get_all("Requires-Dist") or ())
    except PackageNotFoundError:
        return ()


@cache
def _provided_by(extra: str) -> frozenset[str]:
    """Distributions that ``pip install arctic-platform[extra]`` would install."""
    marker = re.compile(rf"""extra\s*==\s*['"]{re.escape(extra)}['"]""")
    names: set[str] = set()
    for line in _requires_dist():
        spec, _, condition = line.partition(";")
        if not marker.search(condition):
            continue
        base, _, nested = _VERSION_SPEC.split(spec.strip(), 1)[0].partition("[")
        if _NAME_SEP.sub("-", base).lower() == _DIST:
            names |= _provided_by(nested.rstrip("]"))  # e.g. arctic_platform[sft]
        else:
            names.add(base)
    return frozenset(names)


@cache
def _installed(name: str) -> bool:
    try:
        distribution(name)
    except PackageNotFoundError:
        return False
    return True


def _missing(extra: str) -> list[str]:
    return sorted(name for name in _provided_by(extra) if not _installed(name))


def require_any_dep_group(*extras: str) -> None:
    """Raise unless at least one of ``extras`` is fully installed.

    Name every extra that provisions the caller, cheapest first. The shared
    training stack is carried by both ``[sft]`` and ``[rl]``, so its gates name
    both rather than leaning on ``[rl]`` continuing to include ``[sft]``.
    """
    if any(not _missing(extra) for extra in extras):
        return
    options = " or ".join(f"'arctic-platform[{extra}]'" for extra in extras)
    raise ImportError(f"missing {', '.join(_missing(extras[0]))} — pip install {options}")
