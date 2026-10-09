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

"""Registry of configurable model options and of the model profiles that settle them.

An option entry says what an option is: its key, where it runs, its goal and why it exists. A model profile
says, for every option, whether a model supports it, does not support it, or cannot have it (not applicable),
with a one-line reason for the last two. ``check_coverage`` fails unless every profile settles every option,
so a new option and a new profile both fail until the table is complete.

Nothing reads this registry at run time. It is pure Python with no torch, transformers, vLLM or DeepSpeed
import, so it can be imported without the training extras.

Command line::

    python -m arctic_platform.common.option_registry list
    python -m arctic_platform.common.option_registry check [--matrix-path PATH]
    python -m arctic_platform.common.option_registry matrix [--write] [--matrix-path PATH]
"""

from __future__ import annotations

import argparse
import fnmatch
import importlib
import re
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from types import MappingProxyType
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence

RUNS_IN = ("trainer", "sampler", "both")
GOALS = ("parity", "memory", "speed", "robustness", "mode", "debug")

Status = Literal["supports", "unsupported", "not_applicable"]
SUPPORTS: Status = "supports"
UNSUPPORTED: Status = "unsupported"
NOT_APPLICABLE: Status = "not_applicable"

# A lowercase dotted path (``lm_head.fp32``) or a single word (``liger``).
_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*$")
_WILDCARD = "*"
# The placeholder a scaffolded profile starts with: ``TODO`` as the first word, such as ``TODO`` or
# ``TODO: decide``. Case-sensitive, so a reason that merely starts with the word "Todo" passes.
_TODO_MARKER = re.compile(r"TODO\b")

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX_PATH = _REPO_ROOT / "docs" / "models" / "matrix.md"
_REGENERATE_HINT = "python -m arctic_platform.common.option_registry matrix --write"

# Modules whose ``register(registry)`` adds the built-in option entries and profiles.
_BUILTIN_MODULES = ("arctic_platform.common.model_options", "arctic_platform.common.profiles")


@dataclass(frozen=True)
class OptionEntry:
    """One configurable option."""

    key: str
    runs_in: str
    goal: str
    why: str
    remove_when: str | None
    source: str


@dataclass(frozen=True)
class Profile:
    """How one model (one loader) settles every option.

    ``supports`` lists keys by name. ``unsupported`` and ``not_applicable`` map a key to a one-line reason, and
    ``not_applicable`` may use a wildcard such as ``moe.*``. ``extends`` names a base profile whose cells this
    one inherits and overrides key by key.
    """

    name: str
    extends: str | None
    supports: tuple[str, ...]
    unsupported: Mapping[str, str]
    not_applicable: Mapping[str, str]


@dataclass(frozen=True)
class Cell:
    """A settled (profile, key) cell."""

    status: Status
    reason: str | None


@dataclass
class Registry:
    """Option entries and profiles. Tests build isolated instances; the module-level ``REGISTRY`` is the real one.

    ``matrix_path`` is the generated matrix file that ``check_coverage`` keeps fresh. ``None`` skips that check.
    """

    matrix_path: Path | None = None
    option_entries: dict[str, OptionEntry] = field(default_factory=dict)
    profile_entries: dict[str, Profile] = field(default_factory=dict)
    builtins_registered: bool = False


REGISTRY = Registry(matrix_path=DEFAULT_MATRIX_PATH)


def _target(registry: Registry | None) -> Registry:
    """The registry to register into: the given one, or the real one."""
    if registry is None:
        return REGISTRY
    return registry


def _loaded(registry: Registry | None) -> Registry:
    """The registry to read from. The first read of the real one registers the built-in entries and profiles.

    The built-in modules register through an explicit ``register(REGISTRY)`` call rather than as an import side
    effect, so this module's ``REGISTRY`` is filled even when it runs as ``__main__`` or after a test has dropped
    ``arctic_platform.common`` modules from ``sys.modules``.

    The built-ins register into a staging registry that is merged in only once every module succeeded, so a failed
    load raises again on every read instead of leaving a half-filled registry behind.
    """
    if registry is not None:
        return registry
    if not REGISTRY.builtins_registered:
        staging = Registry()
        for module in _BUILTIN_MODULES:
            importlib.import_module(module).register(staging)
        for key in staging.option_entries:
            if key in REGISTRY.option_entries:
                raise ValueError(f"option {key!r} already registered")
        for name in staging.profile_entries:
            if name in REGISTRY.profile_entries:
                raise ValueError(f"profile {name!r} already registered")
        REGISTRY.option_entries.update(staging.option_entries)
        REGISTRY.profile_entries.update(staging.profile_entries)
        REGISTRY.builtins_registered = True
    return REGISTRY


def _require_str(value: object, what: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{what} must be a str, got {type(value).__name__}")
    return value


def register_option(
    key: str,
    *,
    runs_in: str,
    goal: str,
    why: str,
    source: str,
    remove_when: str | None = None,
    registry: Registry | None = None,
) -> OptionEntry:
    """Register one option entry. Rejects only what is malformed on its own or a duplicate key."""
    _require_str(key, "option key")
    _require_str(runs_in, f"option {key!r} runs_in")
    _require_str(goal, f"option {key!r} goal")
    _require_str(why, f"option {key!r} why")
    _require_str(source, f"option {key!r} source")
    if remove_when is not None:
        _require_str(remove_when, f"option {key!r} remove_when")
    if _WILDCARD in key:
        raise ValueError(f"option key {key!r} must not contain a wildcard")
    if _KEY_RE.fullmatch(key) is None:
        raise ValueError(f"option key {key!r} must be a lowercase dotted path such as 'lm_head.fp32'")
    if runs_in not in RUNS_IN:
        raise ValueError(f"option {key!r} runs_in must be one of {RUNS_IN}, got {runs_in!r}")
    if goal not in GOALS:
        raise ValueError(f"option {key!r} goal must be one of {GOALS}, got {goal!r}")
    if len(why.strip()) == 0:
        raise ValueError(f"option {key!r} needs a non-empty why")
    target = _target(registry)
    if key in target.option_entries:
        raise ValueError(f"option {key!r} already registered")
    entry = OptionEntry(key=key, runs_in=runs_in, goal=goal, why=why, remove_when=remove_when, source=source)
    target.option_entries[key] = entry
    return entry


def _reason_map(value: Mapping[str, str] | None, what: str) -> Mapping[str, str]:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError(f"{what} must be a mapping of key to reason, got {type(value).__name__}")
    for key, reason in value.items():
        _require_str(key, f"{what} key")
        _require_str(reason, f"{what} reason for {key!r}")
    return MappingProxyType(dict(value))


def register_profile(
    name: str,
    *,
    supports: Sequence[str] = (),
    unsupported: Mapping[str, str] | None = None,
    not_applicable: Mapping[str, str] | None = None,
    extends: str | None = None,
    registry: Registry | None = None,
) -> Profile:
    """Register one model profile.

    Rejects a duplicate name and wrong argument types. Everything that depends on other registrations or on
    the reasons (unknown keys, wildcards outside ``not_applicable``, ``TODO`` reasons, an unknown ``extends``)
    is left to ``check_coverage``, so one check run reports every defect together.
    """
    _require_str(name, "profile name")
    if len(name) == 0:
        raise ValueError("profile name must be non-empty")
    if extends is not None:
        _require_str(extends, f"profile {name!r} extends")
    if isinstance(supports, str) or not isinstance(supports, (list, tuple)):
        raise TypeError(f"profile {name!r} supports must be a list of keys, got {type(supports).__name__}")
    for key in supports:
        _require_str(key, f"profile {name!r} supports key")
    profile = Profile(
        name=name,
        extends=extends,
        supports=tuple(supports),
        unsupported=_reason_map(unsupported, f"profile {name!r} unsupported"),
        not_applicable=_reason_map(not_applicable, f"profile {name!r} not_applicable"),
    )
    target = _target(registry)
    if name in target.profile_entries:
        raise ValueError(f"profile {name!r} already registered")
    target.profile_entries[name] = profile
    return profile


def options(registry: Registry | None = None) -> tuple[OptionEntry, ...]:
    """All option entries, sorted by key."""
    reg = _loaded(registry)
    return tuple(reg.option_entries[key] for key in sorted(reg.option_entries))


def profiles(registry: Registry | None = None) -> tuple[Profile, ...]:
    """All profiles, sorted by name."""
    reg = _loaded(registry)
    return tuple(reg.profile_entries[name] for name in sorted(reg.profile_entries))


def _chain(reg: Registry, profile: Profile) -> tuple[list[Profile], str | None]:
    """The ``extends`` chain from the root base down to ``profile``, and the problem that cut it short, if any."""
    chain = [profile]
    seen = [profile.name]
    current = profile
    while current.extends is not None:
        base_name = current.extends
        if base_name in seen:
            cycle = " -> ".join([*seen, base_name])
            return list(reversed(chain)), f"extends cycle {cycle}"
        base = reg.profile_entries.get(base_name)
        if base is None:
            return list(reversed(chain)), f"extends unknown profile {base_name!r}"
        chain.append(base)
        seen.append(base_name)
        current = base
    return list(reversed(chain)), None


def _own_cells(reg: Registry, profile: Profile) -> dict[str, Cell]:
    """The cells a profile settles itself, over the registered keys. Explicit keys win over wildcards."""
    cells: dict[str, Cell] = {}
    for pattern, reason in profile.not_applicable.items():
        if _WILDCARD in pattern:
            for key in reg.option_entries:
                if fnmatch.fnmatchcase(key, pattern):
                    cells[key] = Cell(status=NOT_APPLICABLE, reason=reason)
    for key, reason in profile.not_applicable.items():
        if _WILDCARD not in key and key in reg.option_entries:
            cells[key] = Cell(status=NOT_APPLICABLE, reason=reason)
    for key, reason in profile.unsupported.items():
        if key in reg.option_entries:
            cells[key] = Cell(status=UNSUPPORTED, reason=reason)
    for key in profile.supports:
        if key in reg.option_entries:
            cells[key] = Cell(status=SUPPORTS, reason=None)
    return cells


def _settle_chain(reg: Registry, chain: Iterable[Profile]) -> dict[str, Cell]:
    cells: dict[str, Cell] = {}
    for member in chain:
        cells.update(_own_cells(reg, member))
    return cells


def settle(profile: str | Profile, registry: Registry | None = None) -> dict[str, Cell]:
    """The settled cells of a profile, keyed by option key, with ``extends`` applied (child overrides base).

    Only registered keys appear. A key the profile never settles is absent. Raises ``ValueError`` for an unknown
    profile or a broken ``extends`` chain; ``check_coverage`` reports those instead of raising.
    """
    reg = _loaded(registry)
    if isinstance(profile, str):
        if profile not in reg.profile_entries:
            raise ValueError(f"unknown profile {profile!r}")
        profile = reg.profile_entries[profile]
    chain, problem = _chain(reg, profile)
    if problem is not None:
        raise ValueError(f"profile {profile.name!r}: {problem}")
    return _settle_chain(reg, chain)


def _reason_problem(reason: str) -> str | None:
    stripped = reason.strip()
    if len(stripped) == 0:
        return "reason is empty"
    if _TODO_MARKER.match(stripped) is not None:
        return "reason is TODO"
    if "\n" in stripped:
        return "reason must be one line"
    return None


def _profile_problems(reg: Registry, profile: Profile) -> list[tuple[str, str]]:
    """(key, problem) pairs for one profile's own entries and its settled cells."""
    problems: list[tuple[str, str]] = []
    # Which of this profile's own entries settle each registered key, to find keys settled more than once.
    settled_by: dict[str, list[str]] = {}

    for key in profile.supports:
        if _WILDCARD in key:
            problems.append((key, "wildcard not allowed in supports; list keys by name"))
        elif key not in reg.option_entries:
            problems.append((key, "unknown key in supports"))
        else:
            settled_by.setdefault(key, []).append("supports")

    for list_name, entries in (("unsupported", profile.unsupported), ("not_applicable", profile.not_applicable)):
        for key, reason in entries.items():
            reason_problem = _reason_problem(reason)
            if reason_problem is not None:
                problems.append((key, f"{list_name} {reason_problem}"))
            if _WILDCARD in key:
                if list_name != "not_applicable":
                    problems.append((key, f"wildcard not allowed in {list_name}; list keys by name"))
                    continue
                matched = [k for k in reg.option_entries if fnmatch.fnmatchcase(k, key)]
                if len(matched) == 0:
                    problems.append((key, "stale wildcard in not_applicable: it matches no registered key"))
                for matched_key in matched:
                    settled_by.setdefault(matched_key, []).append(f"not_applicable {key!r}")
            elif key not in reg.option_entries:
                problems.append((key, f"unknown key in {list_name}"))
            else:
                settled_by.setdefault(key, []).append(list_name)

    for key, sources in settled_by.items():
        if len(sources) > 1:
            problems.append((key, f"settled more than once in one profile: {', '.join(sources)}"))

    chain, chain_problem = _chain(reg, profile)
    if chain_problem is not None:
        problems.append(("", chain_problem))
    cells = _settle_chain(reg, chain)
    for key in reg.option_entries:
        if key not in cells:
            problems.append((key, "not settled: add it to supports, unsupported or not_applicable"))
    return problems


def check_coverage(registry: Registry | None = None) -> list[str]:
    """Every problem that keeps the table from being complete, in a deterministic order. Empty means it passes."""
    reg = _loaded(registry)
    located: list[tuple[str, str, str]] = []
    for name in sorted(reg.profile_entries):
        for key, problem in _profile_problems(reg, reg.profile_entries[name]):
            located.append((name, key, problem))
    problems = []
    for name, key, problem in sorted(located):
        if len(key) == 0:
            problems.append(f"profile {name!r}: {problem}")
        else:
            problems.append(f"profile {name!r}, key {key!r}: {problem}")
    if reg.matrix_path is not None:
        matrix_problem = _matrix_problem(reg, reg.matrix_path)
        if matrix_problem is not None:
            problems.append(matrix_problem)
    return problems


def _display_path(path: Path) -> str:
    """``path`` relative to the checkout when it is inside it, such as ``docs/models/matrix.md``."""
    resolved = path.resolve()
    if resolved.is_relative_to(_REPO_ROOT):
        return str(resolved.relative_to(_REPO_ROOT))
    return str(path)


def _matrix_problem(reg: Registry, path: Path) -> str | None:
    if not path.is_file():
        return f"{_display_path(path)}: missing; generate it with `{_REGENERATE_HINT}`"
    if path.read_text() != render_matrix(reg):
        return f"{_display_path(path)}: stale; regenerate it with `{_REGENERATE_HINT}`"
    return None


_STATUS_LABELS = {SUPPORTS: "yes", UNSUPPORTED: "no", NOT_APPLICABLE: "n/a"}


def render_matrix(registry: Registry | None = None) -> str:
    """The Markdown matrix: one row per option, one column per profile, then the reasons per profile."""
    reg = _loaded(registry)
    entries = options(reg)
    names = [profile.name for profile in profiles(reg)]
    settled: dict[str, dict[str, Cell]] = {}
    for name in names:
        chain, _ = _chain(reg, reg.profile_entries[name])
        settled[name] = _settle_chain(reg, chain)

    lines = [
        "# Model options matrix",
        "",
        f"<!-- Generated by `{_REGENERATE_HINT}`. Do not edit by hand. -->",
        "",
        "One row per option, one column per model profile (one profile per loader). `yes` means the model",
        "supports the option in some context, through a loader option, a `Patches` field or a `ModelSpec` field",
        "that the loader accepts. `no` means the loader offers no way to set it, and `n/a` that the architecture",
        "rules the option out. Restrictions that depend on the platform, the parallelism or the value stay in",
        "each loader's own validation. The reasons follow the table.",
        "",
        "| Option | Runs in | Goal | " + " | ".join(f"`{name}`" for name in names) + " |",
        "|---|---|---|" + "---|" * len(names),
    ]
    for entry in entries:
        row = [f"`{entry.key}`", entry.runs_in, entry.goal]
        for name in names:
            cell = settled[name].get(entry.key)
            if cell is None:
                row.append("?")
            else:
                row.append(_STATUS_LABELS[cell.status])
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Options", ""]
    for entry in entries:
        line = f"- `{entry.key}`: {entry.why} Today: {entry.source}."
        if entry.remove_when is not None:
            line += f" Remove when: {entry.remove_when}."
        lines.append(line)

    for name in names:
        profile = reg.profile_entries[name]
        cells = settled[name]
        unsupported_count = sum(1 for cell in cells.values() if cell.status == UNSUPPORTED)
        lines += ["", f"## `{name}`", ""]
        if profile.extends is not None:
            lines += [f"Extends `{profile.extends}`.", ""]
        lines.append(f"{unsupported_count} of {len(entries)} options unsupported.")
        reasons = [(entry.key, cells[entry.key]) for entry in entries if entry.key in cells]
        reasons = [(key, cell) for key, cell in reasons if cell.reason is not None]
        if len(reasons) > 0:
            lines.append("")
        for key, cell in reasons:
            lines.append(f"- `{key}` ({_STATUS_LABELS[cell.status]}): {cell.reason}")
    return "\n".join(lines) + "\n"


def _cmd_list(reg: Registry) -> int:
    entries = options(reg)
    width = max((len(entry.key) for entry in entries), default=0)
    for entry in entries:
        print(f"{entry.key:<{width}}  {entry.runs_in:<7}  {entry.goal:<10}  {entry.why}")
    return 0


def _cmd_check(reg: Registry) -> int:
    problems = check_coverage(reg)
    for problem in problems:
        print(problem)
    entries = options(reg)
    names = [profile.name for profile in profiles(reg)]
    if len(problems) > 0:
        print(f"check failed: {len(problems)} problem(s) across {len(entries)} options and {len(names)} profiles")
        return 1
    unsupported_count = 0
    for name in names:
        unsupported_count += sum(1 for cell in settle(name, reg).values() if cell.status == UNSUPPORTED)
    print(
        f"check passed: {len(entries)} options x {len(names)} profiles = {len(entries) * len(names)} cells,"
        f" {unsupported_count} unsupported"
    )
    return 0


def _cmd_matrix(reg: Registry, write: bool) -> int:
    text = render_matrix(reg)
    if not write:
        sys.stdout.write(text)
        return 0
    if reg.matrix_path is None:
        print("matrix --write needs a matrix path")
        return 1
    reg.matrix_path.parent.mkdir(parents=True, exist_ok=True)
    reg.matrix_path.write_text(text)
    print(f"wrote {reg.matrix_path}")
    return 0


def main(argv: Sequence[str] | None = None, registry: Registry | None = None) -> int:
    """Command line entry point. ``registry`` defaults to the real one."""
    parser = argparse.ArgumentParser(
        prog="python -m arctic_platform.common.option_registry",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the registered options")
    check_parser = commands.add_parser("check", help="check that every profile settles every option")
    matrix_parser = commands.add_parser("matrix", help="render the options matrix")
    matrix_parser.add_argument("--write", action="store_true", help="write it to the matrix path")
    for sub in (check_parser, matrix_parser):
        sub.add_argument("--matrix-path", type=Path, default=None, help="matrix file (default: the registry's)")
    args = parser.parse_args(argv)

    reg = _loaded(registry)
    if args.command != "list" and args.matrix_path is not None:
        reg = Registry(
            matrix_path=args.matrix_path,
            option_entries=reg.option_entries,
            profile_entries=reg.profile_entries,
        )
    if args.command == "list":
        return _cmd_list(reg)
    if args.command == "check":
        return _cmd_check(reg)
    return _cmd_matrix(reg, args.write)


if __name__ == "__main__":
    sys.exit(main())
