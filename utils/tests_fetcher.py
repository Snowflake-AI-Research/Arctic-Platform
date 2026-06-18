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

"""Diff-based test selector for arctic-platform.

A small, single-suite take on HuggingFace transformers' ``utils/tests_fetcher.py``.
Given a git diff against a base ref, it figures out the *minimal* set of tests
that could be affected by the change, so CI doesn't have to run the whole suite
on every PR:

  1. Collect the Python files changed in the diff.
  2. Build an import-dependency graph over ``arctic_platform/`` + ``tests/``
     (module A "impacts" module B when B imports A, directly or transitively).
  3. Walk that graph backwards from the changed files to every test impacted.
  4. Write the impacted test paths to an output file (one per line), which CI
     feeds straight to ``pytest``.

It deliberately errs on the side of running *more* tests: anything it can't
reason about (a missing base ref, a touched shared fixture / packaging / CI
file, a deleted module, or a ``[test all]`` commit tag) falls back to selecting
the entire suite.

Run it locally to see exactly what CI would run for your branch::

    python utils/tests_fetcher.py --base origin/main
    python -m pytest $(cat test_preparation/test_list.txt) -m "not gpu"
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "arctic_platform"

# Roots whose .py files participate in the import graph.
SOURCE_ROOTS = (PACKAGE, "tests")

# Changing any of these means "we can't safely narrow the suite" -> run everything.
# (Shell globs, matched against repo-root-relative POSIX paths.)
RUN_ALL_GLOBS = (
    "pyproject.toml",
    "tests/conftest.py",  # root conftest: auto-loaded by every test
    ".github/workflows/*",
    "utils/tests_fetcher.py",  # the selector itself
)

# Commit-message tags that force the whole suite (mirrors transformers' parse_commit_message).
RUN_ALL_COMMIT_TAGS = ("[test all]", "[no filter]")


def _run_git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _is_test_file(path: Path) -> bool:
    """A collectible test module: ``tests/**/test_*.py``."""
    try:
        rel = path.relative_to(REPO_ROOT)
    except ValueError:
        return False
    return rel.parts[0] == "tests" and rel.name.startswith("test_") and rel.suffix == ".py"


def _all_test_files() -> list[Path]:
    return sorted(p for p in (REPO_ROOT / "tests").rglob("test_*.py") if _is_test_file(p))


def _all_source_files() -> list[Path]:
    files: list[Path] = []
    for root in SOURCE_ROOTS:
        files.extend((REPO_ROOT / root).rglob("*.py"))
    return sorted(files)


def _module_name(path: Path) -> str | None:
    """Dotted module name for a package file (e.g. arctic_platform/rl/config.py -> arctic_platform.rl.config)."""
    rel = path.relative_to(REPO_ROOT)
    if rel.parts[0] != PACKAGE:
        return None
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _build_indexes(files: list[Path]) -> tuple[dict[str, Path], dict[Path, dict[str, Path]]]:
    """Return (package_module_index, dir_local_index).

    - package_module_index: dotted ``arctic_platform.*`` module name -> file.
    - dir_local_index: directory -> {bare module stem -> file}, used to resolve sibling
      imports inside the test suite (e.g. ``from rl_harness import ...`` in tests/rl/).
    """
    module_index: dict[str, Path] = {}
    dir_local: dict[Path, dict[str, Path]] = {}
    for f in files:
        mod = _module_name(f)
        if mod:
            module_index[mod] = f
        dir_local.setdefault(f.parent, {})[f.stem] = f
    return module_index, dir_local


def _resolve_candidate(name: str, module_index: dict[str, Path]) -> Path | None:
    """Resolve a dotted ``arctic_platform.*`` name to a file via longest-prefix match."""
    if not (name == PACKAGE or name.startswith(PACKAGE + ".")):
        return None
    parts = name.split(".")
    # `from arctic_platform.rl.processors import stats_tracker` yields the symbol name too;
    # try the longest dotted path first, then peel back to the owning module/package.
    while parts:
        hit = module_index.get(".".join(parts))
        if hit is not None:
            return hit
        parts = parts[:-1]
    return None


def _file_intra_deps(
    path: Path,
    module_index: dict[str, Path],
    dir_local: dict[Path, dict[str, Path]],
) -> set[Path]:
    """Intra-repo files that ``path`` imports (package modules + sibling test helpers)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"WARNING: could not parse {path}: {e}", file=sys.stderr)
        return set()

    self_mod = _module_name(path)
    is_init = path.name == "__init__.py"
    deps: set[Path] = set()
    siblings = dir_local.get(path.parent, {})

    def add_dotted(name: str) -> None:
        hit = _resolve_candidate(name, module_index)
        if hit is not None:
            deps.add(hit)

    def add_bare(top: str) -> None:
        # Sibling test-helper module (non-package), e.g. `rl_harness` next to the importer.
        hit = siblings.get(top)
        if hit is not None and hit != path:
            deps.add(hit)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add_dotted(alias.name)
                add_bare(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                module = node.module or ""
                if module:
                    add_dotted(module)
                    add_bare(module.split(".")[0])
                    for alias in node.names:
                        add_dotted(f"{module}.{alias.name}")
                else:
                    for alias in node.names:
                        add_bare(alias.name)
            elif self_mod is not None:
                # Relative import inside the arctic_platform package.
                pkg_parts = self_mod.split(".") if is_init else self_mod.split(".")[:-1]
                base_parts = pkg_parts[: len(pkg_parts) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
                if base:
                    add_dotted(base)
                    for alias in node.names:
                        add_dotted(f"{base}.{alias.name}")
    return deps


def _build_reverse_graph(
    files: list[Path],
    module_index: dict[str, Path],
    dir_local: dict[Path, dict[str, Path]],
) -> dict[Path, set[Path]]:
    """Map each file -> the set of files that import it (directly)."""
    reverse: dict[Path, set[Path]] = {f: set() for f in files}
    for f in files:
        for dep in _file_intra_deps(f, module_index, dir_local):
            reverse.setdefault(dep, set()).add(f)
    return reverse


def _impacted_files(seeds: set[Path], reverse: dict[Path, set[Path]]) -> set[Path]:
    """All files reachable from ``seeds`` by following 'is-imported-by' edges (incl. the seeds)."""
    seen: set[Path] = set()
    stack = list(seeds)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(reverse.get(cur, ()))
    return seen


def _matches_glob(rel_posix: str, globs: tuple[str, ...]) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(rel_posix, g) for g in globs)


def _diff_files(base: str) -> tuple[list[str], list[str]]:
    """Return (changed, deleted) repo-root-relative paths for ``base...HEAD``.

    'changed' = added / modified / renamed-new; 'deleted' = deleted / renamed-old.
    """
    out = _run_git(["diff", "--name-status", "--find-renames", f"{base}...HEAD"])
    changed: list[str] = []
    deleted: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R") and len(parts) >= 3:
            deleted.append(parts[1])
            changed.append(parts[2])
        elif status.startswith("D"):
            deleted.append(parts[1])
        elif len(parts) >= 2:
            changed.append(parts[1])
    return changed, deleted


def _commit_messages(base: str) -> str:
    """Concatenated commit messages on ``base..HEAD`` (every commit the branch adds)."""
    try:
        return _run_git(["log", "--format=%B", f"{base}..HEAD"])
    except subprocess.CalledProcessError:
        return ""


def select_tests(base: str | None, commit_message: str = "") -> tuple[str, list[Path]]:
    """Decide which tests to run.

    Returns ``(mode, test_files)`` where mode is one of:
      - "all"    -> run the entire suite (test_files = every test file),
      - "subset" -> run exactly ``test_files``,
      - "none"   -> nothing is impacted; CI can skip the test job.
    """
    all_tests = _all_test_files()

    if not base:
        print("No base ref provided (e.g. push to main) -> running all tests.")
        return "all", all_tests

    try:
        _run_git(["rev-parse", "--verify", f"{base}^{{commit}}"])
    except subprocess.CalledProcessError:
        print(f"WARNING: base ref {base!r} could not be resolved -> running all tests.")
        return "all", all_tests

    # --- whole-suite fallbacks ------------------------------------------------
    # Scan the explicit message plus every commit the branch adds, so the tag is
    # honored regardless of whether CI checked out the PR head or a merge commit.
    messages = commit_message + "\n" + _commit_messages(base)
    if any(tag in messages for tag in RUN_ALL_COMMIT_TAGS):
        print("Commit message requests the full suite -> running all tests.")
        return "all", all_tests

    changed, deleted = _diff_files(base)

    def _under_sources(p: str) -> bool:
        return p.endswith(".py") and p.split("/", 1)[0] in SOURCE_ROOTS

    if any(_under_sources(p) for p in deleted):
        print("A Python source/test file was deleted -> running all tests (graph can't be trusted).")
        return "all", all_tests

    triggers = [p for p in changed if _matches_glob(p, RUN_ALL_GLOBS)]
    if triggers:
        print(f"Changed shared/infra file(s) {triggers} -> running all tests.")
        return "all", all_tests

    # --- import-graph narrowing ----------------------------------------------
    source_files = _all_source_files()
    module_index, dir_local = _build_indexes(source_files)
    reverse = _build_reverse_graph(source_files, module_index, dir_local)

    selected: set[Path] = set()
    for rel in changed:
        path = (REPO_ROOT / rel).resolve()
        if not _under_sources(rel):
            continue  # non-Python change in source roots (e.g. data) -> no direct test impact
        # A directory-scoped conftest affects every test under that directory.
        if path.name == "conftest.py":
            for t in all_tests:
                if str(t).startswith(str(path.parent) + os.sep):
                    selected.add(t)
            continue
        impacted = _impacted_files({path}, reverse)
        selected.update(f for f in impacted if _is_test_file(f))
        if _is_test_file(path):
            selected.add(path)  # a brand-new test file has no importers yet

    if not selected:
        print("No tests impacted by this diff.")
        return "none", []
    if selected >= set(all_tests):
        print("Diff impacts the whole suite -> running all tests.")
        return "all", sorted(selected)
    return "subset", sorted(selected)


def _emit_github_output(mode: str, rel_paths: list[str]) -> None:
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if not gh_out:
        return
    with open(gh_out, "a", encoding="utf-8") as fh:
        fh.write(f"mode={mode}\n")
        fh.write(f"num_tests={len(rel_paths)}\n")
        # Space-separated so the workflow can feed it straight to pytest.
        fh.write(f"tests={' '.join(rel_paths)}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--base",
        default="origin/main",
        help="Git ref to diff against (merge-base is used). Empty string -> run all tests. Default: origin/main.",
    )
    parser.add_argument(
        "--output-file",
        default="test_preparation/test_list.txt",
        help="Where to write the newline-separated list of test files to run.",
    )
    parser.add_argument(
        "--commit-message",
        default="",
        help="Commit message to scan for [test all] / [no filter] override tags.",
    )
    args = parser.parse_args()

    mode, test_files = select_tests(args.base, args.commit_message)
    rel_paths = [str(p.relative_to(REPO_ROOT)) for p in test_files]

    out_path = (REPO_ROOT / args.output_file).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rel_paths) + ("\n" if rel_paths else ""), encoding="utf-8")

    print(f"\n### MODE: {mode} ###")
    print(f"### {len(rel_paths)} test file(s) -> {out_path.relative_to(REPO_ROOT)} ###")
    for p in rel_paths:
        print(f"  {p}")

    _emit_github_output(mode, rel_paths)


if __name__ == "__main__":
    main()
