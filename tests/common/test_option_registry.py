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

"""Tests for the option registry, its coverage check and its command line. CPU only, no network.

Every test except the real-data ones builds an isolated ``Registry``, so the global one is never touched.
"""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import pytest

from arctic_platform.common import option_registry as reg_mod
from arctic_platform.common.option_registry import NOT_APPLICABLE
from arctic_platform.common.option_registry import SUPPORTS
from arctic_platform.common.option_registry import UNSUPPORTED
from arctic_platform.common.option_registry import Cell
from arctic_platform.common.option_registry import Registry
from arctic_platform.common.option_registry import check_coverage
from arctic_platform.common.option_registry import register_option
from arctic_platform.common.option_registry import register_profile
from arctic_platform.common.option_registry import render_matrix
from arctic_platform.common.option_registry import settle
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import execute_subprocess_async

# The real registry: the 20 keys with where they run and their goal, and the six loaders.
EXPECTED_KEYS = {
    "checkpointing.mode": ("trainer", "memory"),
    "checkpointing.freq": ("trainer", "memory"),
    "checkpointing.targets": ("trainer", "memory"),
    "checkpointing.offload": ("trainer", "memory"),
    "tiled_mlp.token_chunk_size": ("trainer", "memory"),
    "lm_head.fp32": ("both", "parity"),
    "lm_head.token_chunk_size": ("trainer", "memory"),
    "lm_head.vocab_chunk_size": ("trainer", "memory"),
    "lm_head.cross_entropy": ("trainer", "memory"),
    "liger": ("trainer", "speed"),
    "moe.grouped_mm": ("trainer", "speed"),
    "moe.comm_backend": ("trainer", "speed"),
    "moe.comm_sms": ("trainer", "speed"),
    "moe.comm_token_chunk": ("trainer", "speed"),
    "attention.backend": ("trainer", "speed"),
    "attention.sparse_mla": ("trainer", "speed"),
    "numerics.reduce_dtype": ("trainer", "parity"),
    "compile.fullgraph": ("trainer", "speed"),
    "peft": ("trainer", "mode"),
    "zorro_train": ("trainer", "mode"),
}
EXPECTED_PROFILES = {"huggingface", "qwen3_5_moe", "glm_moe_dsa", "generic_moe", "glm5_next", "qwen4_exp"}

SYNTHETIC_KEYS = ("lm_head.fp32", "moe.grouped_mm", "moe.comm_backend", "liger")


def _option(registry: Registry, key: str) -> None:
    register_option(key, runs_in="trainer", goal="speed", why=f"why {key}", source="test", registry=registry)


def _registry(keys: tuple[str, ...] = SYNTHETIC_KEYS) -> Registry:
    registry = Registry()
    for key in keys:
        _option(registry, key)
    return registry


def _complete_registry() -> Registry:
    """A synthetic registry with a base profile, a child that overrides it, and no problems."""
    registry = _registry()
    register_profile(
        "dense",
        supports=["lm_head.fp32"],
        unsupported=dict(liger="no kernels"),
        not_applicable={"moe.*": "no experts"},
        registry=registry,
    )
    register_profile(
        "moe",
        supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend"],
        unsupported=dict(liger="rejected"),
        registry=registry,
    )
    register_profile(
        "moe_child",
        extends="moe",
        unsupported={"moe.grouped_mm": "not yet evaluated"},
        registry=registry,
    )
    return registry


def _problems_naming(problems: list[str], profile: str, key: str, text: str) -> list[str]:
    return [p for p in problems if f"profile {profile!r}" in p and f"key {key!r}" in p and text in p]


class TestRegisterOption:
    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            (dict(key="Lm_head.fp32"), ValueError, "lowercase dotted path"),
            (dict(key="lm_head..fp32"), ValueError, "lowercase dotted path"),
            (dict(key="lm_head.fp32."), ValueError, "lowercase dotted path"),
            (dict(key=""), ValueError, "lowercase dotted path"),
            (dict(key="lm head"), ValueError, "lowercase dotted path"),
            (dict(key="moe.*"), ValueError, "wildcard"),
            (dict(key=7), TypeError, "option key must be a str"),
            (dict(runs_in="gpu"), ValueError, "runs_in must be one of"),
            (dict(goal="fun"), ValueError, "goal must be one of"),
            (dict(why=""), ValueError, "non-empty why"),
            (dict(why="   "), ValueError, "non-empty why"),
            (dict(why=None), TypeError, "why must be a str"),
            (dict(source=None), TypeError, "source must be a str"),
            (dict(remove_when=3), TypeError, "remove_when must be a str"),
        ],
    )
    def test_rejects_malformed_entry(self, kwargs, error, match):
        args = dict(key="lm_head.fp32", runs_in="trainer", goal="parity", why="parity", source="test")
        args.update(kwargs)
        registry = Registry()
        with pytest.raises(error, match=match):
            register_option(args.pop("key"), registry=registry, **args)
        assert len(registry.option_entries) == 0

    def test_rejects_duplicate_key(self):
        registry = _registry(("liger",))
        with pytest.raises(ValueError, match="already registered"):
            _option(registry, "liger")

    @pytest.mark.parametrize("key", ["liger", "peft", "lm_head.fp32", "tiled_mlp.token_chunk_size", "a.b2.c_d"])
    def test_accepts_well_formed_key(self, key):
        registry = Registry()
        entry = register_option(key, runs_in="both", goal="parity", why="w", source="s", registry=registry)
        assert registry.option_entries == {key: entry}
        assert entry.remove_when is None


class TestRegisterProfile:
    @pytest.mark.parametrize(
        "kwargs, error, match",
        [
            (dict(name=3), TypeError, "profile name must be a str"),
            (dict(name=""), ValueError, "non-empty"),
            (dict(supports="liger"), TypeError, "supports must be a list"),
            (dict(supports={"liger"}), TypeError, "supports must be a list"),
            (dict(supports=["liger", 3]), TypeError, "supports key must be a str"),
            (dict(unsupported=["liger"]), TypeError, "unsupported must be a mapping"),
            (dict(unsupported={3: "reason"}), TypeError, "unsupported key must be a str"),
            (dict(unsupported=dict(liger=None)), TypeError, "reason for 'liger' must be a str"),
            (dict(not_applicable={("moe",): "reason"}), TypeError, "not_applicable key must be a str"),
            (dict(extends=1), TypeError, "extends must be a str"),
        ],
    )
    def test_rejects_wrong_types(self, kwargs, error, match):
        args = dict(name="p")
        args.update(kwargs)
        registry = _registry()
        with pytest.raises(error, match=match):
            register_profile(args.pop("name"), registry=registry, **args)
        assert len(registry.profile_entries) == 0

    def test_rejects_duplicate_name(self):
        registry = _complete_registry()
        with pytest.raises(ValueError, match="already registered"):
            register_profile("dense", registry=registry)

    def test_accepts_cross_reference_defects_and_check_reports_them_all(self):
        """Registration stores every defect that depends on other registrations; one check reports them all."""
        registry = _registry()
        register_profile(
            "broken",
            extends="missing_base",
            supports=["lm_head.fp32", "nope.key", "moe.*"],
            unsupported=dict(liger="TODO"),
            not_applicable={"gdn.*": "no gdn"},
            registry=registry,
        )
        problems = check_coverage(registry)
        expected = [
            ("moe.*", "wildcard not allowed in supports"),
            ("nope.key", "unknown key in supports"),
            ("liger", "unsupported reason is TODO"),
            ("gdn.*", "stale wildcard"),
            ("moe.grouped_mm", "not settled"),
            ("moe.comm_backend", "not settled"),
        ]
        for key, text in expected:
            assert len(_problems_naming(problems, "broken", key, text)) == 1, (key, text, problems)
        assert "profile 'broken': extends unknown profile 'missing_base'" in problems
        assert len(problems) == len(expected) + 1


class TestSettle:
    def test_statuses_and_reasons(self):
        registry = _complete_registry()
        assert settle("dense", registry) == {
            "lm_head.fp32": Cell(status=SUPPORTS, reason=None),
            "liger": Cell(status=UNSUPPORTED, reason="no kernels"),
            "moe.grouped_mm": Cell(status=NOT_APPLICABLE, reason="no experts"),
            "moe.comm_backend": Cell(status=NOT_APPLICABLE, reason="no experts"),
        }

    def test_wildcard_expands_over_registered_keys_only(self):
        registry = _complete_registry()
        cells = settle("dense", registry)
        moe_keys = sorted(key for key, cell in cells.items() if cell.status == NOT_APPLICABLE)
        assert moe_keys == ["moe.comm_backend", "moe.grouped_mm"]
        _option(registry, "moe.comm_sms")
        assert settle("dense", registry)["moe.comm_sms"] == Cell(status=NOT_APPLICABLE, reason="no experts")

    def test_extends_child_overrides_base(self):
        registry = _complete_registry()
        assert settle("moe_child", registry) == {
            "lm_head.fp32": Cell(status=SUPPORTS, reason=None),
            "liger": Cell(status=UNSUPPORTED, reason="rejected"),
            "moe.grouped_mm": Cell(status=UNSUPPORTED, reason="not yet evaluated"),
            "moe.comm_backend": Cell(status=SUPPORTS, reason=None),
        }
        assert settle("moe", registry)["moe.grouped_mm"] == Cell(status=SUPPORTS, reason=None)

    def test_child_can_override_a_base_wildcard(self):
        registry = _complete_registry()
        register_profile("dense_child", extends="dense", supports=["moe.grouped_mm"], registry=registry)
        cells = settle("dense_child", registry)
        assert cells["moe.grouped_mm"] == Cell(status=SUPPORTS, reason=None)
        assert cells["moe.comm_backend"] == Cell(status=NOT_APPLICABLE, reason="no experts")
        assert check_coverage(registry) == []

    def test_unknown_profile_and_broken_extends_raise(self):
        registry = _complete_registry()
        with pytest.raises(ValueError, match="unknown profile 'nope'"):
            settle("nope", registry)
        register_profile("orphan", extends="nope", registry=registry)
        with pytest.raises(ValueError, match="extends unknown profile 'nope'"):
            settle("orphan", registry)


class TestCheckCoverage:
    def test_complete_registry_has_no_problems(self):
        assert check_coverage(_complete_registry()) == []

    @pytest.mark.parametrize(
        "profile_kwargs, key, text",
        [
            # a missing cell
            (dict(supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend"]), "liger", "not settled"),
            # a key in two lists
            (
                dict(
                    supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend", "liger"],
                    unsupported=dict(liger="rejected"),
                ),
                "liger",
                "settled more than once in one profile: supports, unsupported",
            ),
            # an explicit key that a wildcard in the same profile also settles
            (
                dict(
                    supports=["lm_head.fp32", "moe.grouped_mm"],
                    unsupported=dict(liger="rejected"),
                    not_applicable={"moe.*": "no experts"},
                ),
                "moe.grouped_mm",
                "settled more than once in one profile: supports, not_applicable 'moe.*'",
            ),
            # an unknown key
            (
                dict(supports=[*SYNTHETIC_KEYS, "gdn.state_fp32"]),
                "gdn.state_fp32",
                "unknown key in supports",
            ),
            (
                dict(supports=list(SYNTHETIC_KEYS), not_applicable={"gdn.state_fp32": "no gdn"}),
                "gdn.state_fp32",
                "unknown key in not_applicable",
            ),
            # a wildcard where it is not allowed
            (
                dict(supports=[*SYNTHETIC_KEYS, "moe.*"]),
                "moe.*",
                "wildcard not allowed in supports",
            ),
            (
                dict(supports=list(SYNTHETIC_KEYS), unsupported={"moe.*": "no experts"}),
                "moe.*",
                "wildcard not allowed in unsupported",
            ),
            # a missing reason
            (
                dict(supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend"], unsupported=dict(liger="")),
                "liger",
                "unsupported reason is empty",
            ),
            (
                dict(supports=["lm_head.fp32", "liger"], not_applicable={"moe.*": "  "}),
                "moe.*",
                "not_applicable reason is empty",
            ),
            (
                dict(supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend"], unsupported=dict(liger="a\nb")),
                "liger",
                "unsupported reason must be one line",
            ),
            # a stale wildcard
            (
                dict(supports=list(SYNTHETIC_KEYS), not_applicable={"gdn.*": "no gdn"}),
                "gdn.*",
                "stale wildcard in not_applicable",
            ),
        ],
    )
    def test_each_defect_names_profile_and_key(self, profile_kwargs, key, text):
        registry = _complete_registry()
        register_profile("defective", registry=registry, **profile_kwargs)
        problems = check_coverage(registry)
        assert len(problems) == 1, problems
        assert _problems_naming(problems, "defective", key, text) == problems

    @pytest.mark.parametrize("list_name", ["unsupported", "not_applicable"])
    @pytest.mark.parametrize("reason", ["TODO", "TODO: decide", " TODO ", "TODO(owner) evaluate"])
    def test_todo_reason_rejected(self, list_name, reason):
        registry = _complete_registry()
        cells = dict(liger=reason)
        register_profile(
            "todo",
            supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend"],
            registry=registry,
            **{list_name: cells},
        )
        problems = check_coverage(registry)
        assert problems == [f"profile 'todo', key 'liger': {list_name} reason is TODO"]

    @pytest.mark.parametrize(
        "reason", ["Todo lists are kept elsewhere", "todo", "TODOS", "see the TODO in the loader"]
    )
    def test_reason_that_is_not_the_todo_marker_passes(self, reason):
        registry = _complete_registry()
        register_profile(
            "fine",
            supports=["lm_head.fp32", "moe.grouped_mm", "moe.comm_backend"],
            unsupported=dict(liger=reason),
            registry=registry,
        )
        assert check_coverage(registry) == []

    def test_unknown_extends(self):
        registry = _complete_registry()
        register_profile("orphan", extends="nope", supports=list(SYNTHETIC_KEYS), registry=registry)
        assert check_coverage(registry) == ["profile 'orphan': extends unknown profile 'nope'"]

    def test_extends_cycle(self):
        registry = _complete_registry()
        register_profile("a", extends="b", supports=list(SYNTHETIC_KEYS), registry=registry)
        register_profile("b", extends="a", supports=list(SYNTHETIC_KEYS), registry=registry)
        register_profile("self", extends="self", supports=list(SYNTHETIC_KEYS), registry=registry)
        assert check_coverage(registry) == [
            "profile 'a': extends cycle a -> b -> a",
            "profile 'b': extends cycle b -> a -> b",
            "profile 'self': extends cycle self -> self",
        ]

    def test_new_option_fails_every_profile(self):
        registry = _complete_registry()
        _option(registry, "compile.fullgraph")
        problems = check_coverage(registry)
        assert problems == [
            f"profile {name!r}, key 'compile.fullgraph': not settled: add it to supports, unsupported or"
            " not_applicable"
            for name in ("dense", "moe", "moe_child")
        ]

    def test_new_empty_profile_fails_every_option(self):
        registry = _complete_registry()
        register_profile("empty", registry=registry)
        problems = check_coverage(registry)
        assert problems == [
            f"profile 'empty', key {key!r}: not settled: add it to supports, unsupported or not_applicable"
            for key in sorted(SYNTHETIC_KEYS)
        ]

    def test_problem_order_is_deterministic(self):
        def build(order):
            registry = Registry()
            for key in order:
                _option(registry, key)
            register_profile("z", registry=registry)
            register_profile("a", unsupported=dict(liger="TODO"), registry=registry)
            return check_coverage(registry)

        first = build(SYNTHETIC_KEYS)
        assert first == build(tuple(reversed(SYNTHETIC_KEYS)))
        assert first[0].startswith("profile 'a'")
        assert first[-1].startswith("profile 'z'")

    def test_stale_and_missing_matrix(self, tmp_path):
        registry = _complete_registry()
        registry.matrix_path = tmp_path / "matrix.md"
        problems = check_coverage(registry)
        assert len(problems) == 1 and "missing" in problems[0] and "matrix.md" in problems[0]

        registry.matrix_path.write_text(render_matrix(registry))
        assert check_coverage(registry) == []

        registry.matrix_path.write_text(render_matrix(registry) + "edited by hand\n")
        problems = check_coverage(registry)
        assert len(problems) == 1 and "stale" in problems[0]

        registry.matrix_path.write_text(render_matrix(registry))
        _option(registry, "compile.fullgraph")
        problems = check_coverage(registry)
        assert problems[-1].endswith(
            "stale; regenerate it with `python -m arctic_platform.common.option_registry matrix --write`"
        )


class TestRenderMatrix:
    def test_one_row_per_option_and_one_column_per_profile(self):
        text = render_matrix(_complete_registry())
        assert "| Option | Runs in | Goal | `dense` | `moe` | `moe_child` |" in text
        assert "| `moe.grouped_mm` | trainer | speed | n/a | yes | no |" in text
        assert "| `liger` | trainer | speed | no | no | no |" in text
        assert "- `moe.grouped_mm` (no): not yet evaluated" in text
        assert text == render_matrix(_complete_registry())


class TestRealRegistry:
    def test_real_data_passes(self):
        assert check_coverage() == []

    def test_real_options_match_key_table(self):
        entries = reg_mod.options()
        assert {entry.key: (entry.runs_in, entry.goal) for entry in entries} == EXPECTED_KEYS
        for entry in entries:
            assert len(entry.why.strip()) > 0
            assert len(entry.source.strip()) > 0

    def test_real_profiles_match_loader_list(self):
        names = {profile.name for profile in reg_mod.profiles()}
        assert names == EXPECTED_PROFILES
        for name in names:
            assert set(settle(name)) == set(EXPECTED_KEYS)

    def test_failed_builtin_load_keeps_failing_until_it_succeeds(self, monkeypatch):
        """A built-in ``register`` that raises midway must raise on every read, never leave a half-filled registry."""
        calls = dict(count=0)

        def register(registry: Registry) -> None:
            calls["count"] += 1
            _option(registry, "first.ok")
            if calls["count"] < 3:
                raise RuntimeError("built-in data module is broken")
            register_profile("only", supports=["first.ok"], registry=registry)

        monkeypatch.setitem(sys.modules, "_fake_builtin_options", types.SimpleNamespace(register=register))
        monkeypatch.setattr(reg_mod, "_BUILTIN_MODULES", ("_fake_builtin_options",))
        monkeypatch.setattr(reg_mod, "REGISTRY", Registry())

        with pytest.raises(RuntimeError, match="broken"):
            reg_mod.options()
        with pytest.raises(RuntimeError, match="broken"):
            check_coverage()
        assert reg_mod.REGISTRY.option_entries == {}
        assert reg_mod.REGISTRY.builtins_registered is False

        assert [entry.key for entry in reg_mod.options()] == ["first.ok"]
        assert check_coverage() == []
        assert calls["count"] == 3

    def test_real_registry_fills_after_common_modules_are_reimported(self, monkeypatch):
        """``test_common_light_imports`` drops ``arctic_platform.common.*`` from ``sys.modules``, after which the
        built-in modules re-import against a fresh copy of this module. The real registry must still fill."""
        import arctic_platform.common as common

        for name in ("option_registry", "model_options", "profiles"):
            module_name = f"arctic_platform.common.{name}"
            if module_name in sys.modules:
                monkeypatch.delitem(sys.modules, module_name)
            if hasattr(common, name):
                monkeypatch.delattr(common, name)
        for module_name in [name for name in sys.modules if name.startswith("arctic_platform.common.profiles.")]:
            monkeypatch.delitem(sys.modules, module_name)
        monkeypatch.setattr(reg_mod, "REGISTRY", Registry(matrix_path=reg_mod.DEFAULT_MATRIX_PATH))

        assert check_coverage() == []
        assert len(reg_mod.options()) == len(EXPECTED_KEYS)
        assert {profile.name for profile in reg_mod.profiles()} == EXPECTED_PROFILES


class TestCommandLine(TestCasePlus):
    def _run(self, *args: str) -> list[str]:
        cmd = [sys.executable, "-m", "arctic_platform.common.option_registry", *args]
        return execute_subprocess_async(cmd, env=self.get_env(), quiet=True, echo=False).stdout

    def _run_expecting_failure(self, *args: str) -> subprocess.CompletedProcess:
        # execute_subprocess_async raises on a non-zero exit and drops stdout, where the problems are printed.
        cmd = [sys.executable, "-m", "arctic_platform.common.option_registry", *args]
        return subprocess.run(cmd, env=self.get_env(), capture_output=True, text=True, timeout=120)

    def test_list(self):
        out = self._run("list")
        assert len(out) == len(EXPECTED_KEYS)
        assert sorted(line.split()[0] for line in out) == sorted(EXPECTED_KEYS)

    def test_check_passes_on_real_data(self):
        out = self._run("check")
        assert len(out) == 1
        assert out[0].startswith("check passed: 20 options x 6 profiles = 120 cells, ")

    def test_matrix_write_reproduces_committed_file(self):
        tmp_dir = Path(self.get_auto_remove_tmp_dir())
        written = tmp_dir / "models" / "matrix.md"
        self._run("matrix", "--write", "--matrix-path", str(written))
        committed = self.repo_root_dir / "docs" / "models" / "matrix.md"
        assert written.read_text() == committed.read_text()
        assert "\n".join(self._run("matrix")) + "\n" == committed.read_text()

    def test_check_fails_on_stale_matrix(self):
        tmp_dir = Path(self.get_auto_remove_tmp_dir())
        stale = tmp_dir / "matrix.md"
        committed = self.repo_root_dir / "docs" / "models" / "matrix.md"
        stale.write_text(committed.read_text().replace("| yes |", "| no |", 1))
        result = self._run_expecting_failure("check", "--matrix-path", str(stale))
        assert result.returncode == 1, result
        assert f"{stale}: stale; regenerate it with" in result.stdout
        assert "check failed: 1 problem(s)" in result.stdout

    def test_check_fails_on_defective_registry(self):
        registry = _complete_registry()
        register_profile("empty", registry=registry)
        with self.assertRaises(SystemExit):
            reg_mod.main(["nope"], registry=registry)
        assert reg_mod.main(["check"], registry=registry) == 1
        assert reg_mod.main(["check"], registry=_complete_registry()) == 0
