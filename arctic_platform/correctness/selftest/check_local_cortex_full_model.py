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

"""Explicit local full-model mode uses the original offline catalog snapshot, never a reduced binding."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import torch
from safetensors.torch import save_file

from arctic_platform.client import CortexConfig
from arctic_platform.client import OnPremConfig
from arctic_platform.correctness.harness import cortex_fixture
from arctic_platform.correctness.harness import execution as cli
from arctic_platform.correctness.harness import runner
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.registry import TestResult as Result
from arctic_platform.correctness.harness.spec import TestSpec as Spec
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import CONFIG
from arctic_platform.correctness.selftest.check_cortex_optimizer_capture import SPEC


@pytest.fixture
def offline_model(tmp_path, monkeypatch):
    from transformers.utils import hub

    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
    root = tmp_path / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / "b968826d9c46dd6066d109eabc6255188de91218"
    root.mkdir(parents=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "attention_bias": False,
                "attention_dropout": 0.0,
                "bos_token_id": 151643,
                "eos_token_id": 151645,
                "head_dim": 128,
                "hidden_act": "silu",
                "hidden_size": 4096,
                "initializer_range": 0.02,
                "intermediate_size": 12288,
                "max_position_embeddings": 40960,
                "max_window_layers": 36,
                "model_type": "qwen3",
                "num_attention_heads": 32,
                "num_hidden_layers": 36,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-6,
                "rope_scaling": None,
                "rope_theta": 1000000,
                "sliding_window": None,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "transformers_version": "4.51.0",
                "use_cache": True,
                "use_sliding_window": False,
                "vocab_size": 151936,
            },
            indent=2,
        )
    )
    (root / "tokenizer_config.json").write_text("{}")
    (root / "tokenizer.json").write_text("{}")
    tensors = {f"model.layers.{i}.weight": torch.ones(1) for i in range(36)}
    save_file(tensors, root / "model-00001-of-00001.safetensors")
    index = {"weight_map": {name: "model-00001-of-00001.safetensors" for name in tensors}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("DSS_ALLOW_REMOTE_MODELS", "1")
    calls = []

    def cached_file(name, filename, **kwargs):
        assert name == cfg.model_name
        assert filename == "config.json"
        assert kwargs["local_files_only"] is True
        calls.append(name)
        return str(root / filename)

    monkeypatch.setattr(hub, "cached_file", cached_file)
    monkeypatch.setattr(requests.Session, "request", lambda *args, **kwargs: pytest.fail("no network"))
    return SimpleNamespace(root=root, cfg=cfg, spec=spec, calls=calls, index=index)


def _execute(case, tmp_path, **kwargs):
    return runner.execute(
        case.cfg,
        case.spec,
        [],
        vocab_size=128,
        select=["e2e-sft-train-validate"],
        workdir=tmp_path,
        client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
        local_gateway_full_model=True,
        **kwargs,
    )


def test_full_runner_preserves_catalog_target_and_spec_with_distinct_reference(offline_model, tmp_path, monkeypatch):
    original = CONFIG.read_bytes(), SPEC.read_bytes()
    observed = []

    def measure(ctx):
        observed.append(ctx)
        assert ctx.local_gateway_full_model is True
        assert ctx.local_full_model_path == str(offline_model.root)
        assert ctx.target_model_name is None
        assert ctx.cfg.model_name == "Qwen/Qwen3-8B"
        assert ctx.spec.model.cache_path == offline_model.spec.model.cache_path
        yield Result(test_id="e2e-sft-train-validate", config_id=ctx.config_id, outcome=Outcome.PASS)

    monkeypatch.setattr(runner.registered_tests()["e2e-sft-train-validate"], "fn", measure)
    monkeypatch.setattr(cortex_fixture, "load_fixture", lambda *args: pytest.fail("no fabricated publication"))
    monkeypatch.setattr(cortex_fixture, "load_local_fixture", lambda *args: pytest.fail("no reduced local binding"))
    results = _execute(offline_model, tmp_path)
    assert len(observed) == 1 and offline_model.calls == ["Qwen/Qwen3-8B"]
    assert [result.outcome for result in results] == [Outcome.PASS]
    assert (CONFIG.read_bytes(), SPEC.read_bytes()) == original


@pytest.mark.parametrize(
    "failure",
    [
        "reduced",
        "five-layers",
        "missing-shard",
        "index-escape",
        "symlink-escape",
        "missing-layer",
        "wrong-revision",
        "wrong-repository",
        "missing-tokenizer",
        "changed-rope",
    ],
)
def test_full_snapshot_invalidity_rejected_before_test(offline_model, tmp_path, monkeypatch, failure):
    root = offline_model.root
    if failure in {"reduced", "five-layers"}:
        config = json.loads((root / "config.json").read_text())
        config["num_hidden_layers"] = 4 if failure == "reduced" else 5
        (root / "config.json").write_text(json.dumps(config))
    elif failure == "missing-shard":
        (root / "model-00001-of-00001.safetensors").unlink()
    elif failure == "index-escape":
        offline_model.index["weight_map"]["model.layers.0.weight"] = "../outside.safetensors"
        (root / "model.safetensors.index.json").write_text(json.dumps(offline_model.index))
    elif failure == "symlink-escape":
        (root / "config.json").unlink()
        (root / "config.json").symlink_to("/tmp/nonexistent-full-model.json")
    elif failure in {"wrong-revision", "wrong-repository"}:
        from transformers.utils import hub

        wrong = (
            root.with_name("b" * 40) if failure == "wrong-revision" else root.parent.parent.with_name("models--other")
        )
        monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(wrong / "config.json"))
    elif failure == "missing-tokenizer":
        (root / "tokenizer.json").unlink()
    elif failure == "changed-rope":
        config = json.loads((root / "config.json").read_text())
        config["rope_theta"] = 10000
        (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    else:
        tensors = {f"model.layers.{i}.weight": torch.ones(1) for i in range(35)}
        save_file(tensors, root / "model-00001-of-00001.safetensors")
    monkeypatch.setattr(
        runner.registered_tests()["e2e-sft-train-validate"],
        "fn",
        lambda *args: pytest.fail("invalid full snapshot reached test"),
    )
    with pytest.raises((ValueError, OSError)):
        _execute(offline_model, tmp_path)


@pytest.mark.parametrize("variable", ["HF_HUB_OFFLINE", "DSS_ALLOW_REMOTE_MODELS"])
def test_full_mode_requires_operational_offline_contract(offline_model, tmp_path, monkeypatch, variable):
    monkeypatch.delenv(variable)
    with pytest.raises(ValueError, match="offline"):
        _execute(offline_model, tmp_path)
    assert offline_model.calls == []


@pytest.mark.parametrize("backend", [OnPremConfig(), CortexConfig(base_url="http://remote.example")])
def test_full_model_mode_is_only_for_verified_local_gateway(offline_model, tmp_path, backend):
    with pytest.raises(ValueError):
        runner.execute(
            offline_model.cfg,
            offline_model.spec,
            [],
            vocab_size=128,
            workdir=tmp_path,
            select=["e2e-sft-train-validate"],
            client_backend=backend,
            local_gateway_full_model=True,
        )
    assert offline_model.calls == []


@pytest.mark.parametrize(
    "selected", [None, ["single-step-optimizer"], ["e2e-sft-train-validate", "single-step-grads"]]
)
def test_full_mode_cannot_bypass_other_fixture_guards(offline_model, tmp_path, selected):
    with pytest.raises(ValueError):
        runner.execute(
            offline_model.cfg,
            offline_model.spec,
            [],
            vocab_size=128,
            workdir=tmp_path,
            select=selected,
            client_backend=CortexConfig(base_url="http://127.0.0.1:8003"),
            local_gateway_full_model=True,
        )
    assert offline_model.calls == []


def test_hosted_e2e_still_requires_reviewed_published_fixture(tmp_path):
    cfg, spec = load_config(CONFIG), Spec.read(SPEC)
    results = runner.execute(
        cfg,
        spec,
        [],
        vocab_size=128,
        select=["e2e-sft-train-validate"],
        client_backend=CortexConfig(base_url="http://remote.example"),
        workdir=tmp_path,
    )
    assert len(results) == 1 and results[0].outcome is Outcome.FAIL
    assert "published Cortex fixture" in results[0].summary


def test_cli_full_mode_explicit_selection_is_eligible(monkeypatch):
    args = SimpleNamespace(
        local_gateway_full_model=True,
        local_cortex_fixture=False,
        cortex_fixture=None,
        config=str(CONFIG),
        all_configs=False,
        test=["e2e-sft-train-validate"],
    )
    monkeypatch.setattr(cli, "_client_backend", lambda args: CortexConfig(base_url="http://127.0.0.1:8003"))

    def stop(*args, **kwargs):
        raise RuntimeError("eligible full-model dispatch")

    monkeypatch.setattr(cli, "_resolve_configs", stop)
    with pytest.raises(RuntimeError, match="eligible full-model"):
        cli.cmd_run(args)


@pytest.mark.parametrize(
    "changes",
    [
        {"test": ["single-step-optimizer"]},
        {"test": None},
        {"local_cortex_fixture": True},
        {"all_configs": True},
        {"cortex_fixture": Path("receipt.json")},
    ],
)
def test_cli_full_mode_invalid_combinations_rejected_before_dispatch(monkeypatch, changes):
    args = SimpleNamespace(
        local_gateway_full_model=True,
        local_cortex_fixture=False,
        cortex_fixture=None,
        config=str(CONFIG),
        all_configs=False,
        test=["e2e-sft-train-validate"],
    )
    vars(args).update(changes)
    monkeypatch.setattr(cli, "_client_backend", lambda args: CortexConfig(base_url="http://127.0.0.1:8003"))
    monkeypatch.setattr(cli, "_resolve_configs", lambda *args, **kwargs: pytest.fail("invalid full mode dispatched"))
    with pytest.raises(SystemExit):
        cli.cmd_run(args)


def test_full_model_vocab_uses_validated_local_snapshot_without_catalog_lookup(offline_model, monkeypatch):
    import transformers

    seen = []

    def config(path, **kwargs):
        assert path == offline_model.root
        assert kwargs["local_files_only"] is True
        seen.append(path)
        return SimpleNamespace(vocab_size=151936)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config)
    monkeypatch.setattr(cli, "_catalog_vocab_size", lambda *args: pytest.fail("full mode attempted catalog lookup"))
    assert (
        cli._run_vocab_size(
            offline_model.cfg,
            offline_model.spec,
            ["e2e-sft-train-validate"],
            CortexConfig(base_url="http://127.0.0.1:8003"),
            local_full_model_path=offline_model.root,
        )
        == 151936
    )
    assert seen == [offline_model.root]
