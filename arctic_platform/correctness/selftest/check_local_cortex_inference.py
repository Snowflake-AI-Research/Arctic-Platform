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

"""Local full-model save and sampling routing, with CPU-only mocked engines."""

import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig
from transformers.utils import hub

from arctic_platform.client import CortexConfig
from arctic_platform.correctness.checks import inference_checkpoint_loss as check
from arctic_platform.correctness.harness import gsm8k
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome

CONFIG = "arctic_platform/correctness/configs/qwen3-8b/h200/train-sft-full-4gpus-64k.config"
FULL_CONFIG = {
    "model_type": "qwen3",
    "architectures": ["Qwen3ForCausalLM"],
    "num_hidden_layers": 36,
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "tie_word_embeddings": False,
}
ROWS = [
    gsm8k.Example(question="validation0", prompt="p", completion=" a", input_ids=(1, 2, 3), n_prompt=1),
    gsm8k.Example(question="validation1", prompt="p", completion=" b", input_ids=(1, 4, 5), n_prompt=1),
]


class Tokenizer:
    special_tokens_map = {"eos_token": "eos"}

    def get_vocab(self):
        return {"p": 1, "a": 2, "b": 4, "eos": 5}

    def save_pretrained(self, path):
        (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": self.get_vocab()}}))
        (path / "tokenizer_config.json").write_text(json.dumps(self.special_tokens_map))


@pytest.fixture
def export_factory(tmp_path, monkeypatch):
    source = tmp_path / "source-index.json"
    index = {"metadata": {"total_size": 4}, "weight_map": {"model.weight": "model.safetensors"}}
    source.write_text(json.dumps(index))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: SimpleNamespace(**FULL_CONFIG))
    monkeypatch.setattr(hub, "cached_file", lambda *args, **kwargs: str(source))
    monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda *args: Tokenizer())

    def create(path):
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps(FULL_CONFIG))
        (path / "model.safetensors.index.json").write_text(json.dumps(index))
        save_file({"model.weight": torch.ones(2, dtype=torch.bfloat16)}, str(path / "model.safetensors"))
        return path

    return create


def context(tmp_path, backend=None):
    cfg = load_config(CONFIG)
    return SimpleNamespace(
        cfg=cfg,
        config_id=cfg.config_id,
        client_backend=backend or CortexConfig(base_url="http://127.0.0.1:8003"),
        attn_implementation="sdpa",
        workdir=tmp_path,
        target_model_name="must-not-use-reduced",
        spec=SimpleNamespace(model=SimpleNamespace(cache_path="/unavailable-reduced")),
    )


def test_local_sampling_uses_exact_export_same_backend_one_gpu_no_source(tmp_path, export_factory):
    ctx = context(tmp_path)
    weights = export_factory(tmp_path / "global_step10")
    config = check.local_cortex_sampling_config(ctx, ctx.client_backend, weights, 10240)
    payload = config.to_cortex()[0]
    assert config.backend == ctx.client_backend
    assert config.hardware == "H200"
    assert payload["model_name"] == str(weights)
    assert payload["inference_config"]["n_gpus"] == 1
    assert config.source_checkpoint_info is None
    assert "source_checkpoint_info" not in payload


@pytest.mark.parametrize(
    "backend",
    [
        CortexConfig(base_url="https://remote"),
        CortexConfig(base_url="http://127.0.0.1:8003", pat="test-not-a-credential"),
        CortexConfig(base_url="http://127.0.0.1:8003/invalid"),
    ],
)
def test_local_sampling_refuses_nonlocal_or_authenticated_backend(tmp_path, backend):
    with pytest.raises(ValueError, match="local Cortex"):
        check.local_cortex_sampling_config(context(tmp_path, backend), backend, tmp_path, 10240)


@pytest.mark.parametrize(
    "broken",
    [
        "reduced_config",
        "missing_config",
        "remote_code",
        "missing_index",
        "missing_tensor",
        "unsafe_shard",
        "missing_shard",
        "extra_shard",
        "symlink_shard",
        "wrong_dtype",
        "wrong_byte_count",
        "invalid_metadata",
        "truncated_weights",
    ],
)
def test_local_export_rejects_incomplete_or_mismatched_weights(tmp_path, export_factory, broken):
    weights = export_factory(tmp_path / "weights")
    if broken in {"reduced_config", "remote_code"}:
        config = dict(FULL_CONFIG)
        config.update({"num_hidden_layers": 4} if broken == "reduced_config" else {"auto_map": {"x": "y"}})
        (weights / "config.json").write_text(json.dumps(config))
    elif broken == "missing_config":
        (weights / "config.json").rename(weights / "missing-config")
    elif broken == "missing_index":
        (weights / "model.safetensors.index.json").rename(weights / "missing-index")
    elif broken in {"missing_tensor", "unsafe_shard", "wrong_byte_count"}:
        index = json.loads((weights / "model.safetensors.index.json").read_text())
        if broken == "missing_tensor":
            index["weight_map"] = {"different": "model.safetensors"}
        elif broken == "unsafe_shard":
            index["weight_map"]["model.weight"] = "../model.safetensors"
        else:
            index["metadata"]["total_size"] = 999
        (weights / "model.safetensors.index.json").write_text(json.dumps(index))
    elif broken == "missing_shard":
        (weights / "model.safetensors").rename(weights / "missing-shard")
    elif broken == "extra_shard":
        save_file({"x": torch.zeros(1)}, str(weights / "extra.safetensors"))
    elif broken == "symlink_shard":
        (weights / "model.safetensors").rename(tmp_path / "outside.safetensors")
        (weights / "model.safetensors").symlink_to(tmp_path / "outside.safetensors")
    elif broken == "wrong_dtype":
        save_file({"model.weight": torch.ones(2)}, str(weights / "model.safetensors"))
    elif broken == "invalid_metadata":
        (weights / "config.json").write_text("[]")
    elif broken == "truncated_weights":
        (weights / "model.safetensors").write_bytes(b"bad")
    with pytest.raises(Exception):
        check.validate_local_inference_export(context(tmp_path), weights)


@pytest.mark.parametrize("broken", ["path", "tag", "version", "wrong_step", "tokenizer"])
def test_save_receipt_and_tokenizer_fail_closed(tmp_path, export_factory, monkeypatch, broken):
    ctx = context(tmp_path)
    root = tmp_path / "save"
    root.mkdir()
    export_factory(root / "global_step10")
    monkeypatch.setattr(check, "correctness_workdir", lambda prefix: root)
    result = {"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10}
    if broken == "path":
        result["checkpoint_path"] = str(tmp_path)
    elif broken == "tag":
        result["checkpoint_tag"] = "cp_unsupported"
    elif broken == "version":
        result["version"] = 9
    elif broken == "wrong_step":
        result.update(checkpoint_tag="global_step9", version=9)
    elif broken == "tokenizer":

        class WrongTokenizer(Tokenizer):
            def get_vocab(self):
                return {"wrong": 999}

        monkeypatch.setattr(check.gsm8k, "load_tokenizer", lambda path: WrongTokenizer())
    client = SimpleNamespace(save_checkpoint=lambda **kwargs: result)
    with pytest.raises(RuntimeError):
        check.save_local_inference_checkpoint(ctx, ctx.client_backend, client, Tokenizer())
    assert root.exists()
    assert (root / "global_step10/model.safetensors").exists()


@pytest.mark.parametrize("malformed", [False, True])
def test_local_execution_preserves_recipe_and_closes_owned_clients(
    tmp_path,
    export_factory,
    monkeypatch,
    malformed,
):
    ctx = context(tmp_path)
    events = []
    saved = []

    def save(**kwargs):
        saved.append(kwargs)
        root = check.Path(kwargs["path"])
        export_factory(root / "global_step10")
        return {"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10}

    @contextmanager
    def gateway(directory, slots, **kwargs):
        assert kwargs["backend"] is ctx.client_backend
        assert slots == 5
        events.append("gateway-enter")
        try:
            yield "session"
        finally:
            events.append("gateway-exit")

    @contextmanager
    def training(session, payload):
        assert payload["model_name"] == "Qwen/Qwen3-8B"
        events.append("training-enter")
        try:
            yield SimpleNamespace(client=SimpleNamespace(save_checkpoint=save))
        finally:
            events.append("training-exit")

    @contextmanager
    def sampling(config):
        assert events[-1] == "training-exit"
        assert config.source_checkpoint_info is None
        assert config.to_cortex()[0]["model_name"].endswith("/global_step10")
        events.append("sampling-enter")
        results = [
            {
                "prompt_len": row.length,
                "prompt_logprobs": [
                    None,
                    *[{str(token): {"logprob": -1.0}} for token in row.input_ids[1:]],
                ],
            }
            for row in ROWS
        ]
        if malformed:
            results[0]["prompt_logprobs"][1] = {999: -1.0}
        try:
            yield SimpleNamespace(generate=lambda *args, **kwargs: results)
        finally:
            events.append("sampling-exit")

    def read_rows(path, count=None):
        return (
            [
                gsm8k.Example(question=f"train{i}", prompt="p", completion=" a", input_ids=(1, 2, 3), n_prompt=1)
                for i in range(count)
            ]
            if count is not None
            else ROWS
        )

    learning_rates = []
    monkeypatch.setattr(check, "gateway", gateway)
    monkeypatch.setattr(check, "running_job", training)
    monkeypatch.setattr(check, "CorrectnessClient", sampling)
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", lambda tokenizer, rows: rows)
    monkeypatch.setattr(check.gsm8k, "padding_token_id", lambda tokenizer: 0)
    monkeypatch.setattr(check.gsm8k, "select_by_token_budget", lambda *args: (ROWS, 3))
    monkeypatch.setattr(
        check,
        "fwd_bwd_step",
        lambda *args, **kwargs: (learning_rates.append(kwargs["learning_rate"]) or SimpleNamespace(avg_loss=2.0)),
    )
    monkeypatch.setattr(check, "forward_logprobs", lambda *args: torch.full((2, 3), -1.0))
    monkeypatch.setattr(check, "save_and_resolve_checkpoint_id", lambda *args: pytest.fail("no hosted catalog"))
    monkeypatch.setattr(check, "_onprem_sampling_and_reload", lambda *args: pytest.fail("no reload diagnostic"))
    result = check.run(ctx)[0]
    assert result.outcome is (TestOutcome.FAIL if malformed else TestOutcome.PASS)
    assert learning_rates == [check.configured_learning_rate(ctx.cfg)] * 10
    assert len(saved) == 1
    assert saved[0]["checkpoint_type"] == "weights-only"
    assert saved[0]["export_hf"] is False
    assert saved[0]["path"].startswith("/data-fast/")
    assert events == [
        "gateway-enter",
        "training-enter",
        "training-exit",
        "sampling-enter",
        "sampling-exit",
        "gateway-exit",
    ]
    if not malformed:
        assert result.metrics["train_steps"] == 10
        assert result.metrics["stated_criterion_abs"] == 1e-3
        assert result.metrics["learning_rate"] == check.configured_learning_rate(ctx.cfg)
