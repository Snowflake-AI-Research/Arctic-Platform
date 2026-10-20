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

"""The two RL checks' verdicts, judged from fabricated zone responses, and the requests they build."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from arctic_platform.common.utils.batch import unpack_batch
from arctic_platform.correctness.checks import rl_router_replay
from arctic_platform.correctness.checks import rl_weight_sync
from arctic_platform.correctness.harness import rl_driver
from arctic_platform.correctness.harness import router_replay_trace
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.rl_driver import rl_batch
from arctic_platform.correctness.harness.rl_driver import rl_gpu_count
from arctic_platform.correctness.harness.rl_driver import rl_sampling_payload
from arctic_platform.correctness.harness.spec import TestSpec as Spec
from arctic_platform.correctness.selftest.config_factory import native_config

from .check_rl_config import CONFIG
from .check_rl_config import SAMPLING_CONFIG
from .check_rl_config import SPEC
from .check_rl_config import TRAINING_CONFIG


def test_rl_shutdown_failure_is_unsafe(monkeypatch, tmp_path):
    from arctic_platform.correctness.harness.dss_driver import UnsafeCleanupError

    failure = RuntimeError("RL cancel failed")
    cfg = load_config(CONFIG)

    class Client:
        def __init__(self, config):
            pass

        def shutdown(self):
            raise failure

    monkeypatch.setattr(rl_driver, "ArcticClient", Client)
    payload = {"training_config": cfg.training, "model_name": cfg.model_name}
    with pytest.raises(UnsafeCleanupError) as caught:
        with rl_driver.running_rl_client(cfg, payload, cfg.model_name, tmp_path):
            pass
    assert caught.value.__cause__ is failure


def test_rl_execution_retains_traceback_for_unwrapped_check_failure(monkeypatch, tmp_path, capsys):
    from arctic_platform.correctness.harness import runner
    from arctic_platform.correctness.harness.registry import registered_tests

    cfg = load_config(CONFIG)
    spec = Spec.read(SPEC)

    def fail(ctx):
        raise RuntimeError("synthetic check failed before returning")

    monkeypatch.setattr(registered_tests()["rl-weight-sync"], "fn", fail)
    results = runner.execute_rl(cfg, spec, vocab_size=128, select=["rl-weight-sync"], workdir=tmp_path)
    assert len(results) == 1 and results[0].outcome is Outcome.FAIL
    assert results[0].error_type == "RuntimeError"
    assert "RuntimeError: synthetic check failed before returning" in capsys.readouterr().err


def _entry(numel, digest):
    return {"shape": [numel], "dtype": "bfloat16", "numel": numel, "sha256": digest}


def _sync(ranks=2):
    sent = {"a.weight": _entry(4, "aa"), "b.weight": _entry(6, "bb")}
    record = {
        "status": "ok",
        "received": {name: {**entry, "destinations": [name]} for name, entry in sent.items()},
        "destinations": {
            "a.weight": {"numel": 2, "uncovered": 0, "mismatched": 0, "equal": True},
            "b.weight": {"numel": 3, "uncovered": 0, "mismatched": 0, "equal": True},
        },
        "untouched_parameters": {},
        "non_parameter_destinations": [],
        "unverifiable_destinations": [],
    }
    return {
        "weight_format": "vllm",
        "targets": [
            {
                "send": {
                    "results": [
                        {"sent_manifest": sent, "trainer_parameters": {"count": 2, "numel": 10}},
                        {"sent_manifest": None},
                    ]
                },
                "recv": {
                    "recv": {
                        "workers": [
                            {"all_rank_weight_sync_verification": [copy.deepcopy(record) for _ in range(ranks)]}
                        ]
                    }
                },
            }
        ],
    }


def _rank(sync, index):
    return sync["targets"][0]["recv"]["recv"]["workers"][0]["all_rank_weight_sync_verification"][index]


def test_weight_sync_agreement_passes():
    failed, metrics = rl_weight_sync.judge(_sync())
    assert failed == []
    assert metrics["sent_elements"] == metrics["trainer_elements"] == 10
    assert metrics["sampler_ranks"] == 2


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda s: s["targets"][0]["send"]["results"][0]["trainer_parameters"].update(numel=12),
            "trainer elements sent",
        ),
        (lambda s: _rank(s, 1)["received"].pop("b.weight"), "sampler rank 1: b.weight not received"),
        (lambda s: _rank(s, 0)["received"]["a.weight"].update(sha256="xx"), "different bytes"),
        (lambda s: _rank(s, 0)["received"]["a.weight"].update(destinations=[]), "written to no parameter"),
        (lambda s: _rank(s, 0)["destinations"]["a.weight"].update(uncovered=1), "no tensor wrote"),
        (lambda s: _rank(s, 0)["destinations"]["b.weight"].update(mismatched=2), "differing from what was written"),
        (lambda s: _rank(s, 0)["untouched_parameters"].update({"visual.w": 5}), "written by no tensor"),
    ],
)
def test_weight_sync_disagreement_fails(mutate, expected):
    sync = _sync()
    mutate(sync)
    failed, _ = rl_weight_sync.judge(sync)
    assert any(expected in m.name for m in failed), [m.name for m in failed]


def test_weight_sync_without_sampler_verification_raises():
    sync = _sync()
    sync["targets"][0]["recv"] = {"recv": {"workers": [{}]}}
    with pytest.raises(RuntimeError, match=rl_weight_sync.VERIFY_ENV):
        rl_weight_sync.judge(sync)


# Two samples, two MoE layers, top-1. Sample 0 is prompt [5, 6] + answer [7]; sample 1 is prompt [8] + [9, 4].
# capture_len is one less than each row, so the last token of each row carries no routing.
def _samples():
    rows = [([5, 6, 7], 2, [[[1], [2]], [[3], [4]]]), ([8, 9, 4], 1, [[[5], [6]], [[7], [0]]])]
    samples = []
    for index, (tokens, prompt_len, routing) in enumerate(rows):
        marker = {
            "replay_id": f"rr1:{index}",
            "prompt_len": prompt_len,
            "generation_len": len(tokens) - prompt_len,
            "capture_len": len(routing),
            "routed_experts": routing,
        }
        samples.append({"tokens": tokens, "prompt_len": prompt_len, "marker": marker, "routing": routing})
    return samples


def _trace(samples):
    tokens = samples[0]["tokens"] + samples[1]["tokens"]
    positions = [0, 1, 2, 0, 1, 2]
    routed = copy.deepcopy(samples[0]["routing"] + [[[0], [0]]] + samples[1]["routing"] + [[[0], [0]]])
    router = [
        {"layer": layer, "phase": phase, "replayed": True, "indices": [row[layer] for row in routed]}
        for phase in ("forward", "recompute")
        for layer in (0, 1)
    ]
    return {
        "rank": 0,
        "received": {s["marker"]["replay_id"]: s["routing"] for s in samples},
        "calls": [
            {
                "padding": False,
                "input_ids": tokens,
                "position_ids": positions,
                "routed_experts": routed,
                "router": router,
            }
        ],
        "unattributed_router_calls": [],
    }


def test_router_replay_agreement_passes():
    samples = _samples()
    failed, metrics = rl_router_replay.judge(samples, [_trace(samples)], full_recompute=True)
    assert failed == []
    assert metrics["compared_positions"] == 4
    assert metrics["forward_invocations"] == metrics["recompute_invocations"] == 2


def test_router_trace_hooks_external_router_implementations():
    import torch

    class TokenChoiceTopKRouter(torch.nn.Module):
        def forward(self, hidden, *, routed_experts=None):
            indices = routed_experts if routed_experts is not None else torch.zeros_like(hidden, dtype=torch.long)
            return hidden, indices, torch.ones(1)

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.router = TokenChoiceTopKRouter()

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = torch.nn.ModuleList([Layer()])

    model = Model()
    trace = router_replay_trace.start(model)
    trace.begin_call(
        {
            "input_ids": torch.tensor([[1]]),
            "position_ids": torch.tensor([[0]]),
            "routed_experts": torch.tensor([[[0]]]),
        },
        padding=False,
    )
    model.layers[0].router(torch.zeros(1, 1), routed_experts=torch.tensor([[3]]))
    exported = router_replay_trace.finish(0)
    assert exported["calls"][0]["router"] == [{"layer": 0, "phase": "forward", "replayed": True, "indices": [[3]]}]

    model.layers[0].router(torch.zeros(1, 1), routed_experts=torch.tensor([[2]]))
    assert len(exported["calls"][0]["router"]) == 1


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda t: t["received"].update({"rr1:1": [[[5], [6]], [[7], [1]]]}), "routing received differs"),
        (lambda t: t["calls"][0]["routed_experts"][1].__setitem__(0, [9]), "routed_experts differ"),
        (lambda t: t["calls"][0]["router"][3].update(indices=[[9]] * 6), "layer 1 recompute tokens"),
        (lambda t: t["calls"][0]["router"][0].update(replayed=False), "computed its own routing"),
        (lambda t: t["calls"][0]["router"].pop(2), "layer 0 has no recompute"),
        (lambda t: t["unattributed_router_calls"].append({"layer": 0}), "outside a traced call"),
        (lambda t: t["received"].pop("rr1:0"), "sample 0: no training rank received"),
    ],
)
def test_router_replay_disagreement_fails(mutate, expected):
    samples = _samples()
    trace = _trace(samples)
    mutate(trace)
    failed, _ = rl_router_replay.judge(samples, [trace], full_recompute=True)
    assert any(expected in m.name for m in failed), [m.name for m in failed]


def test_router_replay_recompute_not_required_without_full_checkpointing():
    samples = _samples()
    trace = _trace(samples)
    trace["calls"][0]["router"] = [r for r in trace["calls"][0]["router"] if r["phase"] == "forward"]
    failed, _ = rl_router_replay.judge(samples, [trace], full_recompute=False)
    assert failed == []
    assert rl_router_replay.expects_full_recompute({"ac_config": {"mode": "full", "freq": 1}})
    assert not rl_router_replay.expects_full_recompute({})


def test_rl_batch_scores_only_answer_predictions():
    batch = rl_batch([[5, 6, 7, 8], [9, 4]], [2, 1], pad_id=0)
    assert batch["kwargs"]["input_ids"].tolist() == [[5, 6, 7, 8], [9, 4, 0, 0]]
    assert batch["kwargs"]["attention_mask"].tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    assert batch["context"]["loss_mask"].tolist() == [[0, 1, 1, 0], [1, 0, 0, 0]]


def test_rl_batch_uses_native_scalar_temperature_boundary():
    request = rl_batch([[5, 6, 7, 8], [9, 4]], [2, 1], pad_id=0)
    _, batch, meta, processing = unpack_batch(request)
    assert "temperature" not in request["kwargs"]
    assert request["context"]["temperature"] == meta["temperature"] == 1.0
    assert batch["input_ids"].tolist() == [[5, 6, 7, 8], [9, 4, 0, 0]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1, 0], [1, 0, 0, 0]]
    assert batch["advantages"].eq(1).all()
    assert processing == {
        "post": ["compute_logprobs"],
        "loss_fn": "grpo",
        "config": {"loss_agg_mode": "token-mean"},
    }


@pytest.mark.parametrize("check", [rl_weight_sync, rl_router_replay], ids=["weight-sync", "router-replay"])
@pytest.mark.parametrize("local_model", [False, True], ids=["frozen-catalog-model", "configured-local-model"])
def test_rl_checks_use_configured_model_for_tokenizer_and_both_zones(tmp_path, monkeypatch, check, local_model):
    cfg = load_config(CONFIG)
    if local_model:
        model_name = "/data-fast/base-models/full-configured-model"
        sub_job = {**cfg.sub_job, "model_name": model_name}
        cfg = replace(
            cfg,
            sub_job=sub_job,
            sub_jobs=[{**job, "model_name": model_name} for job in cfg.sub_jobs],
            native=cfg.native.model_copy(update={"model_name": model_name}, deep=True),
        )
    spec = Spec.read(SPEC)
    original_native = cfg.native.model_dump()
    original_training = copy.deepcopy(cfg.training)
    original_sampling = copy.deepcopy(cfg.sub_jobs[1])
    ctx = SimpleNamespace(
        cfg=cfg,
        spec=spec,
        config_id=cfg.config_id,
        workdir=tmp_path,
        attn_implementation=cfg.attention_implementation,
        target_model_name="published-reduced-reference",
    )
    assert cfg.model_name != spec.model.cache_path
    tokenizer_paths, configs, batches, generations, rates, closed = [], [], [], [], [], []
    rows_per_step = check.configured_rows_per_step(cfg)
    examples = [
        check.gsm8k.Example(
            question=str(i),
            prompt="prompt",
            completion="answer",
            input_ids=(3 * i + 5, 3 * i + 6, 3 * i + 7),
            n_prompt=2,
        )
        for i in range(rows_per_step)
    ]
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    generated_samples = []

    def load_tokenizer(path):
        tokenizer_paths.append(path)
        return tokenizer

    def read_rows(path, *, count):
        assert path == check.TRAIN_SPLIT
        assert count == rows_per_step
        return [("question", "answer")] * count

    def tokenize_examples(actual_tokenizer, rows):
        assert actual_tokenizer is tokenizer
        assert len(rows) == rows_per_step
        return examples

    class Client:
        jobs = SimpleNamespace(sampling="native-sampling-job")

        def __init__(self, config):
            configs.append(config)

        def generate(self, prompts, params):
            generations.append((prompts, params))
            results = []
            for example, prompt, parameters in zip(examples, prompts, params, strict=True):
                assert prompt == list(example.input_ids[: example.n_prompt])
                assert parameters["max_tokens"] == example.answer_tokens
                assert parameters["dss_return_back_router_info"] is True
                marker = {
                    "sample_id": parameters["dss_sample_id"],
                    "replay_id": parameters["dss_router_replay_id"],
                    "prompt_len": len(prompt),
                    "generation_len": 1,
                    "capture_len": 2,
                    "routed_experts": [[[1], [2]], [[3], [4]]],
                }
                result = {"token_ids": list(example.input_ids[example.n_prompt :]), "router_replay": marker}
                results.append(result)
                generated_samples.append(rl_router_replay.sample_from_result(prompt, result))
            return results

        def fwd_bwd(self, batch, *, router_replay=None):
            batches.append((batch, router_replay))
            traces = [_trace(generated_samples[i : i + 2]) for i in range(0, len(generated_samples), 2)]
            return {"avg_loss": 1.0, "router_replay_trace": traces}

        def step(self, learning_rate):
            rates.append(learning_rate)
            return {}

        def sync_weights(self):
            assert check is rl_weight_sync
            return _sync()

        def shutdown(self):
            closed.append(True)

    monkeypatch.setattr(check.gsm8k, "load_tokenizer", load_tokenizer)
    monkeypatch.setattr(check.gsm8k, "read_rows", read_rows)
    monkeypatch.setattr(check.gsm8k, "tokenize_examples", tokenize_examples)
    monkeypatch.setattr(rl_driver, "ArcticClient", Client)
    monkeypatch.setenv(rl_weight_sync.VERIFY_ENV, "test-original")
    result = check._run(ctx)

    assert result.outcome is Outcome.PASS
    assert tokenizer_paths == [cfg.model_name]
    assert len(configs) == len(batches) == len(closed) == 1
    native = configs[0]
    assert native.model_name == cfg.model_name
    assert native.to_onprem("training")["model_name"] == native.to_onprem("sampling")["model_name"] == cfg.model_name
    assert native.sampling.model_dump() == cfg.native.sampling.model_dump()
    assert native.training_gpus == cfg.native.training_gpus == 8
    assert native.sampling_gpus == cfg.native.sampling_gpus == 8
    assert native.max_seq_len == cfg.native.max_seq_len == 32768
    assert native.training.ds_config == cfg.native.training.ds_config
    assert native.training.ds_worker_config["model_provider"] == "prime_rl"
    assert native.training.ds_worker_config["attn_implementation"] == cfg.attention_implementation
    assert native.training.checkpoint_path == str(tmp_path / check.TEST_ID / "gateway" / "checkpoints")
    batch, replay = batches[0]
    assert batch["kwargs"]["input_ids"].tolist() == [list(example.input_ids) for example in examples]
    if check is rl_weight_sync:
        assert native.training.ds_worker_config["debug"]["weight_sync_manifest"] is True
        assert replay is None
        assert rates == [check.configured_learning_rate(cfg)]
        assert generations == []
    else:
        assert native.training.ds_worker_config["debug"]["router_replay_trace"] is True
        assert replay == {"sampling_job_id": "native-sampling-job", "identity_version": 1}
        assert len(generations) == 1
        assert rates == []
        assert batch["kwargs"]["rr_sample_ids"] == [sample["marker"]["replay_id"] for sample in generated_samples]
        assert batch["kwargs"]["rr_discard"] is True
    assert cfg.native.model_dump() == original_native
    assert cfg.training == original_training
    assert cfg.sub_jobs[1] == original_sampling


def test_sampling_payload_keeps_the_config_and_swaps_only_the_model(tmp_path):
    path = tmp_path / "job.config"
    path.write_text(json.dumps(native_config(TRAINING_CONFIG, sampling=SAMPLING_CONFIG)))
    cfg = load_config(path)
    payload = rl_sampling_payload(cfg, "/models/reduced")
    assert payload == {
        "job_type": "sampling",
        "model_name": "/models/reduced",
        "dtype": "bfloat16",
        "seed": 42,
        "inference_config": {"n_gpus": 8, "max_seq_len": 4096, "vllm_config": {}},
    }
    assert rl_gpu_count(cfg) == 16
