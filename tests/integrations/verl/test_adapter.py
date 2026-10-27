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

"""API-surface tests for ``ArcticRLClientWrapper``.

Full integration is covered by the Golden Run E2E smoke; here we pin
the public shape (inheritance, ``from_config`` signature,
``destroy`` idempotency, ``requires_single_forwarder``) without
instantiating the heavy Arctic RL / Ray runtime.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

# adapter.py imports omegaconf + transformers at module top level.
pytest.importorskip("omegaconf")
pytest.importorskip("transformers")


def _adapter(verl_stub):
    """Import the adapter against the shared verl + arctic_platform.rl stubs."""
    return importlib.import_module("arctic_platform.integrations.verl.adapter")


def test_adapter_inherits_from_verl_remote_backend(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    from verl.remote_backend.base import RemoteBackend

    assert issubclass(
        adapter.ArcticRLClientWrapper, RemoteBackend
    ), "ArcticRLClientWrapper must subclass verl's RemoteBackend so verl's trainer can dispatch through the ABC."


def test_from_config_signature(verl_stub) -> None:
    """``from_config`` is the sole public constructor -- its signature is API surface."""
    adapter = _adapter(verl_stub)

    sig = inspect.signature(adapter.ArcticRLClientWrapper.from_config)
    params = sig.parameters

    assert "main_config" in params
    assert "handle" in params
    assert params["handle"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["handle"].default is None


def test_requires_single_forwarder_returns_true(verl_stub) -> None:
    """Arctic owns its own training/sampling parallelism; the verl worker group
    must be a single forwarder. This is the invariant that
    ``RemoteBackendTrainer._enforce_single_forwarder_if_required`` reads.
    """
    adapter = _adapter(verl_stub)

    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    assert wrapper.requires_single_forwarder() is True


def test_destroy_is_idempotent(verl_stub) -> None:
    """Second ``destroy()`` call must not raise or re-shutdown the client.

    verl's trainer may call ``destroy()`` on multiple exception paths;
    a second call has to be safe or the trainer leaks async errors on exit.
    """
    adapter = _adapter(verl_stub)

    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)

    shutdown_calls = 0

    class _FakeClient:
        async def shutdown(self):
            nonlocal shutdown_calls
            shutdown_calls += 1

    wrapper._client = _FakeClient()

    async def _run():
        await wrapper.destroy()
        await wrapper.destroy()

    asyncio.run(_run())
    assert shutdown_calls == 1, "destroy() must be idempotent; second call should be a no-op."
    assert wrapper._client is None


def _balance_wrapper(adapter, *, balance_batch: bool, zorro: bool):
    """A wrapper with the fields the balance stamp reads, and no Ray client."""
    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    wrapper.use_cortex = False
    wrapper.balance_batch = balance_batch
    wrapper.zorro_train_enable = zorro
    wrapper.zorro_train_max_rollouts = 2
    wrapper.drop_position_ids = True
    wrapper._max_token_len_per_gpu = 128
    wrapper.logits_optimization = "none"
    wrapper.logits_optimization_peak_mem_size_in_gib = 4
    wrapper.logits_compute_in_fp32 = False
    wrapper.logits_compute_from_fp32_inputs = False
    actor = SimpleNamespace(ppo_mini_batch_size=2, ppo_epochs=1)
    wrapper.config = SimpleNamespace(
        data=SimpleNamespace(train_batch_size=2),
        actor_rollout_ref=SimpleNamespace(actor=actor),
    )
    return wrapper


def _two_row_batch(*, same_prompt: bool, dense_prompts: bool) -> dict:
    prompt_a = [7, 7]
    prompt_b = [7, 7] if same_prompt else [8, 8]
    response_a = [1, 2]
    response_b = [3, 4]
    if dense_prompts:
        prompts = torch.tensor([prompt_a, prompt_b], dtype=torch.long)
        responses = torch.tensor([response_a, response_b], dtype=torch.long)
    else:
        prompts = torch.nested.nested_tensor(
            [torch.tensor(prompt_a, dtype=torch.long), torch.tensor(prompt_b, dtype=torch.long)],
            layout=torch.jagged,
        )
        responses = torch.nested.nested_tensor(
            [torch.tensor(response_a, dtype=torch.long), torch.tensor(response_b, dtype=torch.long)],
            layout=torch.jagged,
        )
    return {
        "input_ids": torch.nested.nested_tensor(
            [
                torch.tensor(prompt_a + response_a, dtype=torch.long),
                torch.tensor(prompt_b + response_b, dtype=torch.long),
            ],
            layout=torch.jagged,
        ),
        "prompts": prompts,
        "responses": responses,
        "attention_mask": torch.ones((2, 4), dtype=torch.long),
        "response_mask": torch.ones((2, 2), dtype=torch.long),
        "old_log_probs": torch.zeros((2, 2)),
        "advantages": torch.zeros((2, 2)),
        "temperature": 1.0,
    }


class _RecordingClient:
    def __init__(self):
        self.payloads = []

    async def fwd_no_grad(self, payload, reference_model=False):
        self.payloads.append(payload)
        return {"batch": {"logprobs": torch.zeros((2, 4))}, "metrics": {}}

    async def fwd_bwd(self, payload):
        self.payloads.append(payload)
        return {"metrics": {"loss": 1.0}}

    async def step(self):
        return {"metrics": {}}


def test_logprob_stamps_balance_off_like_skyrl(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=True)
    client = _RecordingClient()
    wrapper._client = client

    async def _run():
        await wrapper.compute_log_prob(
            _two_row_batch(same_prompt=True, dense_prompts=False),
            ref=False,
            calculate_entropy=False,
            rollout_n=2,
            temperature=1.0,
            pad_token_id=0,
        )

    asyncio.run(_run())
    assert client.payloads[0]["meta"]["balance_batch"] is False
    assert "prompt_group_ids" not in client.payloads[0]["meta"]
    sent_prompts = client.payloads[0]["batch"]["prompts"]
    assert sent_prompts.ndim == 2 and not sent_prompts.is_nested
    assert torch.equal(sent_prompts, torch.tensor([[7, 7], [7, 7]]))


def test_update_actor_stamps_balance_and_checks_prompt_blocks(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=True)
    client = _RecordingClient()
    wrapper._client = client
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)

    async def _run(batch):
        await wrapper.update_actor(batch, actor_config=actor_config, pad_token_id=0, rollout_n=2, temperature=1.0)

    asyncio.run(_run(_two_row_batch(same_prompt=True, dense_prompts=True)))
    assert client.payloads[0]["meta"]["balance_batch"] is True
    assert client.payloads[0]["meta"]["prompt_group_ids"].tolist() == [0, 0]

    client.payloads.clear()
    asyncio.run(_run(_two_row_batch(same_prompt=False, dense_prompts=True)))
    assert client.payloads[0]["meta"]["balance_batch"] is True
    assert "prompt_group_ids" not in client.payloads[0]["meta"]


def test_update_actor_densifies_jagged_prompts(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=True)
    client = _RecordingClient()
    wrapper._client = client
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)
    batch = _two_row_batch(same_prompt=True, dense_prompts=False)

    async def _run():
        await wrapper.update_actor(batch, actor_config=actor_config, pad_token_id=0, rollout_n=2, temperature=1.0)

    asyncio.run(_run())
    sent_prompts = client.payloads[0]["batch"]["prompts"]
    assert sent_prompts.ndim == 2 and not sent_prompts.is_nested
    assert torch.equal(sent_prompts, torch.tensor([[7, 7], [7, 7]]))
    assert client.payloads[0]["meta"]["prompt_group_ids"].tolist() == [0, 0]


def test_update_actor_balance_off_skips_the_prompt_block_check(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _balance_wrapper(adapter, balance_batch=False, zorro=True)
    client = _RecordingClient()
    wrapper._client = client
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)

    async def _run():
        await wrapper.update_actor(
            _two_row_batch(same_prompt=False, dense_prompts=True),
            actor_config=actor_config,
            pad_token_id=0,
            rollout_n=2,
            temperature=1.0,
        )

    asyncio.run(_run())
    assert client.payloads[0]["meta"]["balance_batch"] is False
    assert "prompt_group_ids" not in client.payloads[0]["meta"]


def _options_wrapper(
    adapter,
    *,
    zorro: bool,
    micro=None,
    pack: bool = True,
    balance_batch: bool = True,
    verl_balance_batch: bool = False,
    logits_optimization: str = "none",
):
    """A wrapper with the config the train-option checks and ds config builders read."""
    from omegaconf import OmegaConf

    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    wrapper.use_cortex = False
    wrapper.zorro_train_enable = zorro
    wrapper.balance_batch = balance_batch
    wrapper.pack = pack
    wrapper.use_liger = False
    wrapper.logits_optimization = logits_optimization
    wrapper.logits_optimization_peak_mem_size_in_gib = 4
    wrapper.logits_compute_in_fp32 = False
    wrapper.logits_compute_from_fp32_inputs = False
    wrapper.config = OmegaConf.create(
        {
            "trainer": {"balance_batch": verl_balance_batch},
            "data": {"train_batch_size": 128, "max_response_length": 4096},
            "actor_rollout_ref": {
                "actor": {
                    "ppo_mini_batch_size": 128,
                    "ppo_micro_batch_size_per_gpu": micro,
                    "ppo_max_token_len_per_gpu": 102400,
                    "fsdp_config": {},
                },
                "rollout": {"n": 16, "temperature": 1.0},
                "model": {
                    "override_config": {"attn_implementation": "flash_attention_2"},
                    "enable_gradient_checkpointing": True,
                },
            },
        }
    )
    wrapper._backend_config = OmegaConf.create({"train": {"deepspeed": {"zero_optimization": {"stage": 3}}}})
    return wrapper


@pytest.mark.parametrize("zorro,micro,accum", [(False, 1, 512), (False, None, 512), (True, 16, 32), (True, 32, 16)])
def test_ds_config_uses_the_configured_micro_batch(verl_stub, zorro, micro, accum) -> None:
    adapter = _adapter(verl_stub)
    ds = _options_wrapper(adapter, zorro=zorro, micro=micro)._create_ds_config(4)
    assert ds["train_micro_batch_size_per_gpu"] == (micro or 1)
    assert ds["train_batch_size"] == 2048
    assert ds["gradient_accumulation_steps"] == accum


@pytest.mark.parametrize("micro", [None, 4, 8, 24])
def test_ds_config_rejects_a_zorro_micro_batch_that_splits_prompt_groups(verl_stub, micro) -> None:
    adapter = _adapter(verl_stub)
    with pytest.raises(ValueError, match="whole prompt groups"):
        _options_wrapper(adapter, zorro=True, micro=micro)._create_ds_config(4)


def test_ds_config_rejects_a_micro_batch_that_does_not_divide_the_step(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    with pytest.raises(ValueError, match="must be divisible"):
        _options_wrapper(adapter, zorro=False, micro=3)._create_ds_config(4)


@pytest.mark.parametrize("zorro", [False, True])
def test_ds_worker_config_carries_pack_false_only(verl_stub, zorro) -> None:
    adapter = _adapter(verl_stub)
    packed = _options_wrapper(adapter, zorro=zorro, micro=16)._create_ds_worker_config()
    unpacked = _options_wrapper(adapter, zorro=zorro, micro=16, pack=False)._create_ds_worker_config()
    assert "pack" not in packed
    assert unpacked["pack"] is False
    assert bool(unpacked.get("zorro_train_enable")) is zorro


def test_arrival_order_rejects_verl_driver_balance(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    _options_wrapper(adapter, zorro=True, balance_batch=True, verl_balance_batch=True)._check_train_options()
    _options_wrapper(adapter, zorro=True, balance_batch=False, verl_balance_batch=False)._check_train_options()
    with pytest.raises(ValueError, match="trainer.balance_batch=False"):
        _options_wrapper(adapter, zorro=True, balance_batch=False, verl_balance_batch=True)._check_train_options()


def test_memory_logits_require_zorro(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    _options_wrapper(adapter, zorro=True, logits_optimization="memory")._check_train_options()
    _options_wrapper(adapter, zorro=False, logits_optimization="compute")._check_train_options()
    with pytest.raises(ValueError, match="requires remote_backend.train.zorro_train.enable"):
        _options_wrapper(adapter, zorro=False, logits_optimization="memory")._check_train_options()


@pytest.mark.parametrize("protocol,cortex", [(None, False), ("ray", False), ("cortex", True), ("http", True)])
def test_comms_protocol_selects_the_backend(verl_stub, protocol, cortex) -> None:
    adapter = _adapter(verl_stub)
    assert adapter._uses_cortex({"protocol": protocol}) is cortex


def test_unknown_comms_protocol_is_refused(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    with pytest.raises(ValueError, match="'ray' \\(on-prem\\) or 'cortex'"):
        adapter._uses_cortex({"protocol": "grpc"})


def _cortex_options_wrapper(adapter, **overrides):
    from omegaconf import OmegaConf

    backend = {
        "colocate": False,
        "log_prob_gpus": 0,
        "weight_sync": {"cuda_ipc": False},
        "rollout": {"zorro_inference": {"enable": False}, "speculative_decoding": {"model": ""}},
    }
    actor = {"use_kl_loss": False}
    algorithm = {"use_kl_in_reward": False}
    for key, value in overrides.items():
        target = {
            "use_kl_loss": actor,
            "entropy_coeff": actor,
            "use_kl_in_reward": algorithm,
            "rollout_correction": algorithm,
        }.get(key, backend)
        if key == "cuda_ipc":
            backend["weight_sync"]["cuda_ipc"] = value
        elif key == "zorro_inference":
            backend["rollout"]["zorro_inference"]["enable"] = value
        else:
            target[key] = value
    wrapper = adapter.ArcticRLClientWrapper.__new__(adapter.ArcticRLClientWrapper)
    wrapper._backend_config = OmegaConf.create(backend)
    wrapper.config = OmegaConf.create({"actor_rollout_ref": {"actor": actor}, "algorithm": algorithm})
    return wrapper


def test_cortex_accepts_a_disabled_rollout_correction_block(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    _cortex_options_wrapper(
        adapter,
        rollout_correction={
            "rollout_is": None,
            "rollout_rs": None,
            "bypass_mode": False,
            "rollout_is_threshold": 2.0,
        },
    )._check_cortex_options()


@pytest.mark.parametrize(
    "override,match",
    [
        ({"colocate": True}, "colocate=False"),
        ({"cuda_ipc": True}, "cuda_ipc=False"),
        ({"log_prob_gpus": 1}, "log_prob_gpus=0"),
        ({"use_kl_loss": True}, "use_kl_loss=False"),
        ({"use_kl_in_reward": True}, "use_kl_in_reward=False"),
        ({"zorro_inference": True}, "zorro_inference.enable=False"),
        ({"rollout_correction": {"rollout_is": "token"}}, "rollout_is=null"),
        ({"rollout_correction": {"bypass_mode": True}}, "bypass_mode=False"),
        ({"entropy_coeff": 0.001}, "entropy_coeff=0"),
    ],
)
def test_cortex_refuses_options_it_cannot_serve(verl_stub, override, match) -> None:
    adapter = _adapter(verl_stub)
    _cortex_options_wrapper(adapter)._check_cortex_options()
    with pytest.raises(ValueError, match=match):
        _cortex_options_wrapper(adapter, **override)._check_cortex_options()


class _CortexRecordingClient:
    """Answers like the Cortex transport: a bare ``logprobs`` forward, ``avg_loss`` fwd_bwd, a flat step."""

    def __init__(self, *, ds_worker_config, logprobs=None):
        self.config = SimpleNamespace(training=SimpleNamespace(ds_worker_config=ds_worker_config))
        self.logprobs = logprobs
        self.forwards = []
        self.fwd_bwds = []
        self.generates = []

    async def fwd_no_grad(self, payload, reference_model=False):
        self.forwards.append(payload)
        logprobs = self.logprobs
        if logprobs is None:
            logprobs = payload["kwargs"]["input_ids"].to(torch.float32)
        return {"job_id": "job.training", "logprobs": logprobs}

    async def fwd_bwd(self, payload):
        self.fwd_bwds.append(payload)
        return {"job_id": "job.training", "avg_loss": 0.5, "metrics": {"actor/pg_loss": 0.25}}

    async def step(self):
        return {
            "job_id": "job.training",
            "global_steps": 1,
            "last_lr": 1e-6,
            "grad_norm": 0.75,
            "update_successful": 1.0,
            "peak_memory": {"allocated": 1, "reserved": 2},
            "gradient_sample_param_name": "model.layers.0.mlp",
        }

    async def generate(self, prompts, sampling_params=None):
        self.generates.append((prompts, sampling_params))
        return [{"token_ids": [5, 6], "finish_reason": "stop"}]


def _cortex_wrapper(adapter, *, zorro=False, pack=False, logprobs=None):
    wrapper = _balance_wrapper(adapter, balance_batch=True, zorro=zorro)
    wrapper.use_cortex = True
    ds_worker_config = {"zorro_train_enable": True} if zorro else {}
    if not pack:
        ds_worker_config["pack"] = False
    wrapper._client = _CortexRecordingClient(ds_worker_config=ds_worker_config, logprobs=logprobs)
    return wrapper


def _padded_batch() -> dict:
    """Two rows whose client layout has a left-padded prompt and a right-padded response."""
    prompts = [[7], [8, 8]]
    responses = [[1, 2], [3]]

    def jagged(rows):
        return torch.nested.nested_tensor([torch.tensor(r, dtype=torch.long) for r in rows], layout=torch.jagged)

    return {
        "input_ids": jagged([p + r for p, r in zip(prompts, responses)]),
        "prompts": jagged(prompts),
        "responses": jagged(responses),
        "attention_mask": torch.tensor([[0, 1, 1, 1], [1, 1, 1, 0]]),
        "temperature": torch.tensor(1.0),
    }


def _log_prob(wrapper, batch, *, ref=False):
    return asyncio.run(
        wrapper.compute_log_prob(batch, ref=ref, calculate_entropy=True, rollout_n=2, temperature=1.0, pad_token_id=0)
    )


def test_cortex_log_prob_keeps_the_client_layout_when_unpacked(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter, pack=False)
    out = _log_prob(wrapper, _padded_batch())

    sent = wrapper._client.forwards[0]
    assert set(sent) == {"args", "kwargs", "context", "processing"}
    assert sent["kwargs"]["input_ids"].tolist() == [[0, 7, 1, 2], [8, 8, 3, 0]]
    assert sent["context"]["temperature"] == 1.0 and not torch.is_tensor(sent["context"]["temperature"])
    assert sent["processing"] == {"post": ["compute_logprobs"], "loss_fn": None}
    assert out["model_output"]["log_probs"].tolist() == [[0.0, 7.0, 1.0, 2.0], [8.0, 8.0, 3.0, 0.0]]
    assert "entropy" not in out["model_output"]


def test_cortex_log_prob_moves_packed_rows_back_to_the_client_layout(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter, pack=True)
    out = _log_prob(wrapper, _padded_batch())

    # The packed executor gets left-aligned rows and answers in that frame.
    assert wrapper._client.forwards[0]["kwargs"]["input_ids"].tolist()[0][:3] == [7, 1, 2]
    assert out["model_output"]["log_probs"].tolist() == [[0.0, 7.0, 1.0, 2.0], [8.0, 8.0, 3.0, 0.0]]


def test_cortex_log_prob_refuses_a_wrong_shape(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter, logprobs=torch.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="client layout"):
        _log_prob(wrapper, _padded_batch())


def test_cortex_refuses_reference_log_probs(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter)
    with pytest.raises(NotImplementedError, match="reference-model"):
        _log_prob(wrapper, _padded_batch(), ref=True)
    assert wrapper._client.forwards == []


def _update(wrapper, batch):
    actor_config = SimpleNamespace(calculate_entropy=False, use_kl_loss=False, policy_loss=None)
    return asyncio.run(
        wrapper.update_actor(batch, actor_config=actor_config, pad_token_id=0, rollout_n=2, temperature=1.0)
    )


def _update_batch() -> dict:
    batch = _two_row_batch(same_prompt=True, dense_prompts=True)
    batch["old_log_probs"] = torch.tensor([[-0.1, -0.2], [-0.3, -0.4]])
    batch["advantages"] = torch.tensor([[1.0, 1.0], [-1.0, -1.0]])
    batch["temperature"] = torch.tensor(1.0)
    return batch


@pytest.mark.parametrize("zorro", [False, True])
def test_cortex_update_sends_unpacked_verl_grpo_and_merges_metrics(verl_stub, zorro) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter, zorro=zorro, pack=False)
    out = _update(wrapper, _update_batch())

    sent = wrapper._client.fwd_bwds[0]
    assert sent["processing"]["loss_fn"] == "verl_grpo"
    if not zorro:
        assert sent["processing"]["post"][-1] == "compute_logprobs"
    context = sent["context"]
    # Token-aligned, in the client columns: the responses sit at columns 2 and 3.
    assert torch.allclose(context["old_log_probs"], torch.tensor([[0, 0, -0.1, -0.2], [0, 0, -0.3, -0.4]]))
    assert context["advantages"].tolist() == [[0, 0, 1, 1], [0, 0, -1, -1]]
    assert context["batch_num_tokens"] == 4 and isinstance(context["batch_num_tokens"], int)
    assert context["temperature"] == 1.0
    assert out["loss"] == 0.5
    assert out["metrics"]["grad_norm"] == 0.75
    assert out["metrics"]["actor/pg_loss"] == 0.25
    assert out["metrics"]["lr"] == 1e-6
    assert "global_steps" not in out["metrics"]
    assert "peak_memory" not in out["metrics"]
    assert "gradient_sample_param_name" not in out["metrics"]


def test_cortex_update_on_a_packed_job_is_rewritten_to_grpo(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter, pack=True)
    _update(wrapper, _update_batch())
    sent = wrapper._client.fwd_bwds[0]
    assert sent["processing"]["loss_fn"] == "grpo"
    assert "old_log_probs" not in sent["context"]


def test_cortex_update_refuses_rollout_correction_weights(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter)
    batch = _update_batch()
    batch["rollout_is_weights"] = torch.ones((2, 2))
    with pytest.raises(NotImplementedError, match="rollout_is_weights"):
        _update(wrapper, batch)


def test_cortex_update_needs_a_loss_in_the_reply(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    with pytest.raises(RuntimeError, match="carries no loss"):
        adapter._merge_cortex_update_metrics({"metrics": {}}, {"grad_norm": 1.0, "update_successful": 1.0})


def test_cortex_update_refuses_a_step_that_did_not_apply(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    with pytest.raises(RuntimeError, match="no update_successful"):
        adapter._merge_cortex_update_metrics({"avg_loss": 0.5}, {"grad_norm": 1.0})
    with pytest.raises(RuntimeError, match="was not applied"):
        adapter._merge_cortex_update_metrics({"avg_loss": 0.5}, {"grad_norm": None, "update_successful": 0.0})


def test_cortex_log_prob_refuses_a_non_positive_temperature(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter)
    batch = _padded_batch()
    batch["temperature"] = torch.tensor(0.0)
    with pytest.raises(ValueError, match="greater than 0"):
        _log_prob(wrapper, batch)


def test_cortex_update_metrics_merge_nested_and_flat_step_replies(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    flat = adapter._merge_cortex_update_metrics(
        {"avg_loss": 0.5},
        {
            "job_id": "j",
            "last_lr": 1e-6,
            "grad_norm": None,
            "update_successful": 1.0,
            "peak_memory": {"a": 1},
        },
    )
    assert flat == {"last_lr": 1e-6, "loss": 0.5}

    nested = adapter._merge_cortex_update_metrics(
        {"avg_loss": 0.5, "metrics": {"loss": 0.4, "actor/pg_loss": 0.2}},
        {"metrics": {"last_lr": 1e-6, "grad_norm": 0.7, "update_successful": 1}},
    )
    assert nested == {"last_lr": 1e-6, "grad_norm": 0.7, "loss": 0.4, "actor/pg_loss": 0.2}


def test_cortex_generate_sends_token_ids(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter)
    wrapper._default_sampling_params = adapter.ArcticRLClientWrapper._default_sampling_params
    wrapper.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens=False: "".join(map(str, ids)))
    out = asyncio.run(wrapper.generate(torch.tensor([3, 4]), {"temperature": 1.0, "logprobs": None, "max_tokens": 8}))

    prompts, params = wrapper._client.generates[0]
    assert prompts == [[3, 4]]
    assert "logprobs" not in params and params["temperature"] == 1.0 and params["max_tokens"] == 8
    assert out == [{"text": "56", "token_ids": [5, 6], "finish_reason": "stop"}]


def test_cortex_reconnect_handle_carries_no_server_state(verl_stub) -> None:
    adapter = _adapter(verl_stub)
    wrapper = _cortex_wrapper(adapter)
    wrapper._client.reconnect_config = lambda: "config-with-job-ids"
    wrapper._client.get_server_state = lambda: pytest.fail("Cortex has no server state")
    assert wrapper.reconnect_handle() == {"rl_client_reconnect_config": "config-with-job-ids", "rl_server_state": None}
