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

"""CPU checks for the TRL e7f6928 adapter: stub, flat collator, loss parity, checkpoints."""

from __future__ import annotations

import os
import queue
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import LlamaConfig
from transformers import PreTrainedModel

pytest.importorskip("trl.experimental.async_grpo.async_grpo_trainer")

from trl.experimental.async_grpo.async_grpo_trainer import DataCollatorForRollout  # noqa: E402
from trl.experimental.async_grpo.packing import TreeRow  # noqa: E402

from arctic_platform.integrations.trl.client import ArcticOptimizer  # noqa: E402
from arctic_platform.integrations.trl.client import ArcticTrainingClient  # noqa: E402
from arctic_platform.integrations.trl.client import _to_server_bs  # noqa: E402
from arctic_platform.integrations.trl.client import clipped_surrogate_sum  # noqa: E402
from arctic_platform.integrations.trl.rollout import ArcticRolloutWorker  # noqa: E402
from arctic_platform.integrations.trl.trainer import _FlatRolloutCollator  # noqa: E402
from arctic_platform.integrations.trl.trainer import build_stub_model  # noqa: E402


def _tiny_config() -> LlamaConfig:
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    config._name_or_path = "tiny-local"
    return config


def test_stub_keeps_width_and_model_id_and_shrinks_depth():
    original = _tiny_config()
    model = build_stub_model(original)
    assert isinstance(model, PreTrainedModel)
    assert model.config.num_hidden_layers == 1
    assert original.num_hidden_layers == 4
    assert model.config.hidden_size == 16
    assert model.config.vocab_size == 32
    assert model.config._name_or_path == "tiny-local"
    assert model._arctic_full_config.num_hidden_layers == 4
    assert next(model.parameters()).dtype == torch.float32
    assert hasattr(model, "set_attn_implementation")


def _sample(ids, mask, old, advantage, group):
    ids = np.asarray(ids, dtype=np.int64)
    return {
        "input_ids": ids,
        "completion_mask": np.asarray(mask, dtype=np.int64),
        "old_log_probs": np.asarray(old, dtype=np.float32),
        "sampling_mask": [[] for _ in ids],
        "advantage": float(advantage),
        "group_id": group,
        "metrics": {"reward": 1.0},
    }


def _shared_prefix_rows():
    # Prefix [1, 2, 3] is shared; completions fork. Masks start at 0 so a packed
    # boundary cannot become a trained token.
    samples = [
        _sample([1, 2, 3, 4, 5], [0, 0, 0, 1, 1], [0, 0, -0.2, -0.4, -0.6], 1.0, 7),
        _sample([1, 2, 3, 6, 7], [0, 0, 0, 1, 1], [0, 0, -0.2, -0.5, -0.7], -0.5, 7),
    ]
    logprobs = [
        [0.10, 0.20, 0.30, 0.40, 9.0],
        [0.10, 0.20, 0.50, 0.60, 9.0],
    ]
    row = TreeRow()
    for sample in samples:
        row.add(sample)
    return row, samples, logprobs


def _trained_terms(samples, logprobs):
    values, old, advantages = [], [], []
    for sample, lp in zip(samples, logprobs, strict=True):
        mask = sample["completion_mask"]
        for index in range(1, len(mask)):
            if mask[index]:
                values.append(lp[index - 1])
                old.append(float(sample["old_log_probs"][index]))
                advantages.append(sample["advantage"])
    return (
        torch.tensor(values, dtype=torch.float32),
        torch.tensor(old, dtype=torch.float32),
        torch.tensor(advantages, dtype=torch.float32),
    )


def test_flat_collator_keeps_trained_count_and_counts_raw_tokens():
    row, samples, _ = _shared_prefix_rows()
    groups: set[int] = set()
    metrics: dict = defaultdict(list)
    collator = _FlatRolloutCollator(
        DataCollatorForRollout(pad_token_id=0, num_processes=1, groups_trained=groups, metrics=metrics, token_budget=0)
    )
    batch = collator([[row]])
    raw = sum(len(sample["input_ids"]) for sample in samples)
    unique = row.pack()["input_ids"].numel()
    assert unique < raw
    assert batch["global_n_tokens"].item() == 4
    assert batch["global_n_forward_tokens"].item() == raw
    assert batch["mean_seq_len"].item() == pytest.approx(raw / len(samples))
    assert groups == {7}
    assert metrics["sample/forwarded_tokens_mean"] == [pytest.approx(raw / len(samples))]
    assert batch["position_ids"][0].tolist()[:5] == [0, 1, 2, 3, 4]
    assert 0 in batch["position_ids"][0].tolist()[5:]


def test_tree_and_flat_trained_token_losses_match():
    row, samples, logprobs = _shared_prefix_rows()
    values, old, advantages = _trained_terms(samples, logprobs)
    reference = clipped_surrogate_sum(
        values.view(1, -1), old.view(1, -1), advantages.view(1, -1), torch.ones_like(values).view(1, -1), 0.2, 0.2
    )

    collator = _FlatRolloutCollator(
        DataCollatorForRollout(
            pad_token_id=0, num_processes=1, groups_trained=set(), metrics=defaultdict(list), token_budget=0
        )
    )
    batch = collator([[row]])
    kept = batch["attention_mask"].bool()
    position_ids = batch["position_ids"][kept].unsqueeze(0)
    completion_mask = batch["completion_mask"][kept].unsqueeze(0)
    shifted_old = batch["old_log_probs"][kept].unsqueeze(0)[:, 1:]
    shifted_adv = batch["advantages"][kept].unsqueeze(0)[:, 1:]
    loss_mask = completion_mask[:, 1:].to(dtype=torch.float32)

    packed_lp = torch.tensor([value for lp in logprobs for value in lp], dtype=torch.float32).view(1, -1)
    seq_lens = torch.tensor([len(lp) for lp in logprobs])
    server_lp = _to_server_bs(packed_lp[:, :-1], seq_lens)

    class _Echo:
        def fwd_no_grad(self, payload):
            del payload
            return {"batch": {"logprobs": server_lp, "entropy": torch.zeros_like(server_lp)}}

        def fwd_bwd(self, payload):
            self.payload = payload

    echo = _Echo()
    client = ArcticTrainingClient(echo, temperature=1.0)
    out = client.forward_backward(
        batch["input_ids"][kept].unsqueeze(0),
        position_ids,
        completion_mask,
        old_log_probs=shifted_old,
        advantages=shifted_adv,
        loss_mask=loss_mask,
        epsilon_low=0.2,
        epsilon_high=0.2,
        tokens_per_rank=batch["global_n_tokens"].reshape(-1)[0],
    )
    assert torch.allclose(out.loss, reference)
    out.loss.backward()
    assert echo.payload["processing"]["loss_fn"].endswith("weighted_logprob_sum")


def test_rollout_sample_schema():
    def reward(**kwargs):
        return [1.0, 0.0]

    worker = ArcticRolloutWorker(
        client=object(),
        dataset=[],
        reward_funcs=reward,
        processing_class=object(),
        num_generations=2,
        old_logprobs_source="sampler",
    )
    worker._emit_group(
        {"prompt": [{"role": "user", "content": "q"}]},
        [11, 12],
        [
            {"token_ids": [3, 4], "text": "a", "logprobs": [{3: {"logprob": -0.2}}, {4: {"logprob": -0.4}}]},
            {"token_ids": [5], "text": "b", "logprobs": [{5: {"logprob": -0.1}}]},
        ],
        group_id=1,
        model_version=2,
    )
    first = worker.rollout_buffer.get_nowait()
    second = worker.rollout_buffer.get_nowait()
    assert first.input_ids.dtype == np.int64
    assert first.completion_mask.dtype == np.int64
    assert first.old_log_probs.dtype == np.float32
    assert len(first.sampling_mask) == len(first.input_ids)
    assert all(mask == [] for mask in first.sampling_mask)
    assert isinstance(first.advantage, float)
    assert first.enqueued_at is not None
    assert len(second.sampling_mask) == len(second.input_ids)


class _Tokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "</s>"
    eos_token_id = 1

    def save_pretrained(self, path):
        with open(os.path.join(path, "tokenizer.saved"), "w", encoding="utf-8") as handle:
            handle.write("ok")


class _CheckpointClient:
    def __init__(self):
        self.calls = []

    def save_checkpoint(self, path=None, **kwargs):
        self.calls.append(("save", path))

    def load_checkpoint(self, path=None, **kwargs):
        self.calls.append(("load", path))

    def step(self):
        return {"metrics": {}}


def _trainer(tmp_path, checkpoint_client):
    from datasets import Dataset
    from trl.experimental.async_grpo.async_grpo_config import AsyncGRPOConfig

    from arctic_platform.integrations.trl.trainer import ArcticAsyncGRPOTrainer

    model = build_stub_model(_tiny_config())
    args = AsyncGRPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=1,
        num_generations=2,
        token_budget=0,
        bf16=False,
        fp16=False,
        use_cpu=True,
        report_to="none",
        logging_steps=1,
        save_strategy="no",
        remove_unused_columns=False,
        vllm_server_base_url="http://127.0.0.1:9",
    )
    dataset = Dataset.from_dict({"prompt": [[{"role": "user", "content": "q"}]], "solution": ["1"]})
    return ArcticAsyncGRPOTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=_Tokenizer(),
        rollout_worker=SimpleNamespace(rollout_buffer=queue.Queue(), check_health=lambda: None),
        weight_transfer=SimpleNamespace(),
        arctic_client=checkpoint_client,
        training_client=ArcticTrainingClient(checkpoint_client),
    )


def test_trainer_initializes_on_cpu_and_roundtrips_the_full_config(tmp_path):
    client = _CheckpointClient()
    trainer = _trainer(tmp_path, client)
    assert trainer.model.config.num_hidden_layers == 1
    assert trainer._original_config.num_hidden_layers == 4
    assert isinstance(trainer.create_optimizer(), ArcticOptimizer)
    dest = tmp_path / "ckpt"
    trainer._save(str(dest))
    trainer._load_from_checkpoint(str(dest))
    assert client.calls == [("save", str(dest)), ("load", str(dest))]
    assert (dest / "tokenizer.saved").is_file()
    assert (dest / "training_args.bin").is_file()
    assert (dest / "config.json").is_file()
    saved = LlamaConfig.from_pretrained(dest)
    assert saved.num_hidden_layers == 4
    names = {path.name for path in dest.iterdir()}
    assert "model.safetensors" not in names
    assert "pytorch_model.bin" not in names
    row, samples, logprobs = _shared_prefix_rows()
    values, old, advantages = _trained_terms(samples, logprobs)
    reference = clipped_surrogate_sum(
        values.view(1, -1), old.view(1, -1), advantages.view(1, -1), torch.ones_like(values).view(1, -1), 0.2, 0.2
    )
    batch = _FlatRolloutCollator(
        DataCollatorForRollout(
            pad_token_id=0, num_processes=1, groups_trained=set(), metrics=defaultdict(list), token_budget=0
        )
    )([[row]])
    packed_lp = torch.tensor([value for lp in logprobs for value in lp], dtype=torch.float32).view(1, -1)
    server_lp = _to_server_bs(packed_lp[:, :-1], torch.tensor([len(lp) for lp in logprobs]))

    class _Echo:
        server_side_loss = False

        def forward_backward(self, input_ids, position_ids, completion_mask, **kwargs):
            return ArcticTrainingClient(_Server(server_lp), temperature=1.0).forward_backward(
                input_ids, position_ids, completion_mask, **kwargs
            )

    class _Server:
        def __init__(self, logprobs):
            self.logprobs = logprobs

        def fwd_no_grad(self, payload):
            del payload
            return {"batch": {"logprobs": self.logprobs, "entropy": torch.zeros_like(self.logprobs)}}

        def fwd_bwd(self, payload):
            del payload

    trainer.training_client = _Echo()
    trainer.current_gradient_accumulation_steps = 2
    trainer.epsilon_low = 0.2
    trainer.epsilon_high = 0.2
    loss = trainer.compute_loss(trainer.model, batch)
    tokens = batch["global_n_tokens"].reshape(-1)[0]
    expected = reference / tokens.to(torch.float32) / 2
    assert torch.allclose(loss, expected)
    assert trainer._step_trained_tokens == pytest.approx(4)
    assert trainer._step_forward_tokens == pytest.approx(10)


def test_trainer_rejects_unsupported_sampling(tmp_path):
    from arctic_platform.integrations.trl.trainer import ArcticAsyncGRPOTrainer

    model = build_stub_model(_tiny_config())
    with pytest.raises(ValueError, match="top_p"):
        ArcticAsyncGRPOTrainer(
            model=model,
            args=SimpleNamespace(top_p=0.9, top_k=0, min_p=None),
            arctic_client=object(),
            training_client=object(),
        )
    with pytest.raises(ValueError, match="top_k"):
        ArcticAsyncGRPOTrainer(
            model=model,
            args=SimpleNamespace(top_p=1.0, top_k=4, min_p=None),
            arctic_client=object(),
            training_client=object(),
        )
    with pytest.raises(ValueError, match="min_p"):
        ArcticAsyncGRPOTrainer(
            model=model,
            args=SimpleNamespace(top_p=1.0, top_k=0, min_p=0.05),
            arctic_client=object(),
            training_client=object(),
        )
    with pytest.raises(ValueError, match="peft_config"):
        ArcticAsyncGRPOTrainer(
            model=model,
            args=SimpleNamespace(top_p=1.0, top_k=0, min_p=None),
            peft_config=object(),
            arctic_client=object(),
            training_client=object(),
        )


def test_server_loss_rejects_multiprocess_world(tmp_path):
    del tmp_path

    class _Client:
        def fwd_bwd(self, payload):
            raise AssertionError(payload)

    adapter = ArcticTrainingClient(_Client(), server_side_loss=True)
    with pytest.raises(RuntimeError, match="single-process"):
        adapter.forward_backward(
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[0, 1, 1]]),
            old_log_probs=torch.zeros(1, 2),
            advantages=torch.ones(1, 2),
            loss_mask=torch.ones(1, 2),
            epsilon_low=0.2,
            epsilon_high=0.2,
            tokens_per_rank=torch.tensor(2.0),
            world_size=2,
        )
