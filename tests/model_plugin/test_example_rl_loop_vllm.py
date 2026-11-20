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

"""Every example plugin through a short RL loop with its vLLM sampler: sample, check, train, sync, repeat.

For each example, a small trainer model is saved as a checkpoint and served by vLLM 0.33 with the engine settings the
plugin's manifest asks for (``sampler_launch``): vLLM's built-in model for the four real architectures, the toy's own
vLLM model for the toy. Then, for a few steps:

1. vLLM samples completions for a group of prompts, recording routed experts for mixture-of-experts models;
2. the trainer's log-probs of those completions, replaying the sampler's routing, must match the sampler's;
3. one GRPO step on group-normalized rewards, with the sampler's log-probs as the old policy;
4. the new weights stream into vLLM through ``load_weights``, which must write every sampler parameter.

Step 2 of the next step checks that the sampler now serves the trained weights; a final check follows the last step.
The trainer runs on one GPU here: ``rl_training`` (``check_rl_training``) covers training under SP and EP.
"""

import os
from functools import partial
from pathlib import Path

import pytest
import torch

from arctic_platform.model_plugin import ROUTER_REPLAY_FRESH
from arctic_platform.model_plugin.admission import JobRequest
from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import prepare_model
from arctic_platform.model_plugin.host import token_logprobs
from arctic_platform.model_plugin.plugin import load_plugin
from arctic_platform.model_plugin.sampler import sampler_launch
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.model_plugin.weights import save_hf_checkpoint
from arctic_platform.testing_utils import require_torch_gpu

from .test_toy_plugin import _save_tokenizer

pytest.importorskip("vllm")
pytestmark = [pytest.mark.vllm, pytest.mark.integration]

STEPS = 3
PROMPTS = [[(7 * i + 3 * j) % 500 + 2 for j in range(length)] for i, length in enumerate([9, 17, 5, 12])]
SAMPLES_PER_PROMPT = 2
NEW_TOKENS = 12
LR = 1e-3
# The trainer runs fp32. An fp32 sampler differs only by vLLM's TF32 dot products (see test_toy_plugin_vllm); a bf16
# sampler by bf16 rounding, ~0.06 here. Every training step must move the log-probs by more than this, or a sampler
# that missed a weight sync could still pass.
LOGPROB_ATOL = {"float32": 5e-3, "bfloat16": 0.1}
# vLLM 0.33's built-in models return NaN log-probs in float32 now and then (OLMoE in every batch beyond a few tokens,
# OLMo-2 intermittently), and are right in bf16, the dtype samplers serve in. The toy's own vLLM model runs float32.
SAMPLER_DTYPE = {"olmo2": "bfloat16", "olmoe": "bfloat16", "lfm2": "bfloat16", "gpt_oss": "bfloat16"}

_COMMON = dict(
    vocab_size=512, hidden_size=256, num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512
)
EXAMPLES = {
    "olmo2": ("olmo2_plugin:PLUGIN", _COMMON | dict(intermediate_size=512, num_hidden_layers=2)),
    "olmoe": (
        "olmoe_plugin:PLUGIN",
        _COMMON | dict(intermediate_size=128, num_hidden_layers=2, num_experts=8, num_experts_per_tok=2),
    ),
    "lfm2": (
        "lfm2_plugin:PLUGIN",
        _COMMON
        | dict(
            intermediate_size=512,
            block_multiple_of=64,
            num_hidden_layers=4,
            layer_types=["conv", "full_attention", "conv", "full_attention"],
            tie_word_embeddings=True,
            # Read by vLLM's built-in LFM2: released checkpoints carry them, transformers 5 defines neither.
            conv_dim=256,
            block_dim=256,
        ),
    ),
    "gpt_oss": (
        "gpt_oss_plugin:PLUGIN",
        _COMMON
        | dict(
            intermediate_size=128,
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
            head_dim=64,
            num_local_experts=8,
            num_experts_per_tok=2,
            rope_parameters=dict(
                rope_type="yarn",
                factor=2.0,
                original_max_position_embeddings=256,
                rope_theta=150000.0,
                beta_fast=32.0,
                beta_slow=1.0,
                truncate=False,
            ),
        ),
    ),
    "toy_hybrid_moe": (
        "toy_hybrid_moe:PLUGIN",
        dict(
            vocab_size=512,
            hidden_size=256,
            layer_types=["linear_attention", "full_attention", "conv", "full_attention"],
            mlp_types=["dense", "moe", "moe", "dense"],
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            recurrence_heads=4,
            recurrence_head_dim=64,
            conv_channels=256,
            intermediate_size=512,
            num_experts=8,
            num_experts_per_tok=2,
            moe_intermediate_size=128,
            max_position_embeddings=1024,
        ),
    ),
}


# apply_model sends its callable to every worker, so it must pickle by reference: module-level functions only.
def _parameter_names(sampler_model) -> list[str]:
    return sorted(name for name, _ in sampler_model.named_parameters())


def _load_weights(sampler_model, stream) -> list[str]:
    return sorted(sampler_model.load_weights(stream))


def _sampler(plugin, path: Path, router_replay: bool, dtype: str):
    """vLLM serving ``path`` with the engine arguments and environment the plugin's manifest asks for."""
    from vllm import LLM

    request = JobRequest(
        job_type="sampling",
        router_replay=router_replay,
        inference=dict(tensor_parallel_size=1, gpu_memory_utilization=0.2, max_model_len=256),
    )
    launch = sampler_launch(plugin, request)
    previous = {key: os.environ.get(key) for key in launch.env}
    os.environ.update(launch.env)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # apply_model sends the weight-sync closure to workers
    try:
        return LLM(model=str(path), dtype=dtype, **launch.engine_args)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key)
            else:
                os.environ[key] = value


def _routing(outputs, moe_layers, lengths, device):
    """The sampler's routed experts per expert layer, packed like the sequences. The last token of each sequence was
    sampled but never run through the model, so it routes freshly (``ROUTER_REPLAY_FRESH``)."""
    rows = {layer: [] for layer in moe_layers}
    for out, length in zip(outputs, lengths):
        captured = torch.as_tensor(out.routed_experts)
        for layer in moe_layers:
            ids = torch.full((length, captured.shape[-1]), ROUTER_REPLAY_FRESH, dtype=torch.long)
            ids[: min(length - 1, captured.shape[0])] = captured[: length - 1, layer]
            rows[layer].append(ids)
    return {layer: torch.cat(ids).to(device) for layer, ids in rows.items()}


def _max_diff(a, b, mask) -> float:
    return (a[mask].double() - b[mask].double()).abs().max().item()


@require_torch_gpu
@pytest.mark.parametrize("example", sorted(EXAMPLES))
def test_rl_loop_with_the_vllm_sampler(example, tmp_path):
    from vllm import SamplingParams

    from arctic_platform.rl.processors.grpo import grpo_loss

    target, fixture = EXAMPLES[example]
    plugin = load_plugin(target)
    torch.manual_seed(0)
    config = plugin.config_class(**fixture)
    model = plugin.build_model(config, plugin.parse_options()).float().cuda()
    binding = prepare_model(model, plugin, config)
    moe_layers = sorted(binding.experts.values())
    save_hf_checkpoint(model, plugin, config, tmp_path)
    _save_tokenizer(tmp_path, config.vocab_size)
    dtype = SAMPLER_DTYPE.get(example, "float32")
    atol = LOGPROB_ATOL[dtype]
    llm = _sampler(plugin, tmp_path, router_replay=len(moe_layers) > 0, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
    sampler_parameters = llm.apply_model(_parameter_names)
    try:
        for step in range(STEPS + 1):
            params = SamplingParams(
                n=SAMPLES_PER_PROMPT,
                temperature=1.0,
                seed=step,
                max_tokens=NEW_TOKENS,
                min_tokens=NEW_TOKENS,
                logprobs=0,
            )
            requests = llm.generate([{"prompt_token_ids": prompt} for prompt in PROMPTS], params, use_tqdm=False)
            outputs = [sample for request in requests for sample in request.outputs]
            prompts = [prompt for prompt in PROMPTS for _ in range(SAMPLES_PER_PROMPT)]
            sequences = [prompt + list(out.token_ids) for prompt, out in zip(prompts, outputs)]
            input_ids, position_ids = pack_sequences([torch.tensor(seq, device="cuda") for seq in sequences])
            labels = next_token_labels(input_ids, position_ids)

            # The sampler's log-prob of each generated token, at the position that predicts it.
            sampled = torch.zeros(labels.shape, device="cuda")
            mask = torch.zeros(labels.shape, dtype=torch.bool, device="cuda")
            start = 0
            for prompt, out, sequence in zip(prompts, outputs, sequences):
                positions = range(start + len(prompt) - 1, start + len(sequence) - 1)
                for position, token, entry in zip(positions, out.token_ids, out.logprobs):
                    sampled[0, position] = entry[token].logprob
                    mask[0, position] = True
                start += len(sequence)

            routed = _routing(outputs, moe_layers, [len(s) for s in sequences], "cuda") if moe_layers else None
            hidden = forward(model, input_ids, position_ids, routed_experts=routed)
            logprobs = token_logprobs(model, hidden, labels)
            diff = _max_diff(logprobs.detach(), sampled, mask)
            assert diff <= atol, f"{example} step {step}: trainer and sampler log-probs differ by {diff}"
            if step == STEPS:  # the sampler serves the last step's weights: done
                break

            # Rewards: the fraction of even token ids; advantages normalized within each prompt's group.
            rewards = torch.tensor([sum(t % 2 == 0 for t in out.token_ids) / NEW_TOKENS for out in outputs])
            grouped = rewards.view(len(PROMPTS), SAMPLES_PER_PROMPT)
            advantages_per_sequence = (grouped - grouped.mean(-1, keepdim=True)).flatten()
            advantages = torch.zeros(labels.shape, device="cuda")
            start = 0
            for advantage, sequence in zip(advantages_per_sequence, sequences):
                advantages[0, start : start + len(sequence)] = advantage
                start += len(sequence)
            loss, _ = grpo_loss(
                dict(logprobs=logprobs),
                dict(old_log_probs_shifted=sampled, advantages=advantages, loss_mask=mask),
                {},
                {},
                "cuda",
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            with torch.no_grad():
                trained = token_logprobs(model, forward(model, input_ids, position_ids, routed_experts=routed), labels)
            moved = _max_diff(trained, logprobs.detach(), mask)
            assert (
                moved > atol
            ), f"{example} step {step}: the step moved log-probs by {moved}, too little to detect a sync"

            stream = [
                (name, tensor.detach().cpu()) for name, tensor in iter_hf_weights(model, plugin, for_sampler=True)
            ]
            for rank, (written, expected) in enumerate(
                zip(llm.apply_model(partial(_load_weights, stream=stream)), sampler_parameters)
            ):
                missing = sorted(set(expected) - set(written))
                assert len(missing) == 0, f"{example} rank {rank}: weight sync left {missing[:5]} unset"
            # Cached prefixes hold keys, values and states computed with the old weights.
            llm.reset_prefix_cache()
    finally:
        del llm
        torch.cuda.empty_cache()
