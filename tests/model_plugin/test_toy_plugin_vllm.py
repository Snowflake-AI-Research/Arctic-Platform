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

"""ToyHybridMoE's vLLM model against its trainer model: the sampler tier on GPUs.

A trainer checkpoint is served by vLLM in fp32, at tensor parallelism 1 and 2, and compared with the trainer on the
same weights:

- prompt log-probs (prefill, several requests in one batch);
- greedy decode log-probs, which resume the recurrence and convolution state step by step;
- after a weight sync of new weights through ``load_weights``, prompt log-probs of the new weights;
- the experts vLLM routed each prompt token to, which router replay feeds back to the trainer.

vLLM 0.33 records routed experts only with its V2 model runner and prefix caching on, which for a hybrid model
means "align"-mode state caching that this plugin's stateful layers do not implement (its manifest declares
``prefix_caching = false``). The routing check therefore runs its own engine with both on, on prompts that never hit
the cache, and checks prompt tokens only: a prefill does not depend on cached state.
"""

import os
from functools import partial
from pathlib import Path

import pytest
import torch

from arctic_platform.model_plugin.host import forward
from arctic_platform.model_plugin.host import next_token_labels
from arctic_platform.model_plugin.host import pack_sequences
from arctic_platform.model_plugin.host import token_logprobs
from arctic_platform.model_plugin.weights import iter_hf_weights
from arctic_platform.model_plugin.weights import save_hf_checkpoint
from arctic_platform.testing_utils import require_torch_gpu

pytest.importorskip("vllm")
pytestmark = [pytest.mark.vllm, pytest.mark.integration]

CONFIG = dict(
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
    final_logit_softcapping=30.0,  # applied by the trainer's compute_logits and by vLLM's logits processor
)
PROMPTS = [[(7 * i + 3 * j) % 500 + 2 for j in range(length)] for i, length in enumerate([17, 5, 40, 9])]
DECODE_TOKENS = 12
# Both sides run fp32, but vLLM's Triton attention and fused-MoE kernels compute fp32 dot products in TF32: the
# difference is uniform across positions (median ~6e-4), not growing with position or concentrated on few tokens.
LOGPROB_ATOL = 5e-3


def _trainer(seed: int):
    from toy_hybrid_moe import PLUGIN
    from toy_hybrid_moe.config import ToyHybridMoEConfig

    torch.manual_seed(seed)
    config = ToyHybridMoEConfig(**CONFIG)
    return config, PLUGIN.build_model(config).float().cuda().eval()


def _trainer_logprobs(model, sequences: list[list[int]]) -> list[torch.Tensor]:
    """Per-token log-probs of each sequence's tokens after the first, from one packed forward."""
    rows = [torch.tensor(sequence, device="cuda") for sequence in sequences]
    input_ids, position_ids = pack_sequences(rows)
    with torch.no_grad():
        labels = next_token_labels(input_ids, position_ids)
        logprobs = token_logprobs(model, forward(model, input_ids, position_ids), labels)
    out, start = [], 0
    for row in rows:
        out.append(logprobs[0, start : start + len(row) - 1].cpu())
        start += len(row)
    return out


def _prompt_logprobs(llm, prompts) -> list[torch.Tensor]:
    from vllm import SamplingParams

    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in prompts],
        SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=0),
        use_tqdm=False,
    )
    return [
        torch.tensor([entry[token].logprob for entry, token in zip(out.prompt_logprobs[1:], out.prompt_token_ids[1:])])
        for out in outputs
    ]


# apply_model sends its callable to every worker, so it must pickle by reference: module-level functions only. They
# return names, not tensors.
def _parameter_names(sampler_model) -> list[str]:
    return sorted(name for name, _ in sampler_model.named_parameters())


def _load_weights(sampler_model, stream) -> list[str]:
    return sorted(sampler_model.load_weights(stream))


def _close(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    diff = (actual.double() - expected.double()).abs().max().item()
    assert actual.shape == expected.shape and diff <= LOGPROB_ATOL, f"{what}: max |diff| {diff}"


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def served(request, tmp_path_factory):
    """A trainer model, its checkpoint, and a vLLM engine serving that checkpoint at the given tensor parallelism."""
    tp = request.param
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPUs")
    from toy_hybrid_moe import PLUGIN
    from vllm import LLM

    from .test_toy_plugin import _save_tokenizer

    config, model = _trainer(seed=0)
    path = Path(tmp_path_factory.mktemp(f"toy-tp{tp}"))
    save_hf_checkpoint(model, PLUGIN, config, path)
    _save_tokenizer(path, config.vocab_size)
    os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"  # apply_model sends the weight-sync closure to workers
    llm = LLM(
        model=str(path),
        dtype="float32",
        enforce_eager=True,
        tensor_parallel_size=tp,
        gpu_memory_utilization=0.15,
        max_model_len=256,
        enable_prefix_caching=False,
    )
    yield config, model, llm
    del llm
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def capturing(tmp_path_factory):
    """A tensor-parallel-1 engine that records routed experts, on the same weights as the trainer model."""
    from toy_hybrid_moe import PLUGIN
    from vllm import LLM

    from .test_toy_plugin import _save_tokenizer

    config, model = _trainer(seed=0)
    path = Path(tmp_path_factory.mktemp("toy-capture"))
    save_hf_checkpoint(model, PLUGIN, config, path)
    _save_tokenizer(path, config.vocab_size)
    previous = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    try:
        llm = LLM(
            model=str(path),
            dtype="float32",
            enforce_eager=True,
            gpu_memory_utilization=0.15,
            max_model_len=256,
            enable_return_routed_experts=True,
            enable_prefix_caching=True,
        )
    finally:
        if previous is None:
            os.environ.pop("VLLM_USE_V2_MODEL_RUNNER")
        else:
            os.environ["VLLM_USE_V2_MODEL_RUNNER"] = previous
    yield config, model, llm
    del llm
    torch.cuda.empty_cache()


@require_torch_gpu
def test_prompt_logprobs_match_trainer(served):
    _, model, llm = served
    for index, (actual, expected) in enumerate(zip(_prompt_logprobs(llm, PROMPTS), _trainer_logprobs(model, PROMPTS))):
        _close(actual, expected, f"prompt {index}")


@require_torch_gpu
def test_decode_logprobs_match_trainer(served):
    from vllm import SamplingParams

    _, model, llm = served
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in PROMPTS],
        SamplingParams(max_tokens=DECODE_TOKENS, min_tokens=DECODE_TOKENS, temperature=0, logprobs=0),
        use_tqdm=False,
    )
    sequences = [prompt + list(out.outputs[0].token_ids) for prompt, out in zip(PROMPTS, outputs)]
    expected = _trainer_logprobs(model, sequences)
    for index, (prompt, out) in enumerate(zip(PROMPTS, outputs)):
        generated = out.outputs[0]
        sampled = torch.tensor([entry[token].logprob for entry, token in zip(generated.logprobs, generated.token_ids)])
        _close(sampled, expected[index][len(prompt) - 1 :], f"decode of prompt {index}")


@require_torch_gpu
def test_routed_experts_match_trainer_routing(capturing):
    from vllm import SamplingParams

    config, model, llm = capturing
    outputs = llm.generate(
        [{"prompt_token_ids": prompt} for prompt in PROMPTS],
        SamplingParams(max_tokens=1, temperature=0),
        use_tqdm=False,
    )
    moe_layers = [index for index, kind in enumerate(config.mlp_types) if kind == "moe"]
    for index, (prompt, out) in enumerate(zip(PROMPTS, outputs)):
        captured = torch.as_tensor(out.outputs[0].routed_experts)[: len(prompt)]
        input_ids, position_ids = pack_sequences([torch.tensor(prompt, device="cuda")])
        with torch.no_grad():
            _, routing = forward(model, input_ids, position_ids, return_routing=True)
        for layer in moe_layers:
            sampler = captured[:, layer].sort(dim=-1).values
            trainer = routing[layer].cpu().sort(dim=-1).values
            agree = (sampler == trainer).all(dim=-1).float().mean().item()
            # Near-tied gate scores may order differently across kernels; anything systematic shows as disagreement.
            assert agree >= 0.95, f"prompt {index} layer {layer}: experts agree on {agree:.0%} of tokens"


@require_torch_gpu
def test_weight_sync_streams_new_weights_into_the_sampler(served):
    from toy_hybrid_moe import PLUGIN

    _, _, llm = served
    _, updated = _trainer(seed=1)
    stream = [(name, tensor.cpu()) for name, tensor in iter_hf_weights(updated, PLUGIN, for_sampler=True)]

    parameters = llm.apply_model(_parameter_names)
    for rank, (written, expected) in enumerate(
        zip(llm.apply_model(partial(_load_weights, stream=stream)), parameters)
    ):
        assert written == expected, f"rank {rank}: weight sync left {sorted(set(expected) - set(written))[:5]} unset"
    for index, (actual, expected) in enumerate(
        zip(_prompt_logprobs(llm, PROMPTS), _trainer_logprobs(updated, PROMPTS))
    ):
        _close(actual, expected, f"prompt {index} after weight sync")
