"""Real GPU generation through InferenceWorker with an in-process vLLM adapter."""

import sys

import pytest

from arctic_platform.testing_utils import TestCasePlus, execute_subprocess_async, require_torch_gpu


SMOKE = """
import asyncio
import sys
import numpy as np
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM
from vllm import LLM
from arctic_platform.testing_utils import set_seed, torch_assert_equal
from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState
from arctic_platform.inference.server.router_replay import RouterReplayCacheTX, RouterReplayPrefixMissingError
from arctic_platform.inference.server.router_replay.cache import RouterReplayCacheFull

set_seed(947)
config = Qwen3MoeConfig(
    vocab_size=512, hidden_size=128, intermediate_size=256,
    moe_intermediate_size=128, num_hidden_layers=2, num_attention_heads=2,
    num_key_value_heads=1, head_dim=64, num_experts=4, num_experts_per_tok=2,
    decoder_sparse_step=1, max_position_embeddings=256,
    tie_word_embeddings=False, bos_token_id=1, eos_token_id=2,
)
Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(sys.argv[1])
llm = LLM(
    model=sys.argv[1], skip_tokenizer_init=True, enable_return_routed_experts=True,
    enable_prefix_caching=True, max_model_len=128, max_num_seqs=4,
    gpu_memory_utilization=.08, enforce_eager=True, disable_log_stats=True, seed=5,
)
worker = InferenceWorker.__ray_metadata__.modified_class()
worker.state = WorkerLifecycleState.READY
worker._replica_label = "test-owner"
raw = []

async def generate_once(prompt, params, request_id, **kwargs):
    assert not params.extra_args
    output = llm.generate(prompt, params, use_tqdm=False)[0]
    raw.append(np.array(output.outputs[0].routed_experts, copy=True))
    return output

worker._generate_once = generate_once

async def request(prompt, replay_id, prefix=None, strict=True):
    extra = {"dss_router_replay_strict": strict}
    if prefix is not None:
        extra["dss_router_replay_prefix_id"] = prefix
        if strict:
            extra["dss_router_replay_prefix_replica"] = "test-owner"
    params = dict(
        temperature=0, max_tokens=4, ignore_eos=True, detokenize=False,
        dss_sample_id="trajectory", dss_router_replay_id=replay_id,
        extra_args=extra,
    )
    return await worker.generate(prompt, params)

async def run():
    for device in ("cpu", "cuda:0"):
        cache = RouterReplayCacheTX(device=torch.device(device), max_bytes=1 << 20)
        worker._router_replay_tx = cache
        llm.reset_prefix_cache()
        prompt = list(range(10, 42))
        first = await request(prompt, "rr1:a")
        assert first["router_replay"]["strict"] is True
        prefix = cache.get("rr1:a")
        # A deliberate route change makes replacement distinguishable from a fresh prefill.
        prefix[0, 0, 0] = (int(prefix[0, 0, 0]) + 1) % 4
        prefix = prefix.clone()
        continuation = prompt + first["token_ids"] + [67, 68]
        cache.max_bytes = prefix.numel()
        llm.reset_prefix_cache()
        try:
            await request(continuation, "rr1:failed", "rr1:a")
            raise AssertionError("cache pressure must fail")
        except RouterReplayCacheFull:
            pass
        torch_assert_equal(cache.get("rr1:a"), prefix)
        assert "rr1:failed" not in cache and not cache._borrowers
        cache.max_bytes = 1 << 20
        llm.reset_prefix_cache()
        second = await request(continuation, "rr1:b", "rr1:a")
        merged = cache.get("rr1:b").cpu()
        n = len(prefix)
        assert second["router_replay"]["prefix_len"] == n
        torch_assert_equal(merged[:n], prefix.cpu())
        torch_assert_equal(merged[n:], torch.from_numpy(raw[-1][n:]))
        retry = await request(continuation, "rr1:retry", "rr1:a")
        assert retry["router_replay"]["prefix_len"] == n
        torch_assert_equal(cache.get("rr1:retry")[:n].cpu(), prefix.cpu())
        third_prompt = continuation + second["token_ids"] + [69, 70]
        third = await request(third_prompt, "rr1:c", "rr1:b")
        assert third["router_replay"]["prefix_len"] == len(merged)
        torch_assert_equal(cache.get("rr1:c")[:len(merged)].cpu(), merged)
        cache.discard(["rr1:a", "rr1:retry"])
        calls = len(raw)
        try:
            await request(continuation, "rr1:missing", "rr1:a")
            raise AssertionError("missing prefix must fail")
        except RouterReplayPrefixMissingError:
            pass
        assert len(raw) == calls
        fallback = await request(continuation, "rr1:fallback", "rr1:a", strict=False)
        assert fallback["router_replay"]["prefix_len"] == 0
        torch_assert_equal(cache.get("rr1:fallback").cpu(), torch.from_numpy(raw[-1]))
        print(f"PASS strict GPU generation cache={device}: pressure, retry, three turns, missing prefix, best effort")

asyncio.run(run())
"""


@pytest.mark.vllm
@require_torch_gpu
class TestStrictRouterReplayGPU(TestCasePlus):
    def test_strict_prefix_replay(self):
        model_dir = self.get_auto_remove_tmp_dir_str()
        env = self.get_env()
        env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        env["VLLM_LOGGING_LEVEL"] = "ERROR"
        execute_subprocess_async([sys.executable, "-c", SMOKE, model_dir], env=env, timeout=240)
