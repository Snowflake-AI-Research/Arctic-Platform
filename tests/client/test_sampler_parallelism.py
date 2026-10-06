# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""How the sampling sub-job spends its GPUs.

``n_gpus`` is the sub-job's GPU budget, not the width of one engine. Leaving
``tensor_parallel_size`` unset shards a single engine across the whole budget,
which for a small model yields one scheduler and one queue: adding concurrent
callers then does nothing, because there is only ever one thing to run. The
value has to survive the trip onto the wire, and it used to be stripped, so it
is worth pinning.
"""

from __future__ import annotations

from arctic_platform.client.config import ArcticRLClientConfig


def _sampling_sub_job(**vllm):
    cfg = ArcticRLClientConfig(model_name="Qwen/Qwen3.5-4B", max_seq_len=4096)
    cfg.sampling.vllm = dict(vllm)
    return cfg._cortex_inference_sub_job("sampling", n_gpus=16)


def test_tensor_parallel_size_reaches_the_wire():
    sub = _sampling_sub_job(tensor_parallel_size=1)
    assert sub["inference_config"]["vllm_config"]["tensor_parallel_size"] == 1


def test_gpu_budget_is_independent_of_shard_width():
    """16 GPUs at width 1 is sixteen replicas, not a contradiction."""
    sub = _sampling_sub_job(tensor_parallel_size=1)
    assert sub["inference_config"]["n_gpus"] == 16


def test_a_wider_shard_is_still_allowed():
    sub = _sampling_sub_job(tensor_parallel_size=4)
    assert sub["inference_config"]["vllm_config"]["tensor_parallel_size"] == 4


def test_max_model_len_is_still_owned_by_max_seq_len():
    """The other popped key stays popped: it genuinely is a duplicate."""
    sub = _sampling_sub_job(tensor_parallel_size=1, max_model_len=999)
    assert "max_model_len" not in sub["inference_config"].get("vllm_config", {})
    assert sub["inference_config"]["max_seq_len"] == 4096


def test_other_engine_settings_pass_through():
    sub = _sampling_sub_job(
        tensor_parallel_size=1, enable_prefix_caching=True, max_num_seqs=56
    )
    vllm = sub["inference_config"]["vllm_config"]
    assert vllm["enable_prefix_caching"] is True
    assert vllm["max_num_seqs"] == 56


def test_no_engine_settings_means_no_vllm_block():
    sub = _sampling_sub_job()
    assert "vllm_config" not in sub["inference_config"]
