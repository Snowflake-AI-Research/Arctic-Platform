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

"""Arctic RL training-engine test: log-prob correctness + GRPO grad flow, one client per matrix cell.

Both ``compute_log_prob`` and ``update_actor`` drive the DeepSpeed training engine on the same topology
(``training_gpus=2``, no sampling job), so per ``(transport x ZoRRO)`` cell we spin up a single client and run both
checks on it. The batch is variable-length (prompts left-padded to ``prompt_len``, responses right-padded to
varying real lengths; the fully-packed row also covers the no-padding case), so pack/unpad masking and per-sequence
position-id reconstruction are exercised. Per cell:

  1. ``fwd_no_grad`` -> per-token log-probs match an independent per-row unpadded reference (run first, while the
     weights are still pristine), compared only at real response positions.
  2. ``fwd_bwd`` + ``step`` -> finite losses and a strictly-positive global ``grad_norm`` (a real backward flowed
     gradients the optimizer stepped on). The clipped-ratio loss keeps any finite ``old_log_probs`` / ``advantages``
     safe, so no prior real log-prob run is needed.

Matrix (see ``train_engine_params``): ``ray`` runs both ZoRRO on/off; ``http`` runs once (its transport is just
serialization, independent of the forward path). Shared infra (config, fake data, lifecycle, ports, skip guard,
GPU lock) lives in ``rl_harness``; does not depend on ``arctic-verl``. Heavyweight GPU test::

    pytest tests/rl/test_train_engine.py -s
"""

from __future__ import annotations

import asyncio

import pytest
from parameterized import parameterized

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import torch_assert_close
from rl_harness import arctic_rl_client_session
from rl_harness import assert_finite_logprobs
from rl_harness import assert_positive_grad_norm
from rl_harness import build_compute_log_prob_payload
from rl_harness import build_update_actor_payload
from rl_harness import cached_padded_batch_and_reference
from rl_harness import cell_tag
from rl_harness import finite_metric
from rl_harness import make_fake_batch
from rl_harness import parameterized_custom_name_func
from rl_harness import response_region
from rl_harness import skip_if_unsupported

# Model + fake-data geometry this test owns and passes into the rl_harness builders. batch_size = num_unique_prompts
# * rollout_n must exceed the training world size so the ZoRRO load balancer (reorg_global_batch) runs bin-packing.
model_name = "Qwen/Qwen3-0.6B"
attn_implementation = "flash_attention_2"
num_unique_prompts = 2
rollout_n = 2
prompt_len = 8
response_len = 8

# Fake data goes straight to the training engine, so no sampling (vLLM) job is created; both GPUs go to training.
training_gpus = 2
sampling_gpus = 0
log_prob_gpus = 0

# atol vs the reference (rtol=0): bf16 kernel/reduction noise is small while a misaligned return is off by several
# nats, so this absorbs bf16 jitter yet rejects garbage.
LOGPROB_ATOL = 0.25

# Trimmed matrix (vs the full 2x2): for a training-only forward, http vs ray is just serialization plumbing and is
# independent of the ZoRRO axis, so ray covers both forward paths and http needs the serialization path checked
# only once. e2e exercises both transports end-to-end anyway.
train_engine_params = [("ray", True), ("ray", False), ("http", True)]


@require_torch_multi_gpu
@pytest.mark.gpu_serial
@pytest.mark.xdist_group("arctic_rl_train")  # keep all cells on one worker so the batch / reference cache is shared
class TestTrainEngine(TestCasePlus):
    @staticmethod
    async def _drive(client, cl_payload: dict, ua_payload: dict) -> tuple[dict, dict, dict]:
        # fwd_no_grad first (pristine weights for the log-prob check), then update actor (fwd_bwd + optimizer).
        logprob_response = await client.fwd_no_grad(cl_payload, reference_model=False)
        fwd_bwd_response = await client.fwd_bwd(ua_payload)
        step_response = await client.step()
        return logprob_response, fwd_bwd_response, step_response

    @staticmethod
    async def _drive_update(client, ua_payload: dict) -> tuple[dict, dict]:
        fwd_bwd_response = await client.fwd_bwd(ua_payload)
        step_response = await client.step()
        return fwd_bwd_response, step_response

    @parameterized.expand(train_engine_params, name_func=parameterized_custom_name_func)
    def test_train_engine(self, comm_protocol, zorro_enable):
        """One client per cell: log-probs match the reference, then update actor flows gradients."""
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus)
        batch, _, response_lens, ref, valid = cached_padded_batch_and_reference(
            model_name, attn_implementation, num_unique_prompts, rollout_n, prompt_len, response_len
        )
        cl_payload = build_compute_log_prob_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)
        ua_payload = build_update_actor_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)
        tag = cell_tag(comm_protocol, zorro_enable)
        with arctic_rl_client_session(
            comm_protocol, zorro_enable, model_name, attn_implementation, prompt_len, response_len, rollout_n,
            training_gpus, sampling_gpus, log_prob_gpus,
        ) as client:
            logprob_response, fwd_bwd_response, step_response = asyncio.run(
                self._drive(client, cl_payload, ua_payload)
            )

        # 1. log-prob numerical correctness on pristine weights (fwd_no_grad ran before the optimizer step).
        logprobs = assert_finite_logprobs(logprob_response, batch)
        got_resp = response_region(logprobs, zorro_enable, prompt_len, response_len)
        print(f"[train-engine] {tag} logprobs response_lens={response_lens}")
        torch_assert_close(got_resp[valid], ref[valid], rtol=0, atol=LOGPROB_ATOL, msg=f"{tag} response tokens")

        # 2. update actor: finite losses + a strictly-positive global grad_norm.
        self.assertIn("metrics", fwd_bwd_response, f"{tag}: fwd_bwd missing 'metrics': {list(fwd_bwd_response)}")
        self.assertIn("avg_loss", fwd_bwd_response, f"{tag}: fwd_bwd missing 'avg_loss': {list(fwd_bwd_response)}")
        fb_metrics = fwd_bwd_response["metrics"]  # combine_metric_shards folds {name}.sum/.tokens -> {name}
        loss = finite_metric(fb_metrics["loss"])
        pg_loss = finite_metric(fb_metrics["actor/pg_loss"])
        avg_loss = finite_metric(fwd_bwd_response["avg_loss"])
        finite_metric(step_response["metrics"]["last_lr"])
        grad_norm = assert_positive_grad_norm(step_response)
        print(
            f"[train-engine] {tag} loss={loss:.4f} pg={pg_loss:.4f} avg={avg_loss:.4f} grad_norm={grad_norm:.4f}"
        )

    def test_microbatch_accumulation(self):
        """Gradient accumulation (>1 forward microbatch per rank) over a batch larger than the world size.

        ``gradient_accumulation_steps=2`` makes each DP rank split its shard into 2 forward microbatches, exercising
        the worker's per-microbatch loop (inter-microbatch ``engine.step()``, ``combine_metric_microbatches`` /
        ``merge_dict_shards``) that the single-microbatch ``test_train_engine`` cells never reach. The 8-row batch
        (4 prompt groups vs 2 ranks) also drives the ZoRRO load balancer's multi-group-per-bin packing
        (``reorg_global_batch``). Asserts finite losses + a strictly-positive grad_norm.
        """
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus)
        batch, _, _ = make_fake_batch(model_name, 4, rollout_n, prompt_len, response_len)
        ua_payload = build_update_actor_payload(batch, True, rollout_n, prompt_len, response_len)
        with arctic_rl_client_session(
            "ray", True, model_name, attn_implementation, prompt_len, response_len, rollout_n,
            training_gpus, sampling_gpus, log_prob_gpus, gradient_accumulation_steps=2,
        ) as client:
            fwd_bwd_response, step_response = asyncio.run(self._drive_update(client, ua_payload))

        self.assertIn("metrics", fwd_bwd_response, f"fwd_bwd missing 'metrics': {list(fwd_bwd_response)}")
        loss = finite_metric(fwd_bwd_response["metrics"]["loss"])
        grad_norm = assert_positive_grad_norm(step_response)
        print(f"[train-engine] grad-accum loss={loss:.4f} grad_norm={grad_norm:.4f}")
