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

"""Arctic RL end-to-end GRPO test: the full client surface over 2 training steps.

Where ``test_generate`` pins the sampling stage and ``test_train_engine`` pins the training-engine forward/update,
this drives the whole GRPO loop against a single live client -- one training (DeepSpeed) job + one sampling (vLLM)
job -- so the
stages share state across steps the way the verl ``ArcticRLClientWrapper`` (arctic-verl,
``verl/trainer/ppo/arctic_rl_client.py`` + ``verl/workers/arctic_workers.py``) drives them. Each of the 2 steps
mirrors the verl ordering: wake_inference -> reset_prefix_cache -> generate -> sleep_inference -> wake_training ->
fwd_no_grad (old log-probs) -> fwd_bwd + step (policy update) -> empty_training_cache -> (sleep_training non_lp ->
CUDA-IPC sync_weights -> sleep_training lp_params) -> sleep_training all, with save_checkpoint and a final CPU-file
sync_weights (cuda_ipc=False) after the loop.

Exercises every ``arctic_platform/rl/ray_client.py`` method reachable from this topology: generate, fwd_no_grad,
fwd_bwd, step, sync_weights (CUDA-IPC bulk + low_memory streaming, and the CPU-file path), sleep/wake_inference,
reset_prefix_cache, sleep_training (all / non_lp / lp_params modes) / wake_training, empty_training_cache,
sleep/wake_log_prob (graceful no-ops -- no log-prob job here), save_checkpoint, reconnect_config /
get_server_state (a second client re-attaches to the live jobs without re-initializing), save_weights (raises
``NotImplementedError`` on the ray client; a graceful warn-on-error disk-reload stub on the http client),
shutdown (via the session). Not exercised: ``log_probs``
(needs a log-prob engine; this 2-GPU training+sampling topology has none -- covered by test_log_prob_engine).
``test_sync_weights_changes_sampler``
additionally proves a weight update actually propagates to the sampler. Fake training data (the clipped-ratio loss
keeps any finite ``old_log_probs`` / ``advantages`` safe); real prompts go through generate. Covers each transport
once (``ray``/``http``); the forward path is numerically certified elsewhere, so this trades the full 4-cell
matrix for a 2-cell diagonal. Heavyweight GPU test; shared infra lives in ``rl_harness``::

    pytest tests/rl/test_e2e.py -s

Tagged ``@pytest.mark.vllm`` + ``xdist_group("arctic_rl_vllm")``: under ``--dist loadgroup`` it shares a worker
with the other vLLM tests (never scheduled against them); ``-m "not vllm"`` drops it from a parallel pool.
"""

from __future__ import annotations

import asyncio

import pytest
from parameterized import parameterized
from rl_harness import arctic_rl_client_session
from rl_harness import assert_finite_logprobs
from rl_harness import assert_generations
from rl_harness import assert_positive_grad_norm
from rl_harness import build_compute_log_prob_payload
from rl_harness import build_update_actor_payload
from rl_harness import cell_tag
from rl_harness import finite_metric
from rl_harness import make_fake_batch
from rl_harness import parameterized_custom_name_func
from rl_harness import skip_if_unsupported

from arctic_platform.rl import create_arctic_rl_client
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import require_torch_multi_gpu

model_name = "Qwen/Qwen3-0.6B"
attn_implementation = "flash_attention_2"
num_unique_prompts = 2
rollout_n = 2
prompt_len = 8
response_len = 8

# Full GRPO loop -> a training (DeepSpeed) job and a sampling (vLLM) job. Production typically colocates them on
# shared GPUs via fractional Ray resources, so the e2e loop runs colocate=True (sync_weights then uses CUDA IPC).
training_gpus = 1
sampling_gpus = 1
log_prob_gpus = 0
colocate = True

num_steps = 2

# This is integration coverage; the ZoRRO/non-ZoRRO forward is numerically certified by test_train_engine over the
# full matrix. What differs at the e2e level is the transport (separate ray_server / http_server lifecycle code), so
# cover each transport once, paired diagonally with one forward path.
e2e_params = [("ray", True), ("http", False)]

# Real prompts for the sampling engine; small max_tokens keeps the rollout within the tiny max_model_len
# (prompt_len + response_len).
gen_prompts = ["2 + 2 =", "The capital of France is"]
gen_sampling_params = {"temperature": 0.0, "max_tokens": 6}

# enable_sleep_mode so sleep/wake_inference work; enable_prefix_caching so reset_prefix_cache is meaningful.
vllm_overrides = {"enable_sleep_mode": True, "enable_prefix_caching": True}

# test_sync_weights_changes_sampler: a large LR over a few steps moves the policy enough that greedy decoding
# visibly shifts once the new weights are synced to the sampler.
sync_test_lr = 0.01
sync_test_steps = 3


@require_torch_multi_gpu
@pytest.mark.gpu_serial
@pytest.mark.vllm
@pytest.mark.xdist_group("arctic_rl_vllm")
class TestE2E(TestCasePlus):
    def _assert_update(self, fwd_bwd_response: dict, step_response: dict, tag: str) -> None:
        self.assertIn("metrics", fwd_bwd_response, f"{tag}: fwd_bwd missing 'metrics': {list(fwd_bwd_response)}")
        loss = finite_metric(fwd_bwd_response["metrics"]["loss"])
        grad_norm = assert_positive_grad_norm(step_response)
        print(f"[e2e] {tag}: loss={loss:.6f} grad_norm={grad_norm:.6f}")

    async def _drive_grpo(self, client, batch: dict, zorro_enable: bool, tag: str) -> None:
        cl_payload = build_compute_log_prob_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)
        ua_payload = build_update_actor_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)

        for step in range(num_steps):
            step_tag = f"{tag}/step{step}"

            # Rollout on the sampling engine.
            await client.wake_inference()
            await client.reset_prefix_cache()
            assert_generations(await client.generate(gen_prompts, gen_sampling_params), len(gen_prompts), step_tag)
            await client.sleep_inference(level=1)

            # Old log-probs on the training engine (fwd_no_grad).
            await client.wake_training()
            assert_finite_logprobs(await client.fwd_no_grad(cl_payload, reference_model=False), batch)

            # No log-prob job in this topology -> these are graceful no-ops.
            self.assertEqual((await client.wake_log_prob()).get("status"), "no_log_prob_job")
            self.assertEqual((await client.sleep_log_prob()).get("status"), "no_log_prob_job")

            # Policy update: forward + loss + backward, then optimizer step.
            fwd_bwd_response = await client.fwd_bwd(ua_payload)
            step_response = await client.step()
            self._assert_update(fwd_bwd_response, step_response, step_tag)

            await client.empty_training_cache()

            # Push updated weights to the sampling engine (wakes inference internally). Under colocate this mirrors
            # the production CUDA-IPC wrap -- offload non-lp state, IPC-gather the still-resident bf16 params, then
            # offload the lp params -- which also exercises the sleep_training non_lp / lp_params modes. Alternate the
            # bulk vs low_memory streaming IPC path (one gathered param at a time) across steps to cover both.
            low_memory = colocate and step % 2 == 1
            if colocate:
                await client.sleep_training(mode="non_lp")
                sync = await client.sync_weights(cuda_ipc=True, low_memory=low_memory)
                await client.sleep_training(mode="lp_params")
            else:
                sync = await client.sync_weights(cuda_ipc=False)
            self.assertIsInstance(sync, dict, f"{step_tag}: sync_weights non-dict")

            # Offload everything between steps.
            await client.sleep_training(mode="all")

        # save_checkpoint needs training state resident.
        await client.wake_training()
        self.assertIsInstance(await client.save_checkpoint(), dict, f"{tag}: save_checkpoint non-dict")

        # Colocate CPU-file weight sync (cuda_ipc=False): gather the full (ZeRO-3) state dict to a shared-memory file
        # and reload it into the sampler -- the disk/CPU sync path, distinct from the CUDA-IPC path the loop drives.
        if colocate:
            cpu_sync = await client.sync_weights(cuda_ipc=False)
            self.assertIsInstance(cpu_sync, dict, f"{tag}: cpu-file sync_weights non-dict")

    def _run_e2e(self, comm_protocol: str, zorro_enable: bool) -> None:
        batch, _, _ = make_fake_batch(model_name, num_unique_prompts, rollout_n, prompt_len, response_len)
        tag = cell_tag(comm_protocol, zorro_enable)
        with arctic_rl_client_session(
            comm_protocol,
            zorro_enable,
            model_name,
            attn_implementation,
            prompt_len,
            response_len,
            rollout_n,
            training_gpus,
            sampling_gpus,
            log_prob_gpus,
            colocate=colocate,
            vllm_overrides=vllm_overrides,
        ) as client:
            asyncio.run(self._drive_grpo(client, batch, zorro_enable, tag))
            asyncio.run(self._assert_client_guards(client, comm_protocol, tag))

    async def _assert_client_guards(self, client, comm_protocol: str, tag: str) -> None:
        """Reconnect contract + intentionally-unimplemented surface, checked on the live client (no new spinup)."""
        # reconnect_config() must round-trip the live job ids -- the serializable handle the verl wrapper passes
        # across process boundaries to re-attach without re-running /initialize.
        rc = client.reconnect_config()
        self.assertEqual(rc.training_job_id, client.training_job_id, f"{tag}: reconnect_config lost training_job_id")
        self.assertEqual(rc.sampling_job_id, client.sampling_job_id, f"{tag}: reconnect_config lost sampling_job_id")

        if comm_protocol == "ray":
            # A second client built from reconnect_config + the live server state attaches to the SAME jobs without
            # spinning up new engines.
            client2 = create_arctic_rl_client(rc, client.get_server_state())
            self.assertEqual(client2.training_job_id, client.training_job_id, f"{tag}: reconnect attached wrong job")
            self.assertIsInstance(await client2.empty_training_cache(), dict, f"{tag}: reconnected client op failed")

            # Disk-based weight reload is deliberately unimplemented on the ray client.
            with self.assertRaises(NotImplementedError):
                await client.save_weights("/tmp/unused")
        else:
            # The http client implements disk-based save_weights as a graceful warn-on-error stub (server-side
            # reload is not fully implemented), so it posts to /sync-weights and must return without raising.
            await client.save_weights("/tmp/arl_unused_ckpt")

    @parameterized.expand(e2e_params, name_func=parameterized_custom_name_func)
    def test_e2e(self, comm_protocol, zorro_enable):
        """Run the full GRPO loop for 2 steps over one live client (one case per transport)."""
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus, colocate)
        self._run_e2e(comm_protocol, zorro_enable)

    async def _perturb_and_sync(self, client, payload: dict) -> tuple[list[str], list[str]]:
        greedy = {"temperature": 0.0, "max_tokens": 8}
        await client.wake_inference()
        before = [r["text"] for r in await client.generate(gen_prompts, greedy)]
        for _ in range(sync_test_steps):  # deliberately move the policy far enough to shift greedy decoding
            await client.fwd_bwd(payload)
            await client.step()
        await client.sync_weights()  # NCCL push to the sampling engine (non-colocate); wakes inference internally
        after = [r["text"] for r in await client.generate(gen_prompts, greedy)]
        return before, after

    def test_sync_weights_changes_sampler(self):
        """sync_weights propagates updated training weights to the sampling engine over NCCL (non-colocate).

        Greedy-generate, run several large-LR update actor steps to deliberately move the policy, sync_weights, then
        greedy-generate again: at least one prompt's completion must change. A no-op sync would leave greedy
        decoding identical, so a changed completion is direct evidence the new weights reached the sampler. This is
        the dedicated separate-GPU / NCCL case; the colocate / CUDA-IPC sync path is covered by the main e2e loop
        (colocate=True), so this test runs non-colocate.
        """
        skip_if_unsupported(training_gpus, sampling_gpus, log_prob_gpus)
        batch, _, _ = make_fake_batch(model_name, num_unique_prompts, rollout_n, prompt_len, response_len)
        payload = build_update_actor_payload(batch, False, rollout_n, prompt_len, response_len)
        with arctic_rl_client_session(
            "ray",
            False,
            model_name,
            attn_implementation,
            prompt_len,
            response_len,
            rollout_n,
            training_gpus,
            sampling_gpus,
            log_prob_gpus,
            vllm_overrides=vllm_overrides,
            lr=sync_test_lr,
        ) as client:
            before, after = asyncio.run(self._perturb_and_sync(client, payload))
        print(f"[e2e] sync_weights: before={before} after={after}")
        self.assertEqual(len(before), len(after), "generation count changed across sync")
        self.assertTrue(
            any(b != a for b, a in zip(before, after)),
            f"sync_weights changed no sampler output: before={before} after={after}",
        )
