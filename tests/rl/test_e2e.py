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
sleep/wake_log_prob (graceful no-ops -- no log-prob job here), save_checkpoint, weight_norm (after the final sync,
asserting the training and sampling engines hold identical weights and produce agreeing log-probs), reconnect_config
/ get_server_state (a second client re-attaches to the live jobs without re-initializing), save_weights (raises
``NotImplementedError`` on the ray client; a graceful warn-on-error disk-reload stub on the http client),
shutdown (via the session). Not exercised: ``log_probs``
(needs a log-prob engine; this 2-GPU training+sampling topology has none -- covered by test_log_prob_engine).
``test_sync_weights_nccl``
additionally proves a weight update actually propagates to the sampler over the NCCL path. Real prompts go through
generate; the update uses fake ``advantages`` but real ``old_log_probs`` (recomputed from the policy each step via
fwd_no_grad), so the clipped ratio starts at 1.0 and every step makes a real gradient step. Covers each transport
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
from rl_harness import assert_weight_norms_match
from rl_harness import build_compute_log_prob_payload
from rl_harness import build_response_logprob_batch
from rl_harness import build_update_actor_payload
from rl_harness import cell_tag
from rl_harness import finite_metric
from rl_harness import inference_response_logprobs
from rl_harness import logprob_kl
from rl_harness import make_fake_batch
from rl_harness import parameterized_custom_name_func
from rl_harness import response_region
from rl_harness import skip_if_unsupported
from rl_harness import tokenize_prompts

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

# A large LR so the update_actor steps move the weights well clear of the initial checkpoint. The post-sync exactness
# check (_assert_weight_sync) only catches a no-op / stale sync if the trainer has diverged from the sampler's
# starting weights by MORE than the norm-equality rtol -- otherwise a sync that did nothing would leave the stale
# sampler within tolerance of the (barely-moved) trainer and pass silently. We additionally assert the realized
# movement exceeds weight_movement_min_rel so this property is enforced, not assumed.
e2e_lr = 1e-2

# Minimum relative weight movement (vs the initial checkpoint) required of the trainer before the final sync, so the
# equal-norms check has teeth. Comfortably above the norm-equality rtol (1e-3); the observed move at e2e_lr is ~1e-2.
weight_movement_min_rel = 5e-3

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

# test_sync_weights_nccl: a large LR over a few steps moves the policy well clear of the initial checkpoint, so the
# post-sync exactness check has teeth (a no-op NCCL push would leave the stale sampler mismatched against the
# updated trainer).
sync_test_lr = 0.01
sync_test_steps = 3

# Post-sync weight-sync verification: greedy-decode a few tokens and recompute their log-probs on the training
# engine. With identical (just-synced) weights the two engines agree to within vLLM-vs-HF kernel + bf16 noise, so
# the observed k3 KL is ~0 (0 for the exact non-zorro path, <1e-3 for zorro); the threshold sits well above that
# noise floor but tight enough to catch a real divergence. max_tokens stays small so prompt+response fits
# max_model_len (prompt_len + response_len).
kl_max_tokens = 4
kl_threshold = 0.05


@require_torch_multi_gpu
@pytest.mark.gpu_serial
@pytest.mark.vllm
@pytest.mark.xdist_group("arctic_rl_vllm")
class TestE2E(TestCasePlus):
    def _assert_update(self, fwd_bwd_response: dict, step_response: dict, tag: str) -> None:
        self.assertIn("metrics", fwd_bwd_response, f"{tag}: fwd_bwd missing 'metrics': {list(fwd_bwd_response)}")
        loss = finite_metric(fwd_bwd_response["metrics"]["loss"])
        # old_log_probs is recomputed from the current policy each step (ratio starts at 1.0), so the unclipped
        # surrogate is active and every step -- not just the first -- must yield a strictly positive gradient.
        grad_norm = assert_positive_grad_norm(step_response)
        print(f"[e2e] {tag}: loss={loss:.6f} grad_norm={grad_norm:.6f}")

    async def _drive_grpo(self, client, batch: dict, zorro_enable: bool, tag: str) -> None:
        cl_payload = build_compute_log_prob_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)

        # Initial (untrained) norm: both engines are freshly initialized and identical here, so this is the baseline
        # the post-sync check measures movement against to prove the update + sync actually changed the weights.
        initial_norm = (await client.weight_norm())["training_norm"]

        for step in range(num_steps):
            step_tag = f"{tag}/step{step}"

            # Rollout on the sampling engine.
            await client.wake_inference()
            await client.reset_prefix_cache()
            assert_generations(await client.generate(gen_prompts, gen_sampling_params), len(gen_prompts), step_tag)
            await client.sleep_inference(level=1)

            # Old log-probs on the training engine (fwd_no_grad) -- the current policy snapshot. Feeding these as the
            # update's ``old_log_probs`` makes the clipped ratio start at exactly 1.0, so the unclipped surrogate is
            # active and this step produces a real gradient (recomputed each step, as verl does).
            await client.wake_training()
            old_lp = assert_finite_logprobs(await client.fwd_no_grad(cl_payload, reference_model=False), batch)
            old_region = response_region(old_lp, zorro_enable, prompt_len, response_len)
            ua_payload = build_update_actor_payload(
                batch, zorro_enable, rollout_n, prompt_len, response_len, old_log_probs=old_region
            )

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
            # Training is awake (save_checkpoint above) and the sampler just received the same weights -> verify the
            # sync actually landed: identical weights on both engines, and agreeing log-probs.
            await self._assert_weight_sync(client, zorro_enable, tag, initial_norm)

    async def _assert_weight_sync(self, client, zorro_enable: bool, tag: str, initial_norm=None) -> None:
        """After a weight sync the training and sampling engines hold identical weights -- prove it three ways.

        1. Global L2 weight norm of both engines must match (layout-invariant, so comparable across DeepSpeed's
           ZeRO-3 sharding and vLLM's fused params).
        2. If ``initial_norm`` (the pre-training baseline) is given, the synced norm must have moved away from it by
           more than ``weight_movement_min_rel`` -- otherwise the matching norms in (1) could be a no-op sync agreeing
           with a barely-trained trainer. This is what makes the equality check able to catch a stale/no-op sync.
        3. Greedy-decode a few tokens on the sampler (capturing its per-token log-probs), recompute those exact
           tokens' log-probs on the training engine, and require a tiny KL -- direct evidence the synced weights
           produce the same policy. The recompute uses the engine's own forward path (``zorro_enable`` matches the
           cell, since the training engine is patched accordingly at init); the two prompts are distinct, so no
           shared-prompt dedup and the zorro first-response-token alignment is exercised on single-rollout groups.
        """
        norms = await client.weight_norm()
        training_norm, sampling_norm = assert_weight_norms_match(norms, tag)
        print(f"[e2e] {tag}: weight_norm training={training_norm:.4f} sampling={sampling_norm:.4f}")

        if initial_norm is not None:
            rel_move = abs(sampling_norm - initial_norm) / initial_norm
            print(f"[e2e] {tag}: weight movement rel={rel_move:.4e} (initial={initial_norm:.4f})")
            self.assertGreater(
                rel_move,
                weight_movement_min_rel,
                f"{tag}: trainer barely moved (rel={rel_move:.2e} <= {weight_movement_min_rel:.0e}); a no-op sync "
                f"would pass the equal-norms check undetected -- raise the LR / step count",
            )

        gen = await client.generate(gen_prompts, {"temperature": 0.0, "max_tokens": kl_max_tokens, "logprobs": 0})
        gen_token_ids, inference_logprobs = inference_response_logprobs(gen)
        prompt_token_ids = tokenize_prompts(model_name, gen_prompts)
        # The KL recompute only aligns if the training engine sees the exact prefix the sampler generated from, so
        # guard that our tokenization reproduces vLLM's prompt ids (a future tokenizer adding specials would
        # otherwise silently shift the prefix and inflate KL with an opaque cause).
        for i, (ids, result) in enumerate(zip(prompt_token_ids, gen)):
            sampler_len = result["prompt_len"]
            self.assertEqual(
                len(ids), sampler_len, f"{tag}: prompt {i} tokenization != sampler ({len(ids)} vs {sampler_len})"
            )
        batch, response_lens = build_response_logprob_batch(prompt_token_ids, gen_token_ids, prompt_len, response_len)
        payload = build_compute_log_prob_payload(batch, zorro_enable, rollout_n, prompt_len, response_len)
        logprobs = assert_finite_logprobs(await client.fwd_no_grad(payload, reference_model=False), batch)
        training_region = response_region(logprobs, zorro_enable, prompt_len, response_len)
        kl, mean_abs_diff = logprob_kl(training_region, inference_logprobs, response_lens)
        num_tokens = sum(response_lens)
        print(f"[e2e] {tag}: train/infer logprob KL={kl:.4e} mean|delta|={mean_abs_diff:.4e} over {num_tokens} tokens")
        self.assertLess(kl, kl_threshold, f"{tag}: train/infer logprob KL too large after sync: {kl}")

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
            lr=e2e_lr,
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

    async def _perturb_sync_and_verify(self, client, payload: dict) -> None:
        initial_norm = (await client.weight_norm())["training_norm"]
        await client.wake_inference()
        for _ in range(sync_test_steps):  # move the policy well clear of the initial checkpoint
            await client.fwd_bwd(payload)
            await client.step()
        await client.sync_weights()  # NCCL push to the sampling engine (non-colocate); wakes inference internally
        await self._assert_weight_sync(client, False, "nccl", initial_norm)

    def test_sync_weights_nccl(self):
        """sync_weights over NCCL (non-colocate) makes the sampler an exact copy of the updated trainer.

        Runs several large-LR update actor steps to move the policy well clear of the initial weights, NCCL-syncs to
        the sampler, then applies the same exactness checks as the main loop (``_assert_weight_sync``): equal global
        weight norms + a tiny train/infer log-prob KL. The large LR is what gives this teeth -- a no-op / stale sync
        would leave the (heavily updated) trainer and the unchanged sampler with clearly different norms, failing the
        equality check. This is the dedicated separate-GPU / NCCL path; the colocate CUDA-IPC + CPU-file sync paths
        are covered by the main e2e loop (colocate=True), so this test runs non-colocate.

        This test is separate, rather than adding non-colocate to the e2e options matrix, since it's much lighter and
        only tests the unique code path not already covered by the colocate e2e path.
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
            asyncio.run(self._perturb_sync_and_verify(client, payload))
