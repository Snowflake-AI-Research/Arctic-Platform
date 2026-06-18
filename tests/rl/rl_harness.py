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

"""Shared harness for the Arctic RL heavyweight GPU integration tests.

Not collected by pytest (doesn't match ``test_*.py``). Holds everything common to the heavyweight RL test modules
(``test_train_engine.py``, ``test_log_prob_engine.py``, ``test_generate.py``, ``test_e2e.py``): fake-data builders,
the per-row HF reference, the ``ArcticRLClientConfig`` factory, the client lifecycle context manager, pytest-xdist
port allocation, the skip guard, and the host-wide GPU-serialization lock (engaged per test by the
``_serialize_gpu_work`` autouse fixture in ``tests/rl/conftest.py``). Model name, geometry, and GPU counts are
owned by each test module and passed in. Does not depend on ``arctic-verl``.

Session lifecycle (``arctic_rl_client_session``): release the driver's torch.distributed group, point ``TMPDIR`` at
a unique per-session root, re-probe a fresh set of ports, start the cluster (ray: in-process head in the driver;
http: a server subprocess that owns its own detached head and inherits that ``TMPDIR``), build the config, create
the client, ``yield`` it, then on exit destroy the jobs / stop the server and reap exactly the cluster this session
spawned (the http head's ``ray_arctic_*`` dir lives under the session ``TMPDIR``; ray uses its module handle). http
retries the spinup a few times (its subprocess can lose a startup race); ray is in-process, so a single attempt.

Port allocation (the central source of cross-run/-worker flakiness, so all of it is re-probed per session, never
fixed at import): each xdist worker owns a contiguous 8-port block via ``get_unique_port_number`` -- conftest takes
``base`` for the torch.distributed ``MASTER_PORT`` and these tests take ``base+1..`` for the http server. Ray
GCS/dashboard (``6379``/``8265``), the DeepSpeed rendezvous ``MASTER_PORT`` (``29500``), and the weight-sync NCCL
``ARL_WEIGHT_SYNC_PORT`` (``30500``) are each probed from a per-worker stride (``+ wid * 50``) so concurrent workers
never overlap and a not-yet-reaped squatter from a prior session is stepped over rather than reused.

pytest-xdist model: two modes, chosen automatically by ``tests/conftest._maybe_partition_gpus`` from the GPU count.
  - Partitioned (enough GPUs to give every worker a disjoint slice of >= the largest single test's need, e.g. 8
    GPUs under ``-n 4``): each worker is pinned to its own GPUs via ``CUDA_VISIBLE_DEVICES``, with its own ports and
    a unique session ``TMPDIR``, so GPU tests run truly in parallel. The ``gpu_serial_lock`` becomes a no-op and
    vLLM claims a larger share of its dedicated slice.
  - Serialized fallback (too few GPUs -- e.g. a 2-GPU box, or ``-n`` larger than gpus/slice): all workers share all
    GPUs, so the host-wide ``gpu_serial_lock`` (engaged per test by ``_serialize_gpu_work`` in
    ``tests/rl/conftest.py``) drives one GPU body at a time, making ``-n N`` safe but GPU-bound.
Either way the vLLM tests carry ``@pytest.mark.vllm`` so ``-m "not vllm"`` can lift them out of a pool, and per-test
isolation (ports / TMPDIR / GPU slice) means a sibling worker's live cluster is never touched on teardown.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import inspect
import math
import os
import shutil
import socket
import subprocess
import tempfile

import pytest
import torch
import torch.distributed as dist
from parameterized import parameterized
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from arctic_platform.rl import ArcticRLClientConfig
from arctic_platform.rl import create_arctic_rl_client
from arctic_platform.rl import ray_cluster
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import get_xdist_worker_count
from arctic_platform.testing_utils import get_xdist_worker_id

# Each xdist worker owns a contiguous 8-port block (get_unique_port_number); conftest claims ``base`` for the
# torch.distributed MASTER_PORT, so these tests claim from ``base+1`` for the http server.
_PORT_BASE = get_unique_port_number()


def _reserve_free_port(start: int, span: int) -> int:
    """First bindable port in ``[start, start + span)``.

    Probed (vs. a fixed port) and re-probed per client session, NOT once at import: a port left squatted by a prior
    cluster in this worker that hasn't been reaped yet is then stepped over rather than reused -- reusing a fixed
    Ray GCS port makes the next ``ray start`` attach to the stale daemon and fail with
    "Session name ... does not match".
    """
    for port in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free port found in [{start}, {start + span})")


# Topology constants that don't vary across tests. The GPU counts (training / sampling / log_prob) and the colocate
# flag are owned by each test module and passed into the builders below.
max_token_len_per_gpu = 4096


def required_gpus(training_gpus: int, sampling_gpus: int, log_prob_gpus: int, colocate: bool = False) -> int:
    # Colocated jobs share GPUs via fractional Ray resources, so the host only needs the largest single job's count.
    if colocate:
        return max(training_gpus, sampling_gpus, log_prob_gpus)
    return training_gpus + sampling_gpus + log_prob_gpus


def parameterized_custom_name_func(func, _param_num, param):
    # Put both params in the sub-test name (parameterized shows only the first).
    param_based_name = parameterized.to_safe_name("_".join(str(x) for x in param.args))
    return f"{func.__name__}_{param_based_name}"


def make_padded_fake_batch(
    vocab_size: int, num_unique_prompts: int, rollout_n: int, prompt_len: int, response_len: int, pad_token_id: int = 0
) -> tuple[dict, list[int], list[int]]:
    """Variable-length batch in the verl convention, plus per-row real-token counts.

    Prompts are LEFT-padded to ``prompt_len`` (so the prompt/response boundary stays at column ``prompt_len`` for
    ZoRRO's fixed-boundary dedup) and responses RIGHT-padded to ``response_len``. The prompt -- including its left
    padding -- is shared across a group's rollouts; real response lengths vary per rollout. Each row is
    ``[pad]*(prompt_len-pl) [prompt pl] [response rl] [pad]*(response_len-rl)``. Returns ``(batch, prompt_lens,
    response_lens)`` where the lens are the real (unpadded) token counts.
    """
    gen = torch.Generator().manual_seed(1234)
    seq_len = prompt_len + response_len
    rows, prompt_rows, masks, prompt_lens, response_lens = [], [], [], [], []
    for group in range(num_unique_prompts):
        pl = max(2, prompt_len - 3 * group)  # vary prompt length per group; >= 2
        left = prompt_len - pl
        real_prompt = torch.randint(1, vocab_size, (pl,), generator=gen, dtype=torch.long)
        prompt_row = torch.full((prompt_len,), pad_token_id, dtype=torch.long)
        prompt_row[left:] = real_prompt
        for rollout in range(rollout_n):
            rl = max(2, response_len - 3 * (group + rollout))  # vary response length per rollout; >= 2
            real_resp = torch.randint(1, vocab_size, (rl,), generator=gen, dtype=torch.long)
            row = torch.full((seq_len,), pad_token_id, dtype=torch.long)
            row[left:prompt_len] = real_prompt
            row[prompt_len : prompt_len + rl] = real_resp
            mask = torch.zeros(seq_len, dtype=torch.long)
            mask[left : prompt_len + rl] = 1
            rows.append(row)
            prompt_rows.append(prompt_row.clone())
            masks.append(mask)
            prompt_lens.append(pl)
            response_lens.append(rl)
    batch = dict(input_ids=torch.stack(rows), attention_mask=torch.stack(masks), prompts=torch.stack(prompt_rows))
    return batch, prompt_lens, response_lens


def make_fake_batch(
    model_name: str, num_unique_prompts: int, rollout_n: int, prompt_len: int, response_len: int
) -> tuple[dict, list[int], list[int]]:
    """``make_padded_fake_batch`` with the vocab size looked up from the model config."""
    vocab_size = AutoConfig.from_pretrained(model_name).vocab_size
    return make_padded_fake_batch(vocab_size, num_unique_prompts, rollout_n, prompt_len, response_len)


def base_meta(pad_token_id: int, zorro_enable: bool, rollout_n: int, prompt_len: int, response_len: int) -> dict:
    """The ``meta`` block all payloads share (update_actor augments it)."""
    return dict(
        zorro_train_enable=zorro_enable,
        zorro_train_max_rollouts=rollout_n,
        rollout_n=rollout_n,
        max_prompt_len=prompt_len,
        max_response_len=response_len,
        max_token_len_per_gpu=max_token_len_per_gpu,
        temperature=1.0,
        calculate_entropy=True,
        pad_token_id=pad_token_id,
        drop_position_ids=True,
        logits_optimization="none",
        logits_optimization_peak_mem_size_in_gib=4,
        logits_compute_in_fp32=False,
    )


def _left_pad(t: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Left-pad a response-only tensor with zero columns up to full sequence length (response_mask is zero
    there too, so the padding is inert)."""
    pad = torch.zeros(t.shape[:-1] + (seq_len - t.shape[-1],), dtype=t.dtype)
    return torch.cat([pad, t], dim=-1)


def build_compute_log_prob_payload(
    batch: dict, zorro_enable: bool, rollout_n: int, prompt_len: int, response_len: int, pad_token_id: int = 0
) -> dict:
    """Payload for ``fwd_no_grad`` (compute_log_prob): fake batch + meta + entropy/logprob post-processing."""
    meta = base_meta(pad_token_id, zorro_enable, rollout_n, prompt_len, response_len)
    processing = {"post": ["compute_entropy_and_logprobs"], "loss_fn": None}
    return dict(batch=batch, meta=meta, processing=processing)


def build_update_actor_payload(
    batch: dict,
    zorro_enable: bool,
    rollout_n: int,
    prompt_len: int,
    response_len: int,
    pad_token_id: int = 0,
    old_log_probs: torch.Tensor | None = None,
) -> dict:
    """Mirror the payload ``update_actor`` sends for one GRPO mini-batch.

    ``meta`` adds ``actor_config`` / ``policy_loss_config`` / per-mini-batch loss normalizers on top of
    ``base_meta`` (``dp_size`` is injected server-side). ``advantages`` / ``response_mask`` are fake and the default
    fake ``old_log_probs`` is random -- fine for a forward/backward smoke since the clipped-ratio loss clamps any
    finite delta, but the resulting ratio is far from 1 so the surrogate sits in its flat clipped region (tiny / zero
    gradient). Pass ``old_log_probs`` = the policy's actual response log-probs (e.g. from ``fwd_no_grad``) to start at
    ratio 1.0 so the unclipped surrogate is active and every step produces a real gradient (mirrors verl, where
    ``old_log_probs`` is the rollout-time policy snapshot). Shape ``[bsz, response_len]``. PG tensors are left-padded
    response-only -> full length, as the wrapper does; ``response_mask`` comes from ``attention_mask`` so right-padded
    response tokens are excluded (a no-op all-ones mask for uniform batches).
    """
    input_ids = batch["input_ids"]
    bsz, seq_len = input_ids.shape
    assert seq_len == prompt_len + response_len, f"unexpected seq_len {seq_len} != {prompt_len + response_len}"

    gen = torch.Generator().manual_seed(4321)
    responses = input_ids[:, prompt_len:].clone()
    response_mask = batch["attention_mask"][:, prompt_len:].to(torch.long)  # real response tokens only
    if old_log_probs is None:
        old_log_probs = -(torch.rand((bsz, response_len), generator=gen) * 5.0 + 0.1)
    else:
        expected_shape = (bsz, response_len)
        assert old_log_probs.shape == expected_shape, f"old_log_probs {tuple(old_log_probs.shape)} != {expected_shape}"
        old_log_probs = old_log_probs.detach().float().cpu()
    advantages = torch.randn((bsz, response_len), generator=gen)

    # Simplest GRPO knobs: no KL loss (no ref_log_prob), no entropy, token-mean.
    actor_config = dict(
        loss_agg_mode="token-mean",
        use_kl_loss=False,
        entropy_coeff=0.0,
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        clip_ratio_c=3.0,
    )
    policy_loss_config = dict(loss_mode="vanilla")

    meta = base_meta(pad_token_id, zorro_enable, rollout_n, prompt_len, response_len)
    meta.update(
        actor_config=actor_config,
        policy_loss_config=policy_loss_config,
        global_batch_size=bsz,
        batch_num_tokens=int(response_mask.sum()),
    )
    out_batch = dict(
        input_ids=input_ids,
        attention_mask=batch["attention_mask"],
        prompts=batch["prompts"],
        responses=responses,
        response_mask=_left_pad(response_mask, seq_len),
        old_log_probs=_left_pad(old_log_probs, seq_len),
        advantages=_left_pad(advantages, seq_len),
    )
    processing = {"post": ["apply_temperature", "compute_entropy_and_logprobs"], "loss_fn": "verl_grpo"}
    return dict(batch=out_batch, meta=meta, processing=processing)


def finite_metric(x) -> float:
    """Reduce a fwd_bwd / step metric to a single finite float.

    ``step`` merges across DP ranks via ``merge_dict_shards``, so a replicated scalar (e.g. ``last_lr`` /
    ``grad_norm``, identical on every rank) can arrive as a per-rank list -- take the first. ``fwd_bwd`` metrics
    are already scalars (``combine_metric_shards``).
    """
    if isinstance(x, (list, tuple)):
        assert len(x) > 0, "expected a non-empty metric list"
        x = x[0]
    if torch.is_tensor(x):
        x = x.flatten()[0].item() if x.numel() > 1 else x.item()
    x = float(x)
    assert math.isfinite(x), f"expected finite value, got {x}"
    return x


def cell_tag(comm_protocol: str, zorro_enable: bool) -> str:
    """Human-readable label for a (transport, forward-path) matrix cell, e.g. ``ray/zorro``."""
    return f"{comm_protocol}/{'zorro' if zorro_enable else 'nonzorro'}"


def assert_generations(results, expected_count: int, tag: str = "") -> list[str]:
    """Validate a ``generate`` response: a list of ``expected_count`` dicts, each with a non-empty ``text``; return
    the texts."""
    assert isinstance(results, list), f"{tag}: expected list of results, got {type(results)}"
    assert len(results) == expected_count, f"{tag}: expected {expected_count} results, got {len(results)}"
    texts = []
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"{tag}: result {i} not a dict: {type(result)}"
        assert "text" in result, f"{tag}: result {i} missing 'text': {list(result)}"
        text = result["text"]
        assert isinstance(text, str), f"{tag}: result {i} 'text' not a str: {type(text)}"
        assert len(text) > 0, f"{tag}: result {i} produced empty text"
        texts.append(text)
    return texts


def assert_finite_logprobs(response: dict, batch: dict) -> torch.Tensor:
    """Validate a ``fwd_no_grad`` response carries a finite ``[B, S]`` log-prob tensor; return it (float cpu)."""
    assert isinstance(response, dict), f"expected dict response, got {type(response)}"
    assert "batch" in response, f"response missing 'batch': {list(response)}"
    logprobs = response["batch"].get("logprobs")
    assert torch.is_tensor(logprobs), f"expected logprobs tensor, got {type(logprobs)}"
    assert logprobs.shape[0] == batch["input_ids"].shape[0], "logprobs batch dim mismatch"
    assert torch.isfinite(logprobs).all(), "logprobs contain non-finite values"
    return logprobs.detach().float().cpu()


def assert_positive_grad_norm(step_response: dict) -> float:
    """A real forward+loss+backward yields a finite, strictly-positive global grad norm in the ``step`` metrics."""
    assert isinstance(step_response, dict), f"step expected dict, got {type(step_response)}"
    metrics = step_response.get("metrics", {})
    assert "grad_norm" in metrics, f"step metrics missing 'grad_norm': {list(metrics)}"
    grad_norm = finite_metric(metrics["grad_norm"])
    assert grad_norm > 0.0, f"expected positive grad_norm, got {grad_norm}"
    return grad_norm


def reference_response_logprobs_padded(
    batch: dict,
    prompt_lens: list[int],
    response_lens: list[int],
    model_name: str,
    attn_implementation: str,
    response_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row unpadded reference for a padded batch: ``([B, response_len] logprobs, [B, response_len] valid mask)``.

    Each row's real tokens (``attention_mask == 1``) are run *alone*, so positions are an unambiguous ``0..len-1``
    arange (no left-pad offset to reconcile, unlike a single batched forward whose default position-ids would be
    wrong for left-padded rows). Response token ``t`` (``0..rl-1``) is predicted from the logits at ``pl-1+t``. Only
    the first ``rl`` columns are marked valid; padded columns are left at zero and excluded from comparison.
    """
    device = "cuda"
    model = (
        AutoModelForCausalLM.from_pretrained(model_name, attn_implementation=attn_implementation, dtype=torch.bfloat16)
        .to(device)
        .eval()
    )
    batch_size = batch["input_ids"].shape[0]
    out = torch.zeros(batch_size, response_len)
    valid = torch.zeros(batch_size, response_len, dtype=torch.bool)
    try:
        with torch.no_grad():
            for row in range(batch_size):
                pl, rl = prompt_lens[row], response_lens[row]
                real = batch["attention_mask"][row].bool()
                ids = batch["input_ids"][row][real].to(device).unsqueeze(0)  # [1, pl+rl], no padding
                log_softmax = torch.log_softmax(model(input_ids=ids).logits.float(), dim=-1)[0]  # [pl+rl, V]
                pred_idx = torch.arange(pl - 1, pl + rl - 1, device=device)
                resp_tokens = ids[0, pl : pl + rl]
                out[row, :rl] = log_softmax[pred_idx].gather(-1, resp_tokens.unsqueeze(-1)).squeeze(-1).cpu()
                valid[row, :rl] = True
    finally:
        del model
        torch.cuda.empty_cache()
    return out, valid


# The deterministic padded batch + its per-row reference depend only on (model, geometry), so build/compute once and
# share across every matrix cell -- and across the train-engine and log-prob-engine modules within a worker.
_REFERENCE_CACHE: dict = {}


def cached_padded_batch_and_reference(
    model_name: str,
    attn_implementation: str,
    num_unique_prompts: int,
    rollout_n: int,
    prompt_len: int,
    response_len: int,
) -> tuple[dict, list[int], list[int], torch.Tensor, torch.Tensor]:
    """``(batch, prompt_lens, response_lens, reference_logprobs, valid_mask)``, memoized on the geometry."""
    key = (model_name, attn_implementation, num_unique_prompts, rollout_n, prompt_len, response_len)
    if key not in _REFERENCE_CACHE:
        batch, prompt_lens, response_lens = make_fake_batch(
            model_name, num_unique_prompts, rollout_n, prompt_len, response_len
        )
        ref, valid = reference_response_logprobs_padded(
            batch, prompt_lens, response_lens, model_name, attn_implementation, response_len
        )
        _REFERENCE_CACHE[key] = (batch, prompt_lens, response_lens, ref, valid)
    return _REFERENCE_CACHE[key]


def response_region(logprobs: torch.Tensor, zorro_enable: bool, prompt_len: int, response_len: int) -> torch.Tensor:
    """Slice the ``[B, response_len]`` response region by the path's alignment convention.

    zorro is position-indexed (response at ``[prompt_len : prompt_len+response_len]``) and zero-fills the prompt
    region; non-zorro uses ``roll(-1)`` (response at ``[prompt_len-1 : prompt_len+response_len-1]``). A path silently
    regressing to the other convention then mismatches the reference at the compared positions.
    """
    if zorro_enable:
        assert (
            torch.count_nonzero(logprobs[:, :prompt_len]).item() == 0
        ), "expected prompt-region log-probs zero-filled"
        return logprobs[:, prompt_len : prompt_len + response_len]
    return logprobs[:, prompt_len - 1 : prompt_len + response_len - 1]


def assert_weight_norms_match(norms: dict, tag: str = "", rtol: float = 1e-3) -> tuple[float, float]:
    """A weight sync makes the two engines bit-identical, so their global L2 norms must agree.

    The norm is sqrt of the sum of squares over all params -- invariant to how each engine sublays/fuses its weights
    -- so the only residual gap is float64 summation order (observed: equal to ~4 decimals), hence the tight rtol.
    The engines legitimately report different ``num_params`` (vLLM fuses QKV / gate_up), so the counts are surfaced
    in the failure message but intentionally not asserted equal. Returns ``(training_norm, sampling_norm)``.
    """
    training = norms["training_norm"]
    sampling = norms["sampling_norm"]
    assert math.isfinite(training) and training > 0, f"{tag}: bad training_norm {training}"
    assert math.isfinite(sampling) and sampling > 0, f"{tag}: bad sampling_norm {sampling}"
    rel = abs(training - sampling) / max(training, sampling)
    assert rel <= rtol, (
        f"{tag}: weight norms diverge after sync -- training={training:.6f} sampling={sampling:.6f} rel={rel:.2e} "
        f"(num_params train={norms.get('training_num_params')} sample={norms.get('sampling_num_params')})"
    )
    return training, sampling


def tokenize_prompts(model_name: str, prompts: list[str]) -> list[list[int]]:
    """Token ids for each prompt (no special tokens), to mirror the ids vLLM fed to ``generate``."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return [tokenizer(prompt, add_special_tokens=False).input_ids for prompt in prompts]


def inference_response_logprobs(results: list[dict]) -> tuple[list[list[int]], list[list[float]]]:
    """From a ``generate(..., {"logprobs": 0})`` response, per row: the generated token ids and the inference
    engine's logprob of each generated (greedy) token. Robust to JSON-stringified dict keys on the http transport.
    """
    gen_token_ids, gen_logprobs = [], []
    for result in results:
        ids = list(result["token_ids"])
        positions = result["logprobs"]
        assert positions is not None, "generate did not return per-token logprobs (pass {'logprobs': 0})"
        row = []
        for tok_id, pos in zip(ids, positions):
            entry = pos.get(tok_id, pos.get(str(tok_id)))
            assert entry is not None, f"generated token {tok_id} missing from its logprob dict keys {list(pos)}"
            row.append(float(entry["logprob"]))
        gen_token_ids.append(ids)
        gen_logprobs.append(row)
    return gen_token_ids, gen_logprobs


def build_response_logprob_batch(
    prompt_token_ids: list[list[int]],
    gen_token_ids: list[list[int]],
    prompt_len: int,
    response_len: int,
    pad_token_id: int = 0,
) -> tuple[dict, list[int]]:
    """Left-pad prompt / right-pad response batch (verl convention) from explicit prompt + generated token ids.

    Feeds ``build_compute_log_prob_payload`` so the training engine recomputes log-probs for the exact tokens the
    sampler generated. Returns ``(batch, response_lens)``.
    """
    seq_len = prompt_len + response_len
    rows, prompt_rows, masks, response_lens = [], [], [], []
    for prompt_ids, gen_ids in zip(prompt_token_ids, gen_token_ids):
        pl, rl = len(prompt_ids), len(gen_ids)
        assert 0 < pl <= prompt_len, f"prompt of {pl} tokens does not fit prompt_len {prompt_len}"
        assert 0 < rl <= response_len, f"response of {rl} tokens does not fit response_len {response_len}"
        left = prompt_len - pl
        row = torch.full((seq_len,), pad_token_id, dtype=torch.long)
        row[left:prompt_len] = torch.tensor(prompt_ids, dtype=torch.long)
        row[prompt_len : prompt_len + rl] = torch.tensor(gen_ids, dtype=torch.long)
        prompt_row = torch.full((prompt_len,), pad_token_id, dtype=torch.long)
        prompt_row[left:] = torch.tensor(prompt_ids, dtype=torch.long)
        mask = torch.zeros(seq_len, dtype=torch.long)
        mask[left : prompt_len + rl] = 1
        rows.append(row)
        prompt_rows.append(prompt_row)
        masks.append(mask)
        response_lens.append(rl)
    batch = dict(input_ids=torch.stack(rows), attention_mask=torch.stack(masks), prompts=torch.stack(prompt_rows))
    return batch, response_lens


def logprob_kl(training_logprobs: torch.Tensor, inference_logprobs: list[list[float]], response_lens: list[int]):
    """k3 KL estimate (and mean abs diff) between the training and inference engines over the realized response
    tokens. ``training_logprobs`` is the ``[B, response_len]`` response region; inference is the per-row list of the
    same greedy tokens' logprobs. The tokens were sampled by the inference engine, so it is the behavior policy; the
    nonnegative k3 estimator ``E[exp(d) - 1 - d]`` (``d = train - infer``) is ~0 iff the two policies agree, which
    they must after a weight sync (only vLLM-vs-HF kernel + bf16 differences remain).
    """
    deltas = []
    for row, response_len in enumerate(response_lens):
        for token in range(response_len):
            deltas.append(float(training_logprobs[row, token]) - inference_logprobs[row][token])
    delta = torch.tensor(deltas)
    kl = torch.mean(torch.exp(delta) - 1.0 - delta).item()
    mean_abs_diff = delta.abs().mean().item()
    return kl, mean_abs_diff


def build_config(
    comm_protocol: str,
    checkpoint_path: str,
    zorro_enable: bool,
    model_name: str,
    attn_implementation: str,
    prompt_len: int,
    response_len: int,
    rollout_n: int,
    training_gpus: int,
    sampling_gpus: int,
    log_prob_gpus: int,
    colocate: bool = False,
    http_port: int | None = None,
    vllm_overrides: dict | None = None,
    lr: float = 1e-6,
    gradient_accumulation_steps: int = 1,
):
    """Minimal hand-rolled ``ArcticRLClientConfig`` (what the verl wrapper builds).

    Same config across transports; ``comm_protocol`` / ``zorro_enable`` / ``colocate`` are the knobs that vary. A
    sampling (vLLM) job is created only when ``sampling_gpus > 0``; ``vllm_overrides`` then merges into its config
    (e.g. ``enable_sleep_mode`` so the e2e test can exercise sleep/wake_inference). ``colocate`` packs training and
    sampling onto shared GPUs via fractional Ray resources (the server forces ``enable_sleep_mode`` in that mode).
    ``gradient_accumulation_steps > 1`` makes each rank split its shard into that many forward microbatches
    (``deepspeed_worker`` ``split_dict`` + per-microbatch metric merge).
    """
    max_length = prompt_len + response_len

    # train_batch_size must == micro_bs * grad_accum * world_size (== training_gpus). Floor the world size at 1 so a
    # log-prob-only topology (training_gpus=0) still yields a valid base config; the forward-only reference engine
    # ignores train_batch_size anyway (ds_inference_config), and no training engine is created to consume it.
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": max(1, training_gpus) * gradient_accumulation_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
    }
    training_config = {
        "optimizer": {"lr": lr, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        "lr_scheduler": {"warmup_ratio": 0.0},
        "training_horizon": 1,
        "max_length": max_length,
        "model_config": None,
        "attn_implementation": attn_implementation,
        "gradient_accumulation_steps": gradient_accumulation_steps,
    }
    # zorro_train_enable toggles the ZoRRO prompt-dedup forward in the DeepSpeed worker.
    ds_worker_config = dict(
        use_liger=False,
        enable_gradient_checkpointing=False,
        attn_implementation=attn_implementation,
        zorro_train_enable=zorro_enable,
        response_len=response_len,
        max_token_len=max_token_len_per_gpu,
        rollout_n=rollout_n,
        temperature=1.0,
        logits_optimization="none",
        logits_optimization_peak_mem_size_in_gib=4,
        logits_compute_from_fp32_inputs=False,
        logits_compute_in_fp32=False,
        use_unpad=True,
        use_autocast=False,
    )
    # vLLM sampling-engine config, only when a sampling job exists. The model is tiny (<2 GiB), so a small share of
    # the GPU is ample and stays tolerant of leftover allocations / CUDA contexts (0.9 would spuriously fail the
    # startup free-memory check the moment any memory is in use). When GPUs are partitioned this worker owns its
    # slice outright (0.3); otherwise workers share all GPUs under the serial lock, so divide by the worker count.
    vllm_config = None
    if sampling_gpus > 0:
        gpu_memory_utilization = 0.3 if gpu_partitioning_active() else 0.3 / get_xdist_worker_count()
        vllm_config = {
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_length,
            "enforce_eager": True,
            "enable_prefix_caching": False,
        }
        if vllm_overrides:
            vllm_config.update(vllm_overrides)

    # http binds a real port (re-probed per session, unique per worker); ray uses in-process actors.
    host_port_kwargs = {"port": http_port} if comm_protocol == "http" else {}

    return ArcticRLClientConfig(
        comm_protocol=comm_protocol,
        backend="local",
        training_gpus=training_gpus,
        sampling_gpus=sampling_gpus,
        log_prob_gpus=log_prob_gpus,
        colocate=colocate,
        log_prob_engine="deepspeed",
        model_name=model_name,
        ds_config=ds_config,
        log_prob_ds_config=None,
        training_config=training_config,
        ds_worker_config=ds_worker_config,
        use_arctic_inference=False,
        vllm_config=vllm_config,
        checkpoint_path=checkpoint_path,
        ray_auto_attach=False,  # force the http server subprocess to start its own cluster
        # Bound the blocking http /initialize: a healthy init is ~40-150s here, so 240s comfortably clears legit
        # (even colocate) startup while turning a wedged multi-GPU NCCL rendezvous into a Timeout the session retry
        # recovers from on fresh ports -- rather than an unbounded hang that only the per-test timeout would catch.
        job_ready_timeout=240.0,
        **host_port_kwargs,
    )


def teardown_client(client) -> None:
    """Destroy jobs / stop the server (ray shutdown is async, http sync)."""
    try:
        maybe_coro = client.shutdown()
        if inspect.isawaitable(maybe_coro):
            asyncio.run(maybe_coro)
    except Exception:  # best-effort; never mask the real test result
        pass


def force_stop_spawned_ray_cluster() -> None:
    # SIGKILL the Ray head init_ray_cluster() spawned -- a bare ray.shutdown() leaves the daemon alive, so the next
    # init attaches to its dead raylet and aborts.
    ray_cluster._shutdown()


def _reap_session_clusters(comm_protocol: str, session_ray_dir: str) -> None:
    """Tear down the Ray cluster this client session spawned so nothing lingers into the next test.

    ray: the driver owns the head -> ``force_stop_spawned_ray_cluster`` (also ``ray.shutdown()`` + reset cached
    address for the driver client). http: the server subprocess starts a detached head whose daemons / vLLM
    ``InferenceWorker`` + ``EngineCore`` actors survive the server's SIGTERM (and a -9 crash) and keep squatting
    GPUs / ports / /dev/shm. The session pinned the head to ``session_ray_dir`` (via ARL_RAY_TEMP_DIR), so SIGKILL
    that cluster by its unique ``--temp-dir`` basename and drop the dir. Keying off this session's own unique dir
    (rather than a global snapshot diff) keeps teardown race-free under parallel workers -- a sibling's live cluster
    carries a different basename and is never matched.
    """
    if comm_protocol == "ray":
        with contextlib.suppress(Exception):
            force_stop_spawned_ray_cluster()
        return
    subprocess.run(["pkill", "-9", "-f", os.path.basename(session_ray_dir)], check=False, timeout=60)
    shutil.rmtree(session_ray_dir, ignore_errors=True)


@contextlib.contextmanager
def arctic_rl_client_session(
    comm_protocol: str,
    zorro_enable: bool,
    model_name: str,
    attn_implementation: str,
    prompt_len: int,
    response_len: int,
    rollout_n: int,
    training_gpus: int,
    sampling_gpus: int,
    log_prob_gpus: int,
    colocate: bool = False,
    vllm_overrides: dict | None = None,
    lr: float = 1e-6,
    gradient_accumulation_steps: int = 1,
):
    """Own the full client lifecycle and ``yield`` a ready client."""
    # Release the driver's torch.distributed group so the rank-0 worker can rebind this worker's MASTER_PORT.
    if dist.is_initialized():
        dist.destroy_process_group()

    # http launches the server as a subprocess that occasionally loses a startup race against a not-yet-reaped
    # cluster (or hits a transient /initialize 500 under GPU contention) and dies; reap its debris and retry on
    # fresh ports. ray spins up in-process (no flaky subprocess), so a single attempt.
    attempts = 3 if comm_protocol == "http" else 1
    with tempfile.TemporaryDirectory(prefix="arl_test_ckpt_") as ckpt_dir:
        for attempt in range(1, attempts + 1):
            wid = get_xdist_worker_id()
            # Pre-create this session's Ray temp dir (shallow, under the default tmp -- Ray's AF_UNIX socket paths
            # must stay under 107 bytes, so we must NOT nest it) and hand it to the cluster via ARL_RAY_TEMP_DIR. The
            # http server subprocess inherits the env and starts its head there, so teardown reaps exactly this
            # cluster by its unique basename -- never a parallel sibling's -- making ``-n N`` partitioning race-free.
            session_ray_dir = tempfile.mkdtemp(prefix="ray_arctic_")
            prev_ray_dir = os.environ.get("ARL_RAY_TEMP_DIR")
            os.environ["ARL_RAY_TEMP_DIR"] = session_ray_dir
            try:
                # Re-probe ports per attempt (not once at import) so a port left squatted by a prior, not-yet-reaped
                # cluster in this worker is skipped rather than reused. Ray GCS/dashboard ports must stay below Ray's
                # worker-port range (>= 10002); 6379/8265 are Ray's defaults, strided per worker so concurrent
                # workers never overlap. Each worker starts its OWN head in its own temp-dir, so address="auto" never
                # resolves to a sibling's cluster; do NOT set RAY_ADDRESS. http_port stays in this worker's port block.
                http_port = _reserve_free_port(_PORT_BASE + 1, span=7) if comm_protocol == "http" else None
                os.environ["RAY_PORT"] = str(_reserve_free_port(6379 + wid * 50, span=50))
                os.environ["RAY_DASHBOARD_PORT"] = str(_reserve_free_port(8265 + wid * 50, span=50))
                # Stride Ray's CoreWorker gRPC port range per worker (init_ray_cluster passes these as
                # --min/--max-worker-port). Ray's default range (10002+) is shared by every cluster on the host, so
                # two concurrent per-worker clusters (partitioned GPU path) collide on a worker port -- a fatal,
                # non-retried CoreWorker bind error that crashes the worker. A 1000-port block per worker is ample
                # for these tiny clusters; the base sits above the http port block (~11000) and below MASTER_PORT
                # (29500+), so the ranges never overlap the other strided ports for any realistic ``-n``.
                ray_worker_port_lo = 12100 + wid * 1000
                os.environ["ARL_RAY_MIN_WORKER_PORT"] = str(ray_worker_port_lo)
                os.environ["ARL_RAY_MAX_WORKER_PORT"] = str(ray_worker_port_lo + 999)
                # DeepSpeed rendezvous port (ray_server / http_server read os.environ["MASTER_PORT"] to hand every
                # rank the same value). Re-probe a fresh free port per session, strided per worker, instead of
                # reusing the single static MASTER_PORT conftest set for the whole run: a SIGKILL-reaped worker from a
                # prior session (e.g. the same heavy test fired repeatedly under pytest-flakefinder) can still squat
                # the old port when the next session's rank-0 worker creates its TCPStore, deadlocking the rendezvous.
                os.environ["MASTER_PORT"] = str(_reserve_free_port(29500 + wid * 50, span=50))
                # Same fix for the training->sampling weight-sync NCCL rendezvous (servers read ARL_WEIGHT_SYNC_PORT);
                # base well clear of MASTER_PORT's window so the two never overlap across workers.
                os.environ["ARL_WEIGHT_SYNC_PORT"] = str(_reserve_free_port(30500 + wid * 50, span=50))

                # ray: start the head in the driver (the server actor re-attaches). http: the server subprocess owns
                # its own cluster. auto_attach=False is essential under xdist or workers attach to each other.
                if comm_protocol == "ray":
                    ray_cluster.init_ray_cluster(auto_attach=False)

                config = build_config(
                    comm_protocol,
                    ckpt_dir,
                    zorro_enable=zorro_enable,
                    model_name=model_name,
                    attn_implementation=attn_implementation,
                    prompt_len=prompt_len,
                    response_len=response_len,
                    rollout_n=rollout_n,
                    training_gpus=training_gpus,
                    sampling_gpus=sampling_gpus,
                    log_prob_gpus=log_prob_gpus,
                    colocate=colocate,
                    http_port=http_port,
                    vllm_overrides=vllm_overrides,
                    lr=lr,
                    gradient_accumulation_steps=gradient_accumulation_steps,
                )
                try:
                    client = create_arctic_rl_client(config)  # http: launches + waits on the server subprocess
                except Exception:
                    _reap_session_clusters(comm_protocol, session_ray_dir)
                    if attempt == attempts:
                        raise
                    continue
                try:
                    yield client
                finally:
                    teardown_client(client)
                    _reap_session_clusters(comm_protocol, session_ray_dir)
                return
            finally:
                if prev_ray_dir is None:
                    os.environ.pop("ARL_RAY_TEMP_DIR", None)
                else:
                    os.environ["ARL_RAY_TEMP_DIR"] = prev_ray_dir
                shutil.rmtree(session_ray_dir, ignore_errors=True)


def skip_if_unsupported(training_gpus: int, sampling_gpus: int, log_prob_gpus: int, colocate: bool = False) -> None:
    """Skip unless CUDA, enough GPUs, and the full inference/training stack exist."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Arctic RL path")
    required = required_gpus(training_gpus, sampling_gpus, log_prob_gpus, colocate)
    if torch.cuda.device_count() < required:
        pytest.skip(f"need >= {required} GPU(s); have {torch.cuda.device_count()}")
    pytest.importorskip("ray")
    pytest.importorskip("arctic_inference")
    pytest.importorskip("vllm")
    pytest.importorskip("deepspeed")


# Host-wide lock path shared across all the GPU test modules so their GPU-heavy bodies serialize against each
# other too. Resolved at import (before any per-session TMPDIR override) so it stays a single host-wide path.
GPU_SERIAL_LOCK_PATH = os.path.join(tempfile.gettempdir(), "arl_test_gpu.lock")


def gpu_partitioning_active() -> bool:
    """True when conftest gave each xdist worker its own disjoint GPU slice (see tests/conftest._maybe_partition_gpus).

    In that mode workers never share GPUs, so the host-wide serial lock is unnecessary and vLLM can claim a larger
    share of its dedicated slice.
    """
    return os.environ.get("ARL_GPU_PARTITIONED") == "1"


@contextlib.contextmanager
def gpu_serial_lock():
    """Serialize GPU-heavy bodies across xdist workers.

    Multiple workers spinning up DeepSpeed engines on the shared GPUs at once contend for VRAM and trip init-time
    memory checks / OOM. Hold a host-wide advisory lock so one worker drives the GPUs at a time. No-op in a serial
    run and when GPUs are partitioned (each worker owns a disjoint slice, so there is nothing to serialize).
    """
    if get_xdist_worker_count() <= 1 or gpu_partitioning_active():
        yield
        return
    with open(GPU_SERIAL_LOCK_PATH, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
