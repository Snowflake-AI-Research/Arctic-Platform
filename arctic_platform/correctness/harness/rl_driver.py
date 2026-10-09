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

"""Drive a reinforcement-learning pair -- the config's training and sampling sub-jobs -- through one gateway.

``dss_driver`` addresses one job at a time. An RL check keeps both sub-jobs of the config alive together and
sends the requests that connect them: generate on the sampling job, the router-replay bootstrap, fwd-bwd in the
RL batch format, and weight sync from the training job into the sampling job.
"""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Dict
from typing import Iterator
from typing import List
from typing import Sequence

from arctic_platform.client import ArcticClient

from .dss_driver import GatewaySession
from .dss_driver import _training_client_config


def sub_job_of_type(cfg, job_type: str) -> dict:
    matches = [sub for sub in cfg.sub_jobs if sub.get("job_type") == job_type]
    if len(matches) != 1:
        raise ValueError(f"{cfg.config_id}: expected one {job_type} sub-job, found {len(matches)}")
    return matches[0]


def rl_sampling_payload(cfg, model_path: str) -> dict:
    """The config's sampling sub-job as written, serving the model the spec names instead of the full one."""
    payload = copy.deepcopy(sub_job_of_type(cfg, "sampling"))
    payload["model_name"] = str(model_path)
    return payload


def rl_gpu_count(cfg) -> int:
    """GPUs the two sub-jobs need at once: each takes its own devices, so the count is the sum."""
    sampling = int(sub_job_of_type(cfg, "sampling")["inference_config"]["n_gpus"])
    return int(cfg.n_gpus) + sampling


@contextlib.contextmanager
def running_rl_client(cfg, training_payload: dict, model_path: str, workdir: Path) -> Iterator[ArcticClient]:
    """Create one native client whose training and sampling jobs share the reviewed config."""
    if cfg.native is None:
        raise ValueError(f"{cfg.config_id}: native ArcticClientConfig is required")
    session = GatewaySession(Path(workdir), rl_gpu_count(cfg))
    training_only = _training_client_config(session, training_payload)
    native = cfg.native.model_copy(
        update={
            "model_name": str(model_path),
            "training": training_only.training,
            "backend": training_only.backend,
        },
        deep=True,
    )
    client = ArcticClient(native)
    try:
        yield client
    finally:
        client.shutdown()


def generate(client: ArcticClient, prompts: Sequence[Sequence[int]], sampling_params: Sequence[dict]) -> List[dict]:
    """Generate one result per prompt, preserving each row's sampling parameters."""
    results = client.generate([list(prompt) for prompt in prompts], [dict(params) for params in sampling_params])
    if not isinstance(results, list) or len(results) != len(prompts):
        raise RuntimeError(f"generate returned {type(results).__name__} for {len(prompts)} prompts")
    return results


def sync_weights(client: ArcticClient) -> dict:
    """Sync the native client's training job into its sampling job."""
    return client.sync_weights()


def rl_fwd_bwd(client: ArcticClient, batch: dict, *, replay_sampling: bool = False) -> dict:
    """Run one RL forward-backward, optionally carrying the sampler identity for router replay."""
    replay = None
    if replay_sampling:
        replay = {"sampling_job_id": str(client.jobs.sampling), "identity_version": 1}
    return client.fwd_bwd(batch, router_replay=replay)


def step(client: ArcticClient, learning_rate: float) -> dict:
    return client.step(learning_rate)


def rl_batch(rows: Sequence[Sequence[int]], loss_starts: Sequence[int], pad_id: int) -> Dict:
    """Right-padded RL batch in the format ``tests/sp/sp_gateway_harness.py:padded_rl_batch`` sends.

    ``loss_mask`` is logit aligned: the entry at ``s`` weights the prediction of the token at ``s + 1``, so a
    row whose scored tokens begin at ``p`` carries ones over ``p - 1`` through ``len(row) - 2``.
    """
    import torch

    width = max(len(row) for row in rows)
    count = len(rows)
    input_ids = torch.full((count, width), int(pad_id), dtype=torch.long)
    attention_mask = torch.zeros(count, width, dtype=torch.long)
    loss_mask = torch.zeros(count, width, dtype=torch.float32)
    for index, (row, start) in enumerate(zip(rows, loss_starts, strict=True)):
        input_ids[index, : len(row)] = torch.tensor(list(row), dtype=torch.long)
        attention_mask[index, : len(row)] = 1
        loss_mask[index, start - 1 : len(row) - 1] = 1.0
    return {
        "args": (),
        "kwargs": {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "temperature": torch.ones(count, width, dtype=torch.float32),
        },
        "context": {
            "input_ids": input_ids.clone(),
            "advantages": torch.ones(count, width, dtype=torch.float32),
            "loss_mask": loss_mask,
        },
        "processing": {
            "post": ["compute_logprobs"],
            "loss_fn": "grpo",
            "config": {"loss_agg_mode": "token-mean"},
        },
    }


def find_key(value, key: str) -> List:
    """Every value stored under ``key`` anywhere in a nested response, in traversal order."""
    found: List = []
    if isinstance(value, dict):
        for name, item in value.items():
            if name == key:
                found.append(item)
            else:
                found.extend(find_key(item, key))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(find_key(item, key))
    return found
