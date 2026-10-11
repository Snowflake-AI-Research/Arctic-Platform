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
"""SkyRL trainer that keeps the generator's loss mask and sends the GRPO config.

SkyRL's Arctic dispatch replaces ``loss_mask`` with ``response_mask`` and does
not forward clip or GSPO settings. Tool observations would then be trained on,
and Cortex would use its default token-mean clip of 0.2. This dispatch sends
the mask the rollout built and the config from ``grpo_processing_config``.

Adapted from ``src/cortex/trainer.py`` in
https://github.com/18jeffreyma/codescout/tree/abab719e08a55dde78c6da864cd24d84fd47bdf2

SkyRL is imported inside :func:`openhands_trainer_cls` so importing this module
does not require a SkyRL checkout.
"""

from __future__ import annotations

import json
import os
import re
import time

from arctic_platform.integrations.openhands.config import grpo_processing_config

_CHECKPOINT_ID_RE = re.compile(r"/checkpoints/(cp_[0-9a-fA-F-]+)/")


def openhands_trainer_cls():
    """The trainer class, built once SkyRL is importable."""
    from integrations.arctic_rl.trainer import ArcticPPOTrainer
    from integrations.arctic_rl.trainer import _ArcticDispatch
    from integrations.arctic_rl.trainer import _run

    class OpenHandsDispatch(_ArcticDispatch):
        def __init__(self, cfg, client) -> None:
            super().__init__(cfg, client)
            self._loss_config = grpo_processing_config(cfg)

        def forward_backward(self, model: str, data, loss_fn=None, loss_fn_config=None) -> dict[str, float]:
            repacked = self._repack_to_verl_shape(data)
            max_prompt = repacked.pop("_max_prompt_len")
            max_response = repacked.pop("_max_response_len")
            seq_len = repacked["input_ids"].shape[-1]
            response_mask = self._left_pad_to_seq(repacked["response_mask"], seq_len)
            loss_mask = self._left_pad_to_seq(repacked["loss_mask"], seq_len)
            loss_mask = (loss_mask > 0) & (response_mask > 0)
            meta = self._build_meta(
                max_prompt_len=max_prompt,
                max_response_len=max_response,
                batch_num_tokens=int(loss_mask.sum().item()),
                global_batch_size=self.cfg.trainer.policy_mini_batch_size * self.cfg.generator.n_samples_per_prompt,
                calculate_entropy=True,
            )
            batch = dict(
                input_ids=repacked["input_ids"],
                attention_mask=repacked["attention_mask"],
                prompts=repacked["prompts"],
                responses=repacked["responses"],
                position_ids=repacked["position_ids"],
                response_mask=response_mask,
                loss_mask=loss_mask,
                advantages=self._left_pad_to_seq(repacked["advantages"], seq_len),
            )
            result = _run(
                self.client.fwd_bwd(dict(batch=batch, meta=meta, processing=dict(config=dict(self._loss_config))))
            )
            result.pop("job_id", None)
            metrics = result.get("metrics", result)
            return metrics

        def get_timing_metrics(self) -> dict[str, float]:
            return {}

        def finalize_pending_saves(self, model: str) -> None:
            return None

    class OpenHandsTrainer(ArcticPPOTrainer):
        def build_models(self, PolicyWorker=None, CriticWorker=None, RefWorker=None):
            super().build_models(PolicyWorker, CriticWorker, RefWorker)
            self.dispatch = OpenHandsDispatch(self.cfg, self._arctic_client)

        def save_checkpoints(self) -> str:
            return self._save_cortex_checkpoint("resumable")

        def save_models(self) -> None:
            self._save_cortex_checkpoint("weights-only")

        def _save_cortex_checkpoint(self, checkpoint_type: str) -> str:
            result = _run(self._arctic_client.save_checkpoint(checkpoint_type=checkpoint_type))
            stage_path = result.get("stage_path") or ""
            match = _CHECKPOINT_ID_RE.search(stage_path)
            record = dict(
                global_step=self.global_step,
                checkpoint_type=checkpoint_type,
                checkpoint_id=match.group(1) if match is not None else None,
                checkpoint_tag=result.get("checkpoint_tag"),
                stage_path=stage_path,
                training_job_id=str(self._arctic_client.training_job_id),
                saved_at=time.time(),
            )
            os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)
            path = os.path.join(self.cfg.trainer.ckpt_path, "cortex_checkpoints.jsonl")
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            return stage_path

    return OpenHandsTrainer
