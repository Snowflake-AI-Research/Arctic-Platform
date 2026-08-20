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

"""Rollout via ``SyncArcticRLClient.generate`` (single-turn, sync rewards)."""

from __future__ import annotations

import asyncio
import inspect
import queue
import threading
import time
from collections.abc import Iterable
from typing import Any

import numpy as np
from trl.experimental.async_grpo.async_rollout_worker import RolloutSample

from arctic_platform.integrations.trl.client import engine_old_log_probs

try:
    from trl.experimental.trainer.utils import get_callable_name
except Exception:  # noqa: BLE001

    def get_callable_name(fn: Any) -> str:
        return getattr(fn, "__name__", repr(fn))


def _sampled_logprob(position: dict, token_id: int) -> float:
    """Sampled-token logprob from one generate position (int or str keys)."""
    if token_id in position:
        return float(position[token_id]["logprob"])
    if str(token_id) in position:
        return float(position[str(token_id)]["logprob"])
    return float(next(iter(position.values()))["logprob"])


class ArcticRolloutWorker:
    """Arctic generate + score loop for ``RolloutWorkerProtocol``.

    ``old_logprobs_source="trainer"`` (default) recomputes old logprobs on the training engine.
    """

    def __init__(
        self,
        client: Any,
        dataset: Iterable[dict],
        reward_funcs: Any,
        processing_class: Any,
        *,
        num_generations: int = 8,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        min_p: float | None = None,
        repetition_penalty: float = 1.0,
        queue_maxsize: int = 0,
        chat_template_kwargs: dict[str, Any] | None = None,
        old_logprobs_source: str = "trainer",
        pad_token_id: int = 0,
        max_token_len_per_gpu: int = 4096,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0 so a group's generations differ (GRPO needs variance).")
        if old_logprobs_source not in ("trainer", "sampler"):
            raise ValueError(f"old_logprobs_source must be 'trainer' or 'sampler', got {old_logprobs_source!r}.")
        self.client = client
        self.dataset = dataset
        self.reward_funcs = reward_funcs if isinstance(reward_funcs, list) else [reward_funcs]
        if not self.reward_funcs:
            raise ValueError("At least one reward function is required.")
        self.reward_func_names = [get_callable_name(f) for f in self.reward_funcs]
        self.tokenizer = processing_class
        self.num_generations = num_generations
        self.chat_template_kwargs = chat_template_kwargs or {}
        self.old_logprobs_source = old_logprobs_source
        self.pad_token_id = pad_token_id
        self.max_token_len_per_gpu = max_token_len_per_gpu
        self._sampling_params: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "logprobs": 0,  # sampled token only
        }
        if min_p is not None:
            self._sampling_params["min_p"] = min_p

        self.rollout_buffer: queue.Queue = queue.Queue(maxsize=queue_maxsize)

        self._model_version = 0
        self._version_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat = time.time()
        self._error: BaseException | None = None

    # ── RolloutWorkerProtocol ────────────────────────────────────────────
    @property
    def model_version(self) -> int:
        with self._version_lock:
            return self._model_version

    def update_model_version(self, model_version: int) -> None:
        with self._version_lock:
            self._model_version = int(model_version)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._heartbeat = time.time()
        self._thread = threading.Thread(target=self._run, name="arctic-rollout-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
            self._thread = None

    def check_health(self, stale_after_s: float) -> None:
        if self._error is not None:
            raise RuntimeError("Arctic rollout worker thread failed; see chained exception.") from self._error
        if self._thread is not None and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("Arctic rollout worker thread has stopped unexpectedly.")
        age = time.time() - self._heartbeat
        if age > stale_after_s:
            raise RuntimeError(f"Arctic rollout worker heartbeat stale: {age:.0f}s > {stale_after_s:.0f}s.")

    # ── internals ────────────────────────────────────────────────────────
    def _run(self) -> None:
        try:
            group_id = 0
            while not self._stop.is_set():
                produced = False
                for row in self.dataset:
                    if self._stop.is_set():
                        return
                    self._heartbeat = time.time()
                    self._rollout_group(row, group_id)
                    group_id += 1
                    produced = True
                if not produced:
                    time.sleep(0.1)
        except BaseException as e:  # noqa: BLE001
            self._error = e
            raise

    def _render_prompt(self, prompt: Any) -> tuple[str, list[int]]:
        """Text prompt plus token ids for the training row."""
        if isinstance(prompt, str):
            text = prompt
        else:
            text = self.tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs
            )
        ids = list(self.tokenizer(text, add_special_tokens=False)["input_ids"])
        return text, ids

    def _rollout_group(self, row: dict, group_id: int) -> None:
        prompt = row["prompt"]
        prompt_text, prompt_ids = self._render_prompt(prompt)

        results = self.client.generate(
            prompts=[prompt_text for _ in range(self.num_generations)],
            sampling_params=dict(self._sampling_params),
        )
        self._heartbeat = time.time()

        completions: list[list[dict[str, str]]] = []
        completion_ids: list[list[int]] = []
        rows: list[dict[str, Any]] = []
        for result in results:
            token_ids = [int(t) for t in result["token_ids"]]
            text = result.get("text", "")
            lps = result.get("logprobs")
            if lps is None:
                raise RuntimeError(
                    "Arctic generate returned no logprobs; set sampling_params['logprobs'] (GRPO needs "
                    "generator log-probs as old_log_probs)."
                )
            token_logprobs = [_sampled_logprob(lps[j], token_ids[j]) for j in range(len(token_ids))]

            n_prompt = len(prompt_ids)
            rows.append(
                {
                    "input_ids": list(prompt_ids) + token_ids,
                    "completion_mask": [0] * n_prompt + [1] * len(token_ids),
                    "old_log_probs": [0.0] * n_prompt + token_logprobs,
                }
            )
            completions.append([{"role": "assistant", "content": text}])
            completion_ids.append(token_ids)

        if self.old_logprobs_source == "trainer" and rows:
            n_prompt = len(prompt_ids)
            engine_lp = engine_old_log_probs(
                self.client,
                [r["input_ids"] for r in rows],
                temperature=self._sampling_params["temperature"],
                pad_token_id=self.pad_token_id,
                rollout_n=self.num_generations,
                max_token_len_per_gpu=self.max_token_len_per_gpu,
            )
            self._heartbeat = time.time()
            for r, lp in zip(rows, engine_lp, strict=True):
                r["old_log_probs"] = [0.0] * n_prompt + list(lp[n_prompt:])

        rewards, per_func_rewards, reward_std = self._score(prompt, completions, completion_ids, row)
        advantages = self._advantages(rewards)

        model_version = self.model_version
        for i, (r, adv) in enumerate(zip(rewards, advantages, strict=True)):
            metrics = {
                "reward": float(r),
                "reward_std": float(reward_std),
                **{f"rewards/{name}": float(per_func_rewards[k, i]) for k, name in enumerate(self.reward_func_names)},
            }
            sample = RolloutSample(
                prompt=prompt,
                completion=completions[i],
                input_ids=rows[i]["input_ids"],
                completion_mask=rows[i]["completion_mask"],
                old_log_probs=rows[i]["old_log_probs"],
                advantage=float(adv),
                model_version=model_version,
                group_id=group_id,
                metrics=metrics,
            )
            self._put(sample)

    def _score(
        self, prompt: Any, completions: list, completion_ids: list, row: dict
    ) -> tuple[np.ndarray, np.ndarray, float]:
        n = len(completions)
        reward_kwargs = {key: [row[key]] * n for key in row if key not in {"prompt", "completion", "completion_ids"}}
        kwargs = dict(prompts=[prompt] * n, completions=completions, completion_ids=completion_ids, **reward_kwargs)

        all_rewards = []
        for func in self.reward_funcs:
            out = func(**kwargs)
            if inspect.isawaitable(out):
                out = asyncio.run(out)
            all_rewards.append([r if r is not None else float("nan") for r in out])

        per_func = np.array(all_rewards, dtype=float)  # [num_funcs, n]
        all_nan = np.all(np.isnan(per_func), axis=0)
        rewards = np.nansum(per_func, axis=0)
        rewards[all_nan] = np.nan
        scored = rewards[~np.isnan(rewards)]
        reward_std = float(scored.std()) if scored.size else float("nan")
        return rewards, per_func, reward_std

    @staticmethod
    def _advantages(rewards: np.ndarray) -> np.ndarray:
        advantages = np.zeros_like(rewards)
        mask = ~np.isnan(rewards)
        if mask.any():
            scored = rewards[mask]
            advantages[mask] = (scored - scored.mean()) / (scored.std() + 1e-8)
        return advantages

    def _put(self, sample: RolloutSample) -> None:
        while not self._stop.is_set():
            try:
                self.rollout_buffer.put(sample, timeout=0.5)
                return
            except queue.Full:
                continue
