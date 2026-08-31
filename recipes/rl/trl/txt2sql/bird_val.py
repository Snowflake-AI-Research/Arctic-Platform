#!/usr/bin/env python
# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Held-out greedy BIRD val (n=1, temp=0), logged as verl ``val-core`` / ``val-aux``.

Do not route these keys through ``AsyncGRPOTrainer.log`` (HF prefixes ``train/``).
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from transformers import TrainerCallback

import bird_task

VAL_CORE_REWARD = "val-core/bird/reward/mean@1"
VAL_AUX_EXEC = "val-aux/bird/execution_success/mean@1"
VAL_AUX_FORMAT = "val-aux/bird/format_correct/mean@1"
VAL_AUX_SCORE = "val-aux/bird/score/mean@1"
VAL_AUX_N = "val-aux/bird/n_prompts"
VAL_AUX_DROPPED = "val-aux/bird/dropped_overlong"
VAL_AUX_TRUNC = "val-aux/bird/truncated_frac"
VAL_AUX_TIME = "val-aux/bird/time_s"

_VAL_CANDIDATES = (
    "/data/snowflakesql/txt2sql/val.parquet",
    "/data/snowflakesql/txt2sql_std32k/val.parquet",
)


def val_env_defaults() -> dict[str, Any]:
    return {
        # Opt-in (0 = off).
        "val_every": int(os.environ.get("VAL_EVERY", "0")),
        "val_parquet": (os.environ.get("BIRD_VAL_PARQUET") or "").strip(),
        "val_max_samples": int(os.environ.get("VAL_MAX_SAMPLES", "0")),
        "val_at_step0": os.environ.get("VAL_AT_STEP0", "0") not in ("0", "false", "False"),
        "val_overlong": (os.environ.get("VAL_OVERLONG") or "drop").strip() or "drop",
        "val_gen_chunk": int(os.environ.get("VAL_GEN_CHUNK", "32")),
        "val_http_workers": int(os.environ.get("VAL_HTTP_WORKERS", "8")),
    }


def resolve_val_parquet(explicit: str | None = None) -> str:
    """Prefer ``BIRD_VAL_PARQUET``, then sibling of train parquet, then known paths."""
    for cand in (
        (explicit or "").strip(),
        (os.environ.get("BIRD_VAL_PARQUET") or "").strip(),
    ):
        if cand and os.path.exists(cand):
            return cand
    train = (os.environ.get("BIRD_TRAIN_PARQUET") or "").strip()
    if train:
        sibling = os.path.join(os.path.dirname(train), "val.parquet")
        if os.path.exists(sibling):
            return sibling
    for cand in _VAL_CANDIDATES:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "BIRD val.parquet not found. Set BIRD_VAL_PARQUET or place val.parquet next to the train parquet."
    )


def render_prompt(
    tokenizer: Any,
    prompt: Any,
    chat_template_kwargs: dict | None = None,
) -> tuple[str, list[int]]:
    """Render prompt text + ids. Thinking stays on (do not set ``enable_thinking=False``)."""
    kw = dict(chat_template_kwargs or {})
    if isinstance(prompt, str):
        text = prompt
    else:
        text = tokenizer.apply_chat_template(
            prompt, add_generation_prompt=True, tokenize=False, **kw
        )
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, ids


def to_verl_val_metrics(
    details: list[dict[str, float]],
    *,
    n_prompts: int,
    dropped_overlong: int = 0,
    truncated: int = 0,
    considered: int = 0,
    time_s: float = 0.0,
) -> dict[str, float]:
    """Means over scored prompts. ``reward`` is verl's core name for ``score``."""
    n = len(details)
    if n == 0:
        reward = exec_s = fmt = score = 0.0
    else:
        reward = sum(float(d.get("score", 0.0)) for d in details) / n
        exec_s = sum(float(d.get("execution_success", 0.0)) for d in details) / n
        fmt = sum(float(d.get("format_correct", 0.0)) for d in details) / n
        score = reward
    denom = considered if considered > 0 else n + dropped_overlong
    return {
        VAL_CORE_REWARD: float(reward),
        VAL_AUX_EXEC: float(exec_s),
        VAL_AUX_FORMAT: float(fmt),
        VAL_AUX_SCORE: float(score),
        VAL_AUX_N: float(n_prompts if n_prompts else n),
        VAL_AUX_DROPPED: float(dropped_overlong),
        VAL_AUX_TRUNC: float(truncated / denom) if denom else 0.0,
        VAL_AUX_TIME: float(time_s),
    }


def prepare_val_rows(
    parquet_path: str,
    tokenizer: Any,
    *,
    max_samples: int = 0,
    seed: int = 42,
    max_prompt_tokens: int,
    overlong: str = "drop",
    chat_template_kwargs: dict | None = None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    """Load val, optional seeded shuffle+cap, drop or left-truncate overlong prompts.

    Returns ``(rows, dropped_overlong, truncated, n_after_slice)``.
    Each kept row has ``_prompt_text`` and ``_prompt_ids``.
    """
    ds = bird_task.load_bird_dataset(parquet_path, num_prompts=-1)
    rows = [dict(ds[i]) for i in range(len(ds))]
    if max_samples and max_samples > 0:
        rng = random.Random(seed)
        rng.shuffle(rows)
        rows = rows[:max_samples]
    n_after_slice = len(rows)

    kept: list[dict[str, Any]] = []
    dropped = 0
    truncated = 0
    for row in rows:
        text, ids = render_prompt(tokenizer, row["prompt"], chat_template_kwargs)
        if len(ids) > max_prompt_tokens:
            if overlong == "left_truncate":
                ids = ids[-max_prompt_tokens:]
                text = tokenizer.decode(ids, skip_special_tokens=False)
                row = dict(row)
                row["_prompt_text"] = text
                row["_prompt_ids"] = ids
                truncated += 1
                kept.append(row)
            else:
                dropped += 1
        else:
            row = dict(row)
            row["_prompt_text"] = text
            row["_prompt_ids"] = ids
            kept.append(row)
    return kept, dropped, truncated, n_after_slice


def assert_val_db_paths(rows: list[dict[str, Any]]) -> None:
    missing = [
        r.get("db_path") for r in rows
        if not r.get("db_path") or not os.path.exists(str(r.get("db_path")))
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} val db_path(s) missing on this node, e.g. {missing[:3]!r}"
        )


def log_val_metrics(trainer: Any, metrics: dict[str, float], step: int) -> None:
    """Append to ``log_history`` and ``wandb.log`` without ``trainer.log`` / ``step=``."""
    record = {"step": int(step), **{k: float(v) for k, v in metrics.items()}}
    trainer.state.log_history.append(record)
    try:
        import wandb

        if wandb.run is not None:
            wandb.log({**{k: float(v) for k, v in metrics.items()}, "train/global_step": int(step)})
    except Exception as exc:  # noqa: BLE001
        print(f"[val] wandb.log skipped: {exc}", flush=True)
    pretty = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    print(f"[val] step={step} {pretty}", flush=True)


def generate_vllm_greedy(
    rows: list[dict[str, Any]],
    *,
    base_url: str,
    model: str,
    tokenizer: Any,
    max_tokens: int,
    chunk: int = 32,
    workers: int = 8,
    timeout: float = 1800.0,
) -> list[str]:
    """C1: POST token-id prompts to stock ``vllm serve`` ``/v1/completions``."""
    if not rows:
        return []
    model_name = _vllm_model_name(base_url, model)

    def _one(row: dict[str, Any]) -> str:
        ids = row.get("_prompt_ids") or render_prompt(tokenizer, row["prompt"])[1]
        payload = {
            "model": model_name,
            "prompt": ids,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "return_token_ids": True,
        }
        body = json.dumps(payload).encode()
        last: Exception | None = None
        for _attempt in range(3):
            try:
                attempt_req = urllib.request.Request(
                    f"{base_url.rstrip('/')}/v1/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(attempt_req, timeout=timeout) as resp:
                    out = json.loads(resp.read())
                choice = (out.get("choices") or [{}])[0]
                text = choice.get("text") or ""
                if not text:
                    tok_ids = choice.get("token_ids") or []
                    if tok_ids:
                        text = tokenizer.decode(tok_ids, skip_special_tokens=True)
                return str(text)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
                time.sleep(1.0)
        raise RuntimeError(f"vLLM greedy val failed after 3 attempts: {last}") from last

    texts = [""] * len(rows)
    workers = max(1, min(workers, len(rows)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bird-val-http") as pool:
        futs = [pool.submit(_one, row) for row in rows]
        for i, fut in enumerate(futs):
            texts[i] = fut.result()
    del chunk
    return texts


def generate_arctic_greedy(
    rows: list[dict[str, Any]],
    *,
    client: Any,
    tokenizer: Any,
    max_tokens: int,
    chunk: int = 32,
    chat_template_kwargs: dict | None = None,
) -> list[str]:
    """C2/C3: ``ArcticRLClient.generate`` with greedy sampling, no logprobs."""
    if not rows:
        return []
    out: list[str] = []
    step = max(1, chunk)
    for i in range(0, len(rows), step):
        batch = rows[i : i + step]
        prompts = []
        for row in batch:
            text = row.get("_prompt_text")
            if not text:
                text, _ids = render_prompt(tokenizer, row["prompt"], chat_template_kwargs)
            prompts.append(text)
        results = client.generate(
            prompts=prompts,
            sampling_params={
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": max_tokens,
            },
        )
        if len(results) != len(prompts):
            raise RuntimeError(f"Arctic greedy val returned {len(results)} results, expected {len(prompts)}")
        for result in results:
            text = result.get("text") or ""
            if not text:
                tok_ids = result.get("token_ids") or []
                if tok_ids:
                    text = tokenizer.decode([int(t) for t in tok_ids], skip_special_tokens=True)
            out.append(str(text))
    return out


def score_texts(rows: list[dict[str, Any]], texts: list[str]) -> list[dict[str, float]]:
    if len(texts) != len(rows):
        raise ValueError(f"score_texts: {len(texts)} texts vs {len(rows)} rows")
    completions = [[{"role": "assistant", "content": t}] for t in texts]
    return bird_task.sql_reward_detailed(
        prompts=[r.get("prompt") for r in rows],
        completions=completions,
        completion_ids=[[] for _ in rows],
        ground_truth=[r.get("ground_truth") for r in rows],
        db_path=[r.get("db_path") for r in rows],
        data_source=[r.get("data_source", "bird") for r in rows],
    )


def _vllm_model_name(base_url: str, fallback: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/models", timeout=30) as resp:
            data = json.loads(resp.read())
        return str(data["data"][0]["id"])
    except Exception:  # noqa: BLE001
        return fallback


def _reduce_details(
    details: list[dict[str, float]],
    accelerator: Any,
) -> list[dict[str, float]]:
    """All-reduce per-sample sums; reconstruct a 1-element mean via counts.

    Ranks score disjoint shards. We only need means, so reduce (sum_score, sum_exec,
    sum_fmt, n) and expand to a single synthetic detail list of length n on rank 0.
    Other ranks get an empty list.
    """
    import torch

    n = float(len(details))
    s_score = sum(float(d.get("score", 0.0)) for d in details)
    s_exec = sum(float(d.get("execution_success", 0.0)) for d in details)
    s_fmt = sum(float(d.get("format_correct", 0.0)) for d in details)
    buf = torch.tensor([s_score, s_exec, s_fmt, n], dtype=torch.float64, device=accelerator.device)
    if accelerator.num_processes > 1:
        buf = accelerator.reduce(buf, reduction="sum")
    if not accelerator.is_main_process:
        return []
    tot_n = int(buf[3].item())
    if tot_n <= 0:
        return []
    # Repeat the global mean tot_n times so to_verl_val_metrics divides back to the mean.
    mean_score = float(buf[0].item()) / tot_n
    mean_exec = float(buf[1].item()) / tot_n
    mean_fmt = float(buf[2].item()) / tot_n
    one = {"score": mean_score, "execution_success": mean_exec, "format_correct": mean_fmt}
    return [one] * tot_n


class BirdValCallback(TrainerCallback):
    """Greedy val after weight sync. Register with ``trainer.add_callback`` *after* construction."""

    def __init__(
        self,
        *,
        rows: list[dict[str, Any]],
        tokenizer: Any,
        generate_fn: Callable[[list[dict[str, Any]]], list[str]],
        val_every: int,
        dropped_overlong: int = 0,
        truncated: int = 0,
        considered: int = 0,
        val_at_step0: bool = False,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.generate_fn = generate_fn
        self.val_every = val_every
        self.dropped_overlong = dropped_overlong
        self.truncated = truncated
        self.considered = considered
        self.val_at_step0 = val_at_step0
        self._last_val_step: int | None = None
        self._trainer: Any = None

    def on_train_begin(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, control, kwargs
        if self.val_at_step0 and self._last_val_step is None:
            self._run(state.global_step or 0)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, control, kwargs
        if self.val_every <= 0:
            return
        step = int(state.global_step)
        if step <= 0 or step % self.val_every != 0:
            return
        self._run(step)

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        del args, control, kwargs
        if self.val_every <= 0:
            return
        step = int(state.global_step)
        if step > 0 and step != self._last_val_step:
            self._run(step)

    def _run(self, step: int) -> None:
        trainer = self._trainer
        if trainer is None:
            return
        accelerator = trainer.accelerator
        t0 = time.perf_counter()
        rank = int(accelerator.process_index)
        world = int(accelerator.num_processes)
        mine = self.rows[rank::world]
        texts = self.generate_fn(mine) if mine else []
        details = score_texts(mine, texts) if mine else []
        merged = _reduce_details(details, accelerator)
        time_s = time.perf_counter() - t0
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            metrics = to_verl_val_metrics(
                merged,
                n_prompts=len(self.rows),
                dropped_overlong=self.dropped_overlong,
                truncated=self.truncated,
                considered=self.considered,
                time_s=time_s,
            )
            log_val_metrics(trainer, metrics, step)
        self._last_val_step = step
        if accelerator.num_processes > 1:
            accelerator.wait_for_everyone()


def attach_bird_val_callback(trainer: Any, callback: BirdValCallback) -> BirdValCallback:
    """Append after construction so val runs *after* ``StepIntervalCallback(_sync_weight)``."""
    callback._trainer = trainer
    trainer.add_callback(callback)
    cbs = list(trainer.callback_handler.callbacks)
    sync_idx = None
    val_idx = None
    for i, cb in enumerate(cbs):
        fn = getattr(cb, "fn", None)
        fn_name = getattr(fn, "__name__", "") if fn is not None else ""
        if type(cb).__name__ == "StepIntervalCallback" and fn_name == "_sync_weight":
            sync_idx = i
        if isinstance(cb, BirdValCallback):
            val_idx = i
    if sync_idx is None:
        raise RuntimeError("weight-sync StepIntervalCallback(_sync_weight) not found; cannot order val after sync")
    if val_idx is None or val_idx <= sync_idx:
        raise RuntimeError(
            f"BirdValCallback must be registered after weight sync (sync_idx={sync_idx} val_idx={val_idx})"
        )
    print(f"[val] callback attached after weight sync (sync_idx={sync_idx} val_idx={val_idx}) "
          f"n={len(callback.rows)} every={callback.val_every}", flush=True)
    return callback


def maybe_attach_val(
    trainer: Any,
    *,
    tokenizer: Any,
    generate_fn: Callable[[list[dict[str, Any]]], list[str]],
    max_completion_length: int,
    max_model_len: int,
    chat_template_kwargs: dict | None = None,
    seed: int = 42,
    val_every: int | None = None,
    val_parquet: str | None = None,
    val_max_samples: int | None = None,
    val_at_step0: bool | None = None,
    val_overlong: str | None = None,
) -> BirdValCallback | None:
    """No-op when ``VAL_EVERY=0``. Otherwise load val, fail fast on missing DBs, attach callback."""
    env = val_env_defaults()
    every = env["val_every"] if val_every is None else val_every
    if every <= 0:
        print("[val] disabled (VAL_EVERY=0)", flush=True)
        return None
    parquet = resolve_val_parquet(val_parquet if val_parquet is not None else env["val_parquet"] or None)
    max_samples = env["val_max_samples"] if val_max_samples is None else val_max_samples
    at0 = env["val_at_step0"] if val_at_step0 is None else val_at_step0
    overlong = env["val_overlong"] if val_overlong is None else val_overlong
    max_prompt = max(1, int(max_model_len) - int(max_completion_length))
    rows, dropped, truncated, considered = prepare_val_rows(
        parquet,
        tokenizer,
        max_samples=max_samples,
        seed=seed,
        max_prompt_tokens=max_prompt,
        overlong=overlong,
        chat_template_kwargs=chat_template_kwargs,
    )
    assert_val_db_paths(rows)
    print(
        f"[val] parquet={parquet} kept={len(rows)} dropped_overlong={dropped} truncated={truncated} "
        f"considered={considered} max_prompt={max_prompt} every={every}",
        flush=True,
    )
    cb = BirdValCallback(
        rows=rows,
        tokenizer=tokenizer,
        generate_fn=generate_fn,
        val_every=every,
        dropped_overlong=dropped,
        truncated=truncated,
        considered=considered,
        val_at_step0=at0,
    )
    return attach_bird_val_callback(trainer, cb)
